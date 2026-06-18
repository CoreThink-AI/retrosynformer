"""Merge all hypertune-*large* study databases into a single merged study.

Creates:
  results/hypermerge/<merged_name>/study.db          — Optuna-compatible merged DB
  results/hypermerge/<merged_name>/trial_NNN/        — symlinks to original trial dirs
  results/hypermerge/<merged_name>/SOURCES.txt       — provenance record

Fixed architecture params that *vary across studies* (e.g. hidden_size=512/640/1024)
are promoted to additional CategoricalDistribution dimensions so the TPE surrogate
can learn their effect on the objective.  Dims that are already in the Optuna search
space (e.g. n_heads, n_layers) are filled from each trial's model.config.yaml for
trials that didn't search them.

RUNNING / WAITING / FAIL trials are added as FAIL (no objective value means
the TPE sampler ignores them; their training data is still accessible via symlinks).

Usage:
    python scripts/merge_large_studies.py
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import FrozenTrial, TrialState

from retrosynformer.models_optuna import (
    Study,
    Trial,
    _collect_param_info,
    connect,
    study_config_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR  = Path("results")
HYPERMERGE   = RESULTS_DIR / "hypermerge"
GLOB_PATTERN = str(RESULTS_DIR / "hypertune-*large*" / "study.db")

# Config keys that are Optuna meta-settings or output paths — not model hyperparameters.
# Don't promote these to merged dimensions.
_EXCLUDE_FIXED_LEAVES = {
    "n_trials",            # Optuna budget setting
    "objective_metric",    # Optuna objective name
    "objective_direction", # Optuna direction string
    "eval_routes_frequency",
    "eval_routes_at_end",
    "results_path",        # output path
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_optuna_dist(rec: dict):
    """Convert a merged-space distribution record to an Optuna distribution."""
    dtype = rec["dist_type"]
    if dtype == "CategoricalDistribution":
        return CategoricalDistribution(choices=rec["choices"])
    if dtype == "FloatDistribution":
        return FloatDistribution(
            low=rec["low"], high=rec["high"], log=bool(rec.get("log") or False)
        )
    if dtype == "IntDistribution":
        return IntDistribution(
            low=int(rec["low"]),
            high=int(rec["high"]),
            log=bool(rec.get("log") or False),
            step=int(rec.get("step") or 1),
        )
    raise ValueError(f"Unknown distribution type: {dtype!r}")


def _trial_state(state_str: str) -> TrialState:
    return {
        "COMPLETE": TrialState.COMPLETE,
        "FAIL":     TrialState.FAIL,
        "RUNNING":  TrialState.FAIL,   # no objective → FAIL in merged DB
        "WAITING":  TrialState.FAIL,
        "PRUNED":   TrialState.PRUNED,
    }[state_str]


def _build_frozen_trial(
    trial: Trial,
    merged_distributions: dict,
    config_flat: dict,
) -> FrozenTrial:
    """Build an Optuna FrozenTrial from our ORM Trial.

    Strategy for filling hyperparameter dimensions:

    1. Start with decoded params from Optuna ``trial_params`` (the searched dims).
    2. For any merged dim NOT yet in params, look it up in ``config_flat``
       (the flattened model.config.yaml for this trial) by:
         a. exact key match (dim_name == config_key)
         b. leaf-name match (dim_name == last segment of config_key after '.')
       This fills two cases:
         - Dims that are searched in *some* studies but fixed in others
           (e.g. n_heads=5 fixed in nonuniform, searched in baseline).
         - Newly promoted dims that vary across studies
           (e.g. hidden_size, head_dim, batch_size).

    RUNNING/WAITING trials are stored as FAIL so the TPE sampler ignores them.
    """
    state = _trial_state(trial.state)
    value = trial.objective_value if state == TrialState.COMPLETE else None

    # Step 1: decoded params from trial_params
    params: dict = {p.param_name: p.decoded_value for p in trial.params}
    distributions: dict = {}
    for p in trial.params:
        rec = merged_distributions.get(p.param_name)
        if rec:
            distributions[p.param_name] = _to_optuna_dist(rec)
        else:
            d = p.distribution
            distributions[p.param_name] = _to_optuna_dist({
                "dist_type": d["name"], **d["attributes"]
            })

    # Step 2: fill missing dims from config_flat
    if config_flat:
        # Pre-build a leaf-name → config value lookup for fast access
        leaf_to_val: dict[str, object] = {}
        for k, v in config_flat.items():
            leaf = k.rsplit(".", 1)[-1]
            if leaf not in leaf_to_val:  # first occurrence wins (most specific)
                leaf_to_val[leaf] = v

        for dim_name, rec in merged_distributions.items():
            if dim_name in params:
                continue  # already filled from trial_params
            # Look up: exact key, then leaf match
            val = config_flat.get(dim_name, leaf_to_val.get(dim_name))
            if val is None:
                continue
            # Validate value falls within the distribution
            if rec["dist_type"] == "CategoricalDistribution":
                if val not in rec["choices"]:
                    continue
            params[dim_name] = val
            distributions[dim_name] = _to_optuna_dist(rec)

    dt_start = trial.datetime_start or datetime.now(timezone.utc)
    # Optuna requires datetime_complete for COMPLETE, FAIL, and PRUNED.
    if state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED):
        dt_complete = trial.datetime_complete or dt_start
    else:
        dt_complete = None

    return FrozenTrial(
        number=-1,
        trial_id=-1,
        state=state,
        value=value,
        values=None,
        datetime_start=dt_start,
        datetime_complete=dt_complete,
        params=params,
        distributions=distributions,
        intermediate_values={},
        system_attrs={},
        user_attrs={"source_study": trial.study.study_name,
                    "source_trial": trial.number},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    db_paths = sorted(glob.glob(GLOB_PATTERN))
    if not db_paths:
        print(f"No study.db files matching {GLOB_PATTERN!r}")
        sys.exit(1)

    sessions = {db: connect(db) for db in db_paths}

    # -----------------------------------------------------------------------
    # Study names and merge name
    # -----------------------------------------------------------------------

    source_names = []
    for db in db_paths:
        study = sessions[db].query(Study).first()
        source_names.append(study.study_name if study else Path(db).parent.name)

    merged_name = "+".join(n.replace("baseline-", "bl-") for n in source_names)
    if len(merged_name) > 80:
        merged_name = "large-merged"

    print(f"Sources ({len(db_paths)}):")
    for db in db_paths:
        s   = sessions[db].query(Study).first()
        t   = sessions[db].query(Trial).all()
        cnt = {}
        for x in t:
            cnt[x.state] = cnt.get(x.state, 0) + 1
        print(f"  {s.study_name}: {cnt}")

    # -----------------------------------------------------------------------
    # Step 1: Merge Optuna trial_params search space (existing approach)
    # -----------------------------------------------------------------------

    merged_space: dict[str, dict] = {}
    for session in sessions.values():
        for name, rec in _collect_param_info(session).items():
            if name not in merged_space:
                merged_space[name] = dict(rec)
                if rec["choices"] is not None:
                    merged_space[name]["choices"] = set(rec["choices"])
            else:
                ex = merged_space[name]
                if rec["dist_type"] == "CategoricalDistribution":
                    ex["choices"] = (ex["choices"] or set()) | (rec["choices"] or set())
                else:
                    if rec["low"]  is not None: ex["low"]  = min(ex["low"]  or rec["low"],  rec["low"])
                    if rec["high"] is not None: ex["high"] = max(ex["high"] or rec["high"], rec["high"])

    # Sort categorical choices
    for rec in merged_space.values():
        if rec["dist_type"] == "CategoricalDistribution" and rec["choices"]:
            try:
                rec["choices"] = sorted(rec["choices"], key=lambda x: (0, float(str(x))))
            except (ValueError, TypeError):
                rec["choices"] = sorted(rec["choices"], key=str)

    # -----------------------------------------------------------------------
    # Step 2: Load per-trial config flats and promote varying fixed params
    # -----------------------------------------------------------------------

    # config_by_trial[(db, trial_number)] = flat config dict (fixed params only)
    config_by_trial: dict[tuple[str, int], dict] = {}
    for db in db_paths:
        for trial_num, flat in study_config_params(db, sessions[db]).items():
            config_by_trial[(db, trial_num)] = flat

    # Collect all distinct values per config dot-key across ALL trials+studies
    vals_by_key: dict[str, list] = {}
    for flat in config_by_trial.values():
        for key, val in flat.items():
            vals_by_key.setdefault(key, []).append(val)

    # Find keys whose leaf name is NOT already a merged_space dim and that vary
    existing_dim_leaves = {name.rsplit(".", 1)[-1] for name in merged_space} | set(merged_space)
    promoted: dict[str, list] = {}  # leaf_name -> sorted distinct values

    for dot_key, vals in vals_by_key.items():
        leaf = dot_key.rsplit(".", 1)[-1]
        if leaf in _EXCLUDE_FIXED_LEAVES:
            continue
        if leaf in existing_dim_leaves:
            continue  # already a searched dim; filled dynamically in _build_frozen_trial
        distinct = sorted(set(vals), key=str)
        if len(distinct) > 1:
            promoted[leaf] = distinct

    # Add promoted dims to merged_space as CategoricalDistribution
    for leaf, distinct in promoted.items():
        # Try numeric sort
        try:
            distinct = sorted(distinct, key=float)
        except (ValueError, TypeError):
            distinct = sorted(distinct, key=str)
        merged_space[leaf] = {
            "dist_type": "CategoricalDistribution",
            "choices": distinct,
            "low": None, "high": None, "log": None, "step": None,
        }

    # Report merged space
    print(f"\nMerged search space ({len(merged_space)} dimensions):")
    optuna_dims  = sorted(set(merged_space) - set(promoted))
    promoted_dims = sorted(promoted)
    for name in optuna_dims:
        rec = merged_space[name]
        if rec["dist_type"] == "CategoricalDistribution":
            print(f"  {name}: choices={rec['choices']}")
        else:
            print(f"  {name}: [{rec['low']}, {rec['high']}] {'(log)' if rec.get('log') else ''}")
    if promoted_dims:
        print(f"  --- promoted from fixed config params ---")
        for name in promoted_dims:
            print(f"  {name}: choices={merged_space[name]['choices']}")

    # -----------------------------------------------------------------------
    # Step 3: Collect all trials
    # -----------------------------------------------------------------------

    all_trials: list[tuple[str, Trial]] = []
    for db in db_paths:
        for trial in sessions[db].query(Trial).order_by(Trial.number).all():
            all_trials.append((db, trial))

    complete = [t for _, t in all_trials if t.state == "COMPLETE"]
    print(f"\nTotal trials to merge: {len(all_trials)}")
    print(f"  COMPLETE (carry objective): {len(complete)}")
    print(f"  Other (stored as FAIL):     {len(all_trials) - len(complete)}")

    # -----------------------------------------------------------------------
    # Step 4: Create merged study.db
    # -----------------------------------------------------------------------

    out_dir = HYPERMERGE / merged_name
    out_dir.mkdir(parents=True, exist_ok=True)

    db_out = out_dir / "study.db"
    if db_out.exists():
        db_out.unlink()
        print(f"\nRemoved existing {db_out}")

    storage = f"sqlite:///{db_out}"
    merged_study = optuna.create_study(
        study_name=merged_name,
        storage=storage,
        direction="maximize",
    )

    print(f"\nAdding trials:")
    added = 0
    skipped_trials = []

    for db, trial in all_trials:
        label = f"{Path(db).parent.name}/trial_{trial.number:03d}"
        cfg   = config_by_trial.get((db, trial.number), {})
        try:
            frozen = _build_frozen_trial(trial, merged_space, cfg)
            merged_study.add_trial(frozen)
            added += 1
        except Exception as exc:
            print(f"  WARNING: skipped {label}: {exc}")
            skipped_trials.append((db, trial))

    print(f"  Added {added} / {len(all_trials)} ({len(skipped_trials)} skipped)")

    # -----------------------------------------------------------------------
    # Step 5: Create symlinks keyed by provenance from merged DB
    # -----------------------------------------------------------------------

    src_dir_map: dict[tuple[str, int], Path] = {}
    for db, trial in all_trials:
        sname    = sessions[db].query(Study).first().study_name
        orig_dir = Path(db).parent / f"trial_{trial.number:03d}"
        if orig_dir.exists():
            src_dir_map[(sname, trial.number)] = orig_dir

    loaded     = optuna.load_study(study_name=merged_name, storage=storage)
    symlink_map: list[tuple[int, Path]] = []
    for mt in loaded.trials:
        sname   = mt.user_attrs.get("source_study")
        src_num = mt.user_attrs.get("source_trial")
        orig    = src_dir_map.get((sname, src_num))
        if orig:
            symlink_map.append((mt.number, orig))

    print(f"\nCreating {len(symlink_map)} symlinks:")
    for merged_num, orig_dir in sorted(symlink_map):
        link = out_dir / f"trial_{merged_num:03d}"
        if link.exists() or link.is_symlink():
            link.unlink()
        rel = os.path.relpath(orig_dir, out_dir)
        link.symlink_to(rel)
        print(f"  trial_{merged_num:03d}/ -> {rel}")

    # -----------------------------------------------------------------------
    # Step 6: SOURCES.txt
    # -----------------------------------------------------------------------

    sources_txt = out_dir / "SOURCES.txt"
    linked_nums = {m for m, _ in symlink_map}
    with sources_txt.open("w") as f:
        f.write(f"Merged study: {merged_name}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n\n")
        f.write("Source studies:\n")
        for db in db_paths:
            st  = sessions[db].query(Study).first()
            ts  = sessions[db].query(Trial).all()
            cnt = {}
            for t in ts:
                cnt[t.state] = cnt.get(t.state, 0) + 1
            f.write(f"  {st.study_name}: {cnt}\n")
        f.write(f"\nMerged search space ({len(merged_space)} dims):\n")
        for name in optuna_dims:
            rec = merged_space[name]
            if rec["dist_type"] == "CategoricalDistribution":
                f.write(f"  {name} (searched): choices={rec['choices']}\n")
            else:
                f.write(f"  {name} (searched): [{rec['low']}, {rec['high']}]\n")
        for name in promoted_dims:
            f.write(f"  {name} (promoted from config): choices={merged_space[name]['choices']}\n")
        f.write("\nTrial provenance:\n")
        for mt in sorted(loaded.trials, key=lambda t: t.number):
            src     = mt.user_attrs.get("source_study", "?")
            src_num = mt.user_attrs.get("source_trial", "?")
            state   = mt.state.name
            has_dir = mt.number in linked_nums
            dims_filled = len(mt.params)
            f.write(f"  trial_{mt.number:03d} [{state:8s}] dims={dims_filled:2d} "
                    f"<- {src}/trial_{src_num}"
                    f"{' (dir)' if has_dir else ''}\n")
        if skipped_trials:
            f.write("\nSkipped (could not add to DB):\n")
            for db, trial in skipped_trials:
                st = sessions[db].query(Study).first()
                f.write(f"  {st.study_name}/trial_{trial.number:03d} [{trial.state}]\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    complete_v = [t for t in loaded.trials if t.state == TrialState.COMPLETE]
    print(f"\nMerged study: {merged_name!r}")
    print(f"  study.db : {db_out}")
    print(f"  sources  : {sources_txt}")
    print(f"  Total trials in DB: {len(loaded.trials)}")
    print(f"  COMPLETE  : {len(complete_v)}")

    print(f"\nComplete trials (dims = hyperparameter dimensions filled):")
    for t in complete_v:
        src     = t.user_attrs.get("source_study", "?")
        src_num = t.user_attrs.get("source_trial", "?")
        print(f"  trial_{t.number:03d}  obj={t.value:.4f}  dims={len(t.params):2d}"
              f"  ← {src}/trial_{src_num}")
        for k, v in sorted(t.params.items()):
            print(f"    {k} = {v!r}")

    best = loaded.best_trial
    print(f"\nBest: trial_{best.number:03d}  value={best.value:.4f}")
    print(f"  params: {best.params}")

    for session in sessions.values():
        session.close()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
