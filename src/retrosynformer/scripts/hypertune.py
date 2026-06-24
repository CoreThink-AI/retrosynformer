#!/usr/bin/env python
"""Hyperparameter search for RetroSynFormer using Optuna.

The first trial is fixed (n_heads=4, n_layers=3, head_dim=256, lr=0.0005) based
on the best standard-dataset finding from trial_001 of the baseline-standard
study (taco, 2026-06-16).
Subsequent trials explore the search space defined in config["optuna"].

Usage:
    python scripts/hypertune.py -c results/config/small.yaml [--n-trials 20]

Results are written to results/hypertune-{study_name}/:
  study.db        Optuna storage (SQLite)
  run.jsonl       Structured log — one JSON object per line
  trial_NNN/      Per-trial model checkpoints and progress files

Multiple studies run simultaneously without conflict because each writes to
its own results/hypertune-{study_name}/ directory.  To add parallel workers
to the *same* study, pass an identical --study-name and --storage to each
process; Optuna coordinates via the shared SQLite file.

Config-driven search space (results/config/*.yaml under the "optuna" key):

    optuna:
      # list of values → suggest_categorical
      n_heads: [1, 2, 4, 8]

      # dict with choices → suggest_categorical (explicit form)
      head_dim:
        choices: [64, 128, 256]

      # dict with low/high (both int) → suggest_int; log: true for log-scale
      n_layers:
        low: 2
        high: 32
        log: true

      # dict with low/high (float) → suggest_float
      lr:
        low: 1.0e-4
        high: 1.0
        log: true

      # float range with a fixed step
      dropout:
        low: 0.0
        high: 0.3
        step: 0.01

      # random_seed — list of ints or an int low/high range (no log scale).
      # Maps to the seed= kwarg of runner.main() and is saved to
      # model.config.yaml as context.random_state before training begins.
      random_seed: [1, 2, 3, 42, 137]
      # or:
      random_seed:
        low: 1
        high: 1000

Any key in the optuna section must match a keyword argument accepted by
runner.main() (n_heads, n_layers, head_dim, lr, dropout, momentum, …),
with the exception of random_seed which is remapped to seed=.
"""
import argparse
import json
import logging
import os
import time

import optuna

import retrosynformer.trainer as _trainer_mod
from retrosynformer.etl import mask_dict_to_list
from retrosynformer.runner import main as train
from retrosynformer.runner import read_config
from retrosynformer.scripts import add_log_args, configure_logging, print_banner

CONFIG_PATH_DEFAULT = "results/config.yaml"

# Fixed first trial — n_heads=4/lr=0.0005 from best standard-dataset finding
# (trial_001 of baseline-standard-lr211-100epochs study, taco 2026-06-16).
# Every value here must be a valid choice in the corresponding optuna config list.
BASELINE_TRIAL = {"n_heads": 4, "n_layers": 3, "head_dim": 256, "lr": 0.0005, "dropout": 0.1}


# ---------------------------------------------------------------------------
# Structured JSONL logging
# ---------------------------------------------------------------------------

def _write(record: dict, run_jsonl: str) -> None:
    """Append one JSON record to run.jsonl (thread-safe at OS level)."""
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(run_jsonl, "a") as f:
        f.write(json.dumps(record) + "\n")


class _JsonlHandler(logging.Handler):
    """Redirect Python log records to run.jsonl."""
    def __init__(self, run_jsonl: str):
        super().__init__()
        self.run_jsonl = run_jsonl

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _write({
                "event": "log",
                "log": {
                    "level": record.levelname,
                    "logger": record.name,
                    "file": f"{record.filename}:{record.lineno}",
                    "msg": self.format(record),
                },
            }, self.run_jsonl)
        except Exception:
            self.handleError(record)


def _setup_jsonl_logging(run_jsonl: str) -> None:
    """Attach the JSONL handler to the root logger (WARNING and above)."""
    handler = _JsonlHandler(run_jsonl)
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# Config preprocessing
# ---------------------------------------------------------------------------

def _restore_numeric_dict_keys(d: dict) -> dict:
    """Convert string keys back to int or float after a JSON round-trip.

    JSON requires string keys, so ``{1.5: 1}`` serialises to ``{"1.5": 1}``.
    This function converts those string keys back to their numeric types:
    a key that round-trips through ``float()`` and equals its integer value
    (e.g. ``"2.0"``) becomes an ``int``; otherwise it becomes a ``float``
    (e.g. ``"1.5"``).  Non-numeric keys are left unchanged.
    """
    result = {}
    for k, v in d.items():
        try:
            f = float(k)
            result[int(f) if f == int(f) else f] = v
        except (ValueError, TypeError):
            result[k] = v
    return result


def _preprocess_optuna_config(config: dict) -> dict:
    """Return a copy of *config* with optuna search-space specs normalised.

    Handles two transformations for ``optuna.layer_shared_resid_dropout``:

    **Integer-keyed dicts → list-of-lists (intra-layer)**
        Each dict maps 0-based layer indices to bool/int values; missing
        indices are filled with ``False``.  The conversion uses
        ``mask_dict_to_list`` so all inner lists reach the same length.

        YAML::

            optuna:
              layer_shared_resid_dropout:
                - {0: true, 2: true}
                - {1: true, 3: true}

        is equivalent to::

            optuna:
              layer_shared_resid_dropout:
                - [true, false, true, false]
                - [false, true, false, true]

    **Float-keyed dicts → list of JSON strings (inter-layer)**
        Each dict maps N.5 boundary keys to group IDs; since dict keys are
        not hashable, each dict is serialised to a JSON string so Optuna's
        ``CategoricalDistribution`` can store and compare choices.
        ``_suggest`` detects the JSON-string form and deserialises the
        result back to a dict (restoring float/int key types).

        YAML::

            optuna:
              layer_shared_resid_dropout:
                - {1.5: 1, 2.5: 1}   # inter-layer tying enabled
                - {}                   # no inter-layer tying (baseline)
    """
    import copy
    import json as _json
    config = copy.deepcopy(config)
    lsrd = config.get("optuna", {}).get("layer_shared_resid_dropout")
    if isinstance(lsrd, list) and lsrd and isinstance(lsrd[0], dict):
        has_float_keys = any(
            isinstance(k, float)
            for d in lsrd if d
            for k in d
        )
        if has_float_keys:
            # Inter-layer (float-keyed): JSON-serialise to hashable string choices.
            config["optuna"]["layer_shared_resid_dropout"] = [
                _json.dumps({str(k): v for k, v in d.items()},
                            separators=(",", ":"), sort_keys=True)
                for d in lsrd
            ]
        else:
            # Intra-layer (integer-keyed): convert to bool lists.
            max_key = max(max(d.keys()) for d in lsrd if d)
            length = max_key + 1
            config["optuna"]["layer_shared_resid_dropout"] = [
                mask_dict_to_list(d, fillna=False, length=length) for d in lsrd
            ]
    return config


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

OBJECTIVE_METRICS = {
    "valid_route_accuracy": ("val_route_acc", "max"),
    "valid_action_accuracy": ("val_acc",      "max"),
    "fraction_solved":       ("fraction_solved", "value"),
}


def _count_discrete_combinations(optuna_cfg: dict) -> int | None:
    """Return the product of all parameter cardinalities if every search param
    is categorical or a stepped integer range; return None if any param is
    continuous (float range without a step, or int range with log=True).

    Reserved keys (objective_metric, etc.) are excluded from the product.
    """
    product = 1
    for name, spec in optuna_cfg.items():
        if name in _RESERVED_OPTUNA_KEYS:
            continue
        if isinstance(spec, list):
            # Flat list or list-of-lists → categorical; cardinality = len(spec).
            product *= len(spec)
        elif isinstance(spec, dict):
            if "choices" in spec:
                product *= len(spec["choices"])
            else:
                low, high = spec["low"], spec["high"]
                log = spec.get("log", False)
                step = spec.get("step")
                is_int = isinstance(low, int) and isinstance(high, int)
                if is_int and not log:
                    s = int(step) if step is not None else 1
                    product *= (high - low) // s + 1
                else:
                    # Continuous or log-scaled range → unbounded.
                    return None
        else:
            # Unknown scalar spec — treat as continuous to be conservative.
            return None
    return product


def _validate_config(config: dict, n_trials: int | None = None) -> None:
    """Raise ValueError for known config inconsistencies before any trial starts.

    Checks:
    - optuna.structured_dropout_bottleneck requires model.use_structured_dropout
    - optuna.objective_metric must be a known metric name
    - optuna.layer_shared_resid_dropout list-of-lists:
        (a) non-jagged: all inner lists have the same length
        (b) each list length >= the largest n_layers value in the search space
        (c) all inner values are bools
        (d) no two lists have the same sequence of values
    - optuna.random_seed, if present, must be a list of ints or an int low/high
      range without log scaling (so the suggested value is always a plain int)
    - if all params are discrete, n_trials must equal total combinations exactly
    """
    optuna_cfg = config.get("optuna", {})
    optuna_keys = set(optuna_cfg)
    use_sd = config.get("model", {}).get("use_structured_dropout", False)
    if not use_sd and "structured_dropout_bottleneck" in optuna_keys:
        raise ValueError(
            "Config conflict: 'optuna.structured_dropout_bottleneck' is in the "
            "search space but 'model.use_structured_dropout' is false. "
            "Either set use_structured_dropout: true or remove "
            "structured_dropout_bottleneck from the optuna section."
        )
    metric = optuna_cfg.get("objective_metric")
    if metric is not None and metric not in OBJECTIVE_METRICS:
        raise ValueError(
            f"Unknown optuna.objective_metric: {metric!r}. "
            f"Valid choices: {sorted(OBJECTIVE_METRICS)}"
        )

    lsrd_spec = optuna_cfg.get("layer_shared_resid_dropout")
    if lsrd_spec is not None and isinstance(lsrd_spec, list) and lsrd_spec and isinstance(lsrd_spec[0], list):
        # (a) Non-jagged: all inner lists must have equal length
        lengths = {len(lst) for lst in lsrd_spec}
        if len(lengths) > 1:
            raise ValueError(
                f"optuna.layer_shared_resid_dropout list-of-lists is jagged: "
                f"found inner lists of lengths {sorted(lengths)}. "
                f"All lists must have the same length."
            )
        list_len = next(iter(lengths))

        # (b) Length >= max n_layers in the optuna search space (or model default)
        n_layers_spec = optuna_cfg.get("n_layers")
        if n_layers_spec is None:
            max_n_layers = config.get("model", {}).get("n_layers", 0)
        elif isinstance(n_layers_spec, list):
            max_n_layers = max(int(v) for v in n_layers_spec)
        elif isinstance(n_layers_spec, dict):
            max_n_layers = int(n_layers_spec.get("high", list_len))
        else:
            max_n_layers = int(n_layers_spec)
        if list_len < max_n_layers:
            raise ValueError(
                f"optuna.layer_shared_resid_dropout inner lists have length {list_len} "
                f"but the maximum n_layers in the search space is {max_n_layers}. "
                f"Each list must be at least as long as the largest n_layers value "
                f"(extra entries beyond the actual n_layers are truncated at runtime)."
            )

        # (c) All inner values must be bool or 0/1
        for i, lst in enumerate(lsrd_spec):
            invalid = [(j, v) for j, v in enumerate(lst) if v not in (True, False, 0, 1)]
            if invalid:
                raise ValueError(
                    f"optuna.layer_shared_resid_dropout[{i}] contains values that are "
                    f"not bool or 0/1: {invalid}"
                )

        # (d) No duplicate lists (same sequence of values)
        seen: dict[tuple, int] = {}
        for i, lst in enumerate(lsrd_spec):
            key = tuple(bool(v) for v in lst)
            if key in seen:
                raise ValueError(
                    f"optuna.layer_shared_resid_dropout[{i}] is a duplicate of "
                    f"entry [{seen[key]}]: {list(key)}"
                )
            seen[key] = i

    # random_seed spec: must be a list of ints or an int low/high range (no log).
    seed_spec = optuna_cfg.get("random_seed")
    if seed_spec is not None:
        if isinstance(seed_spec, list):
            bad = [v for v in seed_spec if not isinstance(v, int)]
            if bad:
                raise ValueError(
                    f"optuna.random_seed list must contain only integers; "
                    f"found non-integer values: {bad}"
                )
        elif isinstance(seed_spec, dict):
            if "choices" in seed_spec:
                bad = [v for v in seed_spec["choices"] if not isinstance(v, int)]
                if bad:
                    raise ValueError(
                        f"optuna.random_seed choices must all be integers; "
                        f"found: {bad}"
                    )
            else:
                low, high = seed_spec.get("low"), seed_spec.get("high")
                if not (isinstance(low, int) and isinstance(high, int)):
                    raise ValueError(
                        f"optuna.random_seed low/high must both be integers, "
                        f"got low={low!r}, high={high!r}"
                    )
                if seed_spec.get("log", False):
                    raise ValueError(
                        "optuna.random_seed does not support log=true; "
                        "seeds must be drawn from a linear int range"
                    )
        else:
            raise ValueError(
                f"optuna.random_seed must be a list of ints or a dict with "
                f"low/high int keys; got {type(seed_spec).__name__}"
            )

    # Discrete-space exact-coverage check: if every parameter is categorical or
    # a stepped-integer range, n_trials must equal the total combinations exactly.
    if n_trials is not None:
        combos = _count_discrete_combinations(optuna_cfg)
        if combos is not None and n_trials != combos:
            raise ValueError(
                f"n_trials={n_trials} must equal the total number of discrete "
                f"parameter combinations ({combos}). Set --n-trials {combos}, "
                f"or add a continuous parameter to the search space."
            )


# ---------------------------------------------------------------------------
# Search-space dispatcher
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec) -> object:
    """Dispatch to the appropriate trial.suggest_* based on the YAML spec.

    Four forms are supported:

    1. Flat list  →  suggest_categorical
         n_heads: [1, 2, 4, 8]

    2. List-of-lists  →  suggest_categorical over list choices
       Each inner list is one complete choice; Optuna picks one per trial.
       Inner lists are serialised to JSON strings for storage compatibility.
         layer_shared_resid_dropout:
           - [true, false, true, ...]
           - [false, false, false, ...]

    3. Dict with choices key  →  suggest_categorical
         n_heads:
           choices: [1, 2, 4, 8]

    4. Scalar (int, float, bool, str)  →  fixed value, returned as-is every trial
         hidden_size: 640
         max_ep_len: 20

    5. Dict with low/high  →  suggest_int or suggest_float
       Type is inferred: both int → suggest_int, otherwise suggest_float.
       Optional keys: log (bool), step (number).
         lr:
           low: 1.0e-4
           high: 1.0
           log: true
         n_layers:
           low: 2
           high: 32
           log: true
         dropout:
           low: 0.0
           high: 0.3
           step: 0.01
    """
    if not isinstance(spec, (list, dict)):
        # Plain scalar — fixed value, not a search dimension; return as-is.
        return spec
    if isinstance(spec, list):
        import json as _json
        # List-of-lists → categorical over list choices.
        # Each inner list is serialised to a JSON string because Optuna's
        # categorical storage requires hashable (scalar) choices.  The result
        # is deserialised back to a Python list before being passed to runner.
        if spec and isinstance(spec[0], list):
            choices = [_json.dumps(lst, separators=(",", ":")) for lst in spec]
            result = trial.suggest_categorical(name, choices)
            return _json.loads(result)
        # List-of-JSON-strings → float-keyed dict choices pre-processed by
        # _preprocess_optuna_config.  Deserialise the chosen string back to a
        # dict and restore numeric key types (JSON forces string keys).
        if spec and isinstance(spec[0], str) and spec[0].startswith("{"):
            result = trial.suggest_categorical(name, spec)
            return _restore_numeric_dict_keys(_json.loads(result))
        return trial.suggest_categorical(name, spec)
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high = spec["low"], spec["high"]
    log = spec.get("log", False)
    step = spec.get("step")
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high,
                                 step=int(step) if step is not None else 1,
                                 log=log)
    return trial.suggest_float(name, low, high, step=step, log=log)


# ---------------------------------------------------------------------------
# Ordered grid enumeration
# ---------------------------------------------------------------------------

_RESERVED_OPTUNA_KEYS = {"objective_metric"}


def _enumerate_ordered_params(optuna_cfg: dict) -> list[dict]:
    """Return every combination of list-typed optuna params in declaration order.

    Only params specified as a flat list or list-of-lists (explicit choices)
    participate.  Dict params with ``low``/``high`` ranges are excluded and
    will be sampled by Optuna's normal sampler for each enqueued trial.

    Inner lists (list-of-lists form, e.g. ``layer_shared_resid_dropout``) are
    JSON-serialised to strings, matching what ``_suggest`` passes to
    ``trial.suggest_categorical`` so that Optuna's trial storage is consistent.

    Returns an empty list when no list-typed params are present.
    """
    import json as _json
    from itertools import product as _product

    names: list[str] = []
    choices_per_param: list[list] = []

    for name, spec in optuna_cfg.items():
        if name in _RESERVED_OPTUNA_KEYS:
            continue
        if isinstance(spec, list):
            if spec and isinstance(spec[0], list):
                # List-of-lists: serialise inner lists to JSON strings.
                names.append(name)
                choices_per_param.append(
                    [_json.dumps(lst, separators=(",", ":")) for lst in spec]
                )
            else:
                names.append(name)
                choices_per_param.append(list(spec))
        elif isinstance(spec, dict) and "choices" in spec:
            names.append(name)
            choices_per_param.append(list(spec["choices"]))
        # dict with low/high → continuous/stepped range; skip

    if not names:
        return []

    return [dict(zip(names, combo)) for combo in _product(*choices_per_param)]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial, config_path: str, n_epochs: int, dataset: str,
              results_base: str, run_jsonl: str, eval_n_batches: int | None = None,
              study_name: str | None = None, n_trials: int | None = None) -> float:
    config = _preprocess_optuna_config(read_config(config_path))
    optuna_config = config.get("optuna", {})
    # Reserved keys configure the study itself and must not be passed to _suggest.
    objective_metric = optuna_config.get("objective_metric", "valid_route_accuracy")
    params = {name: _suggest(trial, name, spec) for name, spec in optuna_config.items()
              if name not in _RESERVED_OPTUNA_KEYS}

    # random_seed is an optuna search param but maps to seed= in runner.main().
    if "random_seed" in params:
        params["seed"] = params.pop("random_seed")

    trial_dir = os.path.join(results_base, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    print(f"\n### Trial {trial.number} params")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    model_params = dict(
        config_path=config_path,
        dataset=dataset,
        n_epochs=n_epochs,
        results_path=trial_dir,
        eval_routes_at_end=True,
        trial_number=trial.number,
        study_name=study_name,
    )
    if eval_n_batches is not None:
        model_params["eval_n_batches"] = eval_n_batches
    model_params.update(params)
    # Keys that are control/path metadata — exclude from the all_params snapshot
    # written to run.jsonl so that rs-plot-learning-curves can display fixed
    # architecture params alongside Optuna-suggested ones.
    _META_KEYS = {"config_path", "results_path", "eval_routes_at_end",
                  "trial_number", "study_name", "eval_n_batches"}
    _write({
        "event": "trial_start",
        "trial": {"number": trial.number, "dir": trial_dir},
        "params": dict(trial.params),
        "all_params": {k: v for k, v in model_params.items() if k not in _META_KEYS},
        "optuna_keys": list(params.keys()),
    }, run_jsonl)
    print("\n#### Model params")
    for k, v in model_params.items():
        print(f"    {k}: {v}")

    # Pre-trial seed check: if random_seed was in the search space, the
    # resolved integer must be present in model_params before training begins
    # so that runner.main() saves it to model.config.yaml.
    if "seed" in model_params and not isinstance(model_params["seed"], int):
        raise RuntimeError(
            f"Trial {trial.number}: random_seed resolved to "
            f"{model_params['seed']!r} which is not an integer. "
            "Check the optuna.random_seed spec in the config."
        )

    t0 = time.time()
    val_loss, val_acc, val_route_acc, fraction_solved = train(**model_params)

    # Post-save update: read model.config.yaml, verify seed, inject n_trials,
    # then write it back so every trial config is self-contained.
    import yaml as _yaml
    saved_cfg_path = os.path.join(trial_dir, "model.config.yaml")
    with open(saved_cfg_path) as _f:
        saved_cfg = _yaml.safe_load(_f)
    if "seed" in model_params:
        saved_seed = saved_cfg.get("context", {}).get("random_state")
        if saved_seed != model_params["seed"]:
            raise RuntimeError(
                f"Trial {trial.number}: seed mismatch after save — "
                f"expected {model_params['seed']} in model.config.yaml "
                f"context.random_state but found {saved_seed!r}."
            )
    if n_trials is not None:
        saved_cfg.setdefault("train", {})["n_trials"] = n_trials
        with open(saved_cfg_path, "w") as _f:
            _yaml.dump(saved_cfg, _f, default_flow_style=False)

    # Compute the Optuna objective from whichever metric is configured.
    # val_route_acc avoids a spurious 0.0 when early stopping fires before
    # route eval; fraction_solved falls back to 0.0 when eval didn't run.
    if objective_metric == "valid_action_accuracy":
        value = max(val_acc) if val_acc else 0.0
    elif objective_metric == "fraction_solved":
        value = fraction_solved if fraction_solved is not None else 0.0
    else:  # default: valid_route_accuracy
        value = max(val_route_acc) if val_route_acc else 0.0
    print(f"\nObjective ({objective_metric}): {value:.4f}")

    trial_results = {
        "event": "trial_end",
        "trial": {"number": trial.number, "dir": trial_dir},
        "params": dict(trial.params),
        "results": {
            "duration_s": round(time.time() - t0, 1),
            "accuracy": {
                "optuna_score": value,
                "fraction_targets_solved": fraction_solved,
                "valid_loss": float(val_loss[-1]) if val_loss else None,
                "valid_action_accuracy": float(val_acc[-1]) if val_acc else None,
                "valid_route_accuracy": float(val_route_acc[-1]) if val_route_acc else None,
            },
        },
    }
    _write(trial_results, run_jsonl)
    print("\n#### Results")
    for k, v in trial_results.items():
        print(f"    {k}: {v}")
    return value


# ---------------------------------------------------------------------------
# Interrupt cleanup
# ---------------------------------------------------------------------------

def _find_last_trial_number(results_base: str) -> int:
    """Return the highest trial_NNN directory number found in results_base."""
    import re
    numbers = [
        int(m.group(1))
        for name in os.listdir(results_base)
        if os.path.isdir(os.path.join(results_base, name))
        for m in [re.fullmatch(r"trial_(\d+)", name)]
        if m
    ]
    if not numbers:
        raise ValueError(f"No trial_NNN directories found in {results_base}")
    return max(numbers)


def _backfill_failed_trials(results_base: str) -> None:
    """Backfill Optuna objective values for FAIL trials that have training data.

    Called automatically after a Ctrl-C stop so the interrupted trial's
    progress is not lost from the TPE surrogate model.
    """
    from pathlib import Path
    from retrosynformer.scripts.cleanup_study import cleanup_study

    db_path = Path(results_base) / "study.db"
    if not db_path.exists():
        return
    try:
        records = cleanup_study(db_path, include_running=False, dry_run=False)
        for rec in records:
            if rec.get("estimated_value") is not None:
                print(
                    f"  [cleanup] trial #{rec['trial']} ({rec['state']}) "
                    f"{rec['metric']}={rec['estimated_value']:.4f} "
                    f"({rec.get('method', '')}) → {rec['action']}"
                )
    except Exception as exc:
        print(f"  [cleanup] warning: could not backfill failed trials: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    print_banner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", default=CONFIG_PATH_DEFAULT, dest="config_path",
        help="Base config.yaml to use for all trials",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Total number of Optuna trials to run",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=200,
        help="Training epochs per trial",
    )
    parser.add_argument(
        "--dataset", default="large", choices=["small", "standard", "large"],
        help="Dataset preset: small=589, standard=1573, large=2957 templates (default: large)",
    )
    parser.add_argument(
        "--eval-n-batches", type=int, default=None, dest="eval_n_batches",
        help="Override evaluation.eval_n_batches from config (reduce for faster CPU runs)",
    )
    parser.add_argument(
        "--study-name", default="retrosynformer_hypertune",
        help="Optuna study name (also determines output directory: results/hypertune-{name}/)",
    )
    parser.add_argument(
        "--storage", default=None,
        help="Optuna storage URL (default: sqlite:///results/hypertune-{study_name}/study.db)",
    )
    parser.add_argument(
        "--resume", type=int, nargs="?", const=-1, default=None, metavar="TRIAL",
        help=(
            "Resume a partially-trained trial before continuing the study. "
            "Pass a trial number (e.g. --resume 3), or omit the number to "
            "resume the latest trial_NNN directory found on disk."
        ),
    )
    add_log_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    # Preprocess then validate config eagerly so bad configs fail before any study setup.
    _validate_config(_preprocess_optuna_config(read_config(args.config_path)), n_trials=args.n_trials)

    results_base = f"results/hypertune-{args.study_name}"
    run_jsonl = os.path.join(results_base, "run.jsonl")
    storage = args.storage or f"sqlite:///{results_base}/study.db"

    os.makedirs(results_base, exist_ok=True)
    is_fresh = not os.path.exists(os.path.join(results_base, "study.db"))

    _setup_jsonl_logging(run_jsonl)
    _write({"event": "study_start" if is_fresh else "study_resume", "config": {
        "n_trials": args.n_trials, "n_epochs": args.n_epochs,
        "dataset": args.dataset, "storage": storage,
        "config_path": args.config_path,
        "eval_n_batches": args.eval_n_batches,
    }}, run_jsonl)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    # Persist n_trials in study.db so it can be retrieved without the CLI args.
    study.set_user_attr("n_trials", args.n_trials)

    # On a fresh study, pre-enqueue all ordered combinations so that list-typed
    # params are tried in declaration order.  Range params are sampled normally
    # by Optuna for each enqueued trial.  Falls back to the fixed baseline when
    # no list-typed params are present (pure continuous / range-only search).
    if len(study.trials) == 0:
        preprocessed_cfg = _preprocess_optuna_config(read_config(args.config_path))
        ordered_combos = _enumerate_ordered_params(preprocessed_cfg.get("optuna", {}))
        if ordered_combos:
            for combo in ordered_combos:
                study.enqueue_trial(combo)
        else:
            study.enqueue_trial(BASELINE_TRIAL)

    if args.resume is not None:
        trial_num = args.resume if args.resume >= 0 else _find_last_trial_number(results_base)
        trial_config = os.path.join(results_base, f"trial_{trial_num:03d}", "model.config.yaml")
        if not os.path.exists(trial_config):
            raise FileNotFoundError(f"No config found for trial {trial_num}: {trial_config}")
        print(f"\n### Resuming trial {trial_num} from {trial_config}")
        _write({"event": "trial_resume", "trial": {"number": trial_num, "config": trial_config}}, run_jsonl)
        train(config_path=trial_config, resume=True, eval_routes_at_end=True,
              trial_number=trial_num, study_name=args.study_name,
              n_epochs=args.n_epochs)
        _backfill_failed_trials(results_base)

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Write a completion record to run.jsonl after each trial."""
        _write({
            "event": "trial_complete" if trial.state == optuna.trial.TrialState.COMPLETE else "trial_fail",
            "trial": {"number": trial.number, "state": trial.state.name},
            "params": dict(trial.params),
            "results": {
                "value": trial.value,
                "duration_s": round(trial.duration.total_seconds(), 1) if trial.duration else None,
            },
        }, run_jsonl)

    def _objective_with_interrupt(trial):
        # Register study.stop as the interrupt callback so a Ctrl-C stops the
        # study after the current trial completes and its result is recorded.
        _trainer_mod.set_interrupt_callback(study.stop)
        try:
            return objective(trial, args.config_path, args.n_epochs, args.dataset,
                             results_base, run_jsonl, args.eval_n_batches, args.study_name,
                             args.n_trials)
        finally:
            _trainer_mod.clear_interrupt_callback()

    study.optimize(
        _objective_with_interrupt,
        n_trials=args.n_trials,
        callbacks=[log_trial],
    )

    # After any stop (Ctrl-C or study.stop()), backfill FAIL trials that have
    # training data but no recorded objective value.  This lets Optuna's TPE
    # sampler use the interrupted run's history for the next trial.
    if _trainer_mod.is_interrupted():
        _backfill_failed_trials(results_base)

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        _write({"event": "study_end", "best": None}, run_jsonl)
        if _trainer_mod.is_interrupted():
            raise KeyboardInterrupt
        return

    best = study.best_trial
    _write({
        "event": "study_end",
        "best": {
            "trial": best.number,
            "value": best.value,
            "params": dict(best.params),
            "dir": os.path.join(results_base, f"trial_{best.number:03d}"),
        },
    }, run_jsonl)
    print("\n=== Best trial ===")
    print(f"  fraction_targets_solved: {best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  results: {os.path.join(results_base, f'trial_{best.number:03d}')}")

    if _trainer_mod.is_interrupted():
        raise KeyboardInterrupt


if __name__ == "__main__":
    main()
