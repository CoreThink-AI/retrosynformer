"""Populate the dashboard meta-DB from Optuna study.db files and JSONL logs.

The Optuna study.db and JSONL files are authoritative. The meta-DB is a
derived cache rebuilt by sync_all() / sync_study().
"""
import json
import os
from datetime import datetime
from pathlib import Path

from .models import Study, Trial, db


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_study_dbs(root: str) -> list[str]:
    """Return absolute paths to all study.db files under *root*, deduped."""
    seen = set()
    result = []
    for p in Path(root).rglob("study.db"):
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            result.append(str(p))
    return sorted(result)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _last_jsonl_record(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return last


def _count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _best_fraction_solved(pred_path: str) -> float | None:
    if not os.path.exists(pred_path):
        return None
    try:
        records = json.loads(Path(pred_path).read_text())
    except (json.JSONDecodeError, ValueError):
        return None
    best = None
    for rec in records:
        results = rec.get("result", [])
        if not results:
            continue
        frac = sum(1 for r in results if r.get("route_solved")) / len(results)
        if best is None or frac > best:
            best = frac
    return best


def _final_metrics_from_jsonl(trial_dir: str) -> dict:
    """Return last-epoch metrics from train_progress.jsonl."""
    path = os.path.join(trial_dir, "train_progress.jsonl")
    last = _last_jsonl_record(path)
    if last is None:
        return {}
    return {
        "valid_loss": last.get("valid_loss"),
        "valid_action_accuracy": last.get("valid_action_accuracy"),
        "valid_route_accuracy": last.get("valid_route_accuracy"),
        "epoch_count": last.get("epoch", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Optuna DB reading
# ---------------------------------------------------------------------------

def _load_optuna(db_path: str) -> dict:
    """Return {study_name, direction, objective_metric, trials: [...]}."""
    try:
        from retrosynformer.study import to_dfs, dfs_to_trials_df
        dfs = to_dfs(db_path)
    except Exception:
        return {}

    studies_df = dfs.get("studies")
    if studies_df is None or studies_df.empty:
        return {}

    study_name = studies_df["study_name"].iloc[0]

    directions_df = dfs.get("study_directions")
    direction = "maximize"
    if directions_df is not None and not directions_df.empty:
        direction = directions_df["direction"].iloc[0].lower()

    # Decode params
    try:
        trials_df = dfs_to_trials_df(dfs)
    except Exception:
        trials_df = None

    trials = []
    raw_trials = dfs.get("trials")
    if raw_trials is not None:
        # Deduplicate: keep highest trial_id per number (study restarts create duplicates)
        if "number" in raw_trials.columns and "trial_id" in raw_trials.columns:
            raw_trials = raw_trials.sort_values("trial_id").drop_duplicates("number", keep="last")
        for _, row in raw_trials.iterrows():
            trial_id = int(row.get("trial_id", row.get("number", 0)))
            trial_number = int(row.get("number", trial_id))
            state = str(row.get("state", "WAITING")).upper()
            dt_start = row.get("datetime_start")
            dt_end = row.get("datetime_complete")

            # Get decoded params from trials_df if available
            params = {}
            if trials_df is not None:
                match = trials_df[trials_df["number"] == trial_number] if "number" in trials_df.columns else trials_df.head(0)
                if not match.empty:
                    param_cols = [c for c in match.columns if c not in
                                  ("number", "state", "datetime_start", "datetime_complete",
                                   "duration", "value", "trial_id")]
                    params = {c: match.iloc[0][c] for c in param_cols
                              if match.iloc[0][c] is not None}

            # Get score from trial_values
            score = None
            tv = dfs.get("trial_values")
            if tv is not None and not tv.empty:
                tv_match = tv[tv["trial_id"] == trial_id]
                if not tv_match.empty:
                    score = float(tv_match["value"].iloc[0])

            trials.append({
                "trial_id": trial_id,
                "trial_number": trial_number,
                "state": state,
                "datetime_start": dt_start,
                "datetime_complete": dt_end,
                "params": params,
                "score": score,
            })

    # Objective metric from system attributes
    objective_metric = "valid_route_accuracy"
    sa = dfs.get("study_system_attributes")
    if sa is not None and not sa.empty:
        row = sa[sa["key"] == "objective_metric"]
        if not row.empty:
            objective_metric = row["value_json"].iloc[0].strip('"')

    return {
        "study_name": study_name,
        "direction": direction,
        "objective_metric": objective_metric,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def sync_all(root: str) -> dict:
    """Sync all discovered study.db files. Returns {synced, skipped, errors}."""
    paths = discover_study_dbs(root)
    synced = skipped = errors = 0
    for db_path in paths:
        try:
            updated = sync_study(db_path, root)
            if updated:
                synced += 1
            else:
                skipped += 1
        except Exception as e:
            db.session.rollback()
            print(f"[sync] ERROR {db_path}: {e}")
            errors += 1
    return {"synced": synced, "skipped": skipped, "errors": errors}


def sync_study(db_path: str, root: str, force: bool = False) -> bool:
    """Sync one study.db into the meta-DB. Returns True if anything changed."""
    db_mtime = os.path.getmtime(db_path)
    study_dir = os.path.dirname(db_path)
    study_folder_name = os.path.basename(study_dir)

    # Check if sync is needed
    existing = Study.query.filter_by(db_path=db_path).first()
    if not force and existing and existing.last_synced_at:
        last_sync_ts = existing.last_synced_at.timestamp()
        if db_mtime <= last_sync_ts:
            return False

    info = _load_optuna(db_path)
    if not info:
        return False

    study_name = info["study_name"]
    run_jsonl = os.path.join(study_dir, "run.jsonl")

    if existing is None:
        # Another db_path may already own this study_name (e.g. rsync'd copy).
        # Prefer the path that's actually a child of root (canonical location).
        name_match = Study.query.filter_by(study_name=study_name).first()
        if name_match is not None:
            root_abs = os.path.abspath(root)
            current_is_canonical = os.path.abspath(db_path).startswith(root_abs + os.sep)
            existing_is_canonical = os.path.abspath(name_match.db_path).startswith(root_abs + os.sep)
            if current_is_canonical and not existing_is_canonical:
                # Takeover: re-point the row to the canonical path
                name_match.db_path = db_path
            existing = name_match
        else:
            existing = Study(study_name=study_name, db_path=db_path)
            db.session.add(existing)

    existing.study_name = study_name
    existing.db_path = db_path
    existing.run_jsonl_path = run_jsonl if os.path.exists(run_jsonl) else None
    existing.objective_metric = info["objective_metric"]
    existing.direction = info["direction"]
    existing.last_synced_at = datetime.utcnow()

    # Look for config
    for name in ("model.config.yaml", "config.yaml"):
        cpath = os.path.join(study_dir, name)
        if os.path.exists(cpath):
            existing.config_path = cpath
            break

    # Sync trials
    _sync_trials(existing, info["trials"], study_dir, root)

    # Recompute aggregates
    complete = [t for t in info["trials"] if t["state"] == "COMPLETE"]
    running = [t for t in info["trials"] if t["state"] == "RUNNING"]
    failed = [t for t in info["trials"] if t["state"] in ("FAIL", "FAILED")]
    scores = [t["score"] for t in complete if t["score"] is not None]

    existing.n_trials = len(info["trials"])
    existing.n_complete = len(complete)
    existing.n_running = len(running)
    existing.n_failed = len(failed)

    if scores:
        import statistics
        existing.best_score = max(scores) if info["direction"] == "maximize" else min(scores)
        existing.mean_score = statistics.mean(scores)
        existing.std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        best_trial = max(complete, key=lambda t: t["score"] or 0) if info["direction"] == "maximize" \
            else min(complete, key=lambda t: t["score"] or 0)
        existing.best_trial_number = best_trial["trial_number"]

    # Count manually-stopped trials in the dashboard (preserved across syncs).
    n_stopped = (
        Trial.query.filter_by(study_id=existing.id, state="STOPPED").count()
        if existing.id else 0
    )
    if running:
        existing.status = "active"
    elif complete:
        existing.status = "complete"
    elif n_stopped > 0:
        existing.status = "stopped"
    else:
        existing.status = "unknown"

    db.session.commit()
    return True


def _sync_trials(study_row: Study, optuna_trials: list[dict],
                 study_dir: str, root: str) -> None:
    # Query directly — the dynamic relationship may miss in-session additions
    existing_rows = Trial.query.filter_by(study_id=study_row.id).all() if study_row.id else []
    existing_map = {t.trial_number: t for t in existing_rows}

    for ot in optuna_trials:
        trial_number = ot["trial_number"]
        trial_dir = os.path.join(study_dir, f"trial_{trial_number:03d}")

        if trial_number not in existing_map:
            t = Trial(study=study_row, trial_number=trial_number)
            db.session.add(t)
        else:
            t = existing_map[trial_number]

        t.optuna_trial_id = ot["trial_id"]
        # Preserve a manually-set STOPPED state — don't let a re-sync overwrite it.
        if t.state != "STOPPED":
            t.state = ot["state"]
        t.params_json = json.dumps(ot["params"]) if ot["params"] else None
        t.optuna_score = ot["score"]
        t.trial_dir = trial_dir if os.path.isdir(trial_dir) else None
        t.synced_at = datetime.utcnow()

        # Parse datetimes
        for attr, val in (("datetime_start", ot["datetime_start"]),
                          ("datetime_complete", ot["datetime_complete"])):
            if val is not None and not isinstance(val, datetime):
                try:
                    val = datetime.fromisoformat(str(val))
                except ValueError:
                    val = None
            setattr(t, attr, val)

        if t.datetime_start and t.datetime_complete:
            delta = (t.datetime_complete - t.datetime_start).total_seconds()
            t.duration_min = round(delta / 60, 1)

        # Read metrics from JSONL (cheap for terminal states; do for all)
        if t.trial_dir:
            metrics = _final_metrics_from_jsonl(t.trial_dir)
            t.valid_loss = metrics.get("valid_loss")
            t.valid_action_accuracy = metrics.get("valid_action_accuracy")
            t.valid_route_accuracy = metrics.get("valid_route_accuracy")
            t.epoch_count = metrics.get("epoch_count", 0)
            t.fraction_targets_solved = _best_fraction_solved(
                os.path.join(t.trial_dir, "pred_routes_train_progress.json")
            )
