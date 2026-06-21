#!/usr/bin/env python
"""rs-cleanup-study — backfill Optuna objective values for stopped/interrupted trials.

When a trial is interrupted (Ctrl-C, crash, SIGKILL), Optuna marks it FAIL
because the objective function never returned a value.  The trial's
``train_progress.jsonl`` often contains enough data to estimate what the score
*would* have been.  This command reads those files, computes the objective
metric the same way the normal run would have, and writes it back into
``study.db`` so Optuna's TPE sampler can use the historical data.

Usage::

    # Dry-run: show what would be changed, make no writes
    rs-cleanup-study --dry-run

    # Process all studies under results/
    rs-cleanup-study

    # One specific study directory
    rs-cleanup-study --study-dir results/hypertune-large-emma-24-26_layer

    # Also backfill stale RUNNING trials (careful: only if the run is no longer active)
    rs-cleanup-study --include-running

    # Use polynomial extrapolation to project forward to n_epochs instead of max-so-far
    rs-cleanup-study --extrapolate
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from retrosynformer.scripts import print_banner

# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _load_jsonl_metric(jsonl_path: Path, metric: str) -> list[tuple[int, float]]:
    """Return sorted (epoch, value) pairs from train_progress.jsonl.

    De-duplicates by epoch (keeps the last value seen — handles append-on-restart
    files). Skips lines where the metric is absent or non-finite.
    """
    import math
    if not jsonl_path.exists():
        return []
    seen: dict[int, float] = {}
    try:
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric not in row:
                continue
            val = row[metric]
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                continue
            epoch = int(row.get("epoch", len(seen)))
            seen[epoch] = float(val)
    except Exception:
        return []
    return sorted(seen.items())


def _objective_value_from_jsonl(
    jsonl_path: Path,
    metric: str,
    extrapolate: bool,
    n_epochs: int,
    direction: str,
) -> Optional[float]:
    """Return the best-so-far metric value (or extrapolated estimate).

    For ``extrapolate=False`` (default): return ``max(metric)`` for MAXIMIZE
    or ``min(metric)`` for MINIMIZE — exactly what the normal objective
    function does when a trial completes (e.g. ``max(val_route_acc)``).

    For ``extrapolate=True``: use polynomial fit to project to *n_epochs*.
    """
    pairs = _load_jsonl_metric(jsonl_path, metric)
    if not pairs:
        return None

    if extrapolate and len(pairs) >= 6:
        try:
            from retrosynformer.models_optuna import _fit_quadratic_estimate
            result = _fit_quadratic_estimate(pairs, n_epochs, direction=direction)
            return result["estimated_value"]
        except Exception:
            pass

    # Simple best-so-far (default, matches how the objective function computes it)
    values = [v for _, v in pairs]
    return max(values) if direction == "MAXIMIZE" else min(values)


# ---------------------------------------------------------------------------
# Objective metric from config
# ---------------------------------------------------------------------------

def _trial_objective_metric(trial_dir: Path) -> str:
    """Return the ``optuna.objective_metric`` from model.config.yaml, or default."""
    for cfg_name in ("model.config.yaml", "config.yaml"):
        cfg_path = trial_dir / cfg_name
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                return cfg.get("optuna", {}).get("objective_metric", "valid_route_accuracy")
            except Exception:
                pass
    return "valid_route_accuracy"


def _trial_n_epochs(trial_dir: Path) -> int:
    """Return the configured n_epochs from model.config.yaml, or 100."""
    for cfg_name in ("model.config.yaml", "config.yaml"):
        cfg_path = trial_dir / cfg_name
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                return int(cfg.get("train", {}).get("n_epochs", 100))
            except Exception:
                pass
    return 100


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

def _backfill_trial(
    db_path: Path,
    trial_number: int,
    trial_id: int,
    value: float,
    dry_run: bool,
) -> None:
    """Insert a TrialValue row and mark the trial COMPLETE in study.db."""
    if dry_run:
        return

    from retrosynformer.models_optuna import Trial, TrialValue, connect

    session = connect(db_path, readonly=False)
    try:
        trial = session.query(Trial).filter_by(trial_id=trial_id).first()
        if trial is None:
            print(f"  [WARNING] trial_id={trial_id} not found in DB — skipping write", file=sys.stderr)
            return

        # Check for existing objective value (shouldn't exist, but guard)
        existing = next((v for v in trial.values if v.objective == 0), None)
        if existing is not None:
            print(f"  [WARNING] trial #{trial_number} already has objective value "
                  f"{existing.value:.4f} — skipping", file=sys.stderr)
            return

        session.add(TrialValue(
            trial_id=trial_id,
            objective=0,
            value=value,
            value_type="FINITE",
        ))
        trial.state = "COMPLETE"
        if trial.datetime_complete is None:
            trial.datetime_complete = datetime.now(timezone.utc).replace(tzinfo=None)
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Per-study cleanup
# ---------------------------------------------------------------------------

def cleanup_study(
    db_path: Path,
    *,
    include_running: bool = False,
    extrapolate: bool = False,
    dry_run: bool = True,
) -> list[dict]:
    """Process one study.db — find FAIL (and optionally RUNNING) trials with no
    objective value, estimate the score from train_progress.jsonl, and write back.

    Returns a list of result records (one per eligible trial).
    """
    from retrosynformer.models_optuna import Study, Trial, connect

    session = connect(db_path, readonly=True)
    try:
        study = session.query(Study).first()
        if study is None:
            return []
        direction = study.direction or "MAXIMIZE"

        states = {"FAIL"}
        if include_running:
            states.add("RUNNING")

        eligible = [
            t for t in study.trials
            if t.state in states and t.objective_value is None
        ]
    finally:
        session.close()

    study_dir = db_path.parent
    results = []

    for trial in eligible:
        trial_dir = study_dir / f"trial_{trial.number:03d}"
        jsonl_path = trial_dir / "train_progress.jsonl"

        metric = _trial_objective_metric(trial_dir)
        n_epochs = _trial_n_epochs(trial_dir)

        pairs = _load_jsonl_metric(jsonl_path, metric)
        n_observed = len(pairs)

        if not pairs:
            results.append({
                "trial": trial.number,
                "state": trial.state,
                "metric": metric,
                "n_epochs_observed": 0,
                "estimated_value": None,
                "action": "skip (no jsonl data)",
            })
            continue

        value = _objective_value_from_jsonl(
            jsonl_path, metric, extrapolate, n_epochs, direction
        )

        if value is None:
            results.append({
                "trial": trial.number,
                "state": trial.state,
                "metric": metric,
                "n_epochs_observed": n_observed,
                "estimated_value": None,
                "action": "skip (could not estimate)",
            })
            continue

        method = "extrapolated" if extrapolate and n_observed >= 6 else f"max over {n_observed} epochs"
        action = "dry-run (no write)" if dry_run else f"wrote {value:.4f} → COMPLETE"

        results.append({
            "trial": trial.number,
            "state": trial.state,
            "metric": metric,
            "n_epochs_observed": n_observed,
            "estimated_value": value,
            "method": method,
            "action": action if dry_run else "pending",
        })

        if not dry_run:
            _backfill_trial(db_path, trial.number, trial.trial_id, value, dry_run=False)
            results[-1]["action"] = f"wrote {value:.4f} → COMPLETE"

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Process only this study directory (must contain study.db). "
             "Default: all hypertune-*/ directories under --results-base.",
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=Path("results"),
        metavar="DIR",
        help="Root directory to search for hypertune-*/study.db files (default: results/).",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also backfill stale RUNNING trials. Only use when the training "
             "process is confirmed stopped.",
    )
    parser.add_argument(
        "--extrapolate",
        action="store_true",
        help="Use polynomial extrapolation to project metrics forward to n_epochs "
             "instead of using the best observed value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be changed but make no writes to study.db.",
    )
    args = parser.parse_args()

    if args.study_dir is not None:
        db_paths = [args.study_dir / "study.db"]
    else:
        db_paths = sorted(args.results_base.glob("hypertune-*/study.db"))

    if not db_paths:
        print("No study.db files found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN — no writes will be made]\n")

    total_written = 0
    total_skipped = 0

    for db_path in db_paths:
        if not db_path.exists():
            print(f"  {db_path}: not found — skipping", file=sys.stderr)
            continue

        study_name = db_path.parent.name.removeprefix("hypertune-")
        print(f"Study: {study_name}  ({db_path})")

        records = cleanup_study(
            db_path,
            include_running=args.include_running,
            extrapolate=args.extrapolate,
            dry_run=args.dry_run,
        )

        if not records:
            print("  No eligible trials (all trials already have objective values or "
                  "no FAIL/RUNNING trials found).\n")
            continue

        for rec in records:
            value_str = f"{rec['estimated_value']:.4f}" if rec["estimated_value"] is not None else "(none)"
            method_str = f" [{rec.get('method', '')}]" if rec.get("method") else ""
            print(
                f"  trial #{rec['trial']:3d}  [{rec['state']:7s}]  "
                f"{rec['metric']}: {value_str}{method_str}  "
                f"epochs={rec['n_epochs_observed']}  → {rec['action']}"
            )
            if "wrote" in rec["action"]:
                total_written += 1
            else:
                total_skipped += 1
        print()

    if args.dry_run:
        eligible = sum(1 for db in db_paths if db.exists())
        print(f"Dry run complete.  Re-run without --dry-run to write changes.")
    else:
        print(f"Done.  {total_written} trial(s) backfilled, {total_skipped} skipped.")
