#!/usr/bin/env python
"""Aggregate all Optuna study.db files into a comprehensive hyperparameter table.

For each trial the output row contains:
  - Optuna-suggested params (what the sampler chose)
  - Actual model params from model.config.yaml (ground truth after overrides)
  - Final & best metrics from train_progress.jsonl
  - Git commit hash that was active when the study started, found by searching
    the log for the most-recent commit at least 1 minute before the first trial.

Implicit hyperparameters (action_dim → dataset size, reward settings, early-stopping
patience, etc.) are included alongside architecture params so the full table can be
analysed for correlated patterns.

Usage
-----
    python scripts/aggregate_hyperparams.py
    python scripts/aggregate_hyperparams.py --output results/all_hyperparams.csv
    python scripts/aggregate_hyperparams.py --results results/ --output all.csv
"""
import argparse
import glob
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _load_git_log(repo_root: str) -> list[tuple[datetime, str, str]]:
    """Return [(utc_datetime, hash, subject), …] sorted oldest-first."""
    result = subprocess.run(
        ["git", "log", "--all", "--format=%aI|%H|%s"],
        capture_output=True, text=True, cwd=repo_root,
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        if "|" not in line:
            continue
        ts_str, hash_, subject = line.split("|", 2)
        try:
            dt = datetime.fromisoformat(ts_str.strip()).astimezone(timezone.utc)
            commits.append((dt, hash_.strip(), subject.strip()))
        except ValueError:
            pass
    commits.sort(key=lambda x: x[0])
    return commits


def _find_git_hash(
    study_start: datetime,
    commits: list[tuple[datetime, str, str]],
    min_before_minutes: float = 1.0,
) -> tuple[str | None, datetime | None, str | None]:
    """Most-recent commit at least *min_before_minutes* before *study_start*."""
    if study_start.tzinfo is None:
        study_start = study_start.replace(tzinfo=timezone.utc)
    cutoff = study_start - timedelta(minutes=min_before_minutes)
    eligible = [(dt, h, m) for dt, h, m in commits if dt <= cutoff]
    if not eligible:
        return None, None, None
    dt, h, m = max(eligible, key=lambda x: x[0])
    return h, dt, m


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _flatten_config(cfg: dict) -> dict:
    """Extract the hyperparameter fields we care about from model.config.yaml."""
    m = cfg.get("model", {})
    t = cfg.get("train", {})
    d = cfg.get("dataset", {})
    o = cfg.get("optimizer", {})
    r = cfg.get("reward", {})
    ev = cfg.get("evaluation", {})
    ctx = cfg.get("context", {})

    action_dim = d.get("action_dim")
    dataset_label = {589: "small", 1573: "standard", 2957: "large"}.get(action_dim, str(action_dim))

    return {
        # model architecture
        "cfg_n_heads": m.get("n_heads"),
        "cfg_n_layers": m.get("n_layers"),
        "cfg_head_dim": m.get("head_dim"),
        "cfg_hidden_size": m.get("hidden_size"),
        "cfg_max_ep_len": m.get("max_ep_len"),
        "cfg_activation_function": m.get("activation_function"),
        "cfg_action_tanh": m.get("action_tanh"),
        # dropout
        "cfg_attn_pdrop": m.get("attn_pdrop"),
        "cfg_embd_pdrop": m.get("embd_pdrop"),
        "cfg_resid_pdrop": m.get("resid_pdrop"),
        "cfg_use_structured_dropout": m.get("use_structured_dropout", False),
        "cfg_structured_dropout_bottleneck": m.get("structured_dropout_bottleneck"),
        "cfg_structured_dropout_rate": m.get("structured_dropout_rate"),
        # optimizer
        "cfg_lr": o.get("lr"),
        "cfg_momentum": o.get("momentum"),
        # training schedule
        "cfg_batch_size": t.get("batch_size"),
        "cfg_n_epochs": t.get("n_epochs"),
        "cfg_early_stopping_patience": t.get("early_stopping_patience"),
        "cfg_lr_scheduler_patience": t.get("lr_scheduler_patience"),
        "cfg_loss": t.get("loss"),
        # dataset
        "cfg_action_dim": action_dim,
        "cfg_dataset": dataset_label,
        "cfg_fp_dim": d.get("fp_dim"),
        "cfg_n_in_state": d.get("n_in_state"),
        "cfg_valid_set": d.get("valid_set"),
        "cfg_random_state": ctx.get("random_state"),
        # reward
        "cfg_bb_reward": r.get("building_block_reward_factor"),
        "cfg_bb_scale": r.get("building_block_scale_with_depth"),
        "cfg_dead_end_reward": r.get("dead_end_reward_factor"),
        "cfg_dead_end_scale": r.get("dead_end_scale_with_depth"),
        "cfg_intermediate_reward": r.get("intermediate_reward_factor"),
        "cfg_intermediate_scale": r.get("intermediate_scale_with_depth"),
        # evaluation
        "cfg_beam_width": ev.get("beam_width"),
        "cfg_eval_n_batches": ev.get("eval_n_batches"),
        "cfg_eval_routes_frequency": ev.get("eval_routes_frequency"),
    }


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _read_jsonl_metrics(jsonl_path: str) -> dict:
    """Return last-epoch and best-epoch metrics from train_progress.jsonl."""
    if not os.path.exists(jsonl_path):
        return {}
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        return {}

    # Handle restarts: keep only the last contiguous run (epoch resets to low value).
    last_reset = 0
    for i in range(1, len(records)):
        if records[i].get("epoch", i) <= records[i - 1].get("epoch", i - 1):
            last_reset = i
    records = records[last_reset:]

    last = records[-1]
    best_vaa = max((r.get("valid_action_accuracy", 0) or 0) for r in records)
    best_vra = max((r.get("valid_route_accuracy", 0) or 0) for r in records)

    return {
        "epoch_count": last.get("epoch", 0) + 1,
        "final_train_loss": last.get("train_loss"),
        "final_train_action_accuracy": last.get("train_action_accuracy"),
        "final_valid_loss": last.get("valid_loss"),
        "final_valid_action_accuracy": last.get("valid_action_accuracy"),
        "final_valid_route_accuracy": last.get("valid_route_accuracy"),
        "best_valid_action_accuracy": best_vaa if best_vaa > 0 else None,
        "best_valid_route_accuracy": best_vra if best_vra > 0 else None,
        "seconds_per_epoch": last.get("seconds_per_epoch"),
    }


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


# ---------------------------------------------------------------------------
# Optuna study.db helpers
# ---------------------------------------------------------------------------

def _decode_param(value: float, dist_json: str) -> object:
    dist = json.loads(dist_json)
    if dist["name"] == "CategoricalDistribution":
        return dist["attributes"]["choices"][int(value)]
    return value


def _load_study(db_path: str) -> dict:
    """Return {study_name, objective_metric, trials: [{...}]}."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        studies = con.execute("SELECT study_id, study_name FROM studies").fetchall()
        if not studies:
            return {}
        # Multi-study DBs: take the first (hypertune-tiny has several test ones)
        study_id, study_name = studies[0]["study_id"], studies[0]["study_name"]

        # Objective metric from system attributes
        obj_row = con.execute(
            "SELECT value_json FROM study_system_attributes "
            "WHERE study_id=? AND key='objective_metric'", (study_id,)
        ).fetchone()
        objective_metric = (
            obj_row["value_json"].strip('"') if obj_row else "valid_route_accuracy"
        )

        # Trial params pivot
        params_rows = con.execute(
            "SELECT tp.trial_id, tp.param_name, tp.param_value, tp.distribution_json "
            "FROM trial_params tp "
            "JOIN trials t ON t.trial_id=tp.trial_id "
            "WHERE t.study_id=?", (study_id,)
        ).fetchall()
        params_by_trial: dict[int, dict] = {}
        for row in params_rows:
            tid = row["trial_id"]
            params_by_trial.setdefault(tid, {})[row["param_name"]] = _decode_param(
                row["param_value"], row["distribution_json"]
            )

        # Trial values (objective scores)
        scores_by_trial: dict[int, float] = {}
        for row in con.execute(
            "SELECT tv.trial_id, tv.value FROM trial_values tv "
            "JOIN trials t ON t.trial_id=tv.trial_id WHERE t.study_id=?", (study_id,)
        ).fetchall():
            scores_by_trial[row["trial_id"]] = row["value"]

        trials = []
        for row in con.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete "
            "FROM trials WHERE study_id=? ORDER BY number", (study_id,)
        ).fetchall():
            tid = row["trial_id"]
            ds = row["datetime_start"]
            dc = row["datetime_complete"]
            duration = None
            if ds and dc:
                try:
                    delta = (
                        datetime.fromisoformat(dc) - datetime.fromisoformat(ds)
                    ).total_seconds()
                    duration = round(delta / 60, 1)
                except ValueError:
                    pass
            trials.append({
                "trial_id": tid,
                "trial_number": row["number"],
                "state": row["state"],
                "datetime_start": ds,
                "datetime_complete": dc,
                "duration_min": duration,
                "optuna_params": params_by_trial.get(tid, {}),
                "optuna_score": scores_by_trial.get(tid),
            })

        return {
            "study_name": study_name,
            "objective_metric": objective_metric,
            "trials": trials,
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(results_root: str, repo_root: str) -> pd.DataFrame:
    commits = _load_git_log(repo_root)
    if not commits:
        print("[warn] No git commits found — git_hash column will be empty.",
              file=sys.stderr)

    db_paths = sorted(glob.glob(
        os.path.join(results_root, "**/study.db"), recursive=True
    ))
    # Deduplicate by real path (handles symlinks / rsynced copies).
    # Note: study.db files that contain multiple Optuna studies (e.g. hypertune-tiny,
    # which holds several short test runs) only contribute their first study — those
    # sub-studies are small test runs and not worth special-casing.
    seen_real = set()
    unique_db_paths = []
    for p in db_paths:
        real = os.path.realpath(p)
        if real not in seen_real:
            seen_real.add(real)
            unique_db_paths.append(p)

    rows = []
    for db_path in unique_db_paths:
        study_dir = os.path.dirname(db_path)
        study_folder = os.path.relpath(study_dir, results_root)

        info = _load_study(db_path)
        if not info:
            continue

        study_name = info["study_name"]
        objective_metric = info["objective_metric"]

        # Study start = earliest trial start
        starts = [
            datetime.fromisoformat(t["datetime_start"])
            for t in info["trials"] if t["datetime_start"]
        ]
        study_start = min(starts) if starts else None

        git_hash, git_commit_time, git_message = (None, None, None)
        if study_start and commits:
            git_hash, git_commit_time, git_message = _find_git_hash(
                study_start, commits
            )

        for trial in info["trials"]:
            tn = trial["trial_number"]
            trial_dir = os.path.join(study_dir, f"trial_{tn:03d}")

            # Model config (ground truth for all actual hyperparameters)
            cfg = _load_yaml(os.path.join(trial_dir, "model.config.yaml"))
            cfg_flat = _flatten_config(cfg)

            # Metrics from JSONL
            jsonl_path = os.path.join(trial_dir, "train_progress.jsonl")
            metrics = _read_jsonl_metrics(jsonl_path)

            # fraction_targets_solved from prediction output
            frac_solved = _best_fraction_solved(
                os.path.join(trial_dir, "pred_routes_train_progress.json")
            )

            # Optuna-suggested params (flattened, prefixed with optuna_)
            op = trial["optuna_params"]
            optuna_flat = {f"optuna_{k}": v for k, v in op.items()}
            # Record which params were Optuna-searched vs fixed
            optuna_searched_keys = "|".join(sorted(op.keys()))

            row = {
                # Study / git provenance
                "study_name": study_name,
                "study_folder": study_folder,
                "objective_metric": objective_metric,
                "git_hash": git_hash,
                "git_hash_short": git_hash[:8] if git_hash else None,
                "git_commit_time": git_commit_time.isoformat() if git_commit_time else None,
                "git_message": git_message,
                # Trial identity
                "trial_number": tn,
                "state": trial["state"],
                "datetime_start": trial["datetime_start"],
                "datetime_complete": trial["datetime_complete"],
                "duration_min": trial["duration_min"],
                # Optuna-recorded score and searched params
                "optuna_score": trial["optuna_score"],
                "optuna_searched": optuna_searched_keys,
                **optuna_flat,
                # Actual config values (ground truth)
                **cfg_flat,
                # Metrics
                **metrics,
                "fraction_targets_solved": frac_solved,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Derived convenience columns
    if "cfg_n_heads" in df.columns and "cfg_head_dim" in df.columns:
        df["cfg_hidden_size"] = df["cfg_hidden_size"].fillna(
            df["cfg_n_heads"] * df["cfg_head_dim"]
        )

    # Sort by best metric descending (use best_valid_action_accuracy when available)
    sort_col = "best_valid_action_accuracy"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", default="results",
        help="Root directory containing hypertune-* study folders (default: results/)",
    )
    parser.add_argument(
        "--output", default="results/all_hyperparams.csv",
        help="Output CSV path (default: results/all_hyperparams.csv)",
    )
    parser.add_argument(
        "--repo", default=".",
        help="Git repo root for commit-hash lookup (default: .)",
    )
    args = parser.parse_args()

    print(f"Scanning {args.results}/ for study.db files…")
    df = aggregate(args.results, args.repo)
    print(f"Found {len(df)} trial rows across {df['study_name'].nunique()} studies.")

    df.to_csv(args.output, index=False)
    print(f"Saved → {args.output}")

    # Quick summary
    print("\nColumn groups:")
    provenance = [c for c in df.columns if c.startswith("git_") or c in ("study_name", "study_folder", "objective_metric")]
    trial_cols = [c for c in df.columns if c in ("trial_number", "state", "duration_min", "datetime_start", "optuna_score", "optuna_searched")]
    metric_cols = [c for c in df.columns if "accuracy" in c or "loss" in c or "solved" in c or "epoch" in c]
    optuna_cols = [c for c in df.columns if c.startswith("optuna_") and c not in ("optuna_score", "optuna_searched")]
    cfg_cols = [c for c in df.columns if c.startswith("cfg_")]
    print(f"  provenance : {provenance}")
    print(f"  trial      : {trial_cols}")
    print(f"  metrics    : {metric_cols}")
    print(f"  optuna_*   : {optuna_cols}")
    print(f"  cfg_*      : {cfg_cols}")

    # Top 10 by best_valid_action_accuracy
    if "best_valid_action_accuracy" in df.columns:
        print("\nTop 10 trials by best_valid_action_accuracy:")
        show = ["study_name", "trial_number", "state", "git_hash_short",
                "best_valid_action_accuracy", "best_valid_route_accuracy",
                "epoch_count", "cfg_n_heads", "cfg_n_layers", "cfg_head_dim",
                "cfg_attn_pdrop", "cfg_embd_pdrop", "cfg_resid_pdrop",
                "cfg_lr", "cfg_dataset"]
        show = [c for c in show if c in df.columns]
        print(df[show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
