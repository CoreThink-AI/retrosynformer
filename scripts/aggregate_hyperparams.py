#!/usr/bin/env python
"""Aggregate all Optuna study.db files into a comprehensive hyperparameter table.

For each trial the output row contains:
  - Optuna-suggested params (what the sampler chose)
  - Actual model params from model.config.yaml (ground truth after overrides)
  - Final & best metrics from train_progress.jsonl
  - Git commit hash that was active when the study started, found by searching
    the log for the most-recent commit at least 1 minute before the first trial.
  - Completeness flags and estimated final metrics for truncated trials.

Implicit hyperparameters (action_dim → dataset size, reward settings, early-stopping
patience, etc.) are included alongside architecture params so the full table can be
analysed for correlated patterns.

Completeness columns
--------------------
epoch_count           : epochs completed in the last (or only) contiguous run
total_jsonl_epochs    : raw line count in train_progress.jsonl (> epoch_count if restarted)
epoch_ran_fraction    : epoch_count / cfg_n_epochs (how much of the scheduled run finished)
max_complete_epoch_in_study : max epoch_count of COMPLETE trials in the same study
is_incomplete         : True when a non-COMPLETE trial ran < 80% of max_complete_epoch
is_early_stopped      : True when a COMPLETE trial ran < 80% of max_complete_epoch
                        (converged early — not misleading, just efficient)
incomplete_reason     : "complete" | "early_stopped" | "nearly_complete" |
                        "killed" | "running"
estimated_valid_action_accuracy : for incomplete trials — estimated final value by
    scaling best_valid_action_accuracy at epoch k by the ratio
    (complete-trial final best) / (complete-trial best up to epoch k),
    averaged over all COMPLETE reference trials in the same study
estimated_valid_route_accuracy  : same, for route accuracy
estimation_n_ref      : number of COMPLETE reference trials used in the estimate

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

import numpy as np
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

def _read_jsonl_full(jsonl_path: str) -> tuple[dict[int, dict], int]:
    """Return (curve, total_lines) where curve is {epoch: record} for the last
    contiguous run and total_lines is the raw count of all valid JSON lines."""
    if not os.path.exists(jsonl_path):
        return {}, 0
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    total_lines = len(records)
    if not records:
        return {}, 0
    # Restart detection: keep the last contiguous run
    last_reset = 0
    for i in range(1, len(records)):
        if records[i].get("epoch", i) <= records[i - 1].get("epoch", i - 1):
            last_reset = i
    records = records[last_reset:]
    curve = {r["epoch"]: r for r in records if "epoch" in r}
    return curve, total_lines


def _read_jsonl_metrics(curve: dict[int, dict]) -> dict:
    """Summarise a pre-loaded JSONL curve dict into scalar metric columns."""
    if not curve:
        return {}
    records = list(curve.values())
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

        obj_row = con.execute(
            "SELECT value_json FROM study_system_attributes "
            "WHERE study_id=? AND key='objective_metric'", (study_id,)
        ).fetchone()
        objective_metric = (
            obj_row["value_json"].strip('"') if obj_row else "valid_route_accuracy"
        )

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
# Completeness analysis (pass 2)
# ---------------------------------------------------------------------------

_INCOMPLETE_THRESHOLD = 0.80  # trial ran < this fraction of max_complete_epoch → flagged


def _best_up_to_epoch(curve: dict[int, dict], last_epoch: int, key: str) -> float | None:
    """Max value of *key* across epochs 0..last_epoch (inclusive) in *curve*."""
    vals = [curve[e][key] for e in range(last_epoch + 1)
            if e in curve and curve[e].get(key) is not None]
    return max(vals) if vals else None


def _add_incomplete_analysis(
    df: pd.DataFrame,
    full_curves: dict[tuple, dict],
) -> pd.DataFrame:
    """Add completeness flags and estimated final metrics to *df* (in-place columns)."""

    # Per study: max epoch_count among COMPLETE trials only.
    max_complete_epoch: dict[str, float | None] = {}
    for sname, grp in df.groupby("study_name"):
        complete_epochs = grp.loc[grp["state"] == "COMPLETE", "epoch_count"].dropna()
        max_complete_epoch[sname] = float(complete_epochs.max()) if len(complete_epochs) else None

    reasons = []
    is_incomplete_col = []
    is_early_stopped_col = []
    is_jsonl_unreliable_col = []
    max_ep_col = []
    epoch_ran_frac_col = []
    est_vaa_col = []
    est_vra_col = []
    est_n_ref_col = []

    for _, row in df.iterrows():
        sname = row["study_name"]
        tn = int(row["trial_number"]) if pd.notna(row.get("trial_number")) else None
        state = row["state"]
        epoch_count = row.get("epoch_count")
        cfg_n_epochs = row.get("cfg_n_epochs")
        total_jsonl = row.get("total_jsonl_epochs")
        max_ep = max_complete_epoch.get(sname)

        # epoch_ran_fraction: how far through the scheduled run this trial got
        epoch_ran_frac = None
        if pd.notna(epoch_count) and pd.notna(cfg_n_epochs) and cfg_n_epochs:
            epoch_ran_frac = round(float(epoch_count) / float(cfg_n_epochs), 4)

        # JSONL restart: epoch_count is from the last contiguous run, but the trial
        # actually ran more epochs across multiple restarts. If total_jsonl_epochs is
        # substantially larger than epoch_count, metrics in the CSV come from a late
        # restart chunk and may not reflect the full training run.
        has_restart = (
            pd.notna(total_jsonl) and pd.notna(epoch_count)
            and float(total_jsonl) > float(epoch_count) * 1.5
        )

        max_ep_col.append(max_ep)
        epoch_ran_frac_col.append(epoch_ran_frac)

        is_jsonl_unreliable = False
        is_inc = False
        is_es = False

        # Classify completeness
        if pd.isna(epoch_count):
            # No JSONL found at all — cannot say anything about training progress
            reason = "no_jsonl"
        elif max_ep is None:
            # No COMPLETE trial in this study to use as a reference for max epochs
            reason = "no_complete_ref"
        elif state == "COMPLETE":
            if has_restart and float(total_jsonl) >= float(max_ep):
                # Trial completed successfully (Optuna=COMPLETE), then was restarted.
                # epoch_count reflects only the last restart chunk, not the full run.
                # Metrics (valid_action_accuracy, etc.) are from that chunk — unreliable.
                # Example: standard-v2-lr0005 trial 6 (total=215, epoch_count=15, cfg=100)
                reason = "complete_restarted"
                is_jsonl_unreliable = True
            elif epoch_count < max_ep * _INCOMPLETE_THRESHOLD:
                reason = "early_stopped"   # converged early via early-stopping — not misleading
                is_es = True
            else:
                reason = "complete"
        else:
            frac_done = float(epoch_count) / float(max_ep)
            if frac_done >= _INCOMPLETE_THRESHOLD:
                reason = "nearly_complete"  # ran almost to completion before dying
            elif state == "RUNNING":
                reason = "running"
                is_inc = True
            else:
                reason = "killed"
                is_inc = True

        reasons.append(reason)
        is_incomplete_col.append(is_inc)
        is_early_stopped_col.append(is_es)
        is_jsonl_unreliable_col.append(is_jsonl_unreliable)

        # Estimate final metrics for genuinely incomplete trials
        est_vaa = est_vra = None
        n_ref = 0

        if is_inc and pd.notna(epoch_count) and epoch_count > 0 and max_ep is not None:
            last_ep = int(epoch_count) - 1   # 0-based index of last epoch the trial saw

            # Reference: COMPLETE trials in the same study
            ref_df = df[(df["study_name"] == sname) & (df["state"] == "COMPLETE")]
            scales_vaa, scales_vra = [], []

            for _, ref_row in ref_df.iterrows():
                ref_tn = int(ref_row["trial_number"]) if pd.notna(ref_row["trial_number"]) else None
                ref_curve = full_curves.get((sname, ref_tn), {})
                if not ref_curve or last_ep not in ref_curve:
                    continue

                # Best the reference trial achieved up to last_ep (fair comparison point)
                ref_best_at_k_vaa = _best_up_to_epoch(ref_curve, last_ep, "valid_action_accuracy")
                ref_final_vaa = ref_row.get("best_valid_action_accuracy")
                if (ref_best_at_k_vaa and ref_best_at_k_vaa > 0
                        and pd.notna(ref_final_vaa) and ref_final_vaa > 0):
                    scales_vaa.append(ref_final_vaa / ref_best_at_k_vaa)

                ref_best_at_k_vra = _best_up_to_epoch(ref_curve, last_ep, "valid_route_accuracy")
                ref_final_vra = ref_row.get("best_valid_route_accuracy")
                if (ref_best_at_k_vra and ref_best_at_k_vra > 0
                        and pd.notna(ref_final_vra) and ref_final_vra > 0):
                    scales_vra.append(ref_final_vra / ref_best_at_k_vra)

            n_ref = max(len(scales_vaa), len(scales_vra))
            inc_curve = full_curves.get((sname, tn), {})
            inc_best_vaa = row.get("best_valid_action_accuracy")
            inc_best_vra = row.get("best_valid_route_accuracy")

            if scales_vaa and pd.notna(inc_best_vaa) and inc_best_vaa:
                est_vaa = round(float(inc_best_vaa) * float(np.mean(scales_vaa)), 6)
            if scales_vra and pd.notna(inc_best_vra) and inc_best_vra:
                est_vra = round(float(inc_best_vra) * float(np.mean(scales_vra)), 6)

        est_vaa_col.append(est_vaa)
        est_vra_col.append(est_vra)
        est_n_ref_col.append(n_ref if is_inc else None)

    df = df.copy()
    df["max_complete_epoch_in_study"] = max_ep_col
    df["epoch_ran_fraction"] = epoch_ran_frac_col
    df["is_incomplete"] = is_incomplete_col
    df["is_early_stopped"] = is_early_stopped_col
    df["is_jsonl_unreliable"] = is_jsonl_unreliable_col
    df["incomplete_reason"] = reasons
    df["estimated_valid_action_accuracy"] = est_vaa_col
    df["estimated_valid_route_accuracy"] = est_vra_col
    df["estimation_n_ref"] = est_n_ref_col
    return df


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

    rows: list[dict] = []
    # full_curves[(study_name, trial_number)] = {epoch: record_dict}
    full_curves: dict[tuple, dict] = {}

    for db_path in unique_db_paths:
        study_dir = os.path.dirname(db_path)
        study_folder = os.path.relpath(study_dir, results_root)

        info = _load_study(db_path)
        if not info:
            continue

        study_name = info["study_name"]
        objective_metric = info["objective_metric"]

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

            cfg = _load_yaml(os.path.join(trial_dir, "model.config.yaml"))
            cfg_flat = _flatten_config(cfg)

            # Full learning curve (pass-2 needs this for estimation)
            jsonl_path = os.path.join(trial_dir, "train_progress.jsonl")
            curve, total_jsonl_epochs = _read_jsonl_full(jsonl_path)
            full_curves[(study_name, tn)] = curve

            # Last epoch index (0-based) from the last contiguous JSONL run, or 0 if none.
            jsonl_last_epoch = max(curve.keys()) if curve else 0

            # Scalar metrics from the curve
            metrics = _read_jsonl_metrics(curve)

            frac_solved = _best_fraction_solved(
                os.path.join(trial_dir, "pred_routes_train_progress.json")
            )

            op = trial["optuna_params"]
            optuna_flat = {f"optuna_{k}": v for k, v in op.items()}
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
                # Optuna
                "optuna_score": trial["optuna_score"],
                "optuna_searched": optuna_searched_keys,
                **optuna_flat,
                # Actual config
                **cfg_flat,
                # Metrics
                **metrics,
                "jsonl_last_epoch": jsonl_last_epoch,
                "total_jsonl_epochs": total_jsonl_epochs if total_jsonl_epochs else None,
                "fraction_targets_solved": frac_solved,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if "cfg_n_heads" in df.columns and "cfg_head_dim" in df.columns:
        df["cfg_hidden_size"] = df["cfg_hidden_size"].fillna(
            df["cfg_n_heads"] * df["cfg_head_dim"]
        )

    # Pass 2: completeness flags and estimated metrics
    df = _add_incomplete_analysis(df, full_curves)

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

    # Completeness summary
    if "incomplete_reason" in df.columns:
        print("\nCompleteness breakdown:")
        print(df["incomplete_reason"].value_counts().to_string())

    print("\nIncomplete trials with estimates:")
    inc = df[df.get("is_incomplete", pd.Series(False, index=df.index))]
    if len(inc):
        show = ["study_name", "trial_number", "state", "epoch_count",
                "max_complete_epoch_in_study", "epoch_ran_fraction",
                "best_valid_action_accuracy", "estimated_valid_action_accuracy",
                "estimation_n_ref"]
        show = [c for c in show if c in inc.columns]
        print(inc[show].to_string(index=False))
    else:
        print("  (none)")

    print("\nEpoch count verification (jsonl_last_epoch vs epoch_count-1 vs total_jsonl_epochs):")
    if "total_jsonl_epochs" in df.columns and "epoch_count" in df.columns:
        restarted = df[df["total_jsonl_epochs"].fillna(0) > df["epoch_count"].fillna(0)]
        if len(restarted):
            print(f"  {len(restarted)} trial(s) had restarts (total_jsonl_epochs > epoch_count):")
            show = [c for c in ["study_name", "trial_number", "state",
                                "jsonl_last_epoch", "epoch_count",
                                "total_jsonl_epochs", "cfg_n_epochs",
                                "incomplete_reason", "is_jsonl_unreliable"]
                    if c in restarted.columns]
            print(restarted[show].to_string(index=False))
        else:
            print("  All epoch_count values match total_jsonl_epochs — no restarts detected.")

    if "is_jsonl_unreliable" in df.columns:
        unreliable = df[df["is_jsonl_unreliable"] == True]
        if len(unreliable):
            print(f"\n  WARNING: {len(unreliable)} COMPLETE trial(s) have unreliable JSONL metrics")
            print("  (metrics come from a late restart chunk, not the full training run):")
            show = [c for c in ["study_name", "trial_number", "jsonl_last_epoch",
                                "epoch_count", "total_jsonl_epochs", "cfg_n_epochs",
                                "best_valid_action_accuracy"]
                    if c in unreliable.columns]
            print(unreliable[show].to_string(index=False))

    # Top 10
    if "best_valid_action_accuracy" in df.columns:
        print("\nTop 10 trials by best_valid_action_accuracy:")
        show = ["study_name", "trial_number", "state", "incomplete_reason",
                "best_valid_action_accuracy", "estimated_valid_action_accuracy",
                "epoch_count", "cfg_n_heads", "cfg_n_layers", "cfg_head_dim",
                "cfg_attn_pdrop", "cfg_embd_pdrop", "cfg_resid_pdrop",
                "cfg_lr", "cfg_dataset"]
        show = [c for c in show if c in df.columns]
        print(df[show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
