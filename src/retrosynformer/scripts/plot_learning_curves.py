#!/usr/bin/env python
"""Plot learning curves from train_progress.jsonl for the top-N Optuna trials.

Loads all study.db files matching a glob, ranks completed trials by Optuna
score (fraction_targets_solved), then overlays the per-epoch metric from each
trial's train_progress.jsonl.

Usage
-----
    retrosynformer-plot-learning-curves
    retrosynformer-plot-learning-curves --top 5 --metric valid_action_accuracy
    retrosynformer-plot-learning-curves "results/hypertune-small*/study.db" --out curves.png
"""
import argparse
import glob
import json
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from retrosynformer.study import dfs_to_trials_df, to_dfs

METRICS = [
    "valid_loss",
    "valid_action_accuracy",
    "valid_route_accuracy",
    "train_loss",
    "train_action_accuracy",
    "train_route_accuracy",
]


def _load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty or "epoch" not in df.columns:
        return df
    # If training was interrupted and restarted, epoch resets to 0.
    # Keep only the last contiguous run by finding the last backwards jump.
    epochs = df["epoch"].to_numpy()
    last_reset = 0
    for i in range(1, len(epochs)):
        if epochs[i] <= epochs[i - 1]:
            last_reset = i
    return df.iloc[last_reset:].reset_index(drop=True)


def _trials_from_db(db_path: str) -> pd.DataFrame:
    """Return completed trials from one db with db_path, db_dir, and study_name added."""
    dfs = to_dfs(db_path)
    df = dfs_to_trials_df(dfs)
    df = df[df["state"] == "COMPLETE"].copy()
    df["db_path"] = db_path
    df["db_dir"] = os.path.dirname(db_path)
    # 'trial' column == original trial number from the db (0-based per study),
    # which maps directly to trial_NNN directories on disk.
    df["original_trial"] = df["trial"]
    # Always carry study_name; for single-study dbs it isn't added by dfs_to_trials_df.
    if "study_name" not in df.columns:
        df["study_name"] = dfs["studies"]["study_name"].iloc[0]
    return df


def _jsonl_path(db_dir: str, trial_number: int) -> str:
    return os.path.join(db_dir, f"trial_{int(trial_number):03d}", "train_progress.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "pattern",
        nargs="?",
        default="results/**/study.db",
        help="Glob for study.db files (default: results/**/study.db)",
    )
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top trials to plot (default: 10)")
    parser.add_argument("--metric", default="valid_action_accuracy", choices=METRICS,
                        help="Metric to rank by and plot on y-axis (default: valid_action_accuracy)")
    parser.add_argument("--also-train", action="store_true",
                        help="Overlay the corresponding train_* metric as a dashed line")
    parser.add_argument("--min-score", type=float, default=None,
                        help="For accuracy metrics: exclude trials below this threshold. "
                             "For loss metrics: exclude trials above this threshold. "
                             "Default: 0.2 for accuracy metrics, no filter for loss metrics.")
    parser.add_argument("--study", metavar="STUDY_NAME", action="append", default=None,
                        help="Only include trials from this study name (repeat for multiple studies)")
    parser.add_argument("--out", default=None,
                        help="Save figure to this path instead of showing interactively")
    parser.add_argument("--root", default=".",
                        help="Root directory for glob resolution (default: .)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, args.pattern), recursive=True))
    if not paths:
        sys.exit(f"No study.db files found matching {args.pattern!r} under {args.root!r}")

    # Deduplicate by realpath so a nested copy (rsync artifact) isn't double-counted.
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)
    paths = unique_paths

    print(f"Found {len(paths)} study.db file(s) (after dedup):")
    for p in paths:
        print(f"  {p}")
    print()

    parts: list[pd.DataFrame] = []
    for db_path in paths:
        try:
            df = _trials_from_db(db_path)
            parts.append(df)
        except Exception as exc:
            print(f"  WARNING: could not load {db_path}: {exc}")

    if not parts:
        sys.exit("No completed trials found across all study.db files.")

    all_trials = pd.concat(parts, ignore_index=True)

    if args.study:
        available = sorted(all_trials["study_name"].unique())
        all_trials = all_trials[all_trials["study_name"].isin(args.study)]
        if all_trials.empty:
            sys.exit(f"No trials found for study name(s): {args.study}\nAvailable: {available}")
        print(f"Filtered to study name(s): {args.study}")
    all_trials["jsonl_path"] = all_trials.apply(
        lambda r: _jsonl_path(r["db_dir"], r["original_trial"]), axis=1
    )
    # Dedup: same (study_name, trial_number) means the same trial even when the
    # same study.db was rsynced into a nested directory.  Keep the row whose
    # jsonl file exists (or the first one if neither / both do).
    all_trials["_jsonl_exists"] = all_trials["jsonl_path"].map(os.path.exists)
    all_trials = (
        all_trials.sort_values(["score", "_jsonl_exists"], ascending=[False, False])
        # Pass 1: drop rsync-duplicate dbs that copied the same study under a nested dir.
        .drop_duplicates(["study_name", "original_trial"])
        # Pass 2: within a multi-study db several studies share the same trial_NNN dir;
        # keep only the highest-scoring row per unique jsonl file.
        .drop_duplicates("jsonl_path")
        .drop(columns=["_jsonl_exists"])
        .reset_index(drop=True)
    )

    # Rank by the selected --metric from the last contiguous training run.
    # Loss metrics: lower is better (min, sort ascending).
    # Accuracy metrics: higher is better (max, sort descending).
    rank_metric = args.metric
    rank_lower_is_better = "loss" in rank_metric
    # Default threshold of 0.2 only for valid_action_accuracy; other metrics have no default filter.
    min_score = args.min_score if args.min_score is not None else (0.2 if rank_metric == "valid_action_accuracy" else None)

    def _rank_value(jsonl_path: str) -> float:
        if not os.path.exists(jsonl_path):
            return float("nan")
        try:
            df = _load_jsonl(jsonl_path)
            if rank_metric in df.columns and not df.empty:
                return float(df[rank_metric].min() if rank_lower_is_better else df[rank_metric].max())
        except Exception:
            pass
        return float("nan")

    all_trials["rank_val"] = all_trials["jsonl_path"].map(_rank_value)
    all_trials = all_trials.sort_values("rank_val", ascending=rank_lower_is_better).reset_index(drop=True)

    if min_score is not None:
        if rank_lower_is_better:
            all_trials = all_trials[all_trials["rank_val"] <= min_score]
            filter_desc = f"{rank_metric} <= {min_score}"
        else:
            all_trials = all_trials[all_trials["rank_val"] >= min_score]
            filter_desc = f"{rank_metric} >= {min_score}"
        if all_trials.empty:
            sys.exit(f"No completed trials with {filter_desc}.")
    top = all_trials.head(args.top)

    # Param columns present across the top trials (exclude bookkeeping columns).
    _non_param = {"trial", "state", "duration_min", "score", "rank_val", "study_name",
                  "db_path", "db_dir", "original_trial", "jsonl_path"}
    PARAM_ORDER = ["dataset", "n_heads", "n_layers", "head_dim", "dropout", "lr",
                   "structured_dropout_bottleneck"]
    present_params = [c for c in PARAM_ORDER if c in top.columns]
    extra_params = [c for c in top.columns if c not in _non_param and c not in present_params]
    param_cols = present_params + extra_params

    def _fmt(col: str, val) -> str:
        if pd.isna(val):
            return "-"
        if col == "lr":
            return f"{val:.2e}"
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    # Shorten the metric name to fit the table column header.
    metric_hdr = rank_metric.replace("valid_", "v_").replace("train_", "t_").replace("_accuracy", "_acc")
    col_w = max(len(metric_hdr), 9)
    direction = "↑" if not rank_lower_is_better else "↓"
    fixed_hdr = f"  {'#':>3}  {metric_hdr+direction:>{col_w}}  {'optuna':>6}  {'trial':>5}  {'study':<28}"
    param_hdr = "  ".join(f"{c:<{max(len(c),6)}}" for c in param_cols)
    print(f"Top {len(top)} completed trials by {rank_metric}:")
    print(f"{fixed_hdr}  {param_hdr}")
    print(f"  {'---':>3}  {'-'*col_w}  {'------':>6}  {'-----':>5}  {'-'*28}  " +
          "  ".join("-" * max(len(c), 6) for c in param_cols))

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        study_short = os.path.basename(row["db_dir"])[:28]
        optuna_score = f"{row['score']:6.4f}" if pd.notna(row.get("score")) else "     -"
        fixed_part = (f"  #{rank:>2}  {row['rank_val']:>{col_w}.4f}  "
                      f"{optuna_score}  "
                      f"{int(row['original_trial']):>5}  {study_short:<28}")
        param_part = "  ".join(
            f"{_fmt(c, row[c]):<{max(len(c),6)}}" for c in param_cols
        )
        print(f"{fixed_part}  {param_part}")
    print()

    # Derive the matching train_* metric for --also-train.
    train_metric: Optional[str] = None
    if args.also_train:
        train_metric = args.metric.replace("valid_", "train_")
        if train_metric == args.metric:
            print("WARNING: --also-train has no effect for train_* metrics.")
            train_metric = None

    sns.set_theme(style="darkgrid", palette="tab10", font_scale=1.6)
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("tab10", n_colors=max(len(top), 10))
    plotted = 0

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        jsonl = row["jsonl_path"]
        if not os.path.exists(jsonl):
            print(f"  SKIP #{rank}: {jsonl} not found")
            continue

        try:
            progress = _load_jsonl(jsonl)
        except Exception as exc:
            print(f"  SKIP #{rank}: could not read {jsonl}: {exc}")
            continue

        if args.metric not in progress.columns:
            print(f"  SKIP #{rank}: column '{args.metric}' missing in {jsonl}")
            continue

        color = palette[plotted % len(palette)]
        study_short = os.path.basename(row["db_dir"])
        if len(study_short) > 28:
            study_short = study_short[:25] + "..."
        label = f"#{rank} t{int(row['original_trial'])} {study_short} ({metric_hdr}={row['rank_val']:.4f})"

        ax.plot(progress["epoch"], progress[args.metric],
                label=label, color=color, linewidth=5.0, alpha=0.5)

        if train_metric and train_metric in progress.columns:
            ax.plot(progress["epoch"], progress[train_metric],
                    color=color, linewidth=2.8, linestyle="--", alpha=0.5)

        plotted += 1

    if plotted == 0:
        sys.exit("No train_progress.jsonl files could be loaded for the top trials.")

    y_label = args.metric.replace("_", " ")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_yscale("log")
    title = f"Learning curves — top {plotted} trials by Optuna score  ({y_label})"
    if train_metric:
        title += f"\nsolid={args.metric}  dashed={train_metric}"
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
