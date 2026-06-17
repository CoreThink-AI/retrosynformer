#!/usr/bin/env python
"""Plot learning curves from train_progress.jsonl for the top-N Optuna trials.

Loads all study.db files matching a glob, ranks completed trials by Optuna
score (fraction_targets_solved), then overlays the per-epoch metric from each
trial's train_progress.jsonl.

Usage
-----
    rs-plot-learning-curves
    rs-plot-learning-curves --top 5 --metric valid_action_accuracy
    rs-plot-learning-curves --metric valid_route_accuracy --metric train_route_accuracy --metric valid_action_accuracy
    rs-plot-learning-curves "results/hypertune-small*/study.db" --out curves.png
"""
import argparse
import glob
import json
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def _find_trial_base(db_dir: str, study_name: str) -> str:
    """Return the directory that contains trial_NNN/ subdirectories.

    For the normal per-study case (one study.db per hypertune run) the trial
    dirs live directly under db_dir, so that's returned unchanged.

    For the shared-storage case (``--storage`` pointing two parallel runs at
    the same SQLite file) the trial dirs live under
    ``results/hypertune-{study_name}/`` which is a sibling of db_dir.  We
    probe candidate locations in priority order and fall back to db_dir.
    """
    db_dir_abs = os.path.abspath(db_dir)
    candidates = [
        # Sibling hypertune-{study_name}/ dir (shared-storage case).
        os.path.join(os.path.dirname(db_dir_abs), f"hypertune-{study_name}"),
        # results/hypertune-{study_name}/ relative to cwd.
        os.path.join("results", f"hypertune-{study_name}"),
        # db_dir itself (per-study db; trial dirs live alongside study.db).
        db_dir_abs,
    ]
    for c in candidates:
        if os.path.isdir(c) and any(
            e.startswith("trial_") and os.path.isdir(os.path.join(c, e))
            for e in os.listdir(c)
        ):
            return c
    return db_dir


def _trials_from_db(db_path: str) -> pd.DataFrame:
    """Return COMPLETE and RUNNING trials from one db with db_path, db_dir, study_name added."""
    dfs = to_dfs(db_path)
    df = dfs_to_trials_df(dfs)
    df = df[df["state"].isin({"COMPLETE", "RUNNING"})].copy()
    db_dir = os.path.dirname(os.path.abspath(db_path))
    df["db_path"] = db_path
    df["db_dir"] = db_dir
    # 'trial' column == original trial number from the db (0-based per study),
    # which maps directly to trial_NNN directories on disk.
    df["original_trial"] = df["trial"]
    # Always carry study_name; for single-study dbs it isn't added by dfs_to_trials_df.
    if "study_name" not in df.columns:
        df["study_name"] = dfs["studies"]["study_name"].iloc[0]
    # Resolve per-study trial directory; may differ from db_dir for shared dbs.
    df["trial_base_dir"] = df["study_name"].map(
        lambda sn: _find_trial_base(db_dir, sn)
    )
    return df


def _jsonl_path(trial_base_dir: str, trial_number: int) -> str:
    return os.path.join(trial_base_dir, f"trial_{int(trial_number):03d}", "train_progress.jsonl")


def _load_run_params(trial_base_dir: str) -> dict[int, tuple[dict, list[str]]]:
    """Read run.jsonl and return {trial_num: (all_params, optuna_keys)} for trial_start events."""
    path = os.path.join(trial_base_dir, "run.jsonl")
    result: dict[int, tuple[dict, list[str]]] = {}
    if not os.path.exists(path):
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "trial_start" and "all_params" in rec:
                n = rec.get("trial", {}).get("number")
                if n is not None:
                    result[int(n)] = (rec["all_params"], rec.get("optuna_keys", []))
    return result


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
    parser.add_argument("--metric", action="append", dest="metrics",
                        metavar="METRIC", choices=METRICS,
                        help="Metric(s) to plot (repeat for multiple); trials are ranked by "
                             "the first metric (default: valid_action_accuracy)")
    parser.add_argument("--also-train", action="store_true",
                        help="For each valid_* metric, also overlay the corresponding train_* metric")
    parser.add_argument("--xscale", default="linear", choices=["linear", "log", "symlog", "logit"],
                        help="X-axis scale (default: linear)")
    parser.add_argument("--yscale", default="log", choices=["linear", "log", "symlog", "logit"],
                        help="Y-axis scale (default: log)")
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
    parser.add_argument("--xmin", type=float, default=None, help="X-axis lower limit")
    parser.add_argument("--xmax", type=float, default=None, help="X-axis upper limit")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis lower limit")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis upper limit")
    args = parser.parse_args()

    metrics: list[str] = args.metrics or ["valid_action_accuracy"]
    if args.also_train:
        extras = [
            m.replace("valid_", "train_") for m in metrics
            if m.startswith("valid_") and m.replace("valid_", "train_") not in metrics
        ]
        if not extras:
            print("WARNING: --also-train has no effect (no valid_* metrics without a train counterpart already listed).")
        metrics = metrics + extras

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
        # Match by substring against both the Optuna study_name and the results
        # directory basename, so users can pass either form or a partial prefix.
        # e.g. "compare_small_standard" matches "compare_small_standard_dropout_baseline"
        #      "hypertune-compare_small_standard" matches the directory name
        def _matches(row: "pd.Series") -> bool:
            targets = [row["study_name"], os.path.basename(row["db_dir"])]
            return any(s in t for s in args.study for t in targets)

        mask = all_trials.apply(_matches, axis=1)
        all_trials = all_trials[mask]
        if all_trials.empty:
            sys.exit(f"No trials found matching study name(s): {args.study}\nAvailable: {available}")
        matched = sorted(all_trials["study_name"].unique())
        print(f"Matched study name(s): {matched}")
    all_trials["jsonl_path"] = all_trials.apply(
        lambda r: _jsonl_path(r["trial_base_dir"], r["original_trial"]), axis=1
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
    rank_metric = metrics[0]
    rank_lower_is_better = "loss" in rank_metric
    # Default threshold of 0.2 only for valid_action_accuracy; other metrics have no default filter.
    min_score = args.min_score if args.min_score is not None else (0.2 if rank_metric == "valid_action_accuracy" else None)

    def _jsonl_stats(jsonl_path: str) -> "pd.Series":
        nan = pd.Series({"rank_val": float("nan"), "n_epochs": 0})
        if not os.path.exists(jsonl_path):
            return nan
        try:
            df = _load_jsonl(jsonl_path)
            if df.empty:
                return nan
            rank_val = float(df[rank_metric].min() if rank_lower_is_better else df[rank_metric].max()) \
                if rank_metric in df.columns else float("nan")
            return pd.Series({"rank_val": rank_val, "n_epochs": len(df)})
        except Exception:
            return nan

    all_trials[["rank_val", "n_epochs"]] = all_trials["jsonl_path"].apply(_jsonl_stats)
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

    # Supplement Optuna trial params with fixed architecture params from run.jsonl.
    _run_cache: dict[str, dict[int, tuple[dict, list[str]]]] = {}
    for tbd in top["trial_base_dir"].unique():
        _run_cache[str(tbd)] = _load_run_params(str(tbd))

    optuna_col_set: set[str] = set()
    new_cols: set[str] = set()
    for _, row in top.iterrows():
        all_p, okeys = _run_cache.get(str(row["trial_base_dir"]), {}).get(int(row["original_trial"]), ({}, []))
        new_cols.update(all_p.keys())
        optuna_col_set.update(okeys)

    for col in new_cols:
        if col not in top.columns:
            top[col] = top.apply(
                lambda r, c=col: _run_cache.get(str(r["trial_base_dir"]), {})
                    .get(int(r["original_trial"]), ({}, []))[0].get(c),
                axis=1,
            )

    # Param columns present across the top trials (exclude bookkeeping columns).
    _non_param = {"trial", "state", "duration_min", "score", "rank_val", "n_epochs",
                  "study_name", "db_path", "db_dir", "trial_base_dir", "original_trial", "jsonl_path"}
    PARAM_ORDER = ["dataset", "n_heads", "n_layers", "head_dim", "dropout", "lr"]
    _HIDDEN_PARAMS = {"structured_dropout_bottleneck", "structured_dropout_rate"}
    present_params = [c for c in PARAM_ORDER if c in top.columns]
    extra_params = [c for c in top.columns if c not in _non_param and c not in present_params and c not in _HIDDEN_PARAMS]
    param_cols = present_params + extra_params

    def _hdr(c: str) -> str:
        return c + "*" if c in optuna_col_set else c

    def _fmt(col: str, val) -> str:
        # Check for lists before pd.isna — isna raises ValueError on list inputs.
        if isinstance(val, list):
            return f"[{val[0]} ... {val[-1]}]" if len(val) > 3 else str(val)
        if isinstance(val, str) and val.startswith("["):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list) and len(parsed) > 3:
                    return f"[{parsed[0]} ... {parsed[-1]}]"
            except (ValueError, json.JSONDecodeError):
                pass
        try:
            if pd.isna(val):
                return "-"
        except (TypeError, ValueError):
            pass
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
    fixed_hdr = f"  {'#':>3}  {metric_hdr+direction:>{col_w}}  {'optuna':>6}  {'ep':>4}  {'trial':>5}  {'study':<40}"
    param_hdr = "  ".join(f"{_hdr(c):<{max(len(_hdr(c)),6)}}" for c in param_cols)
    print(f"Top {len(top)} completed trials by {rank_metric}:")
    print(f"{fixed_hdr}  {param_hdr}")
    print(f"  {'---':>3}  {'-'*col_w}  {'------':>6}  {'----':>4}  {'-----':>5}  {'-'*40}  " +
          "  ".join("-" * max(len(_hdr(c)), 6) for c in param_cols))

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        study_short = str(row["study_name"])[:40]
        optuna_score = f"{row['score']:6.4f}" if pd.notna(row.get("score")) else "     -"
        n_ep = int(row["n_epochs"]) if pd.notna(row.get("n_epochs")) else 0
        fixed_part = (f"  #{rank:>2}  {row['rank_val']:>{col_w}.4f}  "
                      f"{optuna_score}  {n_ep:>4}  "
                      f"{int(row['original_trial']):>5}  {study_short:<40}")
        param_part = "  ".join(
            f"{_fmt(c, row[c]):<{max(len(_hdr(c)),6)}}" for c in param_cols
        )
        print(f"{fixed_part}  {param_part}")
    print()

    LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]

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

        if not any(m in progress.columns for m in metrics):
            print(f"  SKIP #{rank}: none of {metrics} found in {jsonl}")
            continue

        color = palette[plotted % len(palette)]
        study_short = str(row["study_name"])
        if len(study_short) > 28:
            study_short = study_short[:25] + "..."

        for m_idx, metric in enumerate(metrics):
            if metric not in progress.columns:
                print(f"  SKIP #{rank} {metric}: column missing in {jsonl}")
                continue
            ls = LINESTYLES[m_idx % len(LINESTYLES)]
            lw = 4.0 if m_idx == 0 else 2.5
            # Only the first metric line per trial gets a legend entry.
            label = (f"#{rank} t{int(row['original_trial'])} {study_short} ({metric_hdr}={row['rank_val']:.4f})"
                     if m_idx == 0 else "_nolegend_")
            ax.plot(progress["epoch"], progress[metric],
                    label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.5)

        plotted += 1

    if plotted == 0:
        sys.exit("No train_progress.jsonl files could be loaded for the top trials.")

    y_label = metrics[0].replace("_", " ") if len(metrics) == 1 else "metric value"
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)
    ax.set_title(f"Learning curves — top {plotted} trials by {rank_metric.replace('_', ' ')}",
                 fontsize=15)
    ax.tick_params(labelsize=11)

    if len(metrics) > 1:
        # Two-part legend: trial colors (upper-left) + metric linestyles (lower-right).
        trial_legend = ax.legend(fontsize=11, loc="upper left", framealpha=0.8)
        ax.add_artist(trial_legend)
        metric_handles = [
            Line2D([0], [0], color="gray", linestyle=LINESTYLES[i % len(LINESTYLES)],
                   linewidth=2.5, label=m.replace("_", " "))
            for i, m in enumerate(metrics)
        ]
        ax.legend(handles=metric_handles, fontsize=11, loc="lower right",
                  framealpha=0.8, title="metrics", title_fontsize=11)
    else:
        ax.legend(fontsize=11, loc="best", framealpha=0.8)

    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(left=args.xmin, right=args.xmax)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
