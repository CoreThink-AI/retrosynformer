#!/usr/bin/env python
"""Print a merged trial table across all Optuna study.db files found in a glob.

Usage
-----
    rs-show-all-studies
    rs-show-all-studies "results/**/study.db"
    rs-show-all-studies "results/hypertune-*/study.db" --sort score
"""
import argparse
import glob
import os
import sys

import pandas as pd

from retrosynformer.study import (
    concat,
    dfs_to_trials_df,
    inject_estimated_scores,
    inject_train_metrics,
    to_dfs,
)

SCIENTIFIC_COLS = {"lr"}
SCORE_COLS = {"score", "duration_min", "dropout", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"}
HIGHER_IS_BETTER = {"score", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"}
METRIC_COLS = ["n_epochs", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"]


def _fmt_value(col: str, val) -> str:
    if pd.isna(val):
        return "-"
    if isinstance(val, float) and val == int(val) and col not in SCORE_COLS:
        return str(int(val))
    if col in SCIENTIFIC_COLS:
        return f"{val:.2e}"
    if isinstance(val, float):
        prefix = "~" if col == "estimated_score" else ""
        return f"{prefix}{val:.4f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pattern",
        nargs="?",
        default="results/**/study.db",
        help="Glob pattern for study.db files (default: results/**/study.db)",
    )
    parser.add_argument(
        "--sort", default="score",
        help="Column to sort by (default: score)",
    )
    parser.add_argument(
        "--ascending", action=argparse.BooleanOptionalAction, default=None,
        help="Sort order (default: descending for score, ascending otherwise)",
    )
    parser.add_argument(
        "--root", default=".",
        help="Root directory for the glob (default: current directory)",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, args.pattern), recursive=True))
    if not paths:
        sys.exit(f"No study.db files found matching {args.pattern!r} under {args.root!r}")

    print(f"Found {len(paths)} study.db file(s):")
    for p in paths:
        print(f"  {p}")
    print()

    # Load each study, inject train metrics and estimates, then merge all.
    dfs_list = [inject_estimated_scores(inject_train_metrics(to_dfs(p), p), p) for p in paths]
    merged = dfs_list[0]
    for d in dfs_list[1:]:
        merged = concat(merged, d)

    df = dfs_to_trials_df(merged)
    if df.empty:
        print("No trials found.")
        return

    # Sort.
    sort_col = args.sort
    if sort_col not in df.columns:
        sys.exit(f"Unknown sort column '{sort_col}'. Available: {list(df.columns)}")
    ascending = args.ascending
    if ascending is None:
        ascending = sort_col not in HIGHER_IS_BETTER
    df = df.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

    # Mark the best completed trial overall.
    completed = df[df["state"] == "COMPLETE"]
    best_idx = completed["score"].idxmax() if not completed.empty else None

    # Column order: study_name first (if present), then fixed, then params,
    # then any present train-metric cols, then score last.
    fixed = ["trial", "state", "duration_min"]
    meta = ["study_name"] if "study_name" in df.columns else []
    extras = [c for c in METRIC_COLS if c in df.columns]
    param_cols = sorted(c for c in df.columns if c not in meta + fixed + extras + ["score"])
    ordered_cols = meta + fixed + param_cols + extras + ["score"]
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    # Compute column widths.
    col_widths = {}
    for c in ordered_cols:
        values_str = [_fmt_value(c, v) for v in df[c]]
        col_widths[c] = max(len(c), max((len(s) for s in values_str), default=0))

    header = "  ".join(c.ljust(col_widths[c]) for c in ordered_cols)
    sep    = "  ".join("-" * col_widths[c] for c in ordered_cols)
    print(header)
    print(sep)
    for idx, row in df.iterrows():
        marker = " *" if idx == best_idx else "  "
        line = "  ".join(_fmt_value(c, row[c]).ljust(col_widths[c]) for c in ordered_cols)
        print(f"{line}{marker}")

    print()

    # Aggregate stats.
    if not completed.empty:
        print(f"Completed: {len(completed)} / {len(df)} trials across {len(paths)} file(s)")
        print(f"Best  score : {completed['score'].max():.4f}")
        print(f"Mean  score : {completed['score'].mean():.4f}  ± {completed['score'].std():.4f}")
        if best_idx is not None:
            best_row = df.loc[best_idx]
            best_params = {c: _fmt_value(c, best_row[c]) for c in param_cols if c in best_row}
            if "study_name" in best_row:
                print(f"Best  study : {best_row['study_name']}")
            print(f"Best  params: {best_params}")


if __name__ == "__main__":
    main()
