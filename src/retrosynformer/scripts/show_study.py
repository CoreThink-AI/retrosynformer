#!/usr/bin/env python
"""Print a summary table of Optuna trial results from a study.db file.

Usage
-----
    rs-show-study results/hypertune/study.db
    rs-show-study results/hypertune-my-study/study.db --sort score
"""
import argparse
import os
import sys

import pandas as pd

from retrosynformer.study import (
    dfs_to_trials_df,
    inject_estimated_scores,
    inject_train_metrics,
    to_dfs,
)

SCIENTIFIC_COLS = {"lr"}
SCORE_COLS = {"score", "duration_min", "dropout", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"}
HIGHER_IS_BETTER = {"score", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"}
# Appear just before "score" in the output regardless of sort order.
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
    parser.add_argument("db", help="Path to Optuna study.db SQLite file")
    parser.add_argument(
        "--sort", default="trial",
        help="Column to sort by (default: trial). Use 'score' to rank by objective.",
    )
    parser.add_argument(
        "--ascending", action=argparse.BooleanOptionalAction, default=None,
        help="Sort order (default: ascending for trial, descending for score).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"Not found: {args.db}")

    # --- study metadata ------------------------------------------------------
    dfs = to_dfs(args.db)
    study_name = dfs["studies"]["study_name"].iloc[0]
    schema_ver = dfs["version_info"]["schema_version"].iloc[0]
    lib_ver = dfs["version_info"]["library_version"].iloc[0]
    print(f"Study : {study_name}")
    print(f"Optuna: schema {schema_ver}  lib {lib_ver}")
    print(f"DB    : {args.db}")
    print()

    # --- per-trial summary ---------------------------------------------------
    dfs = inject_train_metrics(dfs, args.db)
    dfs = inject_estimated_scores(dfs, args.db)
    df = dfs_to_trials_df(dfs)
    if df.empty:
        print("No trials found.")
        return

    # Sort
    sort_col = args.sort
    if sort_col not in df.columns:
        sys.exit(f"Unknown sort column '{sort_col}'. Available: {list(df.columns)}")
    ascending = args.ascending
    if ascending is None:
        ascending = sort_col not in HIGHER_IS_BETTER
    df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # Mark the best completed trial
    completed = df[df["state"] == "COMPLETE"]
    best_trial = completed.loc[completed["score"].idxmax(), "trial"] if not completed.empty else None

    # Determine column order: fixed cols first, then params alphabetically,
    # then any present train-metric cols, then score last.
    fixed = ["trial", "state", "duration_min"]
    extras = [c for c in METRIC_COLS if c in df.columns]
    param_cols = sorted(c for c in df.columns if c not in fixed + extras + ["score"])
    ordered_cols = fixed + param_cols + extras + ["score"]
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    # Build header
    col_widths = {}
    for c in ordered_cols:
        values_str = [_fmt_value(c, v) for v in df[c]]
        col_widths[c] = max(len(c), max(len(s) for s in values_str))

    header = "  ".join(c.ljust(col_widths[c]) for c in ordered_cols)
    sep = "  ".join("-" * col_widths[c] for c in ordered_cols)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        marker = " *" if row["trial"] == best_trial else "  "
        line = "  ".join(_fmt_value(c, row[c]).ljust(col_widths[c]) for c in ordered_cols)
        print(f"{line}{marker}")

    print()

    # --- aggregate stats over completed trials -------------------------------
    if not completed.empty:
        print(f"Completed: {len(completed)} / {len(df)} trials")
        print(f"Best score: {completed['score'].max():.4f}  (trial {int(best_trial)})")
        print(f"Mean score: {completed['score'].mean():.4f}  ± {completed['score'].std():.4f}")
        if best_trial is not None:
            best_row = completed.loc[completed["trial"] == best_trial].iloc[0]
            best_params = {c: _fmt_value(c, best_row[c]) for c in param_cols if c in best_row}
            print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
