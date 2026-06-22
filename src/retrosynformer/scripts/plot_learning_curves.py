#!/usr/bin/env python
"""Plot learning curves from train_progress.jsonl for the top-N Optuna trials.

Loads all study.db files matching a glob, ranks completed trials by Optuna
score (fraction_targets_solved), then overlays the per-epoch metric from each
trial's train_progress.jsonl.

Output is an interactive Plotly HTML file (or browser window when --out is omitted).
Use --out curves.png to save a static image (requires: pip install kaleido).

Usage
-----
    rs-plot-learning-curves
    rs-plot-learning-curves --top 5 --metric valid_action_accuracy
    rs-plot-learning-curves --metric valid_route_accuracy --metric train_route_accuracy --metric valid_action_accuracy
    rs-plot-learning-curves "results/hypertune-small*/study.db" --out curves.html
    rs-plot-learning-curves "results/hypertune-small*/study.db" --out curves.png
"""
import argparse
import glob
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

from retrosynformer.dataframes import (
    build_trials_df,
    find_trial_base,
    jsonl_best_metrics,
    jsonl_path,
    jsonl_rank_stats,
    load_jsonl,
    load_run_params,
    print_and_save_trials_table,
    trials_df_from_db,
)
from retrosynformer.names import abbreviate
from retrosynformer.scripts import add_log_args, configure_logging, print_banner

METRICS = [
    "valid_loss",
    "valid_action_accuracy",
    "valid_route_accuracy",
    "train_loss",
    "train_action_accuracy",
    "train_route_accuracy",
]

_LINEDASHES = ["solid", "dash", "dot", "dashdot"]


def _resolve_metrics(spec: str) -> list[str]:
    """Expand a partial metric name to a list of full METRICS names.

    Exact names pass through unchanged.  For partial names, substring-match
    against METRICS; if the spec contains no 'train' qualifier, prefer the
    valid_* variants when both exist (e.g. 'accuracy' → valid_action_accuracy +
    valid_route_accuracy rather than all four accuracy metrics).
    """
    if spec in METRICS:
        return [spec]
    matches = [m for m in METRICS if spec in m]
    if not matches:
        return []
    if "train" not in spec:
        valid_only = [m for m in matches if m.startswith("valid_")]
        if valid_only:
            return valid_only
    return matches


def main() -> None:
    print_banner()
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
    parser.add_argument("-m", "--metric", action="append", dest="metrics",
                        metavar="METRIC",
                        help="Metric(s) to plot (repeat for multiple); partial names are expanded "
                             "(e.g. 'accuracy' → valid_action_accuracy + valid_route_accuracy). "
                             "Trials are ranked by the first metric (default: valid_action_accuracy). "
                             f"Full names: {', '.join(METRICS)}")
    parser.add_argument("--also-train", action="store_true",
                        help="For each valid_* metric, also overlay the corresponding train_* metric")
    parser.add_argument("--xscale", default="linear", choices=["linear", "log"],
                        help="X-axis scale (default: linear)")
    parser.add_argument("--yscale", default="log", choices=["linear", "log"],
                        help="Y-axis scale (default: log)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="For accuracy metrics: exclude trials below this threshold. "
                             "For loss metrics: exclude trials above this threshold.")
    parser.add_argument("-s", "--study", metavar="STUDY_NAME", action="append", default=None,
                        help="Only include trials from this study name (repeat for multiple studies)")
    parser.add_argument("--out", default=None,
                        help="Save to file (.html for interactive, .png/.svg for static image). "
                             "Omit to open in browser.")
    parser.add_argument("--root", default=".",
                        help="Root directory for glob resolution (default: .)")
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--estimate", action="store_true",
                        help="Show polynomial extrapolation of the final-epoch objective: "
                             "adds obj_estim column to the table and an x marker on each "
                             "trial's curve at the projected target epoch.")
    add_log_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    raw_specs: list[str] = args.metrics or ["valid_action_accuracy"]
    metrics: list[str] = []
    _seen_metrics: set[str] = set()
    for spec in raw_specs:
        resolved = _resolve_metrics(spec)
        if not resolved:
            parser.error(f"Unknown metric {spec!r}. Full names: {', '.join(METRICS)}")
        for m in resolved:
            if m not in _seen_metrics:
                metrics.append(m)
                _seen_metrics.add(m)

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
            df = trials_df_from_db(db_path)
            parts.append(df)
        except Exception as exc:
            print(f"  WARNING: could not load {db_path}: {exc}")

    parts = [p for p in parts if not p.empty]
    if not parts:
        sys.exit("No completed trials found across all study.db files.")

    all_trials = pd.concat(parts, ignore_index=True)

    if args.study:
        available = sorted(all_trials["study_name"].unique())

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
        lambda r: jsonl_path(r["trial_base_dir"], r["original_trial"]), axis=1
    )
    all_trials["_jsonl_exists"] = all_trials["jsonl_path"].map(os.path.exists)
    all_trials = (
        all_trials.sort_values(["score", "_jsonl_exists"], ascending=[False, False])
        .drop_duplicates(["study_name", "original_trial"])
        .drop_duplicates("jsonl_path")
        .drop(columns=["_jsonl_exists"])
        .reset_index(drop=True)
    )

    rank_metric = metrics[0]
    rank_lower_is_better = "loss" in rank_metric
    min_score = args.min_score if args.min_score is not None else (0.2 if rank_metric == "valid_action_accuracy" else None)

    all_trials[["rank_val", "n_epochs"]] = all_trials["jsonl_path"].apply(
        lambda p: jsonl_rank_stats(p, rank_metric, rank_lower_is_better)
    )
    all_trials = all_trials.sort_values("rank_val", ascending=rank_lower_is_better).reset_index(drop=True)

    # Table metrics: always include valid_action_accuracy and valid_route_accuracy
    # so the table shows each per-epoch metric in its own correctly-labeled column.
    TABLE_METRICS = ["valid_action_accuracy", "valid_route_accuracy"]

    if min_score is not None:
        # Keep trials with NaN rank_val (no local jsonl yet — e.g. RUNNING on remote);
        # only drop trials where we have data showing they're below the threshold.
        if rank_lower_is_better:
            mask = all_trials["rank_val"].isna() | (all_trials["rank_val"] <= min_score)
        else:
            mask = all_trials["rank_val"].isna() | (all_trials["rank_val"] >= min_score)
        all_trials = all_trials[mask]
        if all_trials.empty:
            sys.exit(f"No trials passing the min-score filter.")

    top = all_trials.head(args.top).copy()

    # Load best values for each table metric from each trial's jsonl.
    # Each metric gets its own column _best_<metric> so build_trials_df can
    # show them separately without cross-contaminating column labels.
    all_table_metrics = sorted(set(TABLE_METRICS) | {rank_metric})
    for m in all_table_metrics:
        col = f"_best_{m}"
        if col not in top.columns:
            top[col] = top["jsonl_path"].apply(
                lambda p, _m=m: jsonl_best_metrics(p, [_m]).get(_m)
            )

    # Overwrite _best_{rank_metric} with rank_val (already computed, consistent).
    top[f"_best_{rank_metric}"] = top["rank_val"]

    # Determine the Optuna objective metric from any trial that has a config file.
    # The objective is a study-level property; trials without configs fall back to
    # the default, which we should not let drown out an actual configured value.
    from retrosynformer.models_optuna import (
        _DEFAULT_OBJECTIVE_METRIC as _DOM,
        _trial_objective_metric as _tom,
    )
    obj_metrics = top.apply(
        lambda r: _tom(r["db_path"], int(r["original_trial"])), axis=1
    )
    # Prefer non-default values (i.e., values from trials that actually had configs).
    real_obj_metrics = obj_metrics[obj_metrics != _DOM]
    if not real_obj_metrics.empty:
        optuna_objective_metric = real_obj_metrics.mode().iloc[0]
    elif not obj_metrics.empty:
        optuna_objective_metric = obj_metrics.mode().iloc[0]
    else:
        optuna_objective_metric = rank_metric

    # --estimate: polynomial extrapolation to the final epoch for each trial.
    # Key by (db_path, trial_number) to avoid collisions when multiple studies share
    # the same study_name string stored in their study.db.
    estimates: dict[tuple[str, int], dict] | None = None
    if args.estimate:
        from retrosynformer.models_optuna import connect as _orm_connect, estimate_incomplete_objectives
        estimates = {}
        for db_path in top["db_path"].unique():
            session = _orm_connect(db_path, readonly=True)
            try:
                raw = estimate_incomplete_objectives(
                    db_path, session,
                    metric=rank_metric,
                    states=None,  # include COMPLETE, FAIL, RUNNING
                )
            except Exception as exc:
                print(f"  WARNING: estimate_incomplete_objectives failed for {db_path}: {exc}")
                raw = {}
            finally:
                session.close()
            for trial_num, est in raw.items():
                estimates[(str(db_path), int(trial_num))] = est

    _run_cache: dict[str, dict[int, tuple[dict, list[str]]]] = {}
    for tbd in top["trial_base_dir"].unique():
        _run_cache[str(tbd)] = load_run_params(str(tbd))

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

    _non_param = {"trial", "state", "duration_min", "score", "rank_val", "n_epochs",
                  "study_name", "db_path", "db_dir", "trial_base_dir", "original_trial",
                  "jsonl_path", "score_is_estimated", "_estimated_value", "_target_epoch"}
    # Exclude all _best_* columns from the param list.
    _non_param |= {f"_best_{m}" for m in all_table_metrics}
    PARAM_ORDER = ["dataset", "n_heads", "n_layers", "head_dim", "dropout", "lr"]
    _HIDDEN_PARAMS = {"structured_dropout_bottleneck", "structured_dropout_rate"}
    present_params = [c for c in PARAM_ORDER if c in top.columns]
    extra_params = [c for c in top.columns if c not in _non_param and c not in present_params and c not in _HIDDEN_PARAMS]
    param_cols = present_params + extra_params

    # Merge estimates into top so build_trials_df can access them as row values
    if estimates is not None:
        top["_estimated_value"] = top.apply(
            lambda r: estimates.get((str(r["db_path"]), int(r["original_trial"])), {}).get("estimated_value"),
            axis=1,
        )
        top["_target_epoch"] = top.apply(
            lambda r: estimates.get((str(r["db_path"]), int(r["original_trial"])), {}).get("target_epoch"),
            axis=1,
        )

    rank_metric_short = abbreviate(rank_metric)
    df = build_trials_df(
        top, rank_metric, rank_metric_short, rank_lower_is_better,
        param_cols, optuna_col_set,
        show_estimate=estimates is not None,
        table_metrics=TABLE_METRICS,
        optuna_objective_metric=optuna_objective_metric,
    )
    df = print_and_save_trials_table(df, top, rank_metric)

    palette = pc.qualitative.D3  # 10 distinct colors
    fig = go.Figure()
    plotted = 0

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        jsonl = row["jsonl_path"]
        if not os.path.exists(jsonl):
            print(f"  SKIP #{rank}: {jsonl} not found")
            continue
        try:
            progress = load_jsonl(jsonl)
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
        group_title = f"{study_short} t{int(row['original_trial'])}"

        for m_idx, metric in enumerate(metrics):
            if metric not in progress.columns:
                print(f"  SKIP #{rank} {metric}: column missing in {jsonl}")
                continue
            metric_short = abbreviate(metric)
            lower = "loss" in metric
            if args.xmax is not None and "epoch" in progress.columns:
                _mask = progress["epoch"] <= args.xmax
                _subset = progress[_mask] if _mask.any() else progress
            else:
                _subset = progress
            best_val = float(_subset[metric].min() if lower else _subset[metric].max())
            name = f"{metric_short} ({best_val:.4f})"
            dash = _LINEDASHES[m_idx % len(_LINEDASHES)]
            is_first = m_idx == 0
            fig.add_trace(go.Scatter(
                x=progress["epoch"],
                y=progress[metric],
                name=name,
                legendgroup=f"trial_{rank}",
                legendgrouptitle=dict(text=group_title) if is_first else None,
                showlegend=True,
                line=dict(color=color, dash=dash, width=2.5 if is_first else 1.5),
                opacity=0.75,
                mode="lines",
            ))

        plotted += 1

        # --estimate: add an x marker at the projected final-epoch value
        if estimates is not None:
            est_val = row.get("_estimated_value")
            target_ep = row.get("_target_epoch")
            if est_val is not None and pd.notna(est_val):
                if target_ep is None or pd.isna(target_ep):
                    target_ep = float(progress["epoch"].max())
                est_label = f"#{rank} estimate ({float(est_val):.4f})"
                fig.add_trace(go.Scatter(
                    x=[float(target_ep)],
                    y=[float(est_val)],
                    name=est_label,
                    legendgroup=f"trial_{rank}",
                    showlegend=True,
                    mode="markers",
                    marker=dict(symbol="x", size=14, color=color, line=dict(width=2.5)),
                ))

    if plotted == 0:
        sys.exit("No train_progress.jsonl files could be loaded for the top trials.")

    y_label = metrics[0].replace("_", " ") if len(metrics) == 1 else "metric value"
    fig.update_layout(
        title=f"Learning curves — top {plotted} trials by {rank_metric.replace('_', ' ')}",
        xaxis_title="Epoch",
        yaxis_title=y_label,
        xaxis_type=args.xscale if args.xscale != "linear" else "-",
        yaxis_type="log" if args.yscale == "log" else "-",
        template="plotly_white",
        height=600,
        legend=dict(groupclick="toggleitem"),
        hovermode="x unified",
    )

    if args.xmin is not None or args.xmax is not None:
        fig.update_xaxes(range=[args.xmin, args.xmax])
    if args.ymin is not None or args.ymax is not None:
        fig.update_yaxes(range=[args.ymin, args.ymax])

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        if ext == ".html":
            fig.write_html(args.out)
        else:
            try:
                fig.write_image(args.out)
            except Exception as exc:
                sys.exit(f"Could not write image to {args.out}: {exc}\nHint: pip install kaleido")
        print(f"Saved to {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
