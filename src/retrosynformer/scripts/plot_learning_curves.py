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
import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

from retrosynformer.study import dfs_to_trials_df, to_dfs
from retrosynformer.scripts import print_banner

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
    epochs = df["epoch"].to_numpy()
    last_reset = 0
    for i in range(1, len(epochs)):
        if epochs[i] <= epochs[i - 1]:
            last_reset = i
    return df.iloc[last_reset:].reset_index(drop=True)


def _find_trial_base(db_dir: str, study_name: str) -> str:
    db_dir_abs = os.path.abspath(db_dir)
    candidates = [
        os.path.join(os.path.dirname(db_dir_abs), f"hypertune-{study_name}"),
        os.path.join("results", f"hypertune-{study_name}"),
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
    dfs = to_dfs(db_path)
    df = dfs_to_trials_df(dfs)
    df = df[df["state"].isin({"COMPLETE", "RUNNING", "FAIL"})].copy()
    db_dir = os.path.dirname(os.path.abspath(db_path))
    df["db_path"] = db_path
    df["db_dir"] = db_dir
    df["original_trial"] = df["trial"]
    if "study_name" not in df.columns:
        df["study_name"] = dfs["studies"]["study_name"].iloc[0]
    df["trial_base_dir"] = df["study_name"].map(
        lambda sn: _find_trial_base(db_dir, sn)
    )
    return df


def _jsonl_path(trial_base_dir: str, trial_number: int) -> str:
    return os.path.join(trial_base_dir, f"trial_{int(trial_number):03d}", "train_progress.jsonl")


def _load_run_params(trial_base_dir: str) -> dict[int, tuple[dict, list[str]]]:
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


def _build_table_df(
    top: pd.DataFrame,
    rank_metric: str,
    rank_metric_short: str,
    rank_lower_is_better: bool,
    param_cols: list,
    optuna_col_set: set,
    show_estimate: bool = False,
) -> pd.DataFrame:
    """Build a tidy DataFrame representing the ranked-trials summary table."""
    direction = "↑" if not rank_lower_is_better else "↓"
    metric_col = rank_metric_short + direction

    _ABBREV = {
        "early_stopping_patience": "es_patience",
        "eval_routes_frequency": "eval_freq",
        "objective_metric": "obj_metric",
        "batch_size": "bs",
        "n_layers": "layers",
        "n_heads": "heads",
        "head_dim": "h_dim",
        "n_in_state": "n_state",
        "fp_dim": "fp",
        "weight_decay": "wd",
        "valid_set": "vset",
        "beam_width": "bw",
        "dropout": "drop",
    }

    def _hdr(c: str) -> str:
        short = _ABBREV.get(c, c)
        return short + "*" if c in optuna_col_set else short

    def _fmt(col: str, val) -> str:
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

    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        rec: dict = {"rank": f"#{rank}"}
        rec[metric_col] = f"{row['rank_val']:.4f}" if pd.notna(row.get("rank_val")) else "(no data)"
        rec["optuna"] = f"{row['score']:.4f}" if pd.notna(row.get("score")) else "-"
        if show_estimate:
            est_val = row.get("_estimated_value")
            if est_val is not None and pd.notna(est_val):
                rec["obj_estim"] = f"{est_val:.4f}"
            elif pd.notna(row.get("score")):
                rec["obj_estim"] = f"{row['score']:.4f}"
            else:
                rec["obj_estim"] = "-"
        rec["ep"] = int(row["n_epochs"]) if pd.notna(row.get("n_epochs")) else 0
        rec["state"] = str(row.get("state", ""))
        rec["trial"] = int(row["original_trial"])
        rec["study"] = str(row["study_name"])
        for c in param_cols:
            rec[_hdr(c)] = _fmt(c, row.get(c))
        rows.append(rec)

    return pd.DataFrame(rows)


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
    args = parser.parse_args()

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
            df = _trials_from_db(db_path)
            parts.append(df)
        except Exception as exc:
            print(f"  WARNING: could not load {db_path}: {exc}")

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
        lambda r: _jsonl_path(r["trial_base_dir"], r["original_trial"]), axis=1
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

    _non_param = {"trial", "state", "duration_min", "score", "rank_val", "n_epochs",
                  "study_name", "db_path", "db_dir", "trial_base_dir", "original_trial", "jsonl_path"}
    PARAM_ORDER = ["dataset", "n_heads", "n_layers", "head_dim", "dropout", "lr"]
    _HIDDEN_PARAMS = {"structured_dropout_bottleneck", "structured_dropout_rate"}
    present_params = [c for c in PARAM_ORDER if c in top.columns]
    extra_params = [c for c in top.columns if c not in _non_param and c not in present_params and c not in _HIDDEN_PARAMS]
    param_cols = present_params + extra_params

    # Merge estimates into top so _build_table_df can access them as row values
    if estimates is not None:
        top["_estimated_value"] = top.apply(
            lambda r: estimates.get((str(r["db_path"]), int(r["original_trial"])), {}).get("estimated_value"),
            axis=1,
        )
        top["_target_epoch"] = top.apply(
            lambda r: estimates.get((str(r["db_path"]), int(r["original_trial"])), {}).get("target_epoch"),
            axis=1,
        )

    rank_metric_short = rank_metric.replace("valid_", "v_").replace("train_", "t_").replace("_accuracy", "_acc")
    table_df = _build_table_df(top, rank_metric, rank_metric_short, rank_lower_is_better,
                               param_cols, optuna_col_set, show_estimate=estimates is not None)
    _empty = {"", "-", "nan", "none", "null"}
    table_df = table_df.loc[:, ~table_df.apply(
        lambda col: col.map(lambda v: str(v).strip().lower() in _empty or (isinstance(v, float) and pd.isna(v))).all()
    )]
    print(f"Top {len(top)} trials by {rank_metric}:")
    print(table_df.to_string(index=False))
    print()

    # Save table to CSV in the highest-ranked trial's study directory
    if not table_df.empty:
        top_study = str(top.iloc[0]["study_name"])
        top_base = str(top.iloc[0]["trial_base_dir"])
        csv_path = os.path.join(top_base, f"top_trials_{top_study}.csv")
        table_df.to_csv(csv_path, index=False)
        print(f"Table saved to {csv_path}")
        print()

    palette = pc.qualitative.D3  # 10 distinct colors
    fig = go.Figure()
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
        group_title = f"{study_short} t{int(row['original_trial'])}"

        for m_idx, metric in enumerate(metrics):
            if metric not in progress.columns:
                print(f"  SKIP #{rank} {metric}: column missing in {jsonl}")
                continue
            metric_short = metric.replace("valid_", "v_").replace("train_", "t_").replace("_accuracy", "_acc")
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
