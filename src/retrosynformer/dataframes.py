"""Utilities for loading, manipulating, and displaying trial/study DataFrames."""
import json
import os

import pandas as pd

from retrosynformer.study import dfs_to_trials_df, to_dfs

# ---------------------------------------------------------------------------
# Display constants (shared by show_study, show_all_studies, plot_learning_curves)
# ---------------------------------------------------------------------------

SCIENTIFIC_COLS: set[str] = {"lr"}
SCORE_COLS: set[str] = {
    "score", "duration_min", "dropout",
    "valid_action_accuracy", "valid_route_accuracy", "estimated_score",
}
HIGHER_IS_BETTER: set[str] = {"score", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"}
METRIC_COLS: list[str] = ["n_epochs", "valid_action_accuracy", "valid_route_accuracy", "estimated_score"]


def fmt_trial_value(col: str, val) -> str:
    """Format a single trial table cell for terminal display."""
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


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    """Load a train_progress.jsonl into a DataFrame, discarding epochs before the last restart."""
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


def jsonl_rank_stats(path: str, rank_metric: str, rank_lower_is_better: bool) -> pd.Series:
    """Return rank_val and n_epochs for a single trial's JSONL file."""
    nan = pd.Series({"rank_val": float("nan"), "n_epochs": 0})
    if not os.path.exists(path):
        return nan
    try:
        df = load_jsonl(path)
        if df.empty:
            return nan
        rank_val = (
            float(df[rank_metric].min() if rank_lower_is_better else df[rank_metric].max())
            if rank_metric in df.columns else float("nan")
        )
        return pd.Series({"rank_val": rank_val, "n_epochs": len(df)})
    except Exception:
        return nan


# ---------------------------------------------------------------------------
# Trial directory helpers
# ---------------------------------------------------------------------------

def jsonl_path(trial_base_dir: str, trial_number: int) -> str:
    """Return the expected path to a trial's train_progress.jsonl."""
    return os.path.join(trial_base_dir, f"trial_{int(trial_number):03d}", "train_progress.jsonl")


def find_trial_base(db_dir: str, study_name: str) -> str:
    """Locate the trial base directory relative to a study.db file."""
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


def load_run_params(trial_base_dir: str) -> dict[int, tuple[dict, list[str]]]:
    """Load run.jsonl and return per-trial hyperparameter values and optuna-searched keys."""
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


def trials_df_from_db(db_path: str) -> pd.DataFrame:
    """Load an Optuna study.db and return an enriched one-row-per-trial DataFrame.

    Extends the standard dfs_to_trials_df output with db_path, db_dir,
    study_name (always present), original_trial, and trial_base_dir columns.
    """
    dfs = to_dfs(db_path)
    df = dfs_to_trials_df(dfs)
    df = df[df["state"].isin({"COMPLETE", "RUNNING", "FAIL"})].copy()
    db_dir = os.path.dirname(os.path.abspath(db_path))
    df["db_path"] = db_path
    df["db_dir"] = db_dir
    df["original_trial"] = df["trial"]
    if "study_name" not in df.columns:
        df["study_name"] = dfs["studies"]["study_name"].iloc[0]
    df["trial_base_dir"] = df["study_name"].map(lambda sn: find_trial_base(db_dir, sn))
    return df


# ---------------------------------------------------------------------------
# Table construction and display
# ---------------------------------------------------------------------------

def build_trials_df(
    top: pd.DataFrame,
    rank_metric: str,
    rank_metric_short: str,
    rank_lower_is_better: bool,
    param_cols: list,
    optuna_col_set: set,
    show_estimate: bool = False,
) -> pd.DataFrame:
    """Build a tidy display DataFrame summarising the top-ranked trials."""
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


def print_and_save_trials_table(df: pd.DataFrame, top: pd.DataFrame, rank_metric: str) -> pd.DataFrame:
    """Drop all-empty columns, print, and save the trials summary table; return the filtered DataFrame."""
    _empty = {"", "-", "nan", "none", "null"}
    df = df.loc[:, ~df.apply(
        lambda col: col.map(lambda v: str(v).strip().lower() in _empty or (isinstance(v, float) and pd.isna(v))).all()
    )]
    print(f"Top {len(top)} trials by {rank_metric}:")
    print(df.to_string(index=False))
    print()
    if not df.empty:
        top_study = str(top.iloc[0]["study_name"])
        top_base = str(top.iloc[0]["trial_base_dir"])
        csv_path = os.path.join(top_base, f"top_trials_{top_study}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Table saved to {csv_path}")
        print()
    return df
