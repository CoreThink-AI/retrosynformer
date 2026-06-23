"""Utilities for loading, manipulating, and displaying trial/study DataFrames."""
import json
import os
import sqlite3

import pandas as pd

from retrosynformer.names import abbreviate
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


def jsonl_best_metrics(path: str, metrics: list[str]) -> dict[str, float]:
    """Return best (max) value for each metric from a trial's JSONL file.

    Only metrics that are actually present in the log are returned.
    """
    if not os.path.exists(path):
        return {}
    try:
        df = load_jsonl(path)
        if df.empty:
            return {}
        result = {}
        for m in metrics:
            if m in df.columns:
                lower = "loss" in m
                result[m] = float(df[m].min() if lower else df[m].max())
        return result
    except Exception:
        return {}


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


def _config_yaml_params(trial_dir: str) -> dict:
    """Read fixed (non-Optuna) params from a trial's model.config.yaml.

    Returns a flat dict with keys like ``dataset_name``, ``valid_set``, etc.
    Only extracts params that appear in ``names.ABBREV`` to avoid polluting the
    table with every config key.
    """
    import yaml
    from retrosynformer.names import ABBREV

    cfg_path = os.path.join(trial_dir, "model.config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    params: dict = {}
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if k in ABBREV:
                    params[k] = v
    return params


def load_run_params(trial_base_dir: str) -> dict[int, tuple[dict, list[str]]]:
    """Load run.jsonl and return per-trial hyperparameter values and optuna-searched keys.

    Fixed study-level params (e.g. ``dataset_name``) that are absent from
    ``run.jsonl`` are read from whichever trial has a ``model.config.yaml``
    and merged into every trial's param dict.  This handles studies where
    ``trial_start`` events were only logged for some trials.
    """
    path = os.path.join(trial_base_dir, "run.jsonl")
    result: dict[int, tuple[dict, list[str]]] = {}
    if os.path.exists(path):
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

    # Collect fixed params from any trial that has model.config.yaml.
    # These are study-level constants not captured in run.jsonl trial_start events.
    shared_cfg: dict = {}
    try:
        entries = sorted(os.listdir(trial_base_dir))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.startswith("trial_"):
            continue
        trial_dir = os.path.join(trial_base_dir, entry)
        if not os.path.isdir(trial_dir):
            continue
        cfg_params = _config_yaml_params(trial_dir)
        if cfg_params:
            shared_cfg.update(cfg_params)
            break  # one config file is enough for study-level constants

    if not shared_cfg:
        return result

    # Merge shared cfg params into all known trials (without overwriting Optuna params).
    for n, (params, keys) in list(result.items()):
        merged = dict(shared_cfg)
        merged.update(params)
        result[n] = (merged, keys)

    # For any trial_NNN directory not yet in result, add an entry with shared cfg.
    for entry in entries:
        if not entry.startswith("trial_"):
            continue
        trial_dir = os.path.join(trial_base_dir, entry)
        if not os.path.isdir(trial_dir):
            continue
        try:
            n = int(entry[len("trial_"):])
        except ValueError:
            continue
        if n not in result:
            result[n] = (dict(shared_cfg), [])

    return result


def trials_df_from_db(db_path: str) -> pd.DataFrame:
    """Load an Optuna study.db and return an enriched one-row-per-trial DataFrame.

    Extends the standard dfs_to_trials_df output with db_path, db_dir,
    study_name (always present), original_trial, trial_base_dir, and
    score_is_estimated columns.
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

    # Read rs_estimated_objective user attribute: True when the objective value
    # in trial_values was filled in by complete_stale_trials (proxy-estimated),
    # not measured by an actual route evaluation.
    estimated_set: set[int] = set()
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute(
            "SELECT t.number FROM trials t"
            " JOIN trial_user_attributes a ON t.trial_id = a.trial_id"
            " WHERE a.key = 'rs_estimated_objective' AND a.value_json = '\"1\"'"
        ).fetchall()
        con.close()
        estimated_set = {r[0] for r in rows}
    except Exception:
        pass
    df["score_is_estimated"] = df["original_trial"].isin(estimated_set)

    # dfs_to_trials_df pivots trial_user_attributes into columns.
    # Drop rs_estimated_objective since we've already captured it above.
    df = df.drop(columns=["rs_estimated_objective"], errors="ignore")

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
    table_metrics: list[str] | None = None,
    optuna_objective_metric: str | None = None,
) -> pd.DataFrame:
    """Build a tidy display DataFrame summarising the top-ranked trials.

    Parameters
    ----------
    table_metrics:
        Ordered list of per-epoch metric names to show as columns (best value
        from the trial's jsonl).  Each is shown in its own column named by
        shortening the metric.  ``rank_metric`` is always the first; extras
        are appended without duplication.  Defaults to ``[rank_metric]``.
    optuna_objective_metric:
        The metric name that Optuna optimises (e.g. ``"fraction_solved"``).
        Its column header gets a ``*`` suffix.  When this equals ``rank_metric``
        the ``*`` is added to the rank column instead of a new column.
        When it differs, a new column is added showing the Optuna trial_values
        (``score``); values filled in by proxy extrapolation are prefixed ``~``.
        Defaults to ``rank_metric`` (no extra column).
    """
    direction = "↑" if not rank_lower_is_better else "↓"

    if table_metrics is None:
        table_metrics = [rank_metric]
    else:
        # ensure rank_metric is first without duplication
        seen: set[str] = set()
        ordered: list[str] = []
        for m in [rank_metric] + list(table_metrics):
            if m not in seen:
                ordered.append(m)
                seen.add(m)
        table_metrics = ordered

    if optuna_objective_metric is None:
        optuna_objective_metric = rank_metric

    _metric_short = abbreviate

    def _hdr(c: str) -> str:
        short = abbreviate(c)
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

    # Determine column headers for each table metric.
    # The Optuna objective metric gets a "^" suffix; the rank metric gets "↑"/"↓".
    # When a metric is both the rank metric and the Optuna objective, both markers apply.
    def _metric_col_hdr(m: str) -> str:
        short = _metric_short(m)
        if m == rank_metric:
            short += direction
        if m == optuna_objective_metric:
            short += "^"
        return short

    # Determine whether to add a separate Optuna-objective column.
    # Only needed when the objective is not already a per-epoch table metric.
    obj_col_name: str | None = None
    obj_col_hdr: str | None = None
    if optuna_objective_metric not in table_metrics:
        obj_col_name = _metric_short(optuna_objective_metric) + "^"
        obj_col_hdr = obj_col_name

    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        rec: dict = {"rank": f"#{rank}"}

        # Per-epoch metric columns (from jsonl best values stored in top).
        for m in table_metrics:
            hdr = _metric_col_hdr(m)
            col_key = f"_best_{m}"
            val = row.get(col_key)
            if val is not None and pd.notna(val):
                rec[hdr] = f"{float(val):.4f}"
            else:
                rec[hdr] = "(no data)"

        # Optuna objective column — only when the objective is not a per-epoch metric.
        # Never show proxy-estimated values here: the column is labeled for a
        # specific metric and only real measurements belong in it.
        score = row.get("score")
        is_estimated = bool(row.get("score_is_estimated", False))
        if obj_col_hdr is not None:
            if score is not None and pd.notna(score) and not is_estimated:
                rec[obj_col_hdr] = f"{float(score):.4f}"
            else:
                rec[obj_col_hdr] = "-"

        if show_estimate:
            est_val = row.get("_estimated_value")
            if est_val is not None and pd.notna(est_val):
                rec["obj_estim"] = f"{est_val:.4f}"
            elif score is not None and pd.notna(score) and not is_estimated:
                rec["obj_estim"] = f"{float(score):.4f}"
            else:
                rec["obj_estim"] = "-"

        rec[abbreviate("n_epochs")] = int(row["n_epochs"]) if pd.notna(row.get("n_epochs")) else 0
        rec["state"] = str(row.get("state", ""))
        rec["trial"] = int(row["original_trial"])
        rec["study"] = str(row["study_name"])
        for c in param_cols:
            rec[_hdr(c)] = _fmt(c, row.get(c))
        rows.append(rec)

    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Return a space-aligned Markdown table string from a DataFrame.

    Uses ``pd.DataFrame.to_markdown`` (requires ``tabulate``), which pads every
    column with spaces so the table is human-readable in plain text and renders
    correctly in Markdown.  Always call this instead of hand-writing Markdown
    table syntax or using ``df.to_string``.

    Args:
        df:    DataFrame to render.
        index: Whether to include the row index (default False).

    Returns:
        Multi-line string containing the Markdown table.

    Raises:
        ImportError: if ``tabulate`` is not installed (``uv pip install tabulate``).
    """
    try:
        return df.to_markdown(index=index)
    except ImportError as exc:
        raise ImportError(
            "df_to_markdown requires 'tabulate'. Install with: uv pip install tabulate"
        ) from exc


def print_and_save_trials_table(df: pd.DataFrame, top: pd.DataFrame, rank_metric: str) -> pd.DataFrame:
    """Merge per-trial best-epoch columns into the summary table, then print and save.

    The two logical tables (Optuna summary + per-epoch training metrics) are
    concatenated column-wise into a single display DataFrame so each trial
    occupies exactly one row.  Columns that appear in both tables (trial,
    study, valid_action_accuracy, valid_route_accuracy) are kept from the
    Optuna summary side and dropped from the training-metrics side.  The
    leading columns are reordered to: study, dataset, trial, epochs.
    """
    from retrosynformer.training_display import EPOCH_TABLE_COLS, iter_jsonl_rows

    # jsonl keys whose values already appear in df under different headers.
    _SKIP_JSONL_KEYS = {
        "trial_number",
        "study_name",
        "valid_action_accuracy",
        "valid_route_accuracy",
    }

    lower = "loss" in rank_metric

    # Build one training-metrics row per trial (in top's rank order).
    t2_rows: list[dict] = []
    for _, row in top.iterrows():
        jp = str(row.get("jsonl_path", ""))
        if not os.path.exists(jp):
            t2_rows.append({})
            continue
        best_rec = None
        best_val = float("inf") if lower else float("-inf")
        for rec in iter_jsonl_rows(jp):
            v = rec.get(rank_metric)
            if v is None:
                continue
            if (lower and v < best_val) or (not lower and v > best_val):
                best_val = v
                best_rec = rec
        if best_rec is None:
            t2_rows.append({})
            continue
        t2_row: dict = {}
        for hdr, key, _, fmt_fn in EPOCH_TABLE_COLS:
            if key in _SKIP_JSONL_KEYS:
                continue
            val = best_rec.get(key)
            if val is None:
                t2_row[hdr] = "-"
            else:
                try:
                    t2_row[hdr] = fmt_fn(val)
                except (TypeError, ValueError):
                    t2_row[hdr] = str(val)
        t2_rows.append(t2_row)

    # Concatenate column-wise: summary columns first, then unique training cols.
    t2_df = pd.DataFrame(t2_rows, index=df.index)
    df = pd.concat([df, t2_df], axis=1)

    # Drop display-only columns not useful in the summary table.
    df = df.drop(columns=["rank", "note"], errors="ignore")

    # Reorder leading columns: study, dataset, trial, epochs.
    _LEAD = ["study", "dataset", "trial", "epochs"]
    _rest = [c for c in df.columns if c not in _LEAD]
    df = df[[c for c in _LEAD if c in df.columns] + _rest]

    # Drop all-empty columns.
    _empty = {"", "-", "nan", "none", "null"}
    df = df.loc[:, ~df.apply(
        lambda col: col.map(
            lambda v: str(v).strip().lower() in _empty or (isinstance(v, float) and pd.isna(v))
        ).all()
    )]

    from retrosynformer.names import unabbreviate
    legend = {col: unabbreviate(col.rstrip("*^↑↓")) for col in df.columns}
    legend = {k: v for k, v in legend.items() if v != k.rstrip("*^↑↓")}
    if legend:
        print("Column legend:")
        for abbr, full in legend.items():
            print(f"  {abbr}: {full}")
        print()

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
