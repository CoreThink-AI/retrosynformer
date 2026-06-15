"""Read an Optuna study.db SQLite file into pandas DataFrames.

Public API
----------
to_dfs(db_path)       -> dict[str, DataFrame]   all raw tables
to_trials_df(db_path) -> DataFrame               one row per trial, params decoded
"""
import json
import sqlite3

import pandas as pd


def to_dfs(db_path: str) -> dict[str, pd.DataFrame]:
    """Return every table in an Optuna SQLite study.db as a dict of DataFrames."""
    con = sqlite3.connect(db_path)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        return {t: pd.read_sql(f"SELECT * FROM {t}", con) for t in tables}
    finally:
        con.close()


def _decode_param(param_value: float, distribution_json: str) -> object:
    """Convert a raw Optuna float param_value to its actual Python value.

    Optuna stores all parameter values as floats.  For CategoricalDistribution
    the float is an index into the choices list, not the value itself.
    """
    dist = json.loads(distribution_json)
    if dist["name"] == "CategoricalDistribution":
        return dist["attributes"]["choices"][int(param_value)]
    return param_value


def to_trials_df(db_path: str) -> pd.DataFrame:
    """Return one row per trial with decoded params, score, and duration.

    Columns
    -------
    trial         : int    — trial number (0-based)
    state         : str    — COMPLETE / RUNNING / FAIL / WAITING
    duration_min  : float  — wall-clock minutes (NaN if still running)
    <param_name>  : any    — one column per hyperparameter, decoded
    score         : float  — objective value (fraction_targets_solved), NaN if not yet recorded
    """
    dfs = to_dfs(db_path)

    # --- trials: times → duration -------------------------------------------
    t = dfs["trials"].copy()
    t["datetime_start"] = pd.to_datetime(t["datetime_start"])
    t["datetime_complete"] = pd.to_datetime(t["datetime_complete"])
    t["duration_min"] = (
        (t["datetime_complete"] - t["datetime_start"]).dt.total_seconds() / 60
    )
    trials = t[["trial_id", "number", "state", "duration_min"]].rename(
        columns={"number": "trial"}
    )

    # --- params: pivot wide, decoding categoricals --------------------------
    params = dfs["trial_params"].copy()
    params["decoded"] = params.apply(
        lambda r: _decode_param(r["param_value"], r["distribution_json"]), axis=1
    )
    params_wide = (
        params.pivot_table(
            index="trial_id", columns="param_name", values="decoded", aggfunc="first"
        )
        .reset_index()
    )
    params_wide.columns.name = None

    # --- objective values ----------------------------------------------------
    values = (
        dfs["trial_values"][["trial_id", "value"]]
        .rename(columns={"value": "score"})
    )

    # --- join ----------------------------------------------------------------
    result = trials.merge(params_wide, on="trial_id", how="left")
    result = result.merge(values, on="trial_id", how="left")
    result = result.drop(columns=["trial_id"])
    return result.sort_values("trial").reset_index(drop=True)
