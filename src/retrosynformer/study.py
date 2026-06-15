"""Read an Optuna study.db SQLite file into pandas DataFrames.

Public API
----------
to_dfs(db_path)                    -> dict[str, DataFrame]   all raw tables
to_trials_df(db_path)              -> DataFrame               one row per trial, params decoded
dfs_to_trials_df(dfs)              -> DataFrame               same, from an already-loaded dict
concat(study_a_dfs, study_b_dfs)   -> dict[str, DataFrame]   merge two study dicts
concat_all(pattern, root)          -> dict[str, DataFrame]   merge all study.db files in a glob
"""
import functools
import glob as _glob
import json
import os
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# FK / PK topology constants (derived from PRAGMA foreign_key_list)
# ---------------------------------------------------------------------------

# Tables that carry study_id (as PK in 'studies', FK elsewhere).
_STUDY_ID_TABLES = (
    "studies",
    "study_directions",
    "study_user_attributes",
    "study_system_attributes",
    "trials",
)

# Tables that carry trial_id (as PK in 'trials', FK elsewhere).
_TRIAL_ID_TABLES = (
    "trials",
    "trial_params",
    "trial_values",
    "trial_user_attributes",
    "trial_system_attributes",
    "trial_intermediate_values",
    "trial_heartbeats",
)

# Each of these tables has its own auto-increment PK that is neither
# study_id nor trial_id and must not collide across the two studies.
_OWN_PKS: dict[str, str] = {
    "study_directions":           "study_direction_id",
    "study_user_attributes":      "study_user_attribute_id",
    "study_system_attributes":    "study_system_attribute_id",
    "trial_params":               "param_id",
    "trial_values":               "trial_value_id",
    "trial_user_attributes":      "trial_user_attribute_id",
    "trial_system_attributes":    "trial_system_attribute_id",
    "trial_intermediate_values":  "trial_intermediate_value_id",
    "trial_heartbeats":           "trial_heartbeat_id",
}

# These tables are schema/migration metadata — keep A's copy unchanged.
_SINGLETON_TABLES = {"version_info", "alembic_version"}


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


def dfs_to_trials_df(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per trial with decoded params, score, and duration.

    Accepts an already-loaded dfs dict (e.g. from :func:`to_dfs` or
    :func:`concat_all`).  When *dfs* contains more than one study a
    ``study_name`` column is prepended so rows can be attributed to their
    source study.

    Columns
    -------
    study_name    : str    — present only when multiple studies are merged
    trial         : int    — trial number (sequential across merged studies)
    state         : str    — COMPLETE / RUNNING / FAIL / WAITING
    duration_min  : float  — wall-clock minutes (NaN if still running)
    <param_name>  : any    — one column per hyperparameter, decoded
    score         : float  — objective value (fraction_targets_solved), NaN if not yet recorded
    """
    # --- trials: times → duration -------------------------------------------
    t = dfs["trials"].copy()
    t["datetime_start"] = pd.to_datetime(t["datetime_start"])
    t["datetime_complete"] = pd.to_datetime(t["datetime_complete"])
    t["duration_min"] = (
        (t["datetime_complete"] - t["datetime_start"]).dt.total_seconds() / 60
    )
    trials = t[["trial_id", "study_id", "number", "state", "duration_min"]].rename(
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

    # Add study_name when multiple studies are present.
    studies = dfs["studies"]
    if len(studies) > 1:
        result = result.merge(
            studies[["study_id", "study_name"]], on="study_id", how="left"
        )
        # Move study_name to the front.
        cols = ["study_name"] + [c for c in result.columns if c not in ("study_name", "study_id", "trial_id")]
        result = result[cols]
    else:
        result = result.drop(columns=["study_id"])

    result = result.drop(columns=["trial_id"], errors="ignore")
    return result.sort_values("trial").reset_index(drop=True)


def to_trials_df(db_path: str) -> pd.DataFrame:
    """Return one row per trial with decoded params, score, and duration.

    Thin wrapper around :func:`dfs_to_trials_df` for the single-file case.
    """
    return dfs_to_trials_df(to_dfs(db_path))


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def _col_max(dfs: dict[str, pd.DataFrame], table: str, col: str, default: int = 0) -> int:
    """Return max value of *col* in *table*, or *default* if table/col is absent or empty."""
    df = dfs.get(table)
    if df is None or df.empty or col not in df.columns:
        return default
    v = df[col].max()
    return default if pd.isna(v) else int(v)


def concat(
    study_a_dfs: dict[str, pd.DataFrame],
    study_b_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Concatenate study_b rows into study_a, reindexing PKs and FKs to avoid collisions.

    All auto-increment integer keys in study_b are shifted upward by the
    corresponding maximum value found in study_a, so the combined dict has no
    duplicate key values across any table.  ``trials.number`` (the human-visible
    trial index) is also offset so it remains monotonically increasing across
    both studies.  Singleton metadata tables (``version_info``, ``alembic_version``)
    are taken from study_a unchanged.

    Parameters
    ----------
    study_a_dfs, study_b_dfs:
        Dicts returned by :func:`to_dfs`.

    Returns
    -------
    dict[str, pd.DataFrame]
        A new dict with the same table names; each DataFrame is a fresh copy
        with a clean 0-based RangeIndex.
    """
    # Offsets derived from A's maximum key values.
    study_id_offset     = _col_max(study_a_dfs, "studies", "study_id")
    trial_id_offset     = _col_max(study_a_dfs, "trials",  "trial_id")
    # trial.number is 0-based; default -1 so offset = 0 when A has no trials.
    trial_number_offset = _col_max(study_a_dfs, "trials",  "number", default=-1) + 1

    # Deep-copy B so the caller's dicts are not mutated.
    b = {k: df.copy() for k, df in study_b_dfs.items()}

    # Shift study_id everywhere it appears.
    if study_id_offset:
        for tbl in _STUDY_ID_TABLES:
            if tbl in b and "study_id" in b[tbl].columns:
                b[tbl]["study_id"] += study_id_offset

    # Shift trial_id everywhere it appears.
    if trial_id_offset:
        for tbl in _TRIAL_ID_TABLES:
            if tbl in b and "trial_id" in b[tbl].columns:
                b[tbl]["trial_id"] += trial_id_offset

    # Shift trial number so it stays sequential across the merged result.
    if trial_number_offset and "trials" in b and "number" in b["trials"].columns:
        b["trials"]["number"] += trial_number_offset

    # Shift each table's own PK (none of these are study_id or trial_id).
    for tbl, pk in _OWN_PKS.items():
        if tbl in b and pk in b[tbl].columns:
            offset = _col_max(study_a_dfs, tbl, pk)
            if offset:
                b[tbl][pk] += offset

    # Merge tables; singletons keep A's version.
    result: dict[str, pd.DataFrame] = {}
    for tbl in set(study_a_dfs) | set(b):
        if tbl in _SINGLETON_TABLES:
            result[tbl] = (study_a_dfs.get(tbl) if tbl in study_a_dfs else b[tbl]).copy()
        else:
            frames = [df for df in (study_a_dfs.get(tbl), b.get(tbl)) if df is not None]
            result[tbl] = pd.concat(frames, ignore_index=True)

    return result


# ---------------------------------------------------------------------------
# concat_all
# ---------------------------------------------------------------------------

def concat_all(
    pattern: str = "results/**/study.db",
    root: str = ".",
) -> dict[str, pd.DataFrame]:
    """Load and merge every study.db matched by *pattern* under *root*.

    Files are sorted lexicographically before merging so the result is
    deterministic regardless of filesystem order.

    Parameters
    ----------
    pattern:
        Glob pattern relative to *root* (``**`` is expanded recursively).
    root:
        Directory from which *pattern* is resolved (default: current directory).

    Returns
    -------
    dict[str, pd.DataFrame]
        Merged dfs dict ready for :func:`dfs_to_trials_df`.

    Raises
    ------
    FileNotFoundError
        If the glob matches no files.
    """
    paths = sorted(_glob.glob(os.path.join(root, pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No study.db files found matching {pattern!r} under {root!r}"
        )
    return functools.reduce(concat, (to_dfs(p) for p in paths))
