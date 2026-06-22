"""SQLAlchemy ORM models for an Optuna study.db SQLite file.

All models are read-only by default (``connect(readonly=True)``).

Quick start
-----------
>>> from retrosynformer.models_optuna import connect
>>> session = connect("results/hypertune-foo/study.db")
>>> for study in session.query(Study).all():
...     print(study, study.best_trial)

Merge analysis
--------------
Merging two study.db files into one (different study name, union search
space) is *mostly* possible but has several sharp edges.

**What works cleanly**
- Remapping auto-increment PKs (study_id, trial_id, and all the
  per-table PKs) so there are no collisions — identical to what the
  existing ``study.concat()`` DataFrame approach does.
- Renumbering ``trials.number`` sequentially across both studies.
- Appending ``trial_values``, ``trial_intermediate_values``,
  ``trial_system_attributes``, ``trial_user_attributes`` after remapping
  trial_id.
- Merging FloatDistribution / IntDistribution search spaces: new bounds
  are ``min(A.low, B.low)`` .. ``max(A.high, B.high)``; no re-encoding
  needed because ``param_value`` IS the actual float.

**What requires careful re-encoding**
CategoricalDistribution stores ``param_value`` as an INTEGER INDEX into
the choices list, not the value itself.  If the merged choices list is
``sorted(set(A.choices) | set(B.choices))``, every existing
``param_value`` must be re-mapped:

    new_index = merged_choices.index(decode(old_param_value, old_dist))

Failing to re-encode leaves the historical trials pointing at the wrong
choices, which corrupts the TPE surrogate model.

**Where merging is impossible**
1. ``alembic_version.version_num`` differs — schema incompatibility.
2. ``version_info.schema_version`` differs — ditto.
3. ``study_directions.direction`` differs (one MAXIMIZE, one MINIMIZE) —
   the objective is not comparable; the resulting TPE model is nonsense.
4. Same ``param_name``, different distribution *types* across the two
   studies (e.g., CategoricalDistribution in A, FloatDistribution in B).
   There is no canonical union distribution for this case.
5. Multi-objective studies (``objective > 0`` rows) add complexity:
   directions must match for every objective index, and
   ``trial_values.objective`` must be consistent.

**Incomplete trials in the merged database**
RUNNING, WAITING, and FAIL trials are kept with their original state
unchanged — do NOT relabel them.  They carry no objective value and the
TPE sampler will simply ignore them when fitting its surrogate model.
Their ``trial_params`` rows (if any exist) still describe the
hyperparameter point that was being explored, which is useful for
provenance.  ``trial_heartbeats`` for RUNNING trials are stale but
harmless; drop them only if the target Optuna version complains.

**Using the Optuna API instead**
The cleanest way to avoid all of the above is to use Optuna's own
``study.add_trial()`` after decoding each historical trial to a
``FrozenTrial`` object.  Optuna then handles distribution reconciliation
internally and re-encodes param_values correctly.  The tradeoff is that
you lose the original ``trial_id`` lineage.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Schema / migration metadata
# ---------------------------------------------------------------------------

class VersionInfo(Base):
    __tablename__ = "version_info"

    version_info_id = Column(Integer, primary_key=True)
    schema_version  = Column(Integer)
    library_version = Column(String(256))

    def __repr__(self) -> str:
        return f"<VersionInfo schema={self.schema_version} lib={self.library_version}>"


class AlembicVersion(Base):
    __tablename__ = "alembic_version"

    version_num = Column(String(32), primary_key=True)

    def __repr__(self) -> str:
        return f"<AlembicVersion {self.version_num!r}>"


# ---------------------------------------------------------------------------
# Study-level tables
# ---------------------------------------------------------------------------

class Study(Base):
    __tablename__ = "studies"

    study_id   = Column(Integer, primary_key=True)
    study_name = Column(String(512), nullable=False)

    directions        = relationship("StudyDirection",       back_populates="study", cascade="all, delete-orphan")
    user_attributes   = relationship("StudyUserAttribute",   back_populates="study", cascade="all, delete-orphan")
    system_attributes = relationship("StudySystemAttribute", back_populates="study", cascade="all, delete-orphan")
    trials            = relationship("Trial", back_populates="study",
                                     cascade="all, delete-orphan",
                                     order_by="Trial.number")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def direction(self) -> Optional[str]:
        """Primary optimization direction (MAXIMIZE or MINIMIZE)."""
        return self.directions[0].direction if self.directions else None

    @property
    def complete_trials(self) -> list["Trial"]:
        return [t for t in self.trials if t.state == "COMPLETE"]

    @property
    def best_trial(self) -> Optional["Trial"]:
        ct = self.complete_trials
        if not ct:
            return None
        reverse = self.direction != "MINIMIZE"
        return sorted(ct, key=lambda t: (t.objective_value is None, t.objective_value),
                      reverse=reverse)[0]

    @property
    def search_space(self) -> dict[str, dict]:
        """Union of distributions seen across all complete trials.

        Returns {param_name: distribution_dict} where distribution_dict is
        the parsed ``distribution_json`` from the most-recently-seen trial
        for that param.  For CategoricalDistribution the choices are the
        union across all trials.
        """
        space: dict[str, dict] = {}
        for trial in self.complete_trials:
            for p in trial.params:
                dist = p.distribution
                name = p.param_name
                if name not in space:
                    space[name] = dist
                elif dist["name"] == "CategoricalDistribution":
                    old_choices = set(space[name]["attributes"]["choices"])
                    new_choices = set(dist["attributes"]["choices"])
                    merged = sorted(old_choices | new_choices,
                                    key=lambda x: (str(type(x).__name__), str(x)))
                    space[name]["attributes"]["choices"] = merged
        return space

    def __repr__(self) -> str:
        return f"<Study {self.study_name!r} ({len(self.trials)} trials)>"


class StudyDirection(Base):
    __tablename__ = "study_directions"

    study_direction_id = Column(Integer, primary_key=True)
    direction          = Column(String(8), nullable=False)  # MAXIMIZE | MINIMIZE
    study_id           = Column(Integer, ForeignKey("studies.study_id"))
    objective          = Column(Integer, nullable=False)    # 0 = primary objective

    study = relationship("Study", back_populates="directions")

    def __repr__(self) -> str:
        return f"<StudyDirection obj={self.objective} {self.direction}>"


class StudyUserAttribute(Base):
    __tablename__ = "study_user_attributes"

    study_user_attribute_id = Column(Integer, primary_key=True)
    study_id                = Column(Integer, ForeignKey("studies.study_id"))
    key                     = Column(String(512))
    value_json              = Column(Text)

    study = relationship("Study", back_populates="user_attributes")

    @property
    def value(self):
        return json.loads(self.value_json) if self.value_json else None

    def __repr__(self) -> str:
        return f"<StudyUserAttribute {self.key}={self.value!r}>"


class StudySystemAttribute(Base):
    __tablename__ = "study_system_attributes"

    study_system_attribute_id = Column(Integer, primary_key=True)
    study_id                  = Column(Integer, ForeignKey("studies.study_id"))
    key                       = Column(String(512))
    value_json                = Column(Text)

    study = relationship("Study", back_populates="system_attributes")

    @property
    def value(self):
        return json.loads(self.value_json) if self.value_json else None

    def __repr__(self) -> str:
        return f"<StudySystemAttribute {self.key}={self.value!r}>"


# ---------------------------------------------------------------------------
# Trial-level tables
# ---------------------------------------------------------------------------

class Trial(Base):
    __tablename__ = "trials"

    trial_id          = Column(Integer, primary_key=True)
    number            = Column(Integer)                          # 0-based within study
    study_id          = Column(Integer, ForeignKey("studies.study_id"))
    state             = Column(String(8), nullable=False)        # COMPLETE|RUNNING|FAIL|WAITING
    datetime_start    = Column(DateTime)
    datetime_complete = Column(DateTime)

    study               = relationship("Study", back_populates="trials")
    params              = relationship("TrialParam",             back_populates="trial", cascade="all, delete-orphan")
    values              = relationship("TrialValue",             back_populates="trial", cascade="all, delete-orphan")
    intermediate_values = relationship("TrialIntermediateValue", back_populates="trial", cascade="all, delete-orphan")
    heartbeats          = relationship("TrialHeartbeat",         back_populates="trial", cascade="all, delete-orphan")
    system_attributes   = relationship("TrialSystemAttribute",   back_populates="trial", cascade="all, delete-orphan")
    user_attributes     = relationship("TrialUserAttribute",     back_populates="trial", cascade="all, delete-orphan")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def params_dict(self) -> dict:
        """Decoded hyperparameter values as ``{param_name: value}``."""
        return {p.param_name: p.decoded_value for p in self.params}

    @property
    def objective_value(self) -> Optional[float]:
        """Primary objective value (``objective == 0``), or None."""
        primary = next((v for v in self.values if v.objective == 0), None)
        return primary.value if primary else None

    @property
    def duration_min(self) -> Optional[float]:
        if self.datetime_start and self.datetime_complete:
            return (self.datetime_complete - self.datetime_start).total_seconds() / 60
        return None

    def __repr__(self) -> str:
        score = f"{self.objective_value:.4f}" if self.objective_value is not None else "—"
        return f"<Trial #{self.number} [{self.state}] score={score}>"


class TrialParam(Base):
    __tablename__ = "trial_params"

    param_id          = Column(Integer, primary_key=True)
    trial_id          = Column(Integer, ForeignKey("trials.trial_id"))
    param_name        = Column(String(512))
    param_value       = Column(Float)   # Optuna internal encoding (index for Categorical)
    distribution_json = Column(Text)

    trial = relationship("Trial", back_populates="params")

    @property
    def distribution(self) -> dict:
        return json.loads(self.distribution_json)

    @property
    def decoded_value(self):
        """Actual hyperparameter value.

        CategoricalDistribution stores ``param_value`` as an integer index
        into ``choices``; all other distributions store the value directly.
        """
        dist = self.distribution
        if dist["name"] == "CategoricalDistribution":
            return dist["attributes"]["choices"][int(self.param_value)]
        return self.param_value

    @property
    def choices(self) -> Optional[list]:
        """For CategoricalDistribution: the list of choices; else None."""
        dist = self.distribution
        if dist["name"] == "CategoricalDistribution":
            return dist["attributes"]["choices"]
        return None

    @property
    def bounds(self) -> Optional[tuple]:
        """For Float/IntDistribution: ``(low, high)``; else None."""
        dist = self.distribution
        if dist["name"] in ("FloatDistribution", "IntDistribution"):
            a = dist["attributes"]
            return (a["low"], a["high"])
        return None

    def __repr__(self) -> str:
        return f"<TrialParam {self.param_name}={self.decoded_value!r}>"


class TrialValue(Base):
    __tablename__ = "trial_values"

    trial_value_id = Column(Integer, primary_key=True)
    trial_id       = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    objective      = Column(Integer, nullable=False)          # 0 = primary
    value          = Column(Float)
    value_type     = Column(String(7), nullable=False)        # FINITE|INF|NEG_INF|NAN

    trial = relationship("Trial", back_populates="values")

    def __repr__(self) -> str:
        return f"<TrialValue obj={self.objective} {self.value_type}={self.value}>"


class TrialIntermediateValue(Base):
    __tablename__ = "trial_intermediate_values"

    trial_intermediate_value_id = Column(Integer, primary_key=True)
    trial_id                    = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    step                        = Column(Integer, nullable=False)
    intermediate_value          = Column(Float)
    intermediate_value_type     = Column(String(7), nullable=False)

    trial = relationship("Trial", back_populates="intermediate_values")

    def __repr__(self) -> str:
        return f"<TrialIntermediateValue step={self.step} {self.intermediate_value_type}={self.intermediate_value}>"


class TrialHeartbeat(Base):
    __tablename__ = "trial_heartbeats"

    trial_heartbeat_id = Column(Integer, primary_key=True)
    trial_id           = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    heartbeat          = Column(DateTime, nullable=False)

    trial = relationship("Trial", back_populates="heartbeats")

    def __repr__(self) -> str:
        return f"<TrialHeartbeat trial={self.trial_id} {self.heartbeat}>"


class TrialSystemAttribute(Base):
    __tablename__ = "trial_system_attributes"

    trial_system_attribute_id = Column(Integer, primary_key=True)
    trial_id                  = Column(Integer, ForeignKey("trials.trial_id"))
    key                       = Column(String(512))
    value_json                = Column(Text)

    trial = relationship("Trial", back_populates="system_attributes")

    @property
    def value(self):
        return json.loads(self.value_json) if self.value_json else None

    def __repr__(self) -> str:
        return f"<TrialSystemAttribute {self.key}={self.value!r}>"


class TrialUserAttribute(Base):
    __tablename__ = "trial_user_attributes"

    trial_user_attribute_id = Column(Integer, primary_key=True)
    trial_id                = Column(Integer, ForeignKey("trials.trial_id"))
    key                     = Column(String(512))
    value_json              = Column(Text)

    trial = relationship("Trial", back_populates="user_attributes")

    @property
    def value(self):
        return json.loads(self.value_json) if self.value_json else None

    def __repr__(self) -> str:
        return f"<TrialUserAttribute {self.key}={self.value!r}>"


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def connect(db_path: str | Path, readonly: bool = True) -> Session:
    """Return a SQLAlchemy Session bound to an Optuna SQLite study.db.

    Parameters
    ----------
    db_path:
        Path to a study.db file.
    readonly:
        Open the database in read-only mode (default True).  Prevents
        accidental writes to live study databases.

    Returns
    -------
    sqlalchemy.orm.Session
        Caller is responsible for calling ``session.close()``.
    """
    path = str(Path(db_path).resolve())
    if readonly:
        def _creator():
            return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        engine = create_engine("sqlite://", creator=_creator)
    else:
        engine = create_engine(f"sqlite:///{path}")
    return Session(engine)


# ---------------------------------------------------------------------------
# Merge compatibility check
# ---------------------------------------------------------------------------

class MergeConflict(ValueError):
    """Raised when two studies cannot be safely merged."""


def check_merge_compatibility(session_a: Session, session_b: Session) -> None:
    """Raise MergeConflict if the two databases cannot be merged.

    Checks schema versions, optimization directions, and parameter
    distribution types.  Does NOT check whether the search spaces
    overlap in a statistically useful way.

    Raises
    ------
    MergeConflict
        With a human-readable explanation of the first incompatibility found.
    """
    # --- schema version ---
    av_a = session_a.query(AlembicVersion).first()
    av_b = session_b.query(AlembicVersion).first()
    if av_a and av_b and av_a.version_num != av_b.version_num:
        raise MergeConflict(
            f"alembic_version mismatch: {av_a.version_num!r} vs {av_b.version_num!r}. "
            "Both databases must have been created by the same Optuna schema version."
        )

    vi_a = session_a.query(VersionInfo).first()
    vi_b = session_b.query(VersionInfo).first()
    if vi_a and vi_b and vi_a.schema_version != vi_b.schema_version:
        raise MergeConflict(
            f"schema_version mismatch: {vi_a.schema_version} vs {vi_b.schema_version}."
        )

    # --- optimization directions ---
    studies_a = session_a.query(Study).all()
    studies_b = session_b.query(Study).all()
    dirs_a = {(d.objective, d.direction)
              for s in studies_a for d in s.directions}
    dirs_b = {(d.objective, d.direction)
              for s in studies_b for d in s.directions}
    if dirs_a and dirs_b:
        for obj_idx, dir_a in dirs_a:
            matching = {d for (o, d) in dirs_b if o == obj_idx}
            if matching and dir_a not in matching:
                raise MergeConflict(
                    f"Objective {obj_idx} direction mismatch: "
                    f"{dir_a!r} (A) vs {matching!r} (B). "
                    "Cannot merge MAXIMIZE and MINIMIZE studies."
                )

    # --- distribution type conflicts per param_name ---
    def _dist_types(session: Session) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for p in session.query(TrialParam).all():
            dist_name = p.distribution["name"]
            result.setdefault(p.param_name, set()).add(dist_name)
        return result

    dt_a = _dist_types(session_a)
    dt_b = _dist_types(session_b)
    for param in set(dt_a) & set(dt_b):
        types_a, types_b = dt_a[param], dt_b[param]
        if types_a != types_b:
            raise MergeConflict(
                f"Parameter {param!r} has distribution type {types_a} in A "
                f"but {types_b} in B. Cannot merge incompatible distribution types."
            )


# ---------------------------------------------------------------------------
# Merge planning — analyse trial_params to build a merged search space
# ---------------------------------------------------------------------------

def _collect_param_info(session: Session) -> dict[str, dict]:
    """Return {param_name: {"dist_type": str, "choices": set | None,
                             "low": float|None, "high": float|None,
                             "log": bool|None, "step": float|None}}
    by scanning every TrialParam row in *session*.
    """
    info: dict[str, dict] = {}
    for p in session.query(TrialParam).all():
        dist = p.distribution
        name = p.param_name
        dtype = dist["name"]
        attrs = dist["attributes"]
        if name not in info:
            info[name] = {"dist_type": dtype, "choices": None, "low": None,
                          "high": None, "log": None, "step": None}
        rec = info[name]
        if dtype == "CategoricalDistribution":
            if rec["choices"] is None:
                rec["choices"] = set()
            rec["choices"].update(attrs["choices"])
        else:
            # FloatDistribution or IntDistribution — widen bounds
            lo, hi = attrs["low"], attrs["high"]
            rec["low"]  = lo if rec["low"]  is None else min(rec["low"],  lo)
            rec["high"] = hi if rec["high"] is None else max(rec["high"], hi)
            rec["log"]  = attrs.get("log")
            rec["step"] = attrs.get("step")
    return info


def merge_search_space(
    session_a: Session,
    session_b: Session,
) -> dict[str, dict]:
    """Compute the merged distribution for every parameter seen in A or B.

    Strategy (mirrors the user's intent of analysing the trial_params table):

    * **CategoricalDistribution** — collect every distinct decoded value seen
      across ALL trials in both databases and take the set union.  The merged
      ``choices`` list is sorted for determinism (strings first, then numeric).
    * **FloatDistribution / IntDistribution** — widen bounds to
      ``min(A.low, B.low)`` .. ``max(A.high, B.high)``.  ``log`` and ``step``
      are taken from whichever study has them set (conflict: A wins).
    * Parameters that appear in only one study are included as-is (the
      other study simply never sampled that dimension).

    Returns
    -------
    dict[str, dict]
        ``{param_name: distribution_dict}`` where each value is a dict
        with keys ``dist_type``, and either ``choices`` (Categorical) or
        ``low``/``high``/``log``/``step`` (Float/Int).  Suitable for
        building a new ``distribution_json`` with
        :func:`distribution_dict_to_json`.
    """
    info_a = _collect_param_info(session_a)
    info_b = _collect_param_info(session_b)
    merged: dict[str, dict] = {}

    for name in sorted(set(info_a) | set(info_b)):
        rec_a = info_a.get(name)
        rec_b = info_b.get(name)

        if rec_a is None:
            merged[name] = rec_b
            continue
        if rec_b is None:
            merged[name] = rec_a
            continue

        # Both studies have this param — merge.
        dtype = rec_a["dist_type"]  # check_merge_compatibility guarantees they match
        if dtype == "CategoricalDistribution":
            all_choices = (rec_a["choices"] or set()) | (rec_b["choices"] or set())
            # Sort: try numeric first, fall back to string sort.
            try:
                ordered = sorted(all_choices, key=lambda x: (0, float(str(x))))
            except (ValueError, TypeError):
                ordered = sorted(all_choices, key=str)
            merged[name] = {"dist_type": dtype, "choices": ordered,
                            "low": None, "high": None, "log": None, "step": None}
        else:
            merged[name] = {
                "dist_type": dtype,
                "choices": None,
                "low":  min(rec_a["low"],  rec_b["low"]),
                "high": max(rec_a["high"], rec_b["high"]),
                "log":  rec_a["log"]  if rec_a["log"]  is not None else rec_b["log"],
                "step": rec_a["step"] if rec_a["step"] is not None else rec_b["step"],
            }

    return merged


def distribution_dict_to_json(rec: dict) -> str:
    """Convert a :func:`merge_search_space` record back to Optuna distribution_json."""
    dtype = rec["dist_type"]
    if dtype == "CategoricalDistribution":
        attrs = {"choices": rec["choices"]}
    else:
        attrs = {
            "low":  rec["low"],
            "high": rec["high"],
            "log":  rec["log"] or False,
            "step": rec["step"],
        }
    return json.dumps({"name": dtype, "attributes": attrs})


def encode_param_value(decoded_value, merged_dist_record: dict) -> float:
    """Re-encode a decoded hyperparameter value for a merged distribution.

    For CategoricalDistribution, returns the index of *decoded_value* in the
    merged choices list.  For Float/Int distributions, returns the value as-is
    (no re-encoding needed).

    Raises
    ------
    ValueError
        If *decoded_value* is not found in the merged categorical choices.
    """
    if merged_dist_record["dist_type"] == "CategoricalDistribution":
        choices = merged_dist_record["choices"]
        try:
            return float(choices.index(decoded_value))
        except ValueError:
            raise ValueError(
                f"decoded value {decoded_value!r} not in merged choices {choices!r}"
            )
    return float(decoded_value)


def plan_merge(
    session_a: Session,
    session_b: Session,
    *,
    new_study_name: Optional[str] = None,
) -> dict:
    """Analyse two sessions and return a human-readable merge plan.

    Calls :func:`check_merge_compatibility` first (raises on hard blockers),
    then calls :func:`merge_search_space` to compute the union distributions
    and reports which parameter ranges changed.

    Parameters
    ----------
    session_a, session_b:
        Sessions returned by :func:`connect`.
    new_study_name:
        Name for the merged study.  Defaults to
        ``"<name_a>+<name_b>"``.

    Returns
    -------
    dict with keys:
        ``new_study_name``, ``merged_space``, ``changes`` (list of dicts
        describing widened bounds or new choices per param), ``warnings``
        (list of strings about stale RUNNING/WAITING trials).
    """
    check_merge_compatibility(session_a, session_b)

    name_a = session_a.query(Study).first().study_name if session_a.query(Study).first() else "A"
    name_b = session_b.query(Study).first().study_name if session_b.query(Study).first() else "B"
    merged_name = new_study_name or f"{name_a}+{name_b}"

    merged = merge_search_space(session_a, session_b)
    info_a = _collect_param_info(session_a)
    info_b = _collect_param_info(session_b)

    changes = []
    for param, rec in merged.items():
        ra = info_a.get(param)
        rb = info_b.get(param)
        if ra is None:
            changes.append({"param": param, "change": "new_from_B", "merged": rec})
        elif rb is None:
            changes.append({"param": param, "change": "new_from_A", "merged": rec})
        elif rec["dist_type"] == "CategoricalDistribution":
            added = set(rec["choices"]) - (ra["choices"] or set()) - (rb["choices"] or set())
            new_a = (ra["choices"] or set()) - (rb["choices"] or set())
            new_b = (rb["choices"] or set()) - (ra["choices"] or set())
            if new_a or new_b or added:
                changes.append({
                    "param": param, "change": "widened_choices",
                    "only_in_A": sorted(new_a, key=str),
                    "only_in_B": sorted(new_b, key=str),
                    "merged_choices": rec["choices"],
                    "requires_reencode": True,
                })
        else:
            widened = (rec["low"] < ra["low"] or rec["low"] < rb["low"] or
                       rec["high"] > ra["high"] or rec["high"] > rb["high"])
            if widened:
                changes.append({
                    "param": param, "change": "widened_bounds",
                    "A": (ra["low"], ra["high"]),
                    "B": (rb["low"], rb["high"]),
                    "merged": (rec["low"], rec["high"]),
                    "requires_reencode": False,
                })

    warnings = []
    for session, label in ((session_a, "A"), (session_b, "B")):
        for state in ("RUNNING", "WAITING"):
            n = session.query(Trial).filter_by(state=state).count()
            if n:
                warnings.append(
                    f"Study {label} has {n} {state} trial(s). "
                    f"Their state will be preserved as-is; the TPE sampler "
                    f"ignores trials without a recorded objective value."
                )

    return {
        "new_study_name": merged_name,
        "merged_space": merged,
        "changes": changes,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Config-file augmentation — parameters not stored in study.db
# ---------------------------------------------------------------------------

# Sections of model.config.yaml that carry INPUT parameters.
# The "optuna" section describes the search space, not actual values — skip it.
# The "evaluation" and "train.results_path" sections carry output/path values — skip paths.
_CONFIG_INPUT_SECTIONS = ("model", "optimizer", "train", "dataset", "context", "reward")

# Flat keys whose values are filesystem paths or output artefacts, not inputs.
_CONFIG_PATH_KEYS = {
    "context.building_blocks", "context.templates_path",
    "dataset.routes_path", "dataset.synthetic_routes_path",
    "train.results_path",
}


def trial_config_path(db_path: str | Path, trial_number: int) -> Path:
    """Return the expected path to ``model.config.yaml`` for *trial_number*.

    Convention: configs live at ``<study_dir>/trial_NNN/model.config.yaml``
    where NNN is the zero-padded trial number (e.g. 0 → ``trial_000``).
    Returns the path regardless of whether the file exists.
    """
    return Path(db_path).parent / f"trial_{trial_number:03d}" / "model.config.yaml"


def load_trial_config(config_path: str | Path) -> dict:
    """Load a ``model.config.yaml`` and return the raw nested dict.

    The ``optuna`` section (search-space definition) and any path-valued
    keys are excluded so only actual runtime input values remain.

    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception:
        return {}
    # Keep only input sections.
    return {k: v for k, v in cfg.items() if k in _CONFIG_INPUT_SECTIONS}


def _flatten(nested: dict, prefix: str = "") -> dict[str, object]:
    """Recursively flatten a nested dict using dot-separated keys."""
    out: dict[str, object] = {}
    for k, v in nested.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def study_config_params(
    db_path: str | Path,
    session: Session,
    *,
    exclude_paths: bool = True,
) -> dict[int, dict[str, object]]:
    """Load ``model.config.yaml`` for every trial and return a flat param dict.

    Only parameters NOT already captured in ``trial_params`` (i.e., the
    fixed parameters that Optuna did not search over) are included in the
    returned dicts.  Parameters that Optuna searched over will appear in
    both the config and ``TrialParam`` rows; those are omitted here to
    avoid duplication — use :meth:`Trial.params_dict` for the searched
    params.

    Parameters
    ----------
    db_path:
        Path to the study.db file (used to locate trial directories).
    session:
        ORM session for the same database.
    exclude_paths:
        If True (default), drop keys whose values are filesystem paths
        (e.g. ``context.building_blocks``).

    Returns
    -------
    dict[int, dict[str, object]]
        ``{trial_number: {flat_config_key: value}}`` for parameters that
        are NOT in the trial's ``trial_params`` rows.  Trials with no
        config file are absent from the dict.
    """
    # Collect param_names that Optuna searched, per trial.
    searched_by_trial: dict[int, set[str]] = {}
    for trial in session.query(Trial).all():
        searched_by_trial[trial.number] = {p.param_name for p in trial.params}

    result: dict[int, dict[str, object]] = {}
    for trial in session.query(Trial).all():
        cfg_path = trial_config_path(db_path, trial.number)
        raw = load_trial_config(cfg_path)
        if not raw:
            continue
        flat = _flatten(raw)
        if exclude_paths:
            flat = {k: v for k, v in flat.items() if k not in _CONFIG_PATH_KEYS}
        # Drop keys that correspond to Optuna-searched params.
        # The mapping from config key to param_name is not 1-to-1, so we
        # keep any key whose leaf name (after the last dot) is NOT a searched
        # param_name.  This is conservative: it may keep a few overlapping
        # keys (e.g. model.n_heads vs n_heads), but avoids losing genuinely
        # fixed parameters with similar names.
        searched = searched_by_trial.get(trial.number, set())
        flat = {k: v for k, v in flat.items()
                if k.rsplit(".", 1)[-1] not in searched}
        result[trial.number] = flat
    return result


# ---------------------------------------------------------------------------
# Polynomial estimation for RUNNING / WAITING trials
# ---------------------------------------------------------------------------

_DEFAULT_OBJECTIVE_METRIC = "valid_route_accuracy"


def _load_jsonl_metric(
    jsonl_path: str | Path,
    metric: str,
) -> list[tuple[int, float]]:
    """Load (epoch, value) pairs for *metric* from a train_progress.jsonl.

    De-duplicates by epoch number (keeps the last value seen — handles
    append-on-restart files where epoch 0 may appear several times).
    Skips lines where the metric is absent or non-finite.

    Returns an empty list if the file does not exist or cannot be read.
    """
    import math

    path = Path(jsonl_path)
    if not path.exists():
        return []

    seen: dict[int, float] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric not in row:
                continue
            val = row[metric]
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                continue
            epoch = int(row.get("epoch", len(seen)))
            seen[epoch] = float(val)
    except Exception:
        return []

    return sorted(seen.items())


def _trial_objective_metric(db_path: str | Path, trial_number: int) -> str:
    """Return the ``optuna.objective_metric`` for a trial from its config file.

    Falls back to ``"valid_route_accuracy"`` when the config is absent or
    the key is not set.
    """
    try:
        import yaml
        cfg_path = trial_config_path(db_path, trial_number)
        cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
        return cfg.get("optuna", {}).get("objective_metric", _DEFAULT_OBJECTIVE_METRIC)
    except Exception:
        return _DEFAULT_OBJECTIVE_METRIC


def _fit_quadratic_estimate(
    epoch_value_pairs: list[tuple[int, float]],
    target_epoch: int,
    direction: str = "MAXIMIZE",
) -> dict:
    """Fit a degree-2 polynomial to *epoch_value_pairs* and extrapolate.

    For a downward-opening parabola (``a < 0``) with a MAXIMIZE objective,
    the vertex is the natural extrapolation target — but only if it lies
    between the last observed epoch and the planned final epoch.  Otherwise
    extrapolation goes to *target_epoch*.

    Returns a dict with keys:
        ``estimated_value``, ``target_epoch`` (float, where poly was evaluated),
        ``r_squared``, ``n_points``, ``poly_coeffs``.
    """
    import numpy as np

    epochs = np.array([e for e, _ in epoch_value_pairs], dtype=float)
    values = np.array([v for _, v in epoch_value_pairs], dtype=float)

    coeffs = np.polyfit(epochs, values, 2)
    a, b, _ = coeffs

    extr_epoch = float(target_epoch)
    if abs(a) > 1e-12:
        vertex = -b / (2.0 * a)
        beneficial_vertex = (
            (direction == "MAXIMIZE" and a < 0) or
            (direction == "MINIMIZE" and a > 0)
        )
        if beneficial_vertex and epochs[-1] < vertex <= target_epoch:
            extr_epoch = vertex
        elif beneficial_vertex and vertex <= epochs[-1]:
            # Already past the vertex — current last observed is the peak
            extr_epoch = float(epochs[-1])

    estimated = float(np.polyval(coeffs, extr_epoch))

    y_pred = np.polyval(coeffs, epochs)
    ss_res = float(np.sum((values - y_pred) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    return {
        "estimated_value": estimated,
        "target_epoch": extr_epoch,
        "r_squared": r_squared,
        "n_points": len(epoch_value_pairs),
        "poly_coeffs": coeffs.tolist(),
    }


def estimate_incomplete_objectives(
    db_path: str | Path,
    session: Session,
    *,
    metric: Optional[str] = None,
    min_epochs: int = 6,
    states: Optional[list[str]] = None,
) -> dict[int, dict]:
    """Estimate objective values for trials via polynomial fit on training curves.

    For each trial:

    1. Locate ``train_progress.jsonl`` via the ``trial_NNN/`` convention.
    2. Determine the objective metric from each trial's ``model.config.yaml``
       (``optuna.objective_metric``), or from *metric* if given, or fall back
       to ``"valid_route_accuracy"``.
    3. Skip if fewer than *min_epochs* total epoch rows are available.
    4. Take the **second half** of epoch rows by count.
    5. Fit a degree-2 polynomial (``numpy.polyfit``) to those rows.
    6. Extrapolate to the vertex epoch (when the parabola opens beneficially
       and the vertex is in range) or to ``n_epochs - 1`` from the config.

    Trial states are **never modified** — this is purely analytical.

    Parameters
    ----------
    db_path:
        Path to the study.db file.
    session:
        ORM session for that database.
    metric:
        Override the objective metric column.  If None, each trial's config
        is consulted; falls back to ``"valid_route_accuracy"``.
    min_epochs:
        Minimum total epoch rows required before attempting a fit.  Trials
        with fewer rows get ``skipped=True``.
    states:
        Trial states to include.  Defaults to ``["RUNNING", "WAITING"]``.
        Pass ``None`` to include all states (including ``COMPLETE``).

    Returns
    -------
    dict[int, dict]
        ``{trial_number: result}`` for every matched trial.

        Each result contains:
            ``state``, ``metric``, ``n_epochs_observed``,
            ``estimated_value`` (None if skipped),
            ``se`` (combined standard error, None if skipped),
            ``models`` (per-model breakdown from :func:`extrapolate_objective`,
            None if skipped), ``skipped``, ``skip_reason``.
    """
    study = session.query(Study).first()
    direction = study.direction if study else "MAXIMIZE"

    results: dict[int, dict] = {}

    query = session.query(Trial)
    if states is not None:
        query = query.filter(Trial.state.in_(states))
    # states=None → no filter → all trials
    for trial in query.all():
        trial_metric = metric or _trial_objective_metric(db_path, trial.number)

        try:
            import yaml
            cfg_path = trial_config_path(db_path, trial.number)
            cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
            n_epochs = int(cfg.get("train", {}).get("n_epochs", 100))
        except Exception:
            n_epochs = 100

        jsonl_path = Path(db_path).parent / f"trial_{trial.number:03d}" / "train_progress.jsonl"
        pairs = _load_jsonl_metric(jsonl_path, trial_metric)

        base = {
            "state": trial.state,
            "metric": trial_metric,
            "n_epochs_observed": len(pairs),
        }

        if len(pairs) < min_epochs:
            results[trial.number] = {
                **base,
                "estimated_value": None,
                "n_points_fit": 0,
                "target_epoch": None,
                "r_squared": None,
                "poly_coeffs": None,
                "skipped": True,
                "skip_reason": f"only {len(pairs)} epoch(s) observed (min {min_epochs})",
            }
            continue

        from .extrapolate import extrapolate_objective
        obs_values = [v for _, v in pairs]
        obs_epochs = [e for e, _ in pairs]
        fit = extrapolate_objective(
            obs_values, n_epochs,
            epochs=obs_epochs,
            min_points=min_epochs,
        )

        if fit is None:
            results[trial.number] = {
                **base,
                "estimated_value": None,
                "se": None,
                "models": None,
                "skipped": True,
                "skip_reason": f"extrapolate_objective returned None for {len(pairs)} epoch(s)",
            }
        else:
            results[trial.number] = {
                **base,
                "estimated_value": fit["estimate"],
                "se": fit["se"],
                "target_epoch": fit.get("target_epoch"),
                "models": fit["models"],
                "skipped": False,
                "skip_reason": None,
            }

    return results


# ---------------------------------------------------------------------------
# Stale-trial cleanup
# ---------------------------------------------------------------------------

def _last_jsonl_timestamp(jsonl_path: Path):
    """Return the datetime from the last non-empty line of a train_progress.jsonl.

    Parses the ``timestamp`` field (ISO-8601 string).  Returns None if the
    file does not exist, is empty, or has no timestamp field.
    """
    from datetime import datetime
    try:
        lines = jsonl_path.read_text().splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = row.get("timestamp")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                continue
    return None


def _write_complete_trial(
    db_path: Path,
    trial_num: int,
    value: float,
    datetime_complete,
) -> None:
    """Write COMPLETE state + objective value to the SQLite DB.

    Only inserts a trial_values row when none exists for objective=0.
    Always updates trials.state and datetime_complete.
    """
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT trial_id FROM trials WHERE number=?", (trial_num,)
        ).fetchone()
        if row is None:
            raise ValueError(f"trial number {trial_num} not found in {db_path}")
        trial_id = row[0]

        existing = con.execute(
            "SELECT trial_value_id FROM trial_values WHERE trial_id=? AND objective=0",
            (trial_id,),
        ).fetchone()
        if existing is None:
            con.execute(
                "INSERT INTO trial_values (trial_id, objective, value, value_type)"
                " VALUES (?, 0, ?, 'FINITE')",
                (trial_id, value),
            )

        dt_str = (
            datetime_complete.strftime("%Y-%m-%d %H:%M:%S.%f")
            if datetime_complete is not None
            else None
        )
        con.execute(
            "UPDATE trials SET state='COMPLETE', datetime_complete=? WHERE trial_id=?",
            (dt_str, trial_id),
        )
        con.commit()
    finally:
        con.close()


def complete_stale_trials(
    db_path: str | Path,
    *,
    stale_states: list[str] | None = None,
    proxy_metric: str = "valid_action_accuracy",
    min_epochs: int = 6,
    value_range: tuple[float, float] = (0.0, 1.0),
    dry_run: bool = False,
) -> dict[int, dict]:
    """Mark stale trials as COMPLETE using extrapolated training-curve estimates.

    "Stale" means the trial is in an unresolved non-WAITING state (typically
    RUNNING or FAILED after a crash) and has no objective value in
    ``trial_values``.  Trials that already have an objective value are skipped.

    For each eligible trial the function:

    1. Reads the per-epoch metric series from ``trial_NNN/train_progress.jsonl``.
    2. Determines the objective metric from the trial's ``model.config.yaml``
       (``optuna.objective_metric``).  If that metric is not logged per-epoch
       (e.g. ``fraction_solved`` requires end-of-run route evaluation), falls
       back to *proxy_metric* (default: ``"valid_route_accuracy"``).
    3. Calls :func:`extrapolate_objective` (four-model inverse-variance ensemble)
       to estimate the final-epoch value.
    4. Sets the trial state to COMPLETE with the estimated value, unless
       *dry_run* is True.

    Parameters
    ----------
    db_path:
        Path to the ``study.db`` SQLite file.
    stale_states:
        Trial states to treat as stale.  Defaults to
        ``["RUNNING", "FAILED", "CANCELED"]``.
    proxy_metric:
        Per-epoch metric used when the configured objective metric is absent
        from the training log.  Defaults to ``"valid_action_accuracy"``, which
        is logged every epoch and sits in the same numerical range as
        ``fraction_solved`` (~0.3–0.4).  ``valid_route_accuracy`` is a poor
        proxy because it is an order of magnitude smaller.
    min_epochs:
        Minimum observed epochs before attempting extrapolation.  Trials with
        fewer rows get ``action="skipped"``.
    value_range:
        ``(lo, hi)`` inclusive bounds for accepting an extrapolated estimate.
        Estimates outside this range are treated as extrapolation artifacts and
        the trial is skipped.  Defaults to ``(0.0, 1.0)``, which covers all
        accuracy and fraction metrics.
    dry_run:
        If True, compute estimates without writing to the database.

    Returns
    -------
    dict[int, dict]
        ``{trial_number: result}`` for every stale trial examined.

        Each result contains:
            ``state_before``, ``action`` (``"completed"`` | ``"dry_run"`` |
            ``"skipped"``), ``skip_reason`` (None or str),
            ``metric_used``, ``n_epochs_observed``, ``estimated_value``,
            ``se``, ``datetime_complete`` (ISO str or None).
    """
    if stale_states is None:
        stale_states = ["RUNNING", "FAILED", "CANCELED"]

    db_path = Path(db_path)

    # ── find stale trials: matching state and no objective value ──────────
    ro = connect(db_path, readonly=True)
    stale_nums: list[int] = []
    try:
        for trial in ro.query(Trial).filter(Trial.state.in_(stale_states)).all():
            if trial.objective_value is None:
                stale_nums.append(trial.number)
    finally:
        ro.close()

    if not stale_nums:
        return {}

    # ── extrapolate using the configured objective metric per trial ────────
    ro = connect(db_path, readonly=True)
    try:
        primary = estimate_incomplete_objectives(
            db_path, ro, metric=None, min_epochs=min_epochs, states=None,
        )
        # Retry trials where the configured metric is absent from the log,
        # using proxy_metric instead.
        needs_proxy = [
            n for n in stale_nums if primary.get(n, {}).get("skipped", True)
        ]
        proxy: dict[int, dict] = {}
        if needs_proxy:
            all_proxy = estimate_incomplete_objectives(
                db_path, ro, metric=proxy_metric, min_epochs=min_epochs, states=None,
            )
            proxy = {n: all_proxy[n] for n in needs_proxy if n in all_proxy}
    finally:
        ro.close()

    # ── build result records and optionally write ──────────────────────────
    results: dict[int, dict] = {}
    for trial_num in stale_nums:
        est = primary.get(trial_num, {})
        metric_used = est.get("metric", proxy_metric)

        if est.get("skipped", True) and trial_num in proxy:
            est = proxy[trial_num]
            metric_used = proxy_metric + " (proxy)"

        state_before = est.get("state", "UNKNOWN")
        n_obs = est.get("n_epochs_observed", 0)

        if est.get("skipped") or est.get("estimated_value") is None:
            results[trial_num] = {
                "state_before": state_before,
                "action": "skipped",
                "skip_reason": est.get("skip_reason", "no estimate available"),
                "metric_used": metric_used,
                "n_epochs_observed": n_obs,
                "estimated_value": None,
                "se": None,
                "datetime_complete": None,
            }
            continue

        estimated_value = float(est["estimated_value"])

        # Floor: never report below the trial's actual peak.  Polynomial
        # extrapolation can dip below best-observed when the curve was
        # declining at interruption (e.g. lr not yet reduced).
        actual_metric = metric_used.removesuffix(" (proxy)").strip()
        obs_pairs = _load_jsonl_metric(
            db_path.parent / f"trial_{trial_num:03d}" / "train_progress.jsonl",
            actual_metric,
        )
        if obs_pairs:
            best_observed = max(v for _, v in obs_pairs)
            if best_observed > estimated_value:
                estimated_value = best_observed

        lo, hi = value_range
        if not (lo <= estimated_value <= hi):
            results[trial_num] = {
                "state_before": state_before,
                "action": "skipped",
                "skip_reason": (
                    f"estimate {estimated_value:.4f} outside valid range "
                    f"[{lo}, {hi}] — likely extrapolation artifact"
                ),
                "metric_used": metric_used,
                "n_epochs_observed": n_obs,
                "estimated_value": estimated_value,
                "se": est.get("se"),
                "datetime_complete": None,
            }
            continue
        jsonl_path = db_path.parent / f"trial_{trial_num:03d}" / "train_progress.jsonl"
        dt_complete = _last_jsonl_timestamp(jsonl_path)

        results[trial_num] = {
            "state_before": state_before,
            "action": "dry_run" if dry_run else "completed",
            "skip_reason": None,
            "metric_used": metric_used,
            "n_epochs_observed": n_obs,
            "estimated_value": estimated_value,
            "se": est.get("se"),
            "datetime_complete": dt_complete.isoformat() if dt_complete else None,
        }

        if not dry_run:
            _write_complete_trial(db_path, trial_num, estimated_value, dt_complete)

    return results


def fixed_params_diff(
    db_path_a: str | Path,
    session_a: Session,
    db_path_b: str | Path,
    session_b: Session,
) -> dict[str, dict]:
    """Compare fixed (non-searched) config params between two studies.

    Returns a dict of params that differ between the two studies, with
    the values seen in each.  Params that vary across trials within a
    single study are noted as "varies" rather than a single value.

    A non-empty result is a strong signal that the two studies are not
    directly comparable (e.g. different ``dataset.action_dim`` means
    they used different molecule datasets and their trials cannot be
    meaningfully combined for TPE).

    Returns
    -------
    dict[str, {"A": value_or_"varies", "B": value_or_"varies"}]
    """
    def _summarise(cfg_by_trial: dict[int, dict]) -> dict[str, object]:
        """Collapse per-trial config to study-level: single value or 'varies'."""
        summary: dict[str, object] = {}
        for flat in cfg_by_trial.values():
            for k, v in flat.items():
                if k not in summary:
                    summary[k] = v
                elif summary[k] != v:
                    summary[k] = "varies"
        return summary

    configs_a = study_config_params(db_path_a, session_a)
    configs_b = study_config_params(db_path_b, session_b)
    summary_a = _summarise(configs_a)
    summary_b = _summarise(configs_b)

    diffs: dict[str, dict] = {}
    for key in sorted(set(summary_a) | set(summary_b)):
        va = summary_a.get(key, "<absent>")
        vb = summary_b.get(key, "<absent>")
        if va != vb:
            diffs[key] = {"A": va, "B": vb}
    return diffs
