"""Generate a per-trial ``report.yaml``.

Combines what's already recorded elsewhere for one hypertune trial —
hyperparameters and objective score from ``study.db``, the training curve
from ``train_progress.jsonl`` — with two things that aren't tracked
anywhere else:

- ``model.parameters``: total / per-category learned-parameter counts.
- ``model.complexity``: entropy/complexity estimates of the trained weights
  (see :mod:`retrosynformer.model_stats`), combining a whole-file
  compression-ratio estimate with per-tensor spectral (effective rank /
  stable rank) and value-distribution (histogram, Gaussian differential
  entropy) statistics.

The model-analysis half (:mod:`retrosynformer.model_stats`) works on any
PyTorch ``state_dict`` — this module is the RetroSynFormer-specific part
that knows about ``trial_NNN/`` directory layout, ``model.config.yaml``,
``train_progress.jsonl``, and ``study.db``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import yaml

from retrosynformer.compression import load_model
from retrosynformer.model_stats import weight_complexity_report

# Metrics logged per-epoch in train_progress.jsonl that are worth
# summarising (best + final value) in the report's training_curve section.
_TRAIN_METRIC_KEYS = (
    "val_acc", "val_route_acc", "val_loss", "train_loss",
    "valid_action_accuracy", "valid_route_accuracy", "fraction_solved",
)


def _load_train_progress(trial_dir: Path) -> list[dict]:
    """Return parsed rows from trial_dir/train_progress.jsonl, or [] if absent."""
    jsonl = trial_dir / "train_progress.jsonl"
    if not jsonl.exists():
        return []
    rows = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _training_curve_summary(rows: list[dict]) -> dict:
    """Best + final value per known metric, plus the observed epoch count."""
    summary = {"n_epochs": len(rows)}
    for key in _TRAIN_METRIC_KEYS:
        values = [r[key] for r in rows if key in r and r[key] is not None]
        if values:
            summary[f"best_{key}"] = max(values)
            summary[f"final_{key}"] = values[-1]
    return summary


def _study_db_trial_info(db_path: Optional[Path], trial_number: int) -> dict:
    """Pull this trial's own row from study.db: state, searched params, score, timing."""
    if db_path is None or not Path(db_path).exists():
        return {}
    from retrosynformer.models_optuna import Trial, connect

    session = connect(db_path, readonly=True)
    try:
        trial = session.query(Trial).filter_by(number=trial_number).first()
        if trial is None:
            return {}
        return {
            "state": trial.state,
            "objective_value": trial.objective_value,
            "duration_min": trial.duration_min,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "searched_params": trial.params_dict,
        }
    finally:
        session.close()


def generate_trial_report(
    trial_dir: "str | Path",
    *,
    db_path: "str | Path | None" = None,
    out_path: "str | Path | None" = None,
    checkpoint_name: Optional[str] = None,
    compression_codec: str = "bz2",
    include_per_tensor_stats: bool = False,
    write: bool = True,
) -> dict:
    """Build (and by default write) a ``report.yaml`` for one hypertune trial.

    Parameters
    ----------
    trial_dir:
        Path to a ``trial_NNN/`` directory. Must contain a model checkpoint
        (``model.pth``, ``model.safetensors``, or a compressed variant of
        either — same lookup order as ``runner._find_model_checkpoint``).
        ``model.config.yaml`` and ``train_progress.jsonl`` are included if
        present but are not required.
    db_path:
        Path to the study's ``study.db``. Defaults to ``study.db`` in
        *trial_dir*'s parent (the standard ``rs-hypertune`` layout). If it
        doesn't exist, the ``study_db`` section of the report is empty —
        everything derivable from *trial_dir* alone is still included.
    out_path:
        Where to write the YAML. Defaults to ``trial_dir/report.yaml``.
    checkpoint_name:
        Force a specific checkpoint filename instead of auto-detecting.
    compression_codec:
        Codec for the whole-file compression-ratio estimate: "gz", "bz2"
        (default), or "xz". See :mod:`retrosynformer.compression`.
    include_per_tensor_stats:
        If True, include a full per-tensor breakdown in
        ``model.complexity.per_tensor`` (one entry per weight tensor — verbose;
        off by default). See :func:`retrosynformer.model_stats.weight_complexity_report`.
    write:
        If True (default), write the report to *out_path* as YAML. Set False
        to only compute and return the dict (e.g. from tests).

    Returns
    -------
    dict
        The full report structure. Includes ``"_report_path"`` (str) if it
        was written to disk.

    Raises
    ------
    FileNotFoundError
        If *trial_dir* has no recognisable model checkpoint.
    """
    from retrosynformer.runner import _find_model_checkpoint

    trial_dir = Path(trial_dir)
    m = re.fullmatch(r"trial_(\d+)", trial_dir.name)
    trial_number = int(m.group(1)) if m else None

    if db_path is None:
        candidate = trial_dir.parent / "study.db"
        db_path = candidate if candidate.exists() else None

    config_path = trial_dir / "model.config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    if checkpoint_name is not None:
        checkpoint_path = trial_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
    else:
        found = _find_model_checkpoint(str(trial_dir))
        if found is None:
            raise FileNotFoundError(f"No model checkpoint found in {trial_dir}")
        checkpoint_path = Path(found)

    state_dict = load_model(checkpoint_path)
    weight_report = weight_complexity_report(
        state_dict,
        model_path=checkpoint_path,
        codec=compression_codec,
        include_per_tensor=include_per_tensor_stats,
    )

    report = {
        "trial_dir": str(trial_dir),
        "trial_number": trial_number,
        "checkpoint_file": checkpoint_path.name,
        # optuna.objective_metric describes what "objective_value" below means.
        "objective_metric": config.get("optuna", {}).get("objective_metric"),
        "study_db": (
            _study_db_trial_info(db_path, trial_number) if trial_number is not None else {}
        ),
        # Realised config for this one trial — the "optuna" key (the search
        # space definition, not a value) is dropped as not applicable here.
        "config": {k: v for k, v in config.items() if k != "optuna"},
        "training_curve": _training_curve_summary(_load_train_progress(trial_dir)),
        "model": {
            "parameters": weight_report["parameters"],
            "complexity": {
                k: v for k, v in weight_report.items() if k != "parameters"
            },
        },
    }

    if write:
        out_path = Path(out_path) if out_path is not None else trial_dir / "report.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.dump(report, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=120)
        )
        report["_report_path"] = str(out_path)

    return report
