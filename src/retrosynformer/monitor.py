"""Monitor a running or completed Optuna hypertune study by evaluating trial checkpoints locally.

Typical usage::

    from retrosynformer.monitor import monitor_study

    # Quick look at training curves — no model loading needed.
    df = monitor_study("small_baseline", mode="progress")
    print(df.sort_values("best_valid_accuracy", ascending=False).head())

    # Full greedy evaluation (beam_width=1, 2 validation batches per trial).
    # Provide local data paths when the config was saved with taco-absolute paths.
    overrides = {
        "context.building_blocks": "data/small_building_blocks.csv",
        "context.templates_path": "data/small_reaction_templates.pickle",
        "dataset.routes_path": "data/small_routes.json",
    }
    df = monitor_study("small_baseline", data_overrides=overrides)
    print(df.sort_values("fraction_solved", ascending=False))
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RESULTS_BASE = "results"


def find_trial_dirs(study_name: str, results_base: str = RESULTS_BASE) -> list[Path]:
    """Return sorted list of trial directories that contain a model checkpoint.

    Parameters
    ----------
    study_name:
        Name of the Optuna study (e.g. ``"small_baseline"``).
    results_base:
        Parent directory for hypertune results; default ``"results"``.

    Returns
    -------
    list[Path]
        Sorted trial directories, each containing ``model.pth``.

    Examples
    --------
    >>> find_trial_dirs("__nonexistent__") == []
    True
    """
    study_dir = Path(results_base) / f"hypertune-{study_name}"
    return sorted(p.parent for p in study_dir.glob("trial_*/model.pth"))


def read_trial_progress(trial_dir: str | Path) -> pd.DataFrame:
    """Read ``train_progress.jsonl`` from a trial directory without loading the model.

    Parameters
    ----------
    trial_dir:
        Path to the trial directory (e.g. ``results/hypertune-small/trial_000``).

    Returns
    -------
    pd.DataFrame
        One row per epoch. Columns match the JSONL keys
        (e.g. ``epoch``, ``train_loss``, ``valid_loss``, ``valid_accuracy``).
        Returns an empty DataFrame if the file does not exist yet.
    """
    path = Path(trial_dir) / "train_progress.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _load_config(trial_dir: str | Path) -> dict:
    cfg_path = Path(trial_dir) / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _apply_overrides(config: dict, overrides: dict | None) -> dict:
    """Deep-copy *config* and set dot-notation keys from *overrides*.

    Parameters
    ----------
    config:
        Config dict loaded from ``config.yaml``.
    overrides:
        Mapping of dot-separated key paths to replacement values, e.g.::

            {"context.building_blocks": "data/small_building_blocks.csv"}

    Returns
    -------
    dict
        Modified copy of the config; the original is not mutated.
    """
    if not overrides:
        return config
    config = copy.deepcopy(config)
    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config


def load_trial_model(
    trial_dir: str | Path,
    data_overrides: Optional[dict] = None,
):
    """Load the model checkpoint from a trial directory.

    Parameters
    ----------
    trial_dir:
        Path to the trial directory (must contain ``config.yaml`` and ``model.pth``).
    data_overrides:
        Dot-notation config overrides, e.g.
        ``{"context.building_blocks": "data/small_building_blocks.csv"}``.

    Returns
    -------
    tuple[model, config]
        Loaded PyTorch model and the (possibly overridden) config dict.
    """
    from .runner import init_model

    trial_dir = Path(trial_dir)
    config = _apply_overrides(_load_config(trial_dir), data_overrides)
    model_path = str(trial_dir / "model.pth")
    model = init_model(config, model_path=model_path)
    return model, config


def evaluate_trial(
    trial_dir: str | Path,
    beam_width: int = 1,
    n_batches: int = 2,
    data_overrides: Optional[dict] = None,
) -> dict:
    """Run beam-search evaluation on the checkpoint in *trial_dir*.

    Parameters
    ----------
    trial_dir:
        Path to the trial directory (must contain ``config.yaml`` and ``model.pth``).
    beam_width:
        Beam width for route search.  1 = greedy (fast), 50 = paper setting (slow).
    n_batches:
        Number of validation batches to evaluate.  Overrides the value in the config.
    data_overrides:
        Dot-notation config overrides so local data paths shadow taco-absolute ones::

            {
                "context.building_blocks": "data/small_building_blocks.csv",
                "context.templates_path": "data/small_reaction_templates.pickle",
                "dataset.routes_path": "data/small_routes.json",
            }

    Returns
    -------
    dict
        Keys: ``trial``, ``trial_dir``, ``fraction_solved``, ``n_solved``,
        ``n_routes``, ``beam_width``, ``n_batches``.  Final-epoch training
        metrics from ``train_progress.jsonl`` are included as ``final_*`` keys.
    """
    from .inference import RoutePredictor
    from .runner import create_dataloaders, init_data, init_model

    trial_dir = Path(trial_dir)
    config = _apply_overrides(_load_config(trial_dir), data_overrides)
    config["evaluation"]["beam_width"] = beam_width
    config["evaluation"]["eval_n_batches"] = n_batches

    model_path = str(trial_dir / "model.pth")
    model = init_model(config, model_path=model_path)
    model.eval()

    datasets = init_data(config)
    _, valid_dataloader, _ = create_dataloaders(datasets, config)

    predictor = RoutePredictor(model, config)
    routes = predictor.eval_predicted_routes(valid_dataloader)

    n_routes = len(routes)
    n_solved = sum(1 for r in routes if r.get("route_solved", False))
    fraction_solved = n_solved / n_routes if n_routes else float("nan")

    result: dict = {
        "trial": trial_dir.name,
        "trial_dir": str(trial_dir),
        "fraction_solved": fraction_solved,
        "n_solved": n_solved,
        "n_routes": n_routes,
        "beam_width": beam_width,
        "n_batches": n_batches,
    }

    progress = read_trial_progress(trial_dir)
    if not progress.empty:
        last = progress.iloc[-1].to_dict()
        result.update({f"final_{k}": v for k, v in last.items()})

    return result


def monitor_study(
    study_name: str,
    results_base: str = RESULTS_BASE,
    beam_width: int = 1,
    n_batches: int = 2,
    data_overrides: Optional[dict] = None,
    mode: str = "eval",
) -> pd.DataFrame:
    """Evaluate all trial checkpoints in a hypertune study.

    Parameters
    ----------
    study_name:
        Name of the Optuna study (e.g. ``"small_baseline"``).
    results_base:
        Parent directory for hypertune results; default ``"results"``.
    beam_width:
        Beam width for route prediction (1 = greedy).
    n_batches:
        Number of validation batches per trial (2 = fast sanity check).
    data_overrides:
        Dot-notation config overrides so local data paths shadow taco-absolute ones::

            data_overrides = {
                "context.building_blocks": "data/small_building_blocks.csv",
                "context.templates_path": "data/small_reaction_templates.pickle",
                "dataset.routes_path": "data/small_routes.json",
            }
    mode:
        ``"eval"`` (default) — load each checkpoint and run inference.
        ``"progress"`` — read only ``train_progress.jsonl`` files (no model loading, fast).

    Returns
    -------
    pd.DataFrame
        One row per completed trial.  In ``"eval"`` mode, sorted by
        ``fraction_solved`` descending.  In ``"progress"`` mode, sorted by
        ``best_valid_accuracy`` descending if that column exists.
    """
    trial_dirs = find_trial_dirs(study_name, results_base)
    if not trial_dirs:
        logger.warning("No trial directories with model.pth found for study '%s'", study_name)
        return pd.DataFrame()

    if mode == "progress":
        rows = []
        for td in trial_dirs:
            prog = read_trial_progress(td)
            if prog.empty:
                row: dict = {"trial": td.name, "trial_dir": str(td), "n_epochs": 0}
            else:
                row = {"trial": td.name, "trial_dir": str(td), "n_epochs": len(prog)}
                for col in prog.select_dtypes("number").columns:
                    if "loss" in col:
                        row[f"best_{col}"] = prog[col].min()
                    else:
                        row[f"best_{col}"] = prog[col].max()
                last = prog.iloc[-1].to_dict()
                row.update({f"final_{k}": v for k, v in last.items()})
            rows.append(row)
        df = pd.DataFrame(rows)
        if "best_valid_accuracy" in df.columns:
            df = df.sort_values("best_valid_accuracy", ascending=False)
        return df.reset_index(drop=True)

    rows = []
    for td in trial_dirs:
        logger.info("Evaluating %s …", td.name)
        try:
            row = evaluate_trial(
                td,
                beam_width=beam_width,
                n_batches=n_batches,
                data_overrides=data_overrides,
            )
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", td.name, exc)
            row = {
                "trial": td.name,
                "trial_dir": str(td),
                "fraction_solved": float("nan"),
                "error": str(exc),
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    if "fraction_solved" in df.columns:
        df = df.sort_values("fraction_solved", ascending=False)
    return df.reset_index(drop=True)
