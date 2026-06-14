#!/usr/bin/env python
"""Hyperparameter search for RetroSynFormer using Optuna.

The first trial is fixed (n_heads=1, n_layers=3, head_dim=256, large dataset,
200 epochs) to establish a baseline comparable to early taco-branch runs.
Subsequent trials explore the search space defined below.

Usage:
    python scripts/hypertune.py -c results/config.yaml [--n-trials 20]

Structured logs are written to results/hypertune/run.jsonl — one JSON object
per line covering trial start/end, per-trial accuracy, warnings, and errors.
"""
import argparse
import json
import logging
import os
import time

import optuna

from retrosynformer.runner import main as train

CONFIG_PATH_DEFAULT = "results/config.yaml"
RESULTS_BASE = "results/hypertune"
RUN_JSONL = os.path.join(RESULTS_BASE, "run.jsonl")

# Fixed first trial — matches early taco-branch architecture for comparison.
BASELINE_TRIAL = {"n_heads": 1, "n_layers": 3, "head_dim": 256, "lr": 0.211, "dropout": 0.1}


# ---------------------------------------------------------------------------
# Structured JSONL logging
# ---------------------------------------------------------------------------

def _write(record: dict) -> None:
    """Append one JSON record to run.jsonl (thread-safe at OS level)."""
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(RUN_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


class _JsonlHandler(logging.Handler):
    """Redirect Python log records to run.jsonl."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _write({
                "event": "log",
                "log": {
                    "level": record.levelname,
                    "logger": record.name,
                    "file": f"{record.filename}:{record.lineno}",
                    "msg": self.format(record),
                },
            })
        except Exception:
            self.handleError(record)


def _setup_jsonl_logging() -> None:
    """Attach the JSONL handler to the root logger (WARNING and above)."""
    handler = _JsonlHandler()
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, config_path: str, n_epochs: int, dataset: str, eval_n_batches: int | None = None) -> float:
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    n_layers = trial.suggest_int("n_layers", 2, 32, log=True)
    head_dim = trial.suggest_categorical("head_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.01)

    trial_dir = os.path.join(RESULTS_BASE, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    _write({
        "event": "trial_start",
        "trial": {"number": trial.number, "dir": trial_dir},
        "params": dict(trial.params),
    })
    print(f"\n### Trial {trial.number} params")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    model_params = dict(
        config_path=config_path,
        dataset=dataset,
        n_epochs=n_epochs,
        n_heads=n_heads,
        n_layers=n_layers,
        head_dim=head_dim,
        lr=lr,
        dropout=dropout,
        results_path=trial_dir,
        eval_n_batches=eval_n_batches,
    )
    print("\n#### Model params")
    for k, v in model_params.items():
        print(f"    {k}: {v}")

    t0 = time.time()
    val_loss, val_acc, val_route_acc, fraction_solved = train(**model_params)

    value = fraction_solved if fraction_solved is not None else 0.0
    trial_results = {
        "event": "trial_end",
        "trial": {"number": trial.number, "dir": trial_dir},
        "params": dict(trial.params),
        "results": {
            "duration_s": round(time.time() - t0, 1),
            "accuracy": {
                "fraction_targets_solved": value,
                "valid_loss": float(val_loss[-1]) if val_loss else None,
                "valid_action_accuracy": float(val_acc[-1]) if val_acc else None,
                "valid_route_accuracy": float(val_route_acc[-1]) if val_route_acc else None,
            },
        },
    }
    _write(trial_results)
    print("\n#### Results")
    for k, v in trial_results.items():
        print(f"    {k}: {v}")
    return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", default=CONFIG_PATH_DEFAULT, dest="config_path",
        help="Base config.yaml to use for all trials",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Total number of Optuna trials (including the fixed baseline)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=200,
        help="Training epochs per trial",
    )
    parser.add_argument(
        "--dataset", default="large", choices=["small", "standard", "large"],
        help="Dataset preset: small=589, standard=1573, large=2957 templates (default: large)",
    )
    parser.add_argument(
        "--eval-n-batches", type=int, default=None, dest="eval_n_batches",
        help="Override evaluation.eval_n_batches from config (reduce for faster CPU runs)",
    )
    parser.add_argument(
        "--study-name", default="retrosynformer_hypertune",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage", default=f"sqlite:///{RESULTS_BASE}/study.db",
        help="Optuna storage URL (default: sqlite in results/hypertune/study.db)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_BASE, exist_ok=True)
    _setup_jsonl_logging()
    _write({"event": "study_start", "config": {
        "n_trials": args.n_trials, "n_epochs": args.n_epochs,
        "dataset": args.dataset, "storage": args.storage,
        "config_path": args.config_path,
        "eval_n_batches": args.eval_n_batches,
    }})

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    # Only enqueue the baseline on a fresh study — resuming from storage already has it.
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_TRIAL)

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Write a completion record to run.jsonl after each trial."""
        _write({
            "event": "trial_complete" if trial.state == optuna.trial.TrialState.COMPLETE else "trial_fail",
            "trial": {"number": trial.number, "state": trial.state.name},
            "params": dict(trial.params),
            "results": {
                "value": trial.value,
                "duration_s": round(trial.duration.total_seconds(), 1) if trial.duration else None,
            },
        })

    study.optimize(
        lambda trial: objective(trial, args.config_path, args.n_epochs, args.dataset, args.eval_n_batches),
        n_trials=args.n_trials,
        callbacks=[log_trial],
    )

    best = study.best_trial
    _write({
        "event": "study_end",
        "best": {
            "trial": best.number,
            "value": best.value,
            "params": dict(best.params),
            "dir": os.path.join(RESULTS_BASE, f"trial_{best.number:03d}"),
        },
    })
    print("\n=== Best trial ===")
    print(f"  fraction_targets_solved: {best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  results: {os.path.join(RESULTS_BASE, f'trial_{best.number:03d}')}")


if __name__ == "__main__":
    main()
