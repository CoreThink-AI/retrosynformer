#!/usr/bin/env python
"""Hyperparameter search for RetroSynFormer using Optuna.

The first trial is fixed (n_heads=1, n_layers=3, head_dim=256, small dataset,
200 epochs) to establish a baseline comparable to early taco-branch runs.
Subsequent trials explore the search space defined below.

Usage:
    python scripts/hypertune.py -c results/config.yaml [--n-trials 20]
"""
import argparse
import os

import optuna

from retrosynformer.runner import main as train


CONFIG_PATH_DEFAULT = "results/config.yaml"
RESULTS_BASE = "results/hypertune"

# Fixed first trial — matches early taco-branch architecture for comparison.
BASELINE_TRIAL = {"n_heads": 1, "n_layers": 3, "head_dim": 256, "lr": 0.211, "dropout": 0.1}


def objective(trial: optuna.Trial, config_path: str, n_epochs: int) -> float:
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    n_layers = trial.suggest_int("n_layers", 2, 32, log=True)
    head_dim = trial.suggest_categorical("head_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.01)

    trial_dir = os.path.join(RESULTS_BASE, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    _, _, _, fraction_solved = train(
        config_path=config_path,
        dataset="small",
        n_epochs=n_epochs,
        n_heads=n_heads,
        n_layers=n_layers,
        head_dim=head_dim,
        lr=lr,
        dropout=dropout,
        results_path=trial_dir,
    )

    return fraction_solved if fraction_solved is not None else 0.0


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
        "--study-name", default="retrosynformer_hypertune",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage", default=None,
        help="Optuna storage URL (e.g. sqlite:///hypertune.db) for persistence across runs",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_BASE, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    # Pin the first trial to the taco-branch baseline for direct comparison.
    study.enqueue_trial(BASELINE_TRIAL)

    study.optimize(
        lambda trial: objective(trial, args.config_path, args.n_epochs),
        n_trials=args.n_trials,
    )

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  fraction_targets_solved: {best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  results: {os.path.join(RESULTS_BASE, f'trial_{best.number:03d}')}")


if __name__ == "__main__":
    main()
