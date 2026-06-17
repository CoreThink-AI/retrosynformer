#!/usr/bin/env python
"""Hyperparameter search for RetroSynFormer using Optuna.

The first trial is fixed (n_heads=4, n_layers=3, head_dim=256, lr=0.0005) based
on the best standard-dataset finding from trial_001 of the baseline-standard
study (taco, 2026-06-16).
Subsequent trials explore the search space defined in config["optuna"].

Usage:
    python scripts/hypertune.py -c results/config/small.yaml [--n-trials 20]

Results are written to results/hypertune-{study_name}/:
  study.db        Optuna storage (SQLite)
  run.jsonl       Structured log — one JSON object per line
  trial_NNN/      Per-trial model checkpoints and progress files

Multiple studies run simultaneously without conflict because each writes to
its own results/hypertune-{study_name}/ directory.  To add parallel workers
to the *same* study, pass an identical --study-name and --storage to each
process; Optuna coordinates via the shared SQLite file.

Config-driven search space (results/config/*.yaml under the "optuna" key):

    optuna:
      # list of values → suggest_categorical
      n_heads: [1, 2, 4, 8]

      # dict with choices → suggest_categorical (explicit form)
      head_dim:
        choices: [64, 128, 256]

      # dict with low/high (both int) → suggest_int; log: true for log-scale
      n_layers:
        low: 2
        high: 32
        log: true

      # dict with low/high (float) → suggest_float
      lr:
        low: 1.0e-4
        high: 1.0
        log: true

      # float range with a fixed step
      dropout:
        low: 0.0
        high: 0.3
        step: 0.01

Any key in the optuna section must match a keyword argument accepted by
runner.main() (n_heads, n_layers, head_dim, lr, dropout, momentum, …).
"""
import argparse
import json
import logging
import os
import time

import optuna

import retrosynformer.trainer as _trainer_mod
from retrosynformer.runner import main as train
from retrosynformer.runner import read_config

CONFIG_PATH_DEFAULT = "results/config.yaml"

# Fixed first trial — n_heads=4/lr=0.0005 from best standard-dataset finding
# (trial_001 of baseline-standard-lr211-100epochs study, taco 2026-06-16).
# Every value here must be a valid choice in the corresponding optuna config list.
BASELINE_TRIAL = {"n_heads": 4, "n_layers": 3, "head_dim": 256, "lr": 0.0005, "dropout": 0.1}


# ---------------------------------------------------------------------------
# Structured JSONL logging
# ---------------------------------------------------------------------------

def _write(record: dict, run_jsonl: str) -> None:
    """Append one JSON record to run.jsonl (thread-safe at OS level)."""
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(run_jsonl, "a") as f:
        f.write(json.dumps(record) + "\n")


class _JsonlHandler(logging.Handler):
    """Redirect Python log records to run.jsonl."""
    def __init__(self, run_jsonl: str):
        super().__init__()
        self.run_jsonl = run_jsonl

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
            }, self.run_jsonl)
        except Exception:
            self.handleError(record)


def _setup_jsonl_logging(run_jsonl: str) -> None:
    """Attach the JSONL handler to the root logger (WARNING and above)."""
    handler = _JsonlHandler(run_jsonl)
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

OBJECTIVE_METRICS = {
    "valid_route_accuracy": ("val_route_acc", "max"),
    "valid_action_accuracy": ("val_acc",      "max"),
    "fraction_solved":       ("fraction_solved", "value"),
}


def _validate_config(config: dict) -> None:
    """Raise ValueError for known config inconsistencies before any trial starts.

    Checks:
    - optuna.structured_dropout_bottleneck requires model.use_structured_dropout
    - optuna.objective_metric must be a known metric name
    - optuna.layer_shared_resid_dropout list-of-lists:
        (a) non-jagged: all inner lists have the same length
        (b) each list length >= the largest n_layers value in the search space
        (c) all inner values are bools
    """
    optuna_cfg = config.get("optuna", {})
    optuna_keys = set(optuna_cfg)
    use_sd = config.get("model", {}).get("use_structured_dropout", False)
    if not use_sd and "structured_dropout_bottleneck" in optuna_keys:
        raise ValueError(
            "Config conflict: 'optuna.structured_dropout_bottleneck' is in the "
            "search space but 'model.use_structured_dropout' is false. "
            "Either set use_structured_dropout: true or remove "
            "structured_dropout_bottleneck from the optuna section."
        )
    metric = optuna_cfg.get("objective_metric")
    if metric is not None and metric not in OBJECTIVE_METRICS:
        raise ValueError(
            f"Unknown optuna.objective_metric: {metric!r}. "
            f"Valid choices: {sorted(OBJECTIVE_METRICS)}"
        )

    lsrd_spec = optuna_cfg.get("layer_shared_resid_dropout")
    if lsrd_spec is not None and isinstance(lsrd_spec, list) and lsrd_spec and isinstance(lsrd_spec[0], list):
        # (a) Non-jagged: all inner lists must have equal length
        lengths = {len(lst) for lst in lsrd_spec}
        if len(lengths) > 1:
            raise ValueError(
                f"optuna.layer_shared_resid_dropout list-of-lists is jagged: "
                f"found inner lists of lengths {sorted(lengths)}. "
                f"All lists must have the same length."
            )
        list_len = next(iter(lengths))

        # (b) Length >= max n_layers in the optuna search space (or model default)
        n_layers_spec = optuna_cfg.get("n_layers")
        if n_layers_spec is None:
            max_n_layers = config.get("model", {}).get("n_layers", 0)
        elif isinstance(n_layers_spec, list):
            max_n_layers = max(int(v) for v in n_layers_spec)
        elif isinstance(n_layers_spec, dict):
            max_n_layers = int(n_layers_spec.get("high", list_len))
        else:
            max_n_layers = int(n_layers_spec)
        if list_len < max_n_layers:
            raise ValueError(
                f"optuna.layer_shared_resid_dropout inner lists have length {list_len} "
                f"but the maximum n_layers in the search space is {max_n_layers}. "
                f"Each list must be at least as long as the largest n_layers value "
                f"(extra entries beyond the actual n_layers are truncated at runtime)."
            )

        # (c) All inner values must be bool or 0/1
        for i, lst in enumerate(lsrd_spec):
            invalid = [(j, v) for j, v in enumerate(lst) if v not in (True, False, 0, 1)]
            if invalid:
                raise ValueError(
                    f"optuna.layer_shared_resid_dropout[{i}] contains values that are "
                    f"not bool or 0/1: {invalid}"
                )


# ---------------------------------------------------------------------------
# Search-space dispatcher
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec) -> object:
    """Dispatch to the appropriate trial.suggest_* based on the YAML spec.

    Four forms are supported:

    1. Flat list  →  suggest_categorical
         n_heads: [1, 2, 4, 8]

    2. List-of-lists  →  suggest_categorical over list choices
       Each inner list is one complete choice; Optuna picks one per trial.
       Inner lists are serialised to JSON strings for storage compatibility.
         layer_shared_resid_dropout:
           - [true, false, true, ...]
           - [false, false, false, ...]

    3. Dict with choices key  →  suggest_categorical
         n_heads:
           choices: [1, 2, 4, 8]

    4. Dict with low/high  →  suggest_int or suggest_float
       Type is inferred: both int → suggest_int, otherwise suggest_float.
       Optional keys: log (bool), step (number).
         lr:
           low: 1.0e-4
           high: 1.0
           log: true
         n_layers:
           low: 2
           high: 32
           log: true
         dropout:
           low: 0.0
           high: 0.3
           step: 0.01
    """
    if isinstance(spec, list):
        # List-of-lists → categorical over list choices.
        # Each inner list is serialised to a JSON string because Optuna's
        # categorical storage requires hashable (scalar) choices.  The result
        # is deserialised back to a Python list before being passed to runner.
        if spec and isinstance(spec[0], list):
            import json as _json
            choices = [_json.dumps(lst, separators=(",", ":")) for lst in spec]
            result = trial.suggest_categorical(name, choices)
            return _json.loads(result)
        return trial.suggest_categorical(name, spec)
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high = spec["low"], spec["high"]
    log = spec.get("log", False)
    step = spec.get("step")
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high,
                                 step=int(step) if step is not None else 1,
                                 log=log)
    return trial.suggest_float(name, low, high, step=step, log=log)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

_RESERVED_OPTUNA_KEYS = {"objective_metric"}


def objective(trial: optuna.Trial, config_path: str, n_epochs: int, dataset: str,
              results_base: str, run_jsonl: str, eval_n_batches: int | None = None,
              study_name: str | None = None) -> float:
    config = read_config(config_path)
    _validate_config(config)
    optuna_config = config.get("optuna", {})
    # Reserved keys configure the study itself and must not be passed to _suggest.
    objective_metric = optuna_config.get("objective_metric", "valid_route_accuracy")
    params = {name: _suggest(trial, name, spec) for name, spec in optuna_config.items()
              if name not in _RESERVED_OPTUNA_KEYS}

    trial_dir = os.path.join(results_base, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    print(f"\n### Trial {trial.number} params")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    model_params = dict(
        config_path=config_path,
        dataset=dataset,
        n_epochs=n_epochs,
        results_path=trial_dir,
        eval_routes_at_end=True,
        trial_number=trial.number,
        study_name=study_name,
    )
    if eval_n_batches is not None:
        model_params["eval_n_batches"] = eval_n_batches
    model_params.update(params)
    # Keys that are control/path metadata — exclude from the all_params snapshot
    # written to run.jsonl so that rs-plot-learning-curves can display fixed
    # architecture params alongside Optuna-suggested ones.
    _META_KEYS = {"config_path", "results_path", "eval_routes_at_end",
                  "trial_number", "study_name", "eval_n_batches"}
    _write({
        "event": "trial_start",
        "trial": {"number": trial.number, "dir": trial_dir},
        "params": dict(trial.params),
        "all_params": {k: v for k, v in model_params.items() if k not in _META_KEYS},
        "optuna_keys": list(params.keys()),
    }, run_jsonl)
    print("\n#### Model params")
    for k, v in model_params.items():
        print(f"    {k}: {v}")

    t0 = time.time()
    val_loss, val_acc, val_route_acc, fraction_solved = train(**model_params)

    # Compute the Optuna objective from whichever metric is configured.
    # val_route_acc avoids a spurious 0.0 when early stopping fires before
    # route eval; fraction_solved falls back to 0.0 when eval didn't run.
    if objective_metric == "valid_action_accuracy":
        value = max(val_acc) if val_acc else 0.0
    elif objective_metric == "fraction_solved":
        value = fraction_solved if fraction_solved is not None else 0.0
    else:  # default: valid_route_accuracy
        value = max(val_route_acc) if val_route_acc else 0.0
    print(f"\nObjective ({objective_metric}): {value:.4f}")

    trial_results = {
        "event": "trial_end",
        "trial": {"number": trial.number, "dir": trial_dir},
        "params": dict(trial.params),
        "results": {
            "duration_s": round(time.time() - t0, 1),
            "accuracy": {
                "optuna_score": value,
                "fraction_targets_solved": fraction_solved,
                "valid_loss": float(val_loss[-1]) if val_loss else None,
                "valid_action_accuracy": float(val_acc[-1]) if val_acc else None,
                "valid_route_accuracy": float(val_route_acc[-1]) if val_route_acc else None,
            },
        },
    }
    _write(trial_results, run_jsonl)
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
        help="Optuna study name (also determines output directory: results/hypertune-{name}/)",
    )
    parser.add_argument(
        "--storage", default=None,
        help="Optuna storage URL (default: sqlite:///results/hypertune-{study_name}/study.db)",
    )
    args = parser.parse_args()

    results_base = f"results/hypertune-{args.study_name}"
    run_jsonl = os.path.join(results_base, "run.jsonl")
    storage = args.storage or f"sqlite:///{results_base}/study.db"

    os.makedirs(results_base, exist_ok=True)
    is_fresh = not os.path.exists(os.path.join(results_base, "study.db"))

    _setup_jsonl_logging(run_jsonl)
    _write({"event": "study_start" if is_fresh else "study_resume", "config": {
        "n_trials": args.n_trials, "n_epochs": args.n_epochs,
        "dataset": args.dataset, "storage": storage,
        "config_path": args.config_path,
        "eval_n_batches": args.eval_n_batches,
    }}, run_jsonl)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
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
        }, run_jsonl)

    def _objective_with_interrupt(trial):
        # Register study.stop as the interrupt callback so a Ctrl-C stops the
        # study after the current trial completes and its result is recorded.
        _trainer_mod.set_interrupt_callback(study.stop)
        try:
            return objective(trial, args.config_path, args.n_epochs, args.dataset,
                             results_base, run_jsonl, args.eval_n_batches, args.study_name)
        finally:
            _trainer_mod.clear_interrupt_callback()

    study.optimize(
        _objective_with_interrupt,
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
            "dir": os.path.join(results_base, f"trial_{best.number:03d}"),
        },
    }, run_jsonl)
    print("\n=== Best trial ===")
    print(f"  fraction_targets_solved: {best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  results: {os.path.join(results_base, f'trial_{best.number:03d}')}")

    if _trainer_mod.is_interrupted():
        raise KeyboardInterrupt


if __name__ == "__main__":
    main()
