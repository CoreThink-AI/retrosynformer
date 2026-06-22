"""Canonical abbreviations for all display names used across the codebase.

Single public dict:
  ABBREV  — maps full names to short display labels for both per-epoch metrics
            (valid_action_accuracy → v_a_acc) and model.config.yaml keys
            (n_heads → heads, attn_pdrop → a_pdrop, …).

Two helper functions:
  abbrev(name)       — abbreviate any name; falls back to systematic rules for
                       metric names not in ABBREV
  param_abbrev(name) — abbreviate a config/hyperparameter key; identity fallback
"""

ABBREV: dict[str, str] = {
    # Per-epoch metrics (train_progress.jsonl column names)
    "valid_action_accuracy": "v_a_acc",
    "valid_route_accuracy": "v_r_acc",
    "train_action_accuracy": "t_a_acc",
    "train_route_accuracy": "t_r_acc",
    "fraction_solved": "frac_solved",
    "valid_loss": "v_loss",
    "train_loss": "t_loss",
    # Architecture (model.config.yaml keys)
    "n_heads": "heads",
    "n_layers": "layers",
    "head_dim": "h_dim",
    "n_in_state": "n_state",
    "fp_dim": "fp",
    # Regularisation
    "dropout": "drop",
    "attn_pdrop": "a_pdrop",
    "resid_pdrop": "r_pdrop",
    "weight_decay": "wd",
    # Optimisation
    "batch_size": "bs",
    "learning_rate": "lr",
    # Evaluation / early stopping
    "early_stopping_patience": "es_pat",
    "eval_routes_frequency": "eval_freq",
    "beam_width": "bw",
    # Dataset / study
    "valid_set": "vset",
    "objective_metric": "obj_metric",
}


def abbrev(name: str) -> str:
    """Return the abbreviated display name for a metric or column.

    Falls back to a systematic shortening when the name is not in ABBREV.
    """
    if name in ABBREV:
        return ABBREV[name]
    return (
        name
        .replace("valid_", "v_")
        .replace("train_", "t_")
        .replace("action_accuracy", "a_acc")
        .replace("route_accuracy", "r_acc")
        .replace("_accuracy", "_acc")
    )


def param_abbrev(name: str) -> str:
    """Return the abbreviated display name for a config/hyperparameter key.

    Returns *name* unchanged when not found in ABBREV.
    """
    return ABBREV.get(name, name)
