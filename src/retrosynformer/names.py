"""Canonical abbreviations for all display names used across the codebase.

Public dicts:
  ABBREV    — maps full names → short display labels
  UNABBREV  — inverts ABBREV (short display labels → full names)

Public functions:
  abbreviate(name)   — full name → short label
  unabbreviate(name) — short label → full name

Round-trip doctests
-------------------
Every key in ABBREV survives a full round-trip:

    >>> all(unabbreviate(abbreviate(k)) == k for k in ABBREV)
    True

Every value (short label) in ABBREV survives the reverse round-trip:

    >>> all(abbreviate(unabbreviate(v)) == v for v in ABBREV.values())
    True

Individual examples in both directions:

    >>> abbreviate("valid_action_accuracy")
    'v_a_acc'
    >>> unabbreviate("v_a_acc")
    'valid_action_accuracy'

    >>> abbreviate("n_heads")
    'heads'
    >>> unabbreviate("heads")
    'n_heads'

    >>> abbreviate("beam_width")
    'bw'
    >>> unabbreviate("bw")
    'beam_width'

    >>> abbreviate("learning_rate")
    'lr'
    >>> unabbreviate("lr")
    'learning_rate'

Names absent from ABBREV fall back to systematic rules; the reverse also applies:

    >>> abbreviate("valid_custom_metric")
    'v_custom_metric'
    >>> unabbreviate("v_custom_metric")
    'valid_custom_metric'

    >>> abbreviate("train_custom_accuracy")
    't_custom_acc'
    >>> unabbreviate("t_custom_acc")
    'train_custom_accuracy'

Unknown names pass through unchanged in both directions:

    >>> abbreviate("unknown_key")
    'unknown_key'
    >>> unabbreviate("unknown_key")
    'unknown_key'
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
    # Epoch counts
    "n_epochs": "epochs",
    # Dataset / study
    "valid_set": "vset",
    "objective_metric": "obj_metric",
    "dataset_name": "dataset",
}

UNABBREV: dict[str, str] = {v: k for k, v in ABBREV.items()}


def abbreviate(name: str) -> str:
    """Return the abbreviated display name for any metric or config key.

    Lookup order:
    1. ABBREV exact match.
    2. Systematic metric-name shortening (valid_/train_ prefixes, *_accuracy suffix).
    3. *name* unchanged.
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


def unabbreviate(name: str) -> str:
    """Return the full name for any abbreviated display label.

    Lookup order:
    1. UNABBREV exact match (covers every entry in ABBREV).
    2. Systematic reverse of abbreviate() fallback rules.
    3. *name* unchanged.
    """
    if name in UNABBREV:
        return UNABBREV[name]
    return (
        name
        .replace("v_", "valid_", 1)
        .replace("t_", "train_", 1)
        .replace("a_acc", "action_accuracy")
        .replace("r_acc", "route_accuracy")
        .replace("_acc", "_accuracy")
    )
