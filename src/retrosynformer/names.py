"""Canonical abbreviations for metric and column names used across the codebase."""

NAME_ABBREV: dict[str, str] = {
    "valid_action_accuracy": "v_a_acc",
    "valid_route_accuracy": "v_r_acc",
    "train_action_accuracy": "t_a_acc",
    "train_route_accuracy": "t_r_acc",
    "fraction_solved": "frac_solved",
    "valid_loss": "v_loss",
    "train_loss": "t_loss",
}


def abbrev(name: str) -> str:
    """Return the abbreviated display name for a metric or column.

    Falls back to a systematic shortening when the name is not in NAME_ABBREV.
    """
    if name in NAME_ABBREV:
        return NAME_ABBREV[name]
    return (
        name
        .replace("valid_", "v_")
        .replace("train_", "t_")
        .replace("action_accuracy", "a_acc")
        .replace("route_accuracy", "r_acc")
        .replace("_accuracy", "_acc")
    )
