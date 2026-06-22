"""Shared formatting for the per-epoch training progress table.

Defines the canonical column layout once so that:

* ``RetroTrainer`` can print a live training table to the terminal.
* ``EpochLogger.flush()`` writes exactly the same data to ``train_progress.jsonl``
  (where keys are the full jsonl names, not the abbreviated display names).
* ``hplot`` / ``rs-plot-learning-curves`` can reconstruct the table from jsonl
  records without any separate formatting logic.

Usage
-----
    from retrosynformer.training_display import format_epoch_header, format_epoch_row

    # Once per N epochs (reprint the header):
    print(format_epoch_header(trial=trial_number, study=study_name))

    # Once per epoch (after EpochLogger.flush() returns the record):
    record = EpochLogger.flush()
    print(format_epoch_row(record))
"""

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# Each entry: (display_header, jsonl_key, header_format_spec, value_format_fn)
#
# header_format_spec is used with f"{hdr:{spec}}" to right/left-align the
# column header to match the value width.
#
# The ``trial`` and ``study`` columns are optional (omitted when the run is
# not part of a hypertune study); callers control this via ``include_trial``
# and ``include_study`` flags in the format functions below.

_COL_SEP = "  "

EPOCH_TABLE_COLS: list[tuple[str, str, str, object]] = [
    # (display_hdr, jsonl_key, hdr_fmt_spec, value_fmt_fn)
    ("epoch",  "epoch",                    ">5",  lambda v: f"{int(v):>5}"),
    ("t_loss", "train_loss",               ">7",  lambda v: f"{v:>7.5f}"),
    ("t_acc",  "train_action_accuracy",    ">7",  lambda v: f"{v:>7.5f}"),
    ("t_racc", "train_route_accuracy",     ">7",  lambda v: f"{v:>7.5f}"),
    ("v_loss", "valid_loss",               ">7",  lambda v: f"{v:>7.5f}"),
    ("v_acc",  "valid_action_accuracy",    ">7",  lambda v: f"{v:>7.5f}"),
    ("v_racc", "valid_route_accuracy",     ">7",  lambda v: f"{v:>7.5f}"),
    ("s/ep",   "seconds_per_epoch",        ">6",  lambda v: f"{v:>6.1f}"),
    ("note",   "is_best",                  "<4",  lambda v: f"{'*' if v else '':<4}"),
]

_TRIAL_COL  = ("trial", "trial_number", ">5",  lambda v: f"{int(v):>5}")
_STUDY_COL  = ("study", "study_name",   "",    lambda v: str(v))


def _cols(include_trial: bool, include_study: bool):
    cols = []
    if include_trial:
        cols.append(_TRIAL_COL)
    cols.extend(EPOCH_TABLE_COLS)
    if include_study:
        cols.append(_STUDY_COL)
    return cols


def format_epoch_header(include_trial: bool = False, include_study: bool = False) -> str:
    """Return the formatted column-header line for the training progress table."""
    parts = [f"{hdr:{spec}}" for hdr, _, spec, _ in _cols(include_trial, include_study)]
    return _COL_SEP.join(parts)


def format_epoch_row(record: dict, include_trial: bool = False, include_study: bool = False) -> str:
    """Format one epoch's record as a display row.

    Missing keys in *record* are rendered as ``-`` (or empty for note/study).
    """
    parts = []
    for hdr, key, spec, fmt_fn in _cols(include_trial, include_study):
        val = record.get(key)
        if val is None:
            # Use the spec width to produce a right-aligned dash placeholder.
            width = spec.lstrip("<>^").rstrip("s")
            try:
                placeholder = f"{'-':{spec}}" if spec else "-"
            except (ValueError, TypeError):
                placeholder = "-"
            parts.append(placeholder)
        else:
            try:
                parts.append(fmt_fn(val))
            except (TypeError, ValueError):
                parts.append(f"{str(val):{spec}}" if spec else str(val))
    return _COL_SEP.join(parts)


def iter_jsonl_rows(path: str):
    """Yield each epoch record from a ``train_progress.jsonl`` file."""
    import json
    import os
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
