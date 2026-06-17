# RetroSynFormer 0.1.13 Release Notes

*Branch: `feature-nonuniform-dropout` — June 2026*

---

## Summary

Several improvements to `rs-plot-learning-curves`: multiple metrics can now
be plotted in a single call, fixed architecture params (n_heads, n_layers,
etc.) appear in the trial table alongside Optuna-searched ones, and
Optuna-searched column headers are marked with `*` so fixed vs. searched
params are visually distinct.

---

## Changes

### 1. Multiple `--metric` flags (`src/retrosynformer/scripts/plot_learning_curves.py`)

`--metric` now uses `action="append"` and can be repeated:

```
rs-plot-learning-curves \
    --metric valid_route_accuracy \
    --metric train_route_accuracy \
    --metric valid_action_accuracy
```

- Trials are **ranked by the first metric**.
- Each metric is drawn with a distinct linestyle (solid → dashed → dotted → dashdot) at the same trial color.
- With multiple metrics the legend splits into two: trial colors (upper-left) and metric → linestyle key (lower-right).
- `--also-train` inserts `train_*` counterparts for any `valid_*` metric not already in the list, skipping duplicates.

### 2. Fixed architecture params in trial table

`hypertune.py` now writes two extra fields to each `trial_start` record in `run.jsonl`:

| Field | Content |
|-------|---------|
| `all_params` | Full resolved config minus path/control metadata (`config_path`, `results_path`, `eval_routes_at_end`, `trial_number`, `study_name`, `eval_n_batches`) |
| `optuna_keys` | List of param names actually suggested by Optuna for this trial |

`plot_learning_curves.py` reads `run.jsonl` for each study and adds columns from `all_params` that are not already in the DataFrame — making fixed params like `n_heads`, `n_layers`, `head_dim`, and `lr` visible even when they were not part of the Optuna search space.

### 3. Optuna param column headers marked with `*`

Column headers for Optuna-searched params are suffixed with `*` (e.g. `layer_shared_resid_dropout*`); fixed params use plain names (`n_heads`, `n_layers`).

### 4. Long list params truncated in trial table

List-valued params longer than 3 elements (e.g. `[True, True, True, True, True, True, True, True, True, True]`) are summarised as `[first ... last]` (e.g. `[True ... True]`). Handles both Python list objects and JSON-encoded list strings (the form Optuna uses for categorical list choices).

---

## Compatibility note

Fields `all_params` and `optuna_keys` are only present in `run.jsonl` records written by this version or later. Trials from earlier runs will continue to show only the Optuna-varied params in the table.
