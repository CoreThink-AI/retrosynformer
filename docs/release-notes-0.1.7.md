# RetroSynFormer 0.1.7 Release Notes

*Branch: `feature-structured-dropout` — June 2026*

---

## Summary

New CLI tooling for monitoring and comparing Optuna hypertune runs, a fix so the
Optuna objective score is never spuriously 0.0, and a configurable objective
metric in the YAML config.

---

## Changes

### 1. `rs-plot-learning-curves` CLI

New script at `src/retrosynformer/scripts/plot_learning_curves.py`.

- Globs for all `study.db` files, loads completed trials from every Optuna study,
  ranks by a chosen metric, and overlays the per-epoch learning curves from each
  trial's `train_progress.jsonl`.
- Handles training restarts gracefully: only the last contiguous epoch run in each
  JSONL file is used (earlier rows from a previous run are discarded).
- Deduplicates trials that appear in multiple databases (rsync-nested copies,
  multi-study SQLite files).
- Key flags:

  | Flag | Default | Effect |
  |------|---------|--------|
  | `pattern` | `results/**/study.db` | Glob for study databases |
  | `--top N` | 10 | Plot the top-N trials |
  | `--metric` | `valid_action_accuracy` | Metric to rank and plot |
  | `--also-train` | off | Overlay the matching `train_*` metric (dashed) |
  | `--study NAME` | all | Filter by study name or directory (substring) |
  | `--min-score` | 0.2 (accuracy only) | Exclude trials below/above threshold |
  | `--xscale` / `--yscale` | linear / log | Axis scaling |
  | `--out PATH` | (show) | Save figure to file instead of displaying |

### 2. `rs-sync-results` CLI

New `retrosynformer.rsync` module and `src/retrosynformer/scripts/sync_results.py`.

- Wraps `rsync` to pull only `study.db` and `train_progress.jsonl` from a remote
  host while preserving directory structure.
- rsync filter order guarantees correctness: `--include='*/'` → per-file includes
  → `--exclude='*'`.
- `--include` flag lets you add extra patterns; `--dry-run` / `--quiet` pass through.

```bash
rs-sync-results                          # pull from taco → results/
rs-sync-results --dry-run
rs-sync-results --include run.jsonl --include "*.png"
```

### 3. Optuna objective score is never 0.0 (`trainer.py`, `hypertune.py`)

**Problem**: When `early_stopping_patience` fired before the first
`eval_routes_frequency` epoch, `fraction_targets_solved` was `None`, and
`hypertune.py` reported `0.0` to Optuna. The sampler then treated these trials as
the worst-performing, biasing future suggestions.

**Fix (two parts)**:

1. `trainer.train()` now accepts `eval_routes_at_end=True`. When set, a full
   route evaluation is always run after the training loop finishes (in addition
   to any mid-training evals).
2. `hypertune.py` passes `eval_routes_at_end=True` for every trial, so
   `fraction_targets_solved` is always populated.
3. The Optuna objective is now `max(valid_route_accuracy)` (the running best
   across all epochs) rather than the final `fraction_targets_solved`. This
   metric is always non-zero after the first epoch and better reflects training
   progress when route eval runs only once at the end.

### 4. Configurable Optuna objective metric (`optuna.objective_metric`)

The objective can now be changed per-experiment in `config.yaml` without editing
Python:

```yaml
optuna:
  objective_metric: valid_route_accuracy  # valid_action_accuracy | valid_route_accuracy | fraction_solved
  n_heads: [1, 2, 3, 4]
  ...
```

`_validate_config` rejects unknown metric names with a clear error listing valid
choices. The key is excluded from the Optuna search space (never passed to
`_suggest`). Default when absent: `valid_route_accuracy`.

All six configs in `results/config/` were updated with the documented key.

---

## Upgrade notes

- No API changes to `runner.main()` or `RetroTrainer` beyond the new
  `eval_routes_at_end` keyword argument (defaults to `False`).
- Existing `study.db` files are forward-compatible; Optuna scores from earlier
  trials using `fraction_targets_solved` will look low relative to the new
  `valid_route_accuracy` scores. Start a fresh study when switching objective.
- `rs-plot-learning-curves` and `rs-sync-results` require
  `uv sync` / `pip install -e .` to register the new entry points.
