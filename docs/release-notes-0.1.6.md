# RetroSynFormer 0.1.6 Release Notes

*Branch: `feature-structured-dropout` — June 2026*

---

## Summary

Packaging fixes for PyTorch extras, a crash fix when early stopping fires before the first route-evaluation epoch, and tightened hyperparameter search ranges for the small-dataset comparison runs.

---

## Changes

### 1. PyTorch extras overhaul (`pyproject.toml`)

- **Removed** `torch==2.12.0+cpu` from base `dependencies` — installing without an extra no longer pulls any torch wheel.
- **Added `[cpu]` extra** — routes to the `pytorch-cpu` index:
  ```bash
  uv sync --extra cpu
  ```
- **Added `[rocm]` extra** — canonical name for AMD GPU installs; routes to the `pytorch-rocm` index:
  ```bash
  uv pip install -e .[rocm]
  ```
- **`[amdgpu]` kept** as a backward-compatible alias for `[rocm]`.
- **Fixed phantom version** `torch>=2.12.0` → `torch>=2.0.0` across all extras. PyTorch is currently at 2.5.x; the ROCm 6.2 index tops out at `2.5.1+rocm6.2`. The old pin made the `[amdgpu]` extra permanently unsatisfiable.

### 2. Early-stopping plot crash fix (`src/retrosynformer/trainer.py`)

`trainer.train()` unconditionally called `plot_evaluation_results` after the training loop, but `pred_routes_train_progress.json` is only written when `epoch % eval_routes_frequency == 0`. If early stopping fired before the first route-evaluation epoch, the file was missing (or empty), causing:

```
KeyError: 'epoch'
```

The call is now guarded:

```python
eval_results_path = save_folder + "/pred_routes_train_progress.json"
if os.path.exists(eval_results_path) and os.path.getsize(eval_results_path) > 2:
    utils.plot_evaluation_results(eval_results_path, ...)
```

### 3. Hypertune config tuning (`results/config/small.yaml`, `small_sd.yaml`)

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `n_heads` search space | `[1, 2, 4, 8]` | `[1, 2, 3, 4]` | drop 8-head configs that consistently fail |
| `n_layers` search space | wide (up to 32) | `[2, 3, 4, 5]` | deep models (≥8 layers) always underperform on the small dataset |
| `head_dim` search space | `[64, 128, 256]` | `[64, 128, 256]` | unchanged |
| `early_stopping_patience` | 4 | 6 | allow more epochs before stopping to reduce noise-triggered early exits |

---

## Upgrade notes

- `uv pip install -e .` no longer installs torch — choose `.[cpu]`, `.[rocm]`, or `.[amdgpu]` explicitly.
- No API changes; `trainer.train()` return value is unchanged.
- Stale `study.db` files created with the old `n_heads: [1, 2, 4, 8]` or wide `n_layers` search space are incompatible with the new configs. Delete them before resuming:
  ```bash
  rm results/hypertune-small_baseline/study.db
  rm results/hypertune-small_sd/study.db
  ```
