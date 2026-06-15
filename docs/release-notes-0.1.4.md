# RetroSynFormer 0.1.4 Release Notes

*Branch: `feature-structured-dropout` — June 2026*  
*29 files changed, 2 810 insertions, 182 deletions*

---

## Highlights

- **Structured Dropout** — molecule-conditioned masking of the Decision Transformer's hidden state, the primary research contribution of this branch.
- **Optuna study tooling** — `retrosynformer.study` module + `show_study.py` / `show_all_studies.py` scripts for querying, merging, and displaying hyperparameter search results.
- **Hypertuning robustness** — study-directory locking, interactive resume from `study.db`, verbose JSONL trial logging.
- **Training robustness** — early-stopping patience, correct `valid_loss` restoration on resume, `--start-epoch` / `--batch-size` / `--num-workers` CLI flags.
- **Headless server fix** — `matplotlib.use("Agg")` prevents X11 crash on SSH/ROCm servers.
- **AMD ROCm extras** — `pip install retrosynformer[amdgpu]` wires the ROCm PyTorch index via `[tool.uv.sources]`.
- **Test suite** — four new test files, 700+ lines of unit and integration tests.

---

## 1. Structured Dropout (`src/retrosynformer/structured_dropout.py`)

### Architecture

`MoleculeConditionedMaskGenerator` is a two-layer MLP that maps the target molecule's 1024-bit Morgan fingerprint to a per-channel **drop-probability vector** matching the Decision Transformer's hidden size:

```
fp (1024) → Linear(1024, bottleneck) → ReLU
           → Linear(bottleneck, hidden_size) → Sigmoid → drop_prob (hidden_size,)
```

Key design choices:

| Detail | Value | Rationale |
|--------|-------|-----------|
| Output bias init | −2.2 | sigmoid(−2.2) ≈ 0.10 — ~10% initial drop rate |
| Drop cap | 0.9 | prevents total channel suppression |
| Train mode | inverted Bernoulli mask, scaled ×1/(1−p) | preserves expected activation magnitude |
| Eval mode | deterministic expected-value mask (1−p) | stable, reproducible inference |

`StructuredDropoutDecisionTransformer` wraps the standard `DecisionTransformerModel` and registers a `forward_hook` on `embed_ln` — the LayerNorm applied immediately after all modality embeddings (returns, states, actions, rewards) are stacked — so the mask is applied before any transformer layer without modifying `RetroTrainer`, `RoutePredictor`, or existing training code.

### Activation in `config.yaml`

```yaml
model:
  use_structured_dropout: true
  structured_dropout_bottleneck: 128   # MLP hidden dim; searched by Optuna
```

`runner.py` checks `use_structured_dropout` and instantiates `StructuredDropoutDecisionTransformer` in place of the bare `DecisionTransformerModel`.  A guard in `hypertune.py` raises `ValueError` if `structured_dropout_bottleneck` appears in the Optuna search space but `use_structured_dropout` is `false`.

### Config files added

| File | Purpose |
|------|---------|
| `results/config/small.yaml` | 589-template baseline (standard dropout) |
| `results/config/small_sd.yaml` | 589-template structured-dropout run, bottleneck searched 32–512 |
| `results/config/standard.yaml` | 1573-template config |
| `results/config/large.yaml` | 2957-template config |

### Test statistics (random init, `tests/test_structured_dropout.py`)

At random initialisation the mask generator produces ~10% mean drop rate across all test molecules (methane through cholesterol), confirming the bias-init design.  Molecule-specific spread grows with molecular complexity (cholesterol std 0.005 vs methane 0.001).  Pairwise mask cosine similarity is ~0.999 at init — spatial structure is expected to emerge during training.  See [`docs/structured-dropout-report.md`](structured-dropout-report.md) for full per-molecule statistics and property-correlation analysis.

### Paper baseline for comparison

From Granqvist et al. (*Digital Discovery* 2026, Table 6), using beam width 50 on the small dataset:

| Model | N1 success | N5 success |
|-------|-----------|-----------|
| RetroSynFormer50 | **0.950** | **0.833** |
| AiZynthFinder | 0.923 | 0.917 |

Our 10-epoch greedy hypertune baseline (`trial_000`, n_heads=1, n_layers=3, head_dim=256) scored `fraction_solved=0.352`, consistent with the paper's Fig. 7a (greedy ≈ 0.30 at full training).  The clearest expected signal for structured dropout is an **N5 success-rate gain** (harder, more chemically diverse targets).

---

## 2. Optuna Study Tooling

### `src/retrosynformer/study.py` (new module)

| Function | Description |
|----------|-------------|
| `to_dfs(db_path)` | Load all Optuna SQLite tables into a `dict[str, DataFrame]` |
| `to_trials_df(db_path)` | Join tables into one row per trial; decode categorical params; compute duration |
| `dfs_to_trials_df(dfs)` | Same as above from an already-loaded dict; adds `study_name` column when multiple studies are merged |
| `concat(a_dfs, b_dfs)` | Merge two study dicts, reindexing all PKs and FKs to avoid collisions (`study_id` × 5 tables, `trial_id` × 7 tables, `trials.number`, 9 other table-local PKs) |
| `concat_all(pattern, root)` | Glob `results/**/study.db`, sort paths, reduce with `concat` |

### `scripts/show_study.py` (new)

Prints a formatted per-trial table for a single `study.db` with decoded params, duration, score, `*` on the best trial, and aggregate stats.  Accepts `--sort <col>` and `--ascending/--no-ascending`.

```
python scripts/show_study.py results/hypertune-small_baseline/study.db --sort score
```

### `scripts/show_all_studies.py` (new)

Globs for all `study.db` files, merges them, and prints a single table with a `study_name` column identifying each trial's source.  Defaults to `--sort score` descending.

```
python scripts/show_all_studies.py "results/hypertune-*/study.db"
```

---

## 3. Hypertuning Improvements (`scripts/hypertune.py`)

- **Study-directory lock** — `results/hypertune` is a symlink that acts as a mutex; a second simultaneous run fails fast rather than clobbering results.
- **Interactive resume** — when `results/hypertune` already points to a directory containing `study.db`, the user is prompted `Continue existing study? [y/N]`; answering `y` reuses the existing `study.db` and appends to `run.jsonl`.  The JSONL event is `study_resume` instead of `study_start`.
- **Config validation** — `ValueError` if `optuna.structured_dropout_bottleneck` is present but `model.use_structured_dropout` is false.
- **Verbose JSONL logging** — `trial_start`, `trial_end`, `trial_complete`/`trial_fail`, `study_start`/`study_resume`, `study_end` events; Python `WARNING`+ logs also redirected to `run.jsonl` via a custom `logging.Handler`.
- **`scripts/hypertune_verbose.py`** — standalone verbose variant retained for reference.

---

## 4. Training Improvements

### Early stopping (`src/retrosynformer/trainer.py`)

```yaml
train:
  early_stopping_patience: 4   # stop after 4 epochs with no valid_loss improvement
```

All four dataset configs default to `patience=4`.

### Resume fix

When `--resume` is combined with a mixed-dataset JSONL history (e.g. a standard-dataset run followed by a large-dataset run in the same file), `trainer.py` now scopes `lowest_valid_loss` to the most recent contiguous epoch block by detecting where `epoch[i] < epoch[i-1]`.  Previously, the global minimum could belong to a different (lower-loss) dataset run, preventing the current run from ever saving a checkpoint.

### New CLI flags (`scripts/train.py`, `src/retrosynformer/runner.py`)

| Flag | Purpose |
|------|---------|
| `--start-epoch N` | Resume training from epoch N, loading history from `train_progress.jsonl` |
| `--batch-size N` / `-b N` | Override `train.batch_size` and `evaluation.batch_size` from config |
| `num_workers` (config key) | DataLoader parallel CPU prefetch workers to reduce GPU idle time |

---

## 5. Remote Monitoring (`src/retrosynformer/monitor.py`, new)

`RemoteTrialMonitor` polls a remote server (e.g. `taco`) via `rsync` for the latest trial checkpoint and runs local beam-search evaluation, streaming results back to a local JSONL file.  Intended for watching long-running `hypertune.py` jobs on AMD GPU servers without blocking the training process.

Supporting shell scripts:
- `scripts/monitor_train_progress.sh` — tail `train_progress.jsonl` with formatted columns
- `scripts/train_structured_dropout_comparison.sh` — paired baseline vs SD hypertune runs

---

## 6. Bug Fixes

### Matplotlib X11 crash on headless servers

`matplotlib.use("Agg")` is now called before any `pyplot` import in all three plotting modules (`utils/utils.py`, `utils/evaluation.py`, `utils/evaluation_class.py`).  Prevents `XIO: fatal IO error 22` when SSH X11 forwarding is unavailable or the display disappears mid-run.

### `valid_loss` global-minimum bug

See Section 4 (Resume fix) above.

---

## 7. Tests

| File | Count | Coverage |
|------|-------|---------|
| `tests/test_structured_dropout.py` | 8 tests | `MoleculeConditionedMaskGenerator`, `StructuredDropoutDecisionTransformer` forward pass, drop-rate stats, eval determinism, train/eval mode contrast |
| `tests/test_plotting.py` | 10 tests | All three plot functions, Agg backend guard, single-epoch edge case, output file non-empty |
| `tests/test_utils.py` | 30 tests | `flatten`, `one_hot_encoder`, `get_morgan_fingerprint`, `check_if_building_block`, `flatten_and_crop`, `get_index_values`, `convert_batches_to_action_ids` |
| `tests/test_hypertune.py` | 14 tests | `_suggest` dispatch (all three spec forms), `_validate_config` guard, `_acquire_lock` / `_release_lock` |
| `tests/test_doctests.py` | — | Runs all `doctest`-style examples embedded in `utils.py` |

---

## 8. AMD ROCm Support (`pyproject.toml`)

```toml
[project.optional-dependencies]
amdgpu = ["torch>=2.12.0", "sympy==1.13.1"]

[tool.uv.sources]
torch = [
    { index = "pytorch-rocm", extra = "amdgpu" },
    { index = "pytorch-cpu" },
]

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm6.2"
explicit = true
```

Install on AMD GPU servers:

```bash
uv sync --extra amdgpu
# or: pip install retrosynformer[amdgpu] --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

Verified on AMD Strix Halo iGPU (Radeon 8060S, gfx1151, 32 GB VRAM) with ROCm 7.2.0 — ~16 s/epoch on the small dataset vs ~66 s/epoch on CPU.  See [`docs/TODO-allocate-VRAM-in-bios.md`](TODO-allocate-VRAM-in-bios.md) for full ROCm server setup steps.

---

## Upgrade notes

- `to_trials_df(db_path)` return value is unchanged for single-study calls (no `study_name` column).
- `runner.main()` gains `structured_dropout_bottleneck` as an optional keyword argument; existing callers are unaffected.
- All four dataset configs now carry `early_stopping_patience: 4`; override to `0` to disable.
