# RetroSynFormer Config Reference

RetroSynFormer is configured via a single YAML file passed with `-c` to every
CLI command.  Pre-built configs live in `results/config/`.  This document covers
every section and key, then walks through a complete hyperparameter tuning
workflow.

---

## Quick start

```bash
# Train with default small-dataset config
python -m retrosynformer.runner -c results/config/small.yaml

# Run 20 Optuna trials
rs-hypertune -c results/config/small.yaml --n-trials 20 --n-epochs 100 --study-name my-study

# Plot learning curves for the top 10 trials
rs-plot-learning-curves --study my-study

# Pull results from a remote host
rs-sync-results --host taco
```

---

## Config sections

### `context`

Paths to the raw PaRoutes data files.

```yaml
context:
  building_blocks: data/standard_building_blocks.csv
  random_state: 1
  templates_path: data/standard_reaction_templates.pickle
```

| Key | Type | Description |
|-----|------|-------------|
| `building_blocks` | path | CSV of purchasable building-block SMILES |
| `templates_path` | path | Pickle of reaction SMARTS templates |
| `random_state` | int | RNG seed for reproducible train/valid/test splits |

---

### `dataset`

Controls data loading and the train/valid/test split strategy.

```yaml
dataset:
  action_dim: 589         # must match template count: small=589, standard=1573, large=2957
  drop_duplicates: true
  fp_dim: 1024
  n_in_state: 2
  routes_path: data/standard_routes.json
  shuffle: true
  synthetic_routes_path: false
  valid_set: n1+n5        # or random_split
```

| Key | Type | Description |
|-----|------|-------------|
| `action_dim` | int | Number of reaction templates; **must match the template pickle** |
| `drop_duplicates` | bool | Drop duplicate routes from the dataset before splitting |
| `fp_dim` | int | Morgan fingerprint bit-length (1024 = standard) |
| `n_in_state` | int | How many frontier molecules to concatenate as the state (state_dim = fp_dim × n_in_state) |
| `routes_path` | path | PaRoutes JSON with ground-truth synthesis routes |
| `shuffle` | bool | Shuffle routes before splitting |
| `synthetic_routes_path` | path or `false` | Optional extra synthetic routes to augment training |
| `valid_set` | `n1+n5` or `random_split` | `n1+n5`: use PaRoutes N1+N5 benchmark targets as the held-out test set; `random_split`: random 80/10/10 split |

**`action_dim` quick reference**

| Dataset | Templates |
|---------|-----------|
| small   | 589       |
| standard | 1573     |
| large   | 2957      |

---

### `model`

Decision Transformer architecture hyperparameters.

```yaml
model:
  action_tanh: false
  activation_function: relu
  attn_pdrop: 0.02
  embd_pdrop: 0.2
  head_dim: 256
  max_ep_len: 20
  n_heads: 4
  n_layers: 26
  resid_pdrop: 0.08
  use_structured_dropout: false
# structured_dropout_bottleneck: 128   # MLP hidden dim when use_structured_dropout: true
```

| Key | Type | Description |
|-----|------|-------------|
| `n_heads` | int | Number of attention heads |
| `n_layers` | int | Number of transformer layers |
| `head_dim` | int | Per-head dimension; hidden_size = n_heads × head_dim |
| `max_ep_len` | int | Maximum retrosynthesis depth (route steps) |
| `activation_function` | str | `relu` or `gelu` |
| `attn_pdrop` | float | Attention dropout probability |
| `embd_pdrop` | float | Embedding dropout probability |
| `resid_pdrop` | float | Residual dropout probability |
| `action_tanh` | bool | Apply tanh to action logits (leave false for discrete templates) |
| `use_structured_dropout` | bool | Enable `MoleculeConditionedMaskGenerator` |
| `structured_dropout_bottleneck` | int | MLP hidden dim for the mask generator (fp_dim → bottleneck → hidden_size); only used when `use_structured_dropout: true` |

---

### `optimizer`

SGD optimizer settings.

```yaml
optimizer:
  lr: 0.211
  momentum: 0.98
```

| Key | Type | Description |
|-----|------|-------------|
| `lr` | float | Initial learning rate |
| `momentum` | float | SGD momentum |

The trainer uses `ReduceLROnPlateau` on validation loss, so the `lr` here is
the starting point, not fixed.

---

### `train`

Training loop settings.

```yaml
train:
  batch_size: 512
  loss: crossEntropyLoss
  n_epochs: 200
  early_stopping_patience: 6
  results_path: results/
```

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | int | Training mini-batch size |
| `loss` | str | `crossEntropyLoss` (only option currently) |
| `n_epochs` | int | Maximum training epochs |
| `early_stopping_patience` | int | Stop if validation loss does not improve for this many epochs |
| `results_path` | path | Directory where `model.pth`, logs, and eval JSON are written |

---

### `evaluation`

Controls beam-search route evaluation (the expensive step).

```yaml
evaluation:
  batch_size: 64
  beam_width: 1
  eval_n_batches: 2
  eval_routes_frequency: 100
  max_depth: 6
  test_frac: 0.1
  sort_on: trajectory_prob
  ted_exhaustive_limit: 64
```

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | int | Molecules evaluated per batch during route prediction |
| `beam_width` | int | Beam search width; 1 = greedy, 50 = full beam |
| `eval_n_batches` | int | Number of validation batches used for route eval (reduce for faster hypertune) |
| `eval_routes_frequency` | int | Run full route eval every N epochs (expensive; set high during hypertune) |
| `max_depth` | int | Maximum search depth (tree depth) during beam search |
| `test_frac` | float | Fraction of data held out as the test set when `valid_set: random_split` |
| `sort_on` | str | How to rank candidate routes: `trajectory_prob` or `route_score` |
| `ted_exhaustive_limit` | int | Max tree pairs for exhaustive TED computation |

---

### `reward`

Reward shaping for the retrosynthesis environment.

```yaml
reward:
  building_block_reward_factor: 0
  building_block_scale_with_depth: 2
  dead_end_reward_factor: -2
  dead_end_scale_with_depth: 2
  intermediate_reward_factor: -2
  intermediate_scale_with_depth: 1
```

| Key | Type | Description |
|-----|------|-------------|
| `building_block_reward_factor` | float | Reward when a molecule is a purchasable building block |
| `building_block_scale_with_depth` | float | Multiply BB reward by depth^scale |
| `dead_end_reward_factor` | float | Penalty when a reaction produces no valid products |
| `dead_end_scale_with_depth` | float | Scale dead-end penalty with depth |
| `intermediate_reward_factor` | float | Per-step penalty for non-terminal reactions |
| `intermediate_scale_with_depth` | float | Scale intermediate penalty with depth |

---

### `optuna`

Hyperparameter search space and study configuration.  Only present (and used)
when you run `rs-hypertune`.

```yaml
optuna:
  objective_metric: valid_route_accuracy  # valid_action_accuracy | valid_route_accuracy | fraction_solved
  n_heads: [1, 2, 3, 4]
  n_layers: [2, 3, 4, 5]
  head_dim: [64, 128, 256]
  lr:
    low: 1.0e-4
    high: 0.43
    log: true
  dropout:
    low: 0.03
    high: 0.45
# structured_dropout_bottleneck:
#   low: 32
#   high: 512
#   log: true
```

#### Reserved key: `objective_metric`

Controls what Optuna maximises.  Valid choices:

| Value | Description |
|-------|-------------|
| `valid_route_accuracy` | `max(valid_route_accuracy)` across all training epochs — default, always non-zero |
| `valid_action_accuracy` | `max(valid_action_accuracy)` — fast proxy, correlates with template prediction quality |
| `fraction_solved` | Fraction of validation targets with at least one solved route (from beam search) |

#### Search-space keys

Every other key in the `optuna:` section defines a hyperparameter to search.
The key name must match a keyword argument accepted by `runner.main()`.

**List form** — `suggest_categorical`:
```yaml
n_heads: [1, 2, 3, 4]
```

**Dict with `choices`** — `suggest_categorical` (explicit):
```yaml
head_dim:
  choices: [64, 128, 256]
```

**Dict with `low`/`high` (both int)** — `suggest_int`:
```yaml
n_layers:
  low: 2
  high: 32
  log: true      # optional log-scale
```

**Dict with `low`/`high` (float)** — `suggest_float`:
```yaml
lr:
  low: 1.0e-4
  high: 1.0
  log: true
dropout:
  low: 0.0
  high: 0.3
  step: 0.01     # optional fixed step
```

**Searchable parameters** (must be accepted by `runner.main()`):

| Parameter | Notes |
|-----------|-------|
| `n_heads` | Attention heads |
| `n_layers` | Transformer depth |
| `head_dim` | Per-head hidden dim |
| `lr` | Learning rate |
| `dropout` | Dropout applied to `attn_pdrop`, `embd_pdrop`, `resid_pdrop` |
| `momentum` | SGD momentum |
| `structured_dropout_bottleneck` | MLP bottleneck dim; requires `model.use_structured_dropout: true` |

---

## Pre-built configs

| File | Dataset | Notes |
|------|---------|-------|
| `small.yaml` | Small (589 templates) | Baseline search, no structured dropout |
| `small_standard.yaml` | Small (589 templates) | Narrower LR range for comparing small vs standard |
| `small_structured.yaml` | Small (589 templates) | Structured dropout ON; `structured_dropout_bottleneck` in search space |
| `small_sd.yaml` | Small (589 templates) | Structured dropout search with wider bottleneck range |
| `standard.yaml` | Standard (1573 templates) | Baseline for full-size dataset |
| `large.yaml` | Large (2957 templates) | Widest search space, longest runs |

---

## Running a hyperparameter tuning experiment

### 1. Choose or create a config

Copy the closest pre-built config and adjust paths, `action_dim`, and the
`optuna:` search space:

```bash
cp results/config/small.yaml results/config/my_experiment.yaml
```

### 2. Launch `rs-hypertune`

```bash
rs-hypertune \
  -c results/config/my_experiment.yaml \
  --study-name my-experiment \
  --n-trials 30 \
  --n-epochs 200 \
  --dataset small
```

Results land in `results/hypertune-my-experiment/`:

```
results/hypertune-my-experiment/
├── study.db               # Optuna SQLite storage
├── run.jsonl              # Structured event log (trial start/end, errors)
├── trial_000/
│   ├── train_progress.jsonl   # Per-epoch metrics
│   ├── model.pth              # Best checkpoint for this trial
│   └── pred_routes_train_progress.json
├── trial_001/
│   └── ...
```

`results/hypertune` is a symlink to the active study directory and acts as a
mutual-exclusion lock — remove it once the study finishes:

```bash
rm results/hypertune
```

### 3. Resume an interrupted study

Re-run the same command.  When `study.db` already exists you will be prompted:

```
Existing study found: results/hypertune → results/hypertune-my-experiment
Continue existing study? [y/N]
```

Press `y` to append new trials to the same Optuna study.

### 4. Pull results from a remote machine

```bash
rs-sync-results --host taco

# Preview first:
rs-sync-results --host taco --dry-run
```

This pulls only `study.db` and `train_progress.jsonl` (no model weights) and
mirrors the remote `results/` tree locally.

### 5. Plot learning curves

```bash
# Top 10 trials by valid_action_accuracy
rs-plot-learning-curves

# Filter to one study, rank by route accuracy, save to file
rs-plot-learning-curves \
  --study my-experiment \
  --metric valid_route_accuracy \
  --top 5 \
  --out curves.png
```

### 6. Inspect the Optuna study

```bash
rs-show-study results/hypertune-my-experiment/study.db
rs-show-all-studies "results/**/study.db"
```

---

## Tips

**Keep `eval_routes_frequency` high during hypertune.** Route evaluation via
beam search is expensive. Setting it to 100 or more means it only runs at the
very end of each trial (or not at all before early stopping). The Optuna
objective is based on `valid_route_accuracy` (a cheap per-epoch metric), so
full route eval is not needed to rank trials.

**Reduce `eval_n_batches` for faster hypertune.** Setting `eval_n_batches: 2`
limits each route-evaluation pass to 2 mini-batches, which is fast but noisy.
Use a higher value (or remove the key to evaluate all batches) for a final
training run.

**`n1+n5` as `valid_set` is the standard benchmark.** The PaRoutes N1 and N5
target sets are the accepted evaluation benchmark for retrosynthesis. Using
`valid_set: n1+n5` holds out exactly these targets, giving results comparable
to the published RetroSynFormer paper. Use `random_split` only for ablation
experiments where you need more training data.

**Use `beam_width: 1` during hypertune.** Greedy search is ~50× faster than
`beam_width: 50` and still ranks trials reliably during search. Reserve the
full beam for the final evaluation run.
