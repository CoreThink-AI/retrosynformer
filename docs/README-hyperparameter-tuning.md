# Hyperparameter Tuning Tutorial

This tutorial walks through the full RetroSynFormer training and tuning workflow, from a fast sanity check on a tiny model to reproducing the paper's results on the large dataset.

**Paper:** Granqvist et al., *RetroSynFormer: planning multi-step chemical synthesis routes via a decision transformer*, *Digital Discovery* 2026, 5, 348–362. DOI: [10.1039/d5dd00153f](https://doi.org/10.1039/d5dd00153f)  
**Full paper (markdown):** [d5dd00153f-RetroSynFormer.md](d5dd00153f-RetroSynFormer.md)

---

## Prerequisites

```bash
# Install into a virtual environment
uv sync                       # CPU
uv sync --extra rocm          # AMD GPU (ROCm)
uv sync --extra cpu           # explicit CPU PyTorch

source .venv/bin/activate
```

Download the PaRoutes dataset from <https://zenodo.org/records/17177425> and place the files in `data/`:

```
data/
  standard_routes.json
  standard_building_blocks.csv
  standard_reaction_templates.pickle
```

---

## Concepts

### Decision Transformer recap

RetroSynFormer frames multi-step retrosynthesis as a sequence-modeling problem.  At each step the model predicts the next reaction template (action) given the history of molecules (states) and rewards seen so far.  This is the GPT-2-based Decision Transformer (DT) architecture.

Key dimensions set in `config.yaml`:

| Config key | What it controls |
|------------|-----------------|
| `dataset.action_dim` | Size of the one-hot reaction-template vocabulary (588/1572/2986) |
| `dataset.fp_dim` | Morgan fingerprint bits per molecule (1024) |
| `dataset.n_in_state` | Molecules concatenated per state vector (2) → `state_dim = fp_dim × n_in_state = 2048` |
| `model.n_heads × model.head_dim` | GPT-2 hidden size |
| `model.n_layers` | Transformer depth |
| `evaluation.beam_width` | 1 = greedy; 50 = paper default ("RetroSynFormer50") |

### Three dataset sizes

The paper created three nested datasets from PaRoutes by keeping only the most common reaction templates (Section 3.1, [Table 1](d5dd00153f-RetroSynFormer-table1-datasets.csv)):

| Dataset | Templates | Unique targets | Training routes |
|---------|-----------|----------------|-----------------|
| Small   | 588       | 53 626         | 44 736          |
| Standard | 1 572    | 82 222         | 67 180          |
| Large   | 2 986     | 106 452        | 86 048          |

Small ⊂ Standard ⊂ Large.  Start with **Small** for fast iteration, finish with **Standard** or **Large** for publication-quality results.

---

## Stage 1 — Sanity check: one training epoch

Confirm the pipeline runs end to end before committing to a full run.

```bash
# Edit the epoch count to 1 temporarily, or use the provided config.yaml directly:
python -m retrosynformer.runner -c results/config.yaml
```

`results/config.yaml` is preconfigured with `n_epochs: 1` and `action_dim: 589` (small dataset).  You should see one line of tab-delimited output and a model checkpoint at `results/model.pth`.  Typical wall time: <60 s on CPU.

---

## Stage 2 — Train a baseline small-dataset model

The paper's small-dataset model achieves **95.0% success rate** on the N1 benchmark (Table 6).

```bash
rs-train -c results/config/baseline_small.yaml
```

Or equivalently:

```bash
python -m retrosynformer.runner -c results/config/baseline_small.yaml
```

**What this does:**
- Uses the **small** dataset (`action_dim: 588`, 588 reaction templates)
- Architecture from the paper's Optuna study: `n_heads=4`, `n_layers=26`, `head_dim=256`
- Default reward function: dead-end penalty −2, intermediate penalty −2, building-block reward 0 (all depth-scaled)
- Trains for up to 200 epochs with early stopping (`patience=50`)
- Saves checkpoints and learning curves to `results/`

**Expected training output** (one tab-delimited row per epoch):

```
epoch   t_loss   t_acc    t_racc   v_loss   v_acc    v_racc   s/ep  note
0       3.94521  0.08012  0.00000  3.89234  0.09143  0.00000  12.3  *
1       3.71089  0.11423  0.00412  3.65917  0.12891  0.00521  12.1  *
...
```

`*` in the `note` column means a new best valid loss was saved.

**Evaluate after training:**

```bash
python predict.py -d results/ -w 50              # N1 benchmark
python predict.py -d results/ -w 50 --n1 --n5   # both benchmarks
python src/retrosynformer/utils/evaluation.py    # compute metrics
```

Aim for `success_rate ≈ 0.95` on N1 and `≈ 0.83` on N5 (paper Table 6).

---

## Stage 3 — Hyperparameter search on the small dataset

The paper used Optuna to tune architecture and optimizer hyperparameters (Section 3.3).  The provided config reproduces this search on the fastest (small) dataset.

```bash
rs-hypertune \
    -c results/config/baseline_small_hyperparameter_tuning.yaml \
    --study-name small-hypertune \
    --n-trials 30 \
    --n-epochs 80
```

**What this searches** (see config `optuna:` section):

| Parameter | Type | Range |
|-----------|------|-------|
| `n_heads` | categorical | 1, 2, 4 |
| `n_layers` | categorical | 4, 8, 12, 18, 26 |
| `head_dim` | categorical | 64, 128, 256 |
| `lr` | log-float | 5×10⁻⁴ – 0.5 |

**Beam width during search:** the config uses `beam_width=1` (greedy) to keep each trial fast.  The paper found that greedy and beam-50 route-accuracy rankings are well-correlated, so greedy is fine for comparing trials.

**Results land in** `results/hypertune-small-hypertune/`:
```
study.db           # Optuna SQLite — all trials, parameters, scores
run.jsonl          # structured log — one JSON object per event
trial_000/         # first trial (fixed baseline)
trial_001/
...
```

**Monitor progress:**

```bash
# Plot learning curves for all trials
rs-plot-learning-curves results/hypertune-small-hypertune/study.db

# Show Optuna study summary
rs-show-study results/hypertune-small-hypertune/study.db

# Show all studies
rs-show-all-studies results/hypertune-*/study.db
```

**Run two parallel workers** on the same study (uses SQLite coordination):

```bash
rs-hypertune \
    -c results/config/baseline_small_hyperparameter_tuning.yaml \
    --study-name small-hypertune \
    --storage sqlite:///results/hypertune-small-hypertune/study.db \
    --n-trials 30 --n-epochs 80 &

rs-hypertune \
    -c results/config/baseline_small_hyperparameter_tuning.yaml \
    --study-name small-hypertune \
    --storage sqlite:///results/hypertune-small-hypertune/study.db \
    --n-trials 30 --n-epochs 80 &
```

Each worker picks up where the other leaves off.  The study automatically stops after 30 total trials regardless of how many workers are running.

---

## Stage 4 — Retrain best trial with full epochs and beam width 50

Once the search completes, retrain the best configuration with full training budget and evaluate properly:

```bash
# Find the best trial
rs-show-study results/hypertune-small-hypertune/study.db

# Suppose the best trial was trial_007 with n_heads=4, n_layers=18, head_dim=256, lr=0.15:
rs-train \
    -c results/config/baseline_small.yaml \
    --n-heads 4 --n-layers 18 --head-dim 256 --lr 0.15 \
    --n-epochs 200 \
    --results-path results/small-best/

# Evaluate with beam width 50
python predict.py -d results/small-best/ -w 50 --n1 --n5
python src/retrosynformer/utils/evaluation.py
```

---

## Stage 5 — Scale to standard dataset

Swap `action_dim` and point at the standard dataset.  The templates file is the same; `action_dim` tells the model how many one-hot dimensions the action vector has.

```bash
# Copy the best small-dataset config and adjust:
cp results/config/baseline_small.yaml results/config/baseline_standard.yaml
```

Edit `baseline_standard.yaml`:
```yaml
dataset:
  action_dim: 1572   # standard dataset (was 588)
```

Then train:
```bash
rs-train -c results/config/baseline_standard.yaml
```

**Expected results** (paper Table 6, standard dataset, beam_width=50):

| Test set | Success rate | Top-1 accuracy | TED |
|----------|-------------|----------------|-----|
| N1       | 0.924       | 0.106          | 5.58 |
| N5       | 0.899       | 0.058          | 7.08 |

Run a fresh Optuna search at this scale if you want to find better hyperparameters for the standard dataset specifically:

```bash
rs-hypertune \
    -c results/config/baseline_standard.yaml \
    --study-name standard-hypertune \
    --n-trials 20 --n-epochs 80
```

---

## Stage 6 — Large dataset

The large dataset (`action_dim=2986`) gives marginally higher success rates but lower top-1 accuracy due to the expanded template vocabulary.  It requires more memory and longer training time.

```yaml
# In your config:
dataset:
  action_dim: 2986   # large dataset
```

**Paper results** (Table 6, large dataset, beam_width=50):

| Test set | RSF50 success | AiZyF success |
|----------|--------------|--------------|
| N1       | 0.929        | 0.939        |
| N5       | 0.887        | 0.925        |

The large dataset is where AiZynthFinder's advantage grows most.  The paper notes (Discussion) that the scaling behavior is still an open research question.

---

## Tuning tips from the paper

### Beam width (Section 4.4, Fig. 7)

Success rate grows logarithmically with beam width; search time grows exponentially.

| Beam width | N1 success | Search time / target |
|-----------|-----------|---------------------|
| 1 (greedy) | ~0.30 | <1 s |
| 5 | ~0.78 | ~2 s |
| 10 | ~0.87 | ~5 s |
| 20 | ~0.90 | ~20 s |
| 50 | ~0.92 | ~70 s |

Top-1 accuracy peaks at **beam_width=10** and declines slightly for wider beams — the model finds more routes but they are less likely to exactly match the reference.

### Reward function (Section 4.5, Table 5)

The default reward is robust.  Variations cost at most ~1% success rate.  The paper's default:

```yaml
reward:
  building_block_reward_factor: 0
  building_block_scale_with_depth: 2
  dead_end_reward_factor: -2
  dead_end_scale_with_depth: 2
  intermediate_reward_factor: -2
  intermediate_scale_with_depth: 1
```

Only tune the reward if you have a specific reason (e.g., optimizing for shorter routes rather than success rate).

### Early stopping vs. learning rate scheduler

The default `ReduceLROnPlateau` scheduler halves the learning rate after `lr_scheduler_patience` epochs with no `valid_loss` improvement.  Set `lr_scheduler_patience < early_stopping_patience` to ensure the model gets at least one learning-rate reduction before stopping:

```yaml
train:
  early_stopping_patience: 50   # outer budget
  lr_scheduler_patience: 20     # inner: halve lr after 20 flat epochs
```

### Architecture (from paper's Optuna study)

The winning architecture across datasets was consistently:
- `n_heads: 4`, `n_layers: 26`, `head_dim: 256`
- Learning rate: ~0.211 (SGD + momentum 0.98)

Deeper models (`n_layers ≥ 20`) outperform shallow ones on this task.

---

## Reference configs

| Config file | Dataset | Use case |
|-------------|---------|----------|
| `results/config/baseline_small.yaml` | Small (588 templates) | Reproduce paper Table 6 small-dataset row |
| `results/config/baseline_small_hyperparameter_tuning.yaml` | Small | Optuna hyperparameter search |
| `results/config/small_nodropout_baseline.yaml` | Small | Nodropout baseline for structured-dropout comparison |
| `results/config/small_nodropout_sd.yaml` | Small | Structured dropout ablation |
| `results/config/standard.yaml` | Standard (1572 templates) | Full standard dataset training |
| `results/config/large.yaml` | Large (2986 templates) | Full large dataset training |
| `results/config/example.yaml` | — | Reference: all parameters documented |

---

## Key file locations after training

```
results/
  model.pth                        # best checkpoint (lowest valid_loss)
  train_progress.jsonl             # epoch-by-epoch metrics (JSON lines)
  train_progress_loss.png          # loss curve plot
  train_progress_accuracy.png      # accuracy curve plot
  pred_routes_train_progress.json  # route evaluation results (if run)
  evaluation_target_solved.png     # fraction-solved over epochs

results/hypertune-{study_name}/
  study.db                         # Optuna SQLite database
  run.jsonl                        # structured trial log
  trial_NNN/                       # per-trial results (same layout as above)
```
