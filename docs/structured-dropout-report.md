# Structured Dropout for RetroSynFormer — Feature Report

*Branch: `feature-structured-dropout` — June 2026*

---

## 1. Motivation

RetroSynFormer's Decision Transformer (DT) processes synthesis trajectories as
sequences of (state, action, reward, RTG) tuples.  The state at each timestep
is a concatenation of Morgan fingerprints for the molecules on the current
frontier, so the same 2048-dimensional state vector must simultaneously encode
structural information relevant to wildly different chemical scaffolds.

Standard dropout applies a single uniform rate to every hidden-state channel,
regardless of what chemical information the channel encodes.  Our hypothesis is
that **a mask conditioned on the target molecule's fingerprint** will:

1. Suppress channels that are unreliable or irrelevant for the current
   molecular scaffold, acting as structure-aware regularisation.
2. Encourage the model to learn *which* embedding dimensions carry predictive
   signal for each molecule class — a soft form of feature selection baked into
   training.
3. Improve generalisation on N5 targets (the harder, more complex benchmark
   set) where the model most often fails to find a route.

---

## 2. Architecture

### 2.1 `MoleculeConditionedMaskGenerator`

A two-layer MLP maps the target molecule's Morgan fingerprint to a
per-channel **drop-probability vector** of the same width as the DT's hidden
state:

```
fp (1024-bit) → Linear(1024, bottleneck) → ReLU
              → Linear(bottleneck, hidden_size) → Sigmoid   → drop_prob (hidden_size,)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `fp_dim` | 1024 | radius-2 Morgan fingerprint |
| `hidden_size` | `n_heads × head_dim` | e.g. 256 for (1 head × 256 dim) |
| `bottleneck` | 32–512 (optuna) | default 128 |
| Output bias init | −2.2 | sigmoid(−2.2) ≈ 0.10 → ~10% initial drop |
| Drop cap | 0.9 | never drops >90% of any channel |

**Training mode** — inverted-dropout Bernoulli mask, scaled by 1/(1−p) to
preserve expected activation magnitude.  
**Eval mode** — deterministic expected-value mask (1 − p), giving stable,
reproducible inference.

### 2.2 Integration point

`StructuredDropoutDecisionTransformer` wraps the standard
`DecisionTransformerModel` and installs a `register_forward_hook` on
`embed_ln` — the LayerNorm applied immediately after all modality embeddings
(returns, states, actions, rewards) are stacked.  The hook multiplies every
token's embedding by the molecule mask before the sequence enters any
transformer layer.

```
embed_ln output  (batch, n_tokens, hidden_size)
       ×
molecule mask    (batch,        1, hidden_size)   ← broadcast over tokens
       ↓
masked embeddings → transformer encoder → action logits
```

The hook is installed and removed with `try/finally` inside each `forward`
call, leaving no persistent state between forward passes and requiring no
changes to `RetroTrainer`, `RoutePredictor`, or any other existing code.

### 2.3 Config keys

```yaml
# results/config/small_sd.yaml
model:
  use_structured_dropout: true          # activates StructuredDropoutDecisionTransformer
  structured_dropout_bottleneck: 128    # MLP hidden dim; also searched by Optuna

optuna:
  structured_dropout_bottleneck:
    low: 32
    high: 512
    log: true
```

Validation guard: running `hypertune.py` raises `ValueError` if
`use_structured_dropout: false` but `structured_dropout_bottleneck` appears in
the Optuna search space — an inconsistency that wastes a search dimension.

---

## 3. Test statistics (random initialisation)

The test suite (`tests/test_structured_dropout.py`, 8 tests) verifies
correctness of the mechanism before any training occurs.  Statistics below are
from `pytest -s` with `torch.manual_seed(42)`, `hidden_size=256`,
`bottleneck=128`, `fp_dim=1024`.

### 3.1 Per-molecule drop-probability summary

| Molecule | mean_drop | std | min | max | MW | ArRings |
|----------|-----------|-----|-----|-----|----|---------|
| methane | 0.0998 | 0.0011 | 0.097 | 0.103 | 16 | 0 |
| octane | 0.0998 | 0.0022 | 0.095 | 0.106 | 114 | 0 |
| benzene | 0.0999 | 0.0012 | 0.097 | 0.103 | 78 | 1 |
| naphthalene | 0.0998 | 0.0019 | 0.095 | 0.105 | 128 | 2 |
| glucose | 0.1000 | 0.0026 | 0.094 | 0.106 | 180 | 0 |
| aspirin | 0.0998 | 0.0032 | 0.091 | 0.109 | 180 | 1 |
| ibuprofen | 0.0999 | 0.0032 | 0.092 | 0.110 | 206 | 1 |
| caffeine | 0.0999 | 0.0035 | 0.091 | 0.111 | 194 | 2 |
| cholesterol | 0.0994 | 0.0049 | 0.088 | 0.118 | 387 | 0 |

All molecules hover near **10% mean drop rate** — the bias initialisation is
working as designed.  Within-molecule spread grows with molecular complexity:
cholesterol (std=0.005) is ~4× more spatially varied than methane (std=0.001),
even at init, because its longer, sparser fingerprint projects differently
through the random weights.

### 3.2 Per-channel statistics across molecules

| Statistic | mean | std | min | max |
|-----------|------|-----|-----|-----|
| channel mean drop-prob | 0.0998 | 0.0019 | 0.0946 | 0.1045 |
| channel std across molecules | 0.0021 | 0.0006 | 0.0009 | 0.0054 |
| channel range across molecules | 0.0071 | 0.0023 | 0.0025 | 0.0185 |
| fraction of channels with range > 0.05 | 0.000 | — | — | — |

At random init, no channel has a range >5% across our test molecules — the
masks are nearly spatially uniform.  **This is expected and correct**: spatial
structure only emerges as the model learns which channels are reliably
informative for different chemical scaffolds.

The top-10 most molecule-sensitive channels (ch 169, 156, 79, …) already show
std ~0.004–0.005 across molecules, seeding the structural differentiation that
training should amplify.

### 3.3 Pairwise cosine similarity

| Metric | Value |
|--------|-------|
| mean | 0.9994 |
| std | 0.0003 |
| min (most dissimilar) | 0.9988 — cholesterol ↔ aspirin |
| max (most similar) | 0.9999 — benzene ↔ methane |

Masks are nearly identical at init (cosine ≈ 1.0).  After training, the
expected pattern is that structurally dissimilar molecules (e.g. cholesterol vs
aspirin) have lower cosine similarity and that the most dissimilar pairs
correspond to the most chemically distinct compound classes in the training set.

### 3.4 Property correlation with mean drop probability (random init)

| Property | Pearson r | Interpretation at init |
|----------|-----------|------------------------|
| MW | −0.65 | random artefact — larger FP → different projection |
| HeavyAtoms | −0.66 | same |
| RingCount | −0.64 | same |
| RotBonds | −0.58 | same |
| ArRings | +0.33 | noise |
| HBA, HBD, TPSA | <0.3 | noise |

The moderate negative correlations with size (MW, HeavyAtoms) at init are a
random artefact of the weight initialisation: larger molecules have denser
fingerprints that project to slightly lower mean sigmoid output through the
random weights.  These correlations are not chemically meaningful yet.  After
training, we expect non-zero correlations to reflect *learned* associations
between molecular features and the reliability of each embedding channel — the
structural signal we are trying to induce.

---

## 4. Paper baseline performance (small dataset)

From Granqvist et al. (*Digital Discovery*, 2026, Table 6):

| Model | Test set | Success rate ↑ | Top-1 acc ↑ | TED ↓ | Avg route len ↓ |
|-------|----------|---------------|------------|-------|-----------------|
| RetroSynFormer50 | N1 | **0.950** | 0.182 | 4.43 | 2.29 |
| RetroSynFormer50 | N5 | **0.833** | 0.101 | 5.43 | 2.47 |
| AiZynthFinder | N1 | 0.923 | 0.223 | 4.07 | 2.34 |
| AiZynthFinder | N5 | 0.917 | 0.125 | 7.12 | 2.92 |

Key context:
- **Beam width 50** for all reported numbers; greedy (beam_width=1) gives
  ~0.30 success rate (Fig. 7a).
- Small dataset: 588 templates, 44 736 training routes, 1732/5362 N1/valid
  targets (Table 1).
- Optimal hyperparameters from Table S1 (not reproduced here); our config
  defaults to `n_heads=4, n_layers=26, head_dim=256` which is in that range.
- RetroSynFormer **outperforms AiZynthFinder on success rate** for N1 on the
  small dataset (0.950 vs 0.923) despite the much simpler template space.
- Top-1 accuracy is low (0.182) — the model finds valid routes to building
  blocks but via different specific templates than the reference patent routes.
  This is expected and the paper argues it is not a failure: shorter, cheaper
  routes can be equally valid.

Our 10-epoch greedy hypertune baseline (trial 0, `n_heads=1, n_layers=3,
head_dim=256, lr=0.211, dropout=0.1`) scored **fraction_solved=0.352**, which
is consistent with Fig. 7a (greedy ≈ 0.30 at full training), showing the model
is learning correctly but needs more epochs and beam search to match reported
numbers.

---

## 5. Recommended Optuna comparison run

### Goal

Compare structured-dropout vs standard-dropout RetroSynFormer on the small
dataset at fixed compute, matching the paper's evaluation protocol as closely
as practical.

### Config files

Two configs are needed:

**`results/config/small.yaml`** (baseline, already in repo):
```yaml
model:
  use_structured_dropout: false
# optuna does NOT include structured_dropout_bottleneck
```

**`results/config/small_sd.yaml`** (structured dropout, see below):
```yaml
model:
  use_structured_dropout: true
  structured_dropout_bottleneck: 128
# optuna includes structured_dropout_bottleneck
```

### Run commands

```bash
# 1 — Baseline (standard dropout)
python scripts/hypertune.py \
  -c results/config/small.yaml \
  --study-name small_baseline \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small

# 2 — Structured dropout
python scripts/hypertune.py \
  -c results/config/small_sd.yaml \
  --study-name small_sd \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small
```

### Why these settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--n-epochs 200` | 200 | Paper default; enough for convergence on small dataset |
| `--n-trials 10` | 10 | 1 fixed baseline + 9 Optuna trials; captures most of the search surface |
| `--dataset small` | small | 588 templates, fastest iteration; paper reports best absolute gain vs AiZynthFinder on N1 |
| `beam_width` (config) | 1 (greedy) during training eval; set to 50 for final predict.py | Match paper's evaluation protocol |

### Evaluation after training

After each study completes, run beam-search prediction on the N1 and N5 test
sets to get the numbers comparable to Table 6:

```bash
# predict on best trial from baseline study
python predict.py -d results/hypertune-small_baseline/trial_000/ -w 50 --n1
python predict.py -d results/hypertune-small_baseline/trial_000/ -w 50 --n5

# predict on best trial from SD study
python predict.py -d results/hypertune-small_sd/trial_XXX/ -w 50 --n1
python predict.py -d results/hypertune-small_sd/trial_XXX/ -w 50 --n5
```

### What to look for

| Outcome | Interpretation |
|---------|---------------|
| SD success rate ≥ baseline | Structured dropout improves or maintains route-finding |
| SD top-1 accuracy ≥ baseline | Masks focus the model on chemically correct templates |
| SD TED ↓ (lower) | Routes are structurally closer to reference patent routes |
| SD N5 > N1 gain | Expected: N5 has harder, more diverse targets where molecule-specific masking helps most |
| SD cosine similarity between masks ↓ after training | Learned spatial structure; masks have diverged across chemical space |
| Bottleneck Optuna optimum | Small bottleneck (32–64) → compact routing, large (256–512) → expressive |

The clearest signal of success would be an **N5 success rate improvement**
(harder targets), since these are the molecules where uniform dropout most
hurts and molecule-conditioned attention to reliable features should help most.

---

## 6. Limitations and next steps

1. **No route-feature conditioning yet.** `MoleculeConditionedMaskGenerator`
   uses only the *target molecule* fingerprint.  A richer signal would include
   features of the current frontier (e.g. partial-route depth, number of open
   branches), allowing the mask to adapt mid-route.

2. **Mask applied at `embed_ln` only.** Applying independent masks per
   transformer layer (conditioned on the target) would be more expressive but
   requires more parameters.

3. **Correlation analysis requires a trained model.** The property-correlation
   statistics in Section 3.4 are meaningless at init.  Rerunning
   `pytest -s tests/test_structured_dropout.py::test_dropout_distribution_statistics`
   after loading a trained checkpoint will show whether the mask generator has
   learned chemically coherent channel specialisation.

4. **Bottleneck size.** A bottleneck of 128 adds `1024×128 + 128×256 = 163 840`
   parameters (≈0.6% of a 26-layer model).  If it proves unhelpful, a bottleneck
   of 32 reduces that to ≈42 K parameters.

---

*Generated 2026-06-14. Paper reference: Granqvist, Mercado & Genheden,
"RetroSynFormer: planning multi-step chemical synthesis routes via a decision
transformer", Digital Discovery, 2026, 5, 348–362.*
