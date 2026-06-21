# CLAUDE.md

Guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RetroSynFormer

RetroSynFormer is a retrosynthesis route planner that frames multi-step chemical synthesis as a sequence-modeling problem. A Decision Transformer (via HuggingFace `transformers`) iteratively selects reaction templates to decompose a target molecule into purchasable building blocks. Routes are scored with a reward function during beam-search (tree search) inference to predict a retrosynthesis reaction pathway that can be used to manufacture the target molecule (SMILES), usually a drug or bio-relevant compound.

Paper: Granqvist et al., *RetroSynFormer*, chemRxiv 2025, DOI: 10.26434/chemrxiv-2025-kd6gb
Paper converted to markdown: /home/hobs/code/corethink/retrosynformer/docs/d5dd00153f-RetroSynFormer.md

## Setup

```bash
uv pip install --editable ".[cpu]"   # CPU-only
uv pip install --editable ".[rocm]"  # AMD ROCm GPU (taco)
uv pip install --editable ".[cuda]"  # NVIDIA CUDA GPU
source .venv/bin/activate
```

**Always use `uv pip install --editable` rather than `uv sync`.** The lockfile is generated on a non-ROCm machine and locks the CPU torch build; `uv sync --extra rocm` therefore installs CPU torch even on taco. `uv pip install` bypasses the lockfile and resolves torch from the correct index for the chosen extra.

Python ≥ 3.10 required. Key pinned deps: `reaction-utils==1.9.3`, `rdchiral` (from git at `../rdchiral`). `torch` is **not** in base deps — must pick one extra: `cpu`, `rocm`, or `cuda`.

`rdchiral` must be cloned locally at `../rdchiral`; `reaction-utils==1.9.3` comes from PyPI.

Download the PaRoutes dataset from <https://zenodo.org/records/17177425> and place files in `data/`. Update all paths in `results/config.yaml` to match.

## Commands

```bash
# Preprocess raw routes into a trainable dataset
python process_routes.py \
  -r data/standard_routes.json \
  -bb data/standard_building_blocks.csv \
  -t data/standard_reaction_templates.pickle \
  -tl data/standard_reaction_templates.pickle \
  -s data/processed_routes.json

# Train (preferred entry point — runner.py wires everything together)
python -m retrosynformer.runner -c results/config.yaml

# Predict routes for validation set
python predict.py -d results/ -w 50

# Predict for PaRoutes N1 benchmark set
python predict.py -d results/ -w 50 --n1

# Predict for N5 benchmark set
python predict.py -d results/ -w 50 --n5

# Evaluate results (produces metrics and plots)
python src/retrosynformer/utils/evaluation.py

# Format code
black src/ *.py
```

## Architecture

The pipeline has two phases: data preprocessing and training/inference.

**Data preprocessing** (`process_routes.py` → `src/retrosynformer/data.py`)  
`RouteDataset` loads PaRoutes JSON routes, maps each reaction step to a reaction template index, and assigns rewards per step. The result is a serialized DataFrame consumed by `RouteDatasetTorch`.

**Training** (`runner.py` is the real entry point; `train.py` at the repo root is an older wrapper with a mismatched `RetroTrainer` call signature — use `runner.py` instead)

1. `runner.read_config` → loads YAML
2. `runner.init_model` → builds `DecisionTransformerModel` with custom `act_dim` (= number of reaction templates) and `state_dim` (= `fp_dim × n_in_state` Morgan fingerprint bits)
3. `runner.init_data` → loads routes, applies reward mapping, splits train/valid/test (either random split or n1+n5 as the held-out test set)
4. `runner.create_dataloaders` → wraps `RouteDatasetTorch` in DataLoaders with custom `collate_fn` for variable-length episodes
5. `RetroTrainer.train` → SGD + ReduceLROnPlateau, saves best model to `results/model.pth`

**State representation**: each molecule in the current synthesis frontier is encoded as a 1024-bit Morgan fingerprint. The state tensor concatenates `n_in_state` fingerprints (default 2).

**Actions**: one-hot vectors over the reaction template vocabulary. Template counts differ by dataset size: small=589, standard=1573, large=2957 — must match `action_dim` in `config.yaml`.

**Inference** (`inference.py`, `RoutePredictor`)  
Beam search (`beam_width` controls how many candidate routes are explored in parallel). At each step, the model scores all templates; the environment (`RetroGymEnvironment`) applies the chosen SMARTS template via `rdchiral`, checks if products are building blocks, and marks branches as solved or dead. Routes are sorted by `trajectory_prob` (default) or another key.

**Environment** (`environment.py`, `RetroGymEnvironment`)  
Stateful simulator for one retrosynthesis search. `set_target_compound` initializes state; `step` applies a template and returns the new frontier, reward, and done flag. `copy()` enables beam search branching.

**Reward functions** (`utils/reward_functions.py`)  
Rewards are configured via `config.yaml` under `reward:`. Each step yields a building-block reward, dead-end penalty, or intermediate penalty, optionally scaled with depth.

**Evaluation** (`utils/evaluation.py`, `utils/evaluation_compare_aizynth.py`)  
Computes fraction of targets solved, valid routes, and TED distance to ground-truth routes. `evaluation_compare_aizynth.py` compares against AiZynthFinder baseline.

## Recent changes

| Version | Summary |
|---------|---------|
| **0.1.7** | `rs-plot-learning-curves` and `rs-sync-results` CLI commands; configurable Optuna objective (`objective_metric`); `eval_routes_at_end` flag; rocm extra pinned to `torch 2.5.1+rocm6.2`; `amdgpu` made alias for `rocm` |
| **0.1.6** | `[cpu]` and `[rocm]` torch extras added; torch removed from base deps; fixed `KeyError: 'epoch'` crash when early stopping fires before first route-eval epoch; tightened hypertune search space (`n_heads/n_layers`) and raised `early_stopping_patience` to 6 |
| **0.1.5** | Scripts installable as CLI commands (`rs-train`, `rs-hypertune`, etc.) via `[project.scripts]`; logic moved to `src/retrosynformer/scripts/`; `scripts/*.py` are now thin shims |
| **0.1.4** | Structured dropout (`MoleculeConditionedMaskGenerator`); Optuna study tooling (`study.py`, `show_study.py`, `show_all_studies.py`); early-stopping patience; `RemoteTrialMonitor`; AMD ROCm `[rocm]`/`[amdgpu]` extras; 700+ lines of tests |

See [`CHANGELOG.md`](../CHANGELOG.md) and [`docs/`](docs/) for full release notes.

## Config keys to know

| Key | Meaning |
|-----|---------|
| `dataset.action_dim` | Must match template count (589/1573/2957) |
| `dataset.valid_set` | `"n1+n5"` uses PaRoutes benchmark targets as test; `"random_split"` splits randomly |
| `context.building_blocks` / `context.templates_path` | Absolute paths to data files |
| `train.results_path` | Where model checkpoints and logs are written |
| `evaluation.beam_width` | 1 = greedy; 50 = full beam search |
| `evaluation.eval_routes_frequency` | How often (in epochs) to run full route evaluation (expensive) |
