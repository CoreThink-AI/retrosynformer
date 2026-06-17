# RetroSynFormer 0.1.9 Release Notes

*Branch: `feature-nonuniform-dropout` — June 2026*

---

## Summary

Layer-shared residual dropout for the Decision Transformer, plus atomic and
complete model checkpointing. The dropout change is a new architectural
option — controlled per-layer — that ties the attention and MLP residual masks
within each transformer block so the same hidden units are suppressed at both
residual sites. Checkpointing now writes `model.last.pth` after every epoch
and uses atomic temp-file + rename saves throughout, eliminating any risk of
a corrupt checkpoint from a concurrent rsync.

---

## Changes

### 1. Layer-shared residual dropout (`src/retrosynformer/dropout.py`)

New module implementing `SharedResidMaskDropout` and
`apply_layer_shared_resid_dropout`.

Standard GPT-2 samples **two independent** Bernoulli masks per block per
forward pass — one after the attention output projection
(`attn.resid_dropout`) and one after the MLP output projection
(`mlp.dropout`).  With layer-shared residual dropout a **single** mask is
sampled once at the start of each block's forward pass and reused at both
residual sites, so the same set of hidden units is suppressed across the
full layer.  Different layers still receive independent masks; only the
within-layer correlation structure changes.  The probability `p` is
unchanged.

Implementation uses `register_forward_pre_hook` on each block — no
HuggingFace subclassing required, and `load_state_dict` continues to work
without modification because no parameters are added or removed.

**Config key**: `model.layer_shared_resid_dropout`

- `true` / `false` — apply uniformly to all layers
- `[true, false, true, …]` — per-layer list (length ≥ `n_layers`; extra
  entries are truncated at runtime)
- `0` and `1` are accepted as aliases for `false`/`true` throughout
  (YAML parses integer literals without quoting)

### 2. Optuna search over dropout patterns

`hypertune._suggest()` now handles **list-of-lists** specs: each inner
list is one complete `layer_shared_resid_dropout` choice.  Optuna requires
hashable categorical values, so inner lists are JSON-serialised to strings
for storage and deserialised back to `list[bool]` before being passed to
`runner.main()`.

Example config:

```yaml
optuna:
  layer_shared_resid_dropout:
    - [true,  true,  true,  …]   # all-tied
    - [false, false, false, …]   # none-tied (baseline)
    - [true,  false, false, …]   # first layer only
    - [false, false, false, true] # last layer only
```

### 3. Pre-flight validation (`runner._validate_layer_shared_resid_dropout`, `hypertune._validate_config`)

Three new checks run before any trial starts:

- **Non-jagged**: all inner lists in an Optuna list-of-lists must have the
  same length.
- **Length ≥ max n\_layers**: each list must be at least as long as the
  largest `n_layers` value in the search space so truncation always yields
  a valid per-layer flag vector.
- **Valid values**: every element must be `True`, `False`, `0`, or `1`.

### 4. `results/config/small_nonuniform_dropout.yaml`

New study config that fixes the best architecture found in
`standard-v2-dropout-details` trial 001
(`n_heads=5`, `n_layers=10`, `head_dim=128`, `lr=9.685e-4`) and searches
only the four dropout-pattern choices above on the small dataset for 50
epochs.

### 5. Atomic model checkpointing (`src/retrosynformer/trainer.py`)

Previously `torch.save(state_dict, path)` wrote directly to `model.pth`,
leaving a window where a concurrent rsync could read a partially-written
or zero-length file.

The new scheme:

1. **Every epoch**: serialize state\_dict to a temp file in `save_folder/`,
   then `os.replace(tmp, model.last.pth)` — atomic on POSIX.
2. **On new best**: `shutil.copyfile(model.last.pth, tmp2)` then
   `os.replace(tmp2, model.pth)` — also atomic, and avoids a second
   `torch.save` call.

`model.last.pth` therefore always reflects the weights from the most
recently completed epoch, making it safe to rsync at any time during
training.  `model.pth` always contains a complete best-loss checkpoint.
