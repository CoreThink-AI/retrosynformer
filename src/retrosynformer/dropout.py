"""Layer-shared residual dropout for the Decision Transformer GPT-2 blocks.

Standard GPT-2 uses independent dropout at two residual sites per block:
  1. after the attention output projection  (attn.resid_dropout)
  2. after the MLP output projection        (mlp.dropout)

Both use the same scalar probability p but sample *independent* masks, so
different hidden units may be zeroed at each site within the same layer.

With layer-shared residual dropout, a *single* mask is sampled once at the
start of each block's forward pass and reused for both residual sites.  This
means the same set of hidden units is suppressed across the full layer — in
both the attention and the MLP sub-layers — while different layers still
receive independent masks.

The probability p is unchanged; only the correlation structure differs.

Usage
-----
    from retrosynformer.dropout import apply_layer_shared_resid_dropout
    n = apply_layer_shared_resid_dropout(model, p=config["model"]["resid_pdrop"])
    print(f"Patched {n} blocks with layer-shared resid dropout")

Config flag
-----------
Set ``model.layer_shared_resid_dropout: true`` in model.config.yaml.
Optuna search: ``layer_shared_resid_dropout: [true, false]``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedResidMaskDropout(nn.Module):
    """Dropout that reuses a mask set externally rather than sampling one per call.

    A forward pre-hook on the enclosing transformer block calls ``set_mask``
    once before any sub-layer runs.  Both residual dropout sites in the block
    then call ``forward`` and apply the same mask.

    When ``training=False`` or ``p == 0``, behaves identically to ``nn.Dropout``.
    If ``set_mask`` was never called (e.g. during the very first forward before
    the hook fires), falls back to standard functional dropout.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p
        # Not a buffer — transient per-forward-pass state, generated on the
        # input device and discarded after the block's forward returns.
        self._mask: torch.Tensor | None = None

    def set_mask(self, mask: torch.Tensor | None) -> None:
        self._mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self._mask is None:
            # Fallback: independent mask (shouldn't happen in normal usage)
            return F.dropout(x, p=self.p, training=True)
        return x * self._mask

    def extra_repr(self) -> str:
        return f"p={self.p}, layer_shared=True"


def _make_pre_hook(shared: "SharedResidMaskDropout"):
    """Return a forward pre-hook that generates a fresh mask for the block."""
    def hook(module, args):
        hidden_states = args[0] if isinstance(args, tuple) else args
        if shared.training and shared.p > 0.0:
            # Inverted dropout mask: Bernoulli(1-p) / (1-p) preserves expected value.
            mask = torch.bernoulli(
                torch.full(
                    hidden_states.shape,
                    1.0 - shared.p,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            ).div_(1.0 - shared.p)
        else:
            mask = None
        shared.set_mask(mask)
    return hook


def apply_layer_shared_resid_dropout(
    model: nn.Module,
    p: float,
    flags: "bool | list[bool]" = True,
) -> int:
    """Patch a DecisionTransformerModel so selected blocks share resid masks.

    Replaces ``block.attn.resid_dropout`` and ``block.mlp.dropout`` in each
    enabled GPT-2 block with a single ``SharedResidMaskDropout`` instance,
    then registers a forward pre-hook to generate the shared mask before each
    forward pass.  Layers where ``flags`` is ``False`` are left untouched.

    Different blocks always get independent ``SharedResidMaskDropout``
    instances — masks are independent across layers but tied within each layer.

    Safe to call on a freshly-constructed or loaded model: no parameters are
    added or removed, so ``load_state_dict`` works normally.

    Args:
        model: ``DecisionTransformerModel`` instance.
        p:     Dropout probability (should equal ``config["model"]["resid_pdrop"]``).
        flags: Per-layer enable flags.  May be:
               - A single ``bool``: applied uniformly to all layers.
               - A ``list[bool]`` of length ``n_layers``: enables tying
                 independently for each layer (``True`` = tied, ``False`` = keep
                 independent standard dropout).
               Defaults to ``True`` (tie all layers).

    Returns:
        Number of blocks patched.
    """
    if p <= 0.0:
        return 0

    blocks = model.encoder.h
    n_layers = len(blocks)

    # Normalise flags to a per-layer list
    if isinstance(flags, bool):
        per_layer: list[bool] = [flags] * n_layers
    else:
        if len(flags) != n_layers:
            raise ValueError(
                f"layer_shared_resid_dropout list length ({len(flags)}) "
                f"must match n_layers ({n_layers})"
            )
        per_layer = [bool(f) for f in flags]

    n_patched = 0
    for block, enabled in zip(blocks, per_layer):
        if not enabled:
            continue
        shared = SharedResidMaskDropout(p)
        block.attn.resid_dropout = shared
        block.mlp.dropout = shared
        block.register_forward_pre_hook(_make_pre_hook(shared))
        n_patched += 1

    return n_patched
