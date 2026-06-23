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
    logger.info("Patched %d blocks with layer-shared resid dropout", n)

Config flag
-----------
Set ``model.layer_shared_resid_dropout: true`` in model.config.yaml.
Optuna search: ``layer_shared_resid_dropout: [true, false]``.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


def _validate_interlayer_ties(ties: dict, n_layers: int) -> None:
    """Validate the float-key (inter-layer) portion of a layer_shared_resid_dropout dict.

    Rules:
    - All keys must be floats of the form N.5 (exactly, e.g. 1.5, 2.5).
    - All values must be integers (group IDs).
    - Both the left block (int(K)-1) and the right block (int(K)) must be
      valid 0-based indices into the transformer stack.
    """
    bad_keys = [k for k in ties if not (isinstance(k, float) and k - int(k) == 0.5)]
    if bad_keys:
        raise ValueError(
            f"layer_shared_resid_dropout float keys must be of the form N.5 "
            f"(e.g. 1.5, 2.5). Invalid keys: {bad_keys}"
        )
    bad_vals = [v for v in ties.values() if not isinstance(v, int)]
    if bad_vals:
        raise ValueError(
            f"layer_shared_resid_dropout inter-layer values must be integers (group IDs). "
            f"Invalid values: {bad_vals}"
        )
    for k in ties:
        left_idx = int(k) - 1
        right_idx = int(k)
        if left_idx < 0:
            raise ValueError(
                f"layer_shared_resid_dropout boundary {k} implies left block index "
                f"{left_idx} which is out of range (n_layers={n_layers})."
            )
        if right_idx >= n_layers:
            raise ValueError(
                f"layer_shared_resid_dropout boundary {k} implies right block index "
                f"{right_idx} which is out of range (n_layers={n_layers})."
            )


def apply_interlayer_tied_dropout(
    model: nn.Module,
    p: float,
    ties: dict,
) -> int:
    """Tie residual dropout masks across adjacent GPT-2 blocks.

    For boundary ``K`` (e.g. ``1.5``), the MLP output dropout of the
    left block and the attention residual dropout of the right block share
    one ``SharedResidMaskDropout`` instance (and thus one Bernoulli mask):

        blocks[int(K)-1].mlp.dropout          — 1-based layer N output
        blocks[int(K)].attn.resid_dropout     — 1-based layer N+1 input

    Boundaries that share the same group ID (the dict value) are assigned
    the *same* ``SharedResidMaskDropout`` instance.  A single forward
    pre-hook on the earliest left-block in each group generates the mask
    once before that block's forward pass, so the mask is ready for all
    patched dropout sites in the group.

    Args:
        model: ``DecisionTransformerModel`` (or wrapper) instance.
        p:     Dropout probability.
        ties:  ``{boundary: group_id}`` mapping, e.g. ``{1.5: 1}`` or
               ``{1.5: 1, 2.5: 1, 3.5: 2}``.

    Returns:
        Number of boundaries patched.
    """
    if p <= 0.0:
        return 0

    blocks = model.encoder.h
    n_layers = len(blocks)
    _validate_interlayer_ties(ties, n_layers)

    from collections import defaultdict
    groups: dict[int, list[float]] = defaultdict(list)
    for boundary, group_id in ties.items():
        groups[group_id].append(boundary)

    n_patched = 0
    for group_id, boundaries in groups.items():
        shared = SharedResidMaskDropout(p)
        sorted_boundaries = sorted(boundaries)
        earliest_left = int(sorted_boundaries[0]) - 1
        for k in sorted_boundaries:
            left_idx = int(k) - 1
            right_idx = int(k)
            blocks[left_idx].mlp.dropout = shared
            blocks[right_idx].attn.resid_dropout = shared
            n_patched += 1
        blocks[earliest_left].register_forward_pre_hook(_make_pre_hook(shared))

    return n_patched


def apply_shared_resid_dropout(
    model: nn.Module,
    p: float,
    spec,
) -> tuple[int, int]:
    """Unified entry point for the ``layer_shared_resid_dropout`` config key.

    ``spec`` controls which residual dropout sites share masks:

    * **``list[bool | int]``** — per-layer intra-layer flags, 0-indexed
      (backward-compatible with the original list format).  Delegates to
      ``apply_layer_shared_resid_dropout``.
    * **``bool``** — apply intra-layer tying uniformly to all layers.
    * **``dict``** — unified intra + inter-layer specification:

      - Integer key ``N`` (1-based): intra-layer tying for layer N.
        Value must be truthy (``1`` / ``True``) or falsy (``0`` / ``False``).
      - Float key ``N.5``: inter-layer tying at the boundary between 1-based
        layer N (output) and layer N+1 (input).
        Value is an integer group-ID; boundaries with the same group-ID share
        one Bernoulli mask.

    Dict example — tie layers 1 and 2 intra-layer *and* their shared boundary::

        {1: 1, 2: 1, 1.5: 1}

    Returns:
        ``(n_intra, n_inter)`` — number of intra-layer blocks patched and
        inter-layer boundaries patched.
    """
    if isinstance(spec, (bool, list)):
        return apply_layer_shared_resid_dropout(model, p, flags=spec), 0

    if not isinstance(spec, dict):
        raise TypeError(
            f"layer_shared_resid_dropout must be bool, list, or dict "
            f"(got {type(spec).__name__!r})"
        )

    intra = {k: v for k, v in spec.items() if isinstance(k, int)}
    inter = {k: v for k, v in spec.items() if isinstance(k, float)}

    n_intra = 0
    if intra:
        n_layers = len(model.encoder.h)
        flags = [bool(intra.get(i + 1, False)) for i in range(n_layers)]
        n_intra = apply_layer_shared_resid_dropout(model, p, flags=flags)

    n_inter = 0
    if inter:
        n_inter = apply_interlayer_tied_dropout(model, p, inter)

    return n_intra, n_inter
