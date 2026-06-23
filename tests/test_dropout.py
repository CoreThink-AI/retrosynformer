"""Tests for layer_shared_resid_dropout — intra- and inter-layer mask tying.

The unified ``apply_shared_resid_dropout`` accepts three spec formats:
  - list[bool|int]: per-layer intra-layer flags (0-indexed, backward-compat)
  - bool:           apply intra-layer tying to all layers
  - dict:           int keys N (1-based) → intra-layer; float keys N.5 → inter-layer

The lower-level helpers ``apply_layer_shared_resid_dropout`` and
``apply_interlayer_tied_dropout`` are also tested directly.

Run with:
    pytest tests/test_dropout.py -v
"""

import pytest
import torch
import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from retrosynformer.dropout import (
    SharedResidMaskDropout,
    _validate_interlayer_ties,
    apply_interlayer_tied_dropout,
    apply_layer_shared_resid_dropout,
    apply_shared_resid_dropout,
)

# Tiny model dimensions — fast to construct, no GPU required.
N_LAYERS = 4
HIDDEN_SIZE = 32
N_HEADS = 2
ACT_DIM = 8
STATE_DIM = 16
P = 0.1


@pytest.fixture(scope="module")
def tiny_model():
    cfg = DecisionTransformerConfig(
        bos_token_id=None,
        eos_token_id=None,
        act_dim=ACT_DIM,
        state_dim=STATE_DIM,
        hidden_size=HIDDEN_SIZE,
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        max_ep_len=20,
        resid_pdrop=P,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        action_tanh=False,
    )
    return DecisionTransformerModel(cfg)


def _fresh_model():
    """Return a freshly constructed tiny model (not shared — for patch tests)."""
    cfg = DecisionTransformerConfig(
        bos_token_id=None,
        eos_token_id=None,
        act_dim=ACT_DIM,
        state_dim=STATE_DIM,
        hidden_size=HIDDEN_SIZE,
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        max_ep_len=20,
        resid_pdrop=P,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        action_tanh=False,
    )
    return DecisionTransformerModel(cfg)


def _dummy_forward(model):
    """Run one forward pass with minimal batch/seq input."""
    B, T = 2, 3
    states = torch.zeros(B, T, STATE_DIM)
    actions = torch.zeros(B, T, ACT_DIM)
    rewards = torch.zeros(B, T, 1)
    rtg = torch.zeros(B, T, 1)
    timesteps = torch.zeros(B, T, dtype=torch.long)
    with torch.no_grad():
        model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=rtg,
            timesteps=timesteps,
        )


# ---------------------------------------------------------------------------
# Structural assertions (no forward pass needed)
# ---------------------------------------------------------------------------

def test_single_boundary_same_instance():
    """blocks[0].mlp.dropout and blocks[1].attn.resid_dropout must share one object."""
    model = _fresh_model()
    n = apply_interlayer_tied_dropout(model, P, {1.5: 1})
    assert n == 1
    blocks = model.encoder.h
    assert blocks[0].mlp.dropout is blocks[1].attn.resid_dropout
    assert isinstance(blocks[0].mlp.dropout, SharedResidMaskDropout)


def test_independent_sites_unchanged():
    """Sites NOT in the tie spec must remain plain nn.Dropout."""
    model = _fresh_model()
    apply_interlayer_tied_dropout(model, P, {1.5: 1})
    blocks = model.encoder.h
    # Unpatch sites
    assert isinstance(blocks[0].attn.resid_dropout, nn.Dropout)
    assert isinstance(blocks[1].mlp.dropout, nn.Dropout)
    assert isinstance(blocks[2].attn.resid_dropout, nn.Dropout)
    assert isinstance(blocks[2].mlp.dropout, nn.Dropout)


def test_two_groups_different_instances():
    """{1.5: 1, 2.5: 2} must produce two distinct SharedResidMaskDropout instances."""
    model = _fresh_model()
    n = apply_interlayer_tied_dropout(model, P, {1.5: 1, 2.5: 2})
    assert n == 2
    blocks = model.encoder.h
    shared_g1 = blocks[0].mlp.dropout
    shared_g2 = blocks[1].mlp.dropout
    assert shared_g1 is not shared_g2
    assert shared_g1 is blocks[1].attn.resid_dropout
    assert shared_g2 is blocks[2].attn.resid_dropout


def test_same_group_multiple_boundaries():
    """{1.5: 1, 2.5: 1}: all four sites share one instance."""
    model = _fresh_model()
    n = apply_interlayer_tied_dropout(model, P, {1.5: 1, 2.5: 1})
    assert n == 2
    blocks = model.encoder.h
    shared = blocks[0].mlp.dropout
    assert isinstance(shared, SharedResidMaskDropout)
    assert blocks[1].attn.resid_dropout is shared
    assert blocks[1].mlp.dropout is shared
    assert blocks[2].attn.resid_dropout is shared


def test_zero_p_returns_zero():
    """With p=0.0 no sites should be patched."""
    model = _fresh_model()
    n = apply_interlayer_tied_dropout(model, 0.0, {1.5: 1})
    assert n == 0
    blocks = model.encoder.h
    assert isinstance(blocks[0].mlp.dropout, nn.Dropout)
    assert isinstance(blocks[1].attn.resid_dropout, nn.Dropout)


# ---------------------------------------------------------------------------
# Forward-pass assertions (mask actually fires)
# ---------------------------------------------------------------------------

def test_mask_is_set_during_training_forward():
    """After a training-mode forward, the shared instance's _mask must be non-None."""
    model = _fresh_model()
    apply_interlayer_tied_dropout(model, P, {1.5: 1})
    blocks = model.encoder.h
    shared = blocks[0].mlp.dropout  # same object as blocks[1].attn.resid_dropout

    model.train()
    _dummy_forward(model)
    assert shared._mask is not None, "Pre-hook did not fire — mask was never set"


def test_mask_is_none_in_eval_forward():
    """In eval mode the pre-hook sets mask=None (dropout is off)."""
    model = _fresh_model()
    apply_interlayer_tied_dropout(model, P, {1.5: 1})
    blocks = model.encoder.h
    shared = blocks[0].mlp.dropout

    model.eval()
    _dummy_forward(model)
    assert shared._mask is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validation_rejects_integer_key():
    with pytest.raises(ValueError, match="N.5"):
        _validate_interlayer_ties({1: 1}, n_layers=4)


def test_validation_rejects_non_half_float():
    with pytest.raises(ValueError, match="N.5"):
        _validate_interlayer_ties({1.3: 1}, n_layers=4)


def test_validation_rejects_non_integer_group_id():
    with pytest.raises(ValueError, match="group IDs"):
        _validate_interlayer_ties({1.5: "a"}, n_layers=4)


def test_validation_rejects_left_block_out_of_range():
    """Boundary 0.5 implies left block index -1 — must raise."""
    with pytest.raises(ValueError, match="out of range"):
        _validate_interlayer_ties({0.5: 1}, n_layers=4)


def test_validation_rejects_right_block_out_of_range():
    """Boundary 4.5 implies right block index 4 for n_layers=4 — must raise."""
    with pytest.raises(ValueError, match="out of range"):
        _validate_interlayer_ties({4.5: 1}, n_layers=4)


def test_validation_accepts_valid_ties():
    """Valid boundaries 1.5 and 3.5 for a 4-layer model must not raise."""
    _validate_interlayer_ties({1.5: 1, 3.5: 2}, n_layers=4)


# ---------------------------------------------------------------------------
# apply_shared_resid_dropout — unified API
# ---------------------------------------------------------------------------

class TestUnifiedAPI:
    """apply_shared_resid_dropout routes correctly and preserves old behaviour."""

    # -- list / bool (backward-compat) --------------------------------------

    def test_list_spec_returns_intra_only(self):
        """List spec → n_intra > 0, n_inter == 0."""
        model = _fresh_model()
        flags = [True, False, True, False]
        n_intra, n_inter = apply_shared_resid_dropout(model, P, flags)
        assert n_intra == 2
        assert n_inter == 0

    def test_list_spec_patches_correct_blocks(self):
        """List spec patches the enabled blocks intra-layer, leaves others alone."""
        model = _fresh_model()
        apply_shared_resid_dropout(model, P, [True, False, False, False])
        blocks = model.encoder.h
        assert isinstance(blocks[0].attn.resid_dropout, SharedResidMaskDropout)
        assert blocks[0].attn.resid_dropout is blocks[0].mlp.dropout
        assert isinstance(blocks[1].attn.resid_dropout, nn.Dropout)

    def test_bool_true_spec_patches_all_blocks(self):
        """bool True → all layers get intra-layer tying."""
        model = _fresh_model()
        n_intra, n_inter = apply_shared_resid_dropout(model, P, True)
        assert n_intra == N_LAYERS
        assert n_inter == 0
        for block in model.encoder.h:
            assert isinstance(block.attn.resid_dropout, SharedResidMaskDropout)
            assert block.attn.resid_dropout is block.mlp.dropout

    def test_bool_false_spec_patches_nothing(self):
        """bool False → no blocks patched."""
        model = _fresh_model()
        n_intra, n_inter = apply_shared_resid_dropout(model, P, False)
        assert n_intra == 0
        assert n_inter == 0

    # -- dict, intra-only ---------------------------------------------------

    def test_dict_int_keys_intra_only(self):
        """Dict with only int keys → intra-layer only."""
        model = _fresh_model()
        # 1-based keys: layers 1 and 3
        n_intra, n_inter = apply_shared_resid_dropout(model, P, {1: 1, 3: 1})
        assert n_intra == 2
        assert n_inter == 0
        blocks = model.encoder.h
        assert isinstance(blocks[0].attn.resid_dropout, SharedResidMaskDropout)
        assert blocks[0].attn.resid_dropout is blocks[0].mlp.dropout
        assert isinstance(blocks[1].attn.resid_dropout, nn.Dropout)  # layer 2 untouched
        assert isinstance(blocks[2].attn.resid_dropout, SharedResidMaskDropout)
        assert blocks[2].attn.resid_dropout is blocks[2].mlp.dropout

    def test_dict_int_key_false_value_skips_layer(self):
        """Int key with falsy value (0) must not patch that layer."""
        model = _fresh_model()
        apply_shared_resid_dropout(model, P, {1: 1, 2: 0})
        blocks = model.encoder.h
        assert isinstance(blocks[0].attn.resid_dropout, SharedResidMaskDropout)
        assert isinstance(blocks[1].attn.resid_dropout, nn.Dropout)

    # -- dict, inter-only ---------------------------------------------------

    def test_dict_float_keys_inter_only(self):
        """Dict with only float keys → inter-layer only."""
        model = _fresh_model()
        n_intra, n_inter = apply_shared_resid_dropout(model, P, {1.5: 1})
        assert n_intra == 0
        assert n_inter == 1
        blocks = model.encoder.h
        assert blocks[0].mlp.dropout is blocks[1].attn.resid_dropout
        assert isinstance(blocks[0].mlp.dropout, SharedResidMaskDropout)
        # Intra-layer sites of block 0 must remain independent
        assert isinstance(blocks[0].attn.resid_dropout, nn.Dropout)

    # -- dict, combined -----------------------------------------------------

    def test_dict_combined_intra_and_inter(self):
        """Dict with int and float keys patches both intra- and inter-layer sites."""
        model = _fresh_model()
        # Layer 1 intra-layer + boundary 1.5 inter-layer
        n_intra, n_inter = apply_shared_resid_dropout(model, P, {1: 1, 1.5: 1})
        assert n_intra == 1
        assert n_inter == 1
        blocks = model.encoder.h
        # Intra-layer: block 0 attn and mlp share a mask
        assert isinstance(blocks[0].attn.resid_dropout, SharedResidMaskDropout)
        # Note: block 0 mlp.dropout is overwritten by inter-layer patching,
        # so it is now the inter-layer shared instance (different from the
        # intra-layer instance on block 0 attn).
        assert isinstance(blocks[0].mlp.dropout, SharedResidMaskDropout)
        # Inter-layer: block 0 mlp tied to block 1 attn
        assert blocks[0].mlp.dropout is blocks[1].attn.resid_dropout

    def test_dict_combined_non_overlapping(self):
        """Non-overlapping intra + inter: all sites correctly assigned."""
        model = _fresh_model()
        # Layer 2 intra-layer + boundary 3.5 inter-layer (no overlap)
        apply_shared_resid_dropout(model, P, {2: 1, 3.5: 1})
        blocks = model.encoder.h
        assert isinstance(blocks[1].attn.resid_dropout, SharedResidMaskDropout)
        assert blocks[1].attn.resid_dropout is blocks[1].mlp.dropout  # intra, layer 2
        assert blocks[2].mlp.dropout is blocks[3].attn.resid_dropout  # inter, boundary 3.5

    # -- zero p -------------------------------------------------------------

    def test_zero_p_dict_patches_nothing(self):
        """p=0.0 must leave all sites untouched regardless of spec."""
        model = _fresh_model()
        n_intra, n_inter = apply_shared_resid_dropout(model, 0.0, {1: 1, 1.5: 1})
        assert n_intra == 0
        assert n_inter == 0
