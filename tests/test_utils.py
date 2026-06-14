"""Unit tests for retrosynformer.utils.utils."""

import torch
import pytest

from retrosynformer.utils.utils import (
    check_if_building_block,
    convert_batches_to_action_ids,
    convert_batches_to_action_ids_batch,
    flatten,
    flatten_and_crop,
    flatten_list,
    get_device,
    get_index_values,
    get_morgan_fingerprint,
    one_hot_encoder,
)

BENZENE_INCHI = "UHOVQNZJYSORNB-UHFFFAOYSA-N"
ASPIRIN_INCHI = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def test_get_device_returns_valid_string():
    assert get_device() in ("cuda", "mps", "cpu")


def test_get_device_is_consistent():
    assert get_device() == get_device()


# ---------------------------------------------------------------------------
# flatten / flatten_list
# ---------------------------------------------------------------------------

def test_flatten_one_level():
    assert flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_flatten_preserves_order():
    assert flatten([["a", "b"], ["c"]]) == ["a", "b", "c"]


def test_flatten_empty_sublists():
    assert flatten([[], [1], []]) == [1]


def test_flatten_list_deep():
    assert flatten_list([1, [2, [3, [4]]]]) == [1, 2, 3, 4]


def test_flatten_list_already_flat():
    assert flatten_list([1, 2, 3]) == [1, 2, 3]


def test_flatten_list_empty():
    assert flatten_list([]) == []


# ---------------------------------------------------------------------------
# one_hot_encoder
# ---------------------------------------------------------------------------

def test_one_hot_encoder_shape():
    assert one_hot_encoder(3, 7).shape == torch.Size([7])


def test_one_hot_encoder_sum_is_one():
    assert one_hot_encoder(0, 5).sum().item() == 1.0


def test_one_hot_encoder_correct_position():
    v = one_hot_encoder(4, 6)
    assert v[4].item() == 1.0
    assert v[:4].sum().item() == 0.0
    assert v[5].item() == 0.0


def test_one_hot_encoder_last_position():
    v = one_hot_encoder(4, 5)
    assert v[4].item() == 1.0


# ---------------------------------------------------------------------------
# get_morgan_fingerprint
# ---------------------------------------------------------------------------

def test_morgan_fp_length():
    fp = get_morgan_fingerprint("C", radius=2, n_bits=512)
    assert fp.shape == torch.Size([512])


def test_morgan_fp_is_binary():
    fp = get_morgan_fingerprint("c1ccccc1", radius=2)
    assert set(fp.unique().tolist()).issubset({0, 1})


def test_morgan_fp_distinct_molecules():
    fp_benzene = get_morgan_fingerprint("c1ccccc1", radius=2)
    fp_methane = get_morgan_fingerprint("C", radius=2)
    assert not torch.equal(fp_benzene, fp_methane)


def test_morgan_fp_same_molecule_reproducible():
    fp1 = get_morgan_fingerprint("c1ccccc1", radius=2)
    fp2 = get_morgan_fingerprint("c1ccccc1", radius=2)
    assert torch.equal(fp1, fp2)


# ---------------------------------------------------------------------------
# check_if_building_block
# ---------------------------------------------------------------------------

def test_building_block_hit():
    assert check_if_building_block("c1ccccc1", {BENZENE_INCHI}) is True


def test_building_block_miss():
    assert check_if_building_block("CC(=O)Oc1ccccc1C(=O)O", {BENZENE_INCHI}) is False


def test_building_block_empty_set():
    assert check_if_building_block("c1ccccc1", set()) is False


def test_building_block_multiple_entries():
    bb = {BENZENE_INCHI, ASPIRIN_INCHI}
    assert check_if_building_block("c1ccccc1", bb) is True
    assert check_if_building_block("CC(=O)Oc1ccccc1C(=O)O", bb) is True
    assert check_if_building_block("C", bb) is False


# ---------------------------------------------------------------------------
# flatten_and_crop
# ---------------------------------------------------------------------------

def test_flatten_and_crop_removes_zeros():
    a, b = flatten_and_crop([[1, 0, 2]], [[10, 20, 30]])
    assert a == [1, 2]
    assert b == [10, 20]  # cropped to len(a)=2


def test_flatten_and_crop_crops_to_shorter():
    a, b = flatten_and_crop([[1, 2, 3]], [[10, 20]])
    assert a == [1, 2]
    assert b == [10, 20]


def test_flatten_and_crop_multiple_routes():
    a, b = flatten_and_crop([[1, 0, 2], [3, 4]], [[10, 20, 30], [40, 50, 60]])
    assert a == [1, 2, 3, 4]
    assert b == [10, 20, 40, 50]


def test_flatten_and_crop_skips_non_list_items():
    # non-list items at the top level are silently skipped
    a, b = flatten_and_crop([42, [1, 2]], [99, [10, 20]])
    assert a == [1, 2]
    assert b == [10, 20]


def test_flatten_and_crop_empty():
    assert flatten_and_crop([], []) == ([], [])


# ---------------------------------------------------------------------------
# get_index_values
# ---------------------------------------------------------------------------

def test_get_index_values_index_zero():
    a, b = get_index_values([[[1, 2], [3, 4]]], [[[10, 20], [30, 40]]], idx=0)
    assert a == [1, 2]
    assert b == [10, 20]


def test_get_index_values_index_one():
    a, b = get_index_values([[[1, 2], [3, 4]]], [[[10, 20], [30, 40]]], idx=1)
    assert a == [3, 4]
    assert b == [30, 40]


def test_get_index_values_skips_short_routes():
    # first route has 2 steps, second has 1 — idx=1 skips the second
    a, b = get_index_values(
        [[[1, 2], [3, 4]], [[5, 6]]],
        [[[10, 20], [30, 40]], [[50, 60]]],
        idx=1,
    )
    assert a == [3, 4]
    assert b == [30, 40]


def test_get_index_values_beyond_all_routes():
    a, b = get_index_values([[[1]]], [[[10]]], idx=5)
    assert a == []
    assert b == []


def test_get_index_values_multiple_routes():
    a, b = get_index_values(
        [[[1], [2]], [[3], [4]]],
        [[[10], [20]], [[30], [40]]],
        idx=0,
    )
    assert a == [1, 3]
    assert b == [10, 30]


# ---------------------------------------------------------------------------
# convert_batches_to_action_ids
# ---------------------------------------------------------------------------

def _onehot(idx, dim):
    t = torch.zeros(dim)
    t[idx] = 1.0
    return t


def test_convert_action_ids_basic():
    actions = [[_onehot(2, 5)]]
    preds   = [[torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1])]]
    ids, pred_ids, probs = convert_batches_to_action_ids(actions, preds)
    assert ids == [2]
    assert pred_ids == [2]
    assert len(probs[0]) == 5


def test_convert_action_ids_skips_padding():
    actions = [[_onehot(1, 3), torch.zeros(3)]]
    preds   = [[torch.tensor([0.1, 0.8, 0.1]), torch.zeros(3)]]
    ids, pred_ids, _ = convert_batches_to_action_ids(actions, preds)
    assert ids == [1]      # only the non-padded step
    assert pred_ids == [1]


def test_convert_action_ids_multiple_routes():
    actions = [[_onehot(0, 3)], [_onehot(2, 3)]]
    preds   = [[torch.tensor([0.9, 0.05, 0.05])], [torch.tensor([0.1, 0.1, 0.8])]]
    ids, pred_ids, _ = convert_batches_to_action_ids(actions, preds)
    assert ids == [0, 2]
    assert pred_ids == [0, 2]


def test_convert_action_ids_pred_differs_from_true():
    actions = [[_onehot(1, 3)]]
    preds   = [[torch.tensor([0.8, 0.1, 0.1])]]
    ids, pred_ids, _ = convert_batches_to_action_ids(actions, preds)
    assert ids == [1]
    assert pred_ids == [0]


# ---------------------------------------------------------------------------
# convert_batches_to_action_ids_batch
# ---------------------------------------------------------------------------

def test_convert_batch_groups_by_route():
    actions = [[_onehot(0, 3), _onehot(2, 3)], [_onehot(1, 3)]]
    preds   = [[torch.tensor([0.9, 0.05, 0.05]), torch.tensor([0.1, 0.1, 0.8])],
               [torch.tensor([0.1, 0.8, 0.1])]]
    ids, pred_ids = convert_batches_to_action_ids_batch(actions, preds)
    assert ids == [[0, 2], [1]]
    assert pred_ids == [[0, 2], [1]]


def test_convert_batch_skips_padding():
    actions = [[_onehot(1, 3), torch.zeros(3)]]
    preds   = [[torch.tensor([0.1, 0.8, 0.1]), torch.zeros(3)]]
    ids, pred_ids = convert_batches_to_action_ids_batch(actions, preds)
    assert ids == [[1]]
    assert pred_ids == [[1]]
