"""Unit tests for retrosynformer.etl and hypertune._preprocess_optuna_config."""
import pytest

from retrosynformer.etl import mask_dict_to_list
from retrosynformer.scripts import hypertune as ht


# ---------------------------------------------------------------------------
# mask_dict_to_list
# ---------------------------------------------------------------------------

def test_basic_bool_mask():
    assert mask_dict_to_list({0: True, 2: True}, fillna=False) == [True, False, True]


def test_explicit_length_pads():
    assert mask_dict_to_list({0: True, 2: True}, fillna=False, length=5) == [
        True, False, True, False, False
    ]


def test_fillna_type_coercion():
    result = mask_dict_to_list({1: 1, 3: 1}, fillna=0)
    assert result == [0, 1, 0, 1]
    assert all(isinstance(v, int) for v in result)


def test_bool_fillna_coerces_int_values():
    result = mask_dict_to_list({0: 1, 2: 1}, fillna=False)
    assert result == [True, False, True]
    assert all(isinstance(v, bool) for v in result)


def test_empty_dict_returns_empty():
    assert mask_dict_to_list({}, fillna=False) == []


def test_single_entry():
    assert mask_dict_to_list({3: True}, fillna=False) == [False, False, False, True]


def test_explicit_length_shorter_than_keys_stops_early():
    # length=2 means range(2) = [0, 1] — key 3 is beyond the window
    assert mask_dict_to_list({0: True, 3: True}, fillna=False, length=2) == [True, False]


# ---------------------------------------------------------------------------
# _preprocess_optuna_config — list-of-dicts → list-of-lists
# ---------------------------------------------------------------------------

def test_preprocess_converts_list_of_dicts():
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [
                {0: True, 2: True},
                {1: True, 3: True},
            ]
        }
    }
    out = ht._preprocess_optuna_config(cfg)
    lsrd = out["optuna"]["layer_shared_resid_dropout"]
    assert lsrd == [
        [True, False, True, False],
        [False, True, False, True],
    ]


def test_preprocess_pads_to_common_length():
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [
                {0: True},       # max key = 0 → length would be 1 alone
                {3: True},       # max key = 3 → drives length to 4
            ]
        }
    }
    out = ht._preprocess_optuna_config(cfg)
    lsrd = out["optuna"]["layer_shared_resid_dropout"]
    assert len(lsrd[0]) == len(lsrd[1]) == 4
    assert lsrd[0] == [True, False, False, False]
    assert lsrd[1] == [False, False, False, True]


def test_preprocess_leaves_list_of_lists_unchanged():
    original = [[True, False], [False, True]]
    cfg = {"optuna": {"layer_shared_resid_dropout": original}}
    out = ht._preprocess_optuna_config(cfg)
    assert out["optuna"]["layer_shared_resid_dropout"] == original


def test_preprocess_does_not_mutate_input():
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [{0: True, 2: True}]
        }
    }
    ht._preprocess_optuna_config(cfg)
    assert isinstance(cfg["optuna"]["layer_shared_resid_dropout"][0], dict)


def test_preprocess_no_lsrd_key_unchanged():
    cfg = {"optuna": {"n_heads": [1, 2, 4]}}
    out = ht._preprocess_optuna_config(cfg)
    assert out == cfg


def test_validate_config_accepts_converted_list_of_dicts():
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [
                {0: True, 2: True},
                {1: True, 3: True},
            ]
        }
    }
    ht._validate_config(ht._preprocess_optuna_config(cfg))  # must not raise
