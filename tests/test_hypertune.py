"""Unit tests for retrosynformer.scripts.hypertune helper functions."""

from unittest.mock import MagicMock

import pytest

from retrosynformer.scripts import hypertune as ht


# ---------------------------------------------------------------------------
# _suggest — search-space dispatcher
# ---------------------------------------------------------------------------

@pytest.fixture
def trial():
    """Mock Optuna trial whose suggest_* methods record their calls."""
    t = MagicMock()
    t.suggest_categorical.side_effect = lambda name, choices: choices[0]
    t.suggest_int.side_effect = lambda name, low, high, **kw: low
    t.suggest_float.side_effect = lambda name, low, high, **kw: low
    return t


def test_suggest_list_calls_categorical(trial):
    result = ht._suggest(trial, "n_heads", [1, 2, 4])
    trial.suggest_categorical.assert_called_once_with("n_heads", [1, 2, 4])
    assert result == 1


def test_suggest_choices_dict_calls_categorical(trial):
    result = ht._suggest(trial, "head_dim", {"choices": [64, 128, 256]})
    trial.suggest_categorical.assert_called_once_with("head_dim", [64, 128, 256])
    assert result == 64


def test_suggest_int_range(trial):
    ht._suggest(trial, "n_layers", {"low": 2, "high": 32})
    trial.suggest_int.assert_called_once_with("n_layers", 2, 32, step=1, log=False)
    trial.suggest_float.assert_not_called()


def test_suggest_float_range(trial):
    ht._suggest(trial, "dropout", {"low": 0.0, "high": 0.3})
    trial.suggest_float.assert_called_once_with("dropout", 0.0, 0.3, step=None, log=False)
    trial.suggest_int.assert_not_called()


def test_suggest_float_log_scale(trial):
    ht._suggest(trial, "lr", {"low": 1e-4, "high": 1.0, "log": True})
    trial.suggest_float.assert_called_once_with("lr", 1e-4, 1.0, step=None, log=True)


def test_suggest_float_with_step(trial):
    ht._suggest(trial, "dropout", {"low": 0.0, "high": 0.3, "step": 0.05})
    trial.suggest_float.assert_called_once_with("dropout", 0.0, 0.3, step=0.05, log=False)


def test_suggest_int_log_scale(trial):
    ht._suggest(trial, "bottleneck", {"low": 32, "high": 512, "log": True})
    trial.suggest_int.assert_called_once_with("bottleneck", 32, 512, step=1, log=True)


def test_suggest_int_with_step(trial):
    ht._suggest(trial, "layers", {"low": 2, "high": 16, "step": 2})
    trial.suggest_int.assert_called_once_with("layers", 2, 16, step=2, log=False)


def test_suggest_int_inferred_from_python_int_types(trial):
    # YAML parses `2` as int and `2.0` as float; verify type-based dispatch
    ht._suggest(trial, "x", {"low": 2, "high": 8})
    trial.suggest_int.assert_called()
    trial.suggest_float.assert_not_called()


def test_suggest_float_inferred_from_python_float_types(trial):
    ht._suggest(trial, "x", {"low": 2.0, "high": 8.0})
    trial.suggest_float.assert_called()
    trial.suggest_int.assert_not_called()


# ---------------------------------------------------------------------------
# _validate_config
# ---------------------------------------------------------------------------

def test_validate_config_passes_sd_enabled_with_bottleneck():
    cfg = {
        "model": {"use_structured_dropout": True},
        "optuna": {"structured_dropout_bottleneck": {"low": 32, "high": 512}},
    }
    ht._validate_config(cfg)  # must not raise


def test_validate_config_passes_sd_disabled_without_bottleneck():
    cfg = {
        "model": {"use_structured_dropout": False},
        "optuna": {"lr": {"low": 1e-4, "high": 1.0}},
    }
    ht._validate_config(cfg)  # must not raise


def test_validate_config_passes_no_model_key():
    cfg = {"optuna": {"lr": {"low": 1e-4, "high": 1.0}}}
    ht._validate_config(cfg)  # use_structured_dropout defaults to False; no bottleneck → ok


def test_validate_config_raises_sd_disabled_with_bottleneck():
    cfg = {
        "model": {"use_structured_dropout": False},
        "optuna": {"structured_dropout_bottleneck": {"low": 32, "high": 512}},
    }
    with pytest.raises(ValueError, match="structured_dropout_bottleneck"):
        ht._validate_config(cfg)


def test_validate_config_error_message_mentions_remedy():
    cfg = {
        "model": {"use_structured_dropout": False},
        "optuna": {"structured_dropout_bottleneck": [32, 64, 128]},
    }
    with pytest.raises(ValueError, match="use_structured_dropout"):
        ht._validate_config(cfg)


def test_validate_config_lsrd_duplicate_raises():
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [
                [True, False, True],
                [False, True, False],
                [True, False, True],  # duplicate of index 0
            ]
        }
    }
    with pytest.raises(ValueError, match="duplicate.*\\[0\\]"):
        ht._validate_config(cfg)


def test_validate_config_lsrd_duplicate_01_and_bool_raises():
    """0/1 and True/False with same sequence must be caught as duplicates."""
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [
                [True, False],
                [1, 0],  # same as index 0 after bool normalisation
            ]
        }
    }
    with pytest.raises(ValueError, match="duplicate"):
        ht._validate_config(cfg)


def test_validate_config_lsrd_no_duplicates_passes():
    cfg = {
        "optuna": {
            "layer_shared_resid_dropout": [
                [True, False, True],
                [False, True, False],
                [True, True, False],
            ]
        }
    }
    ht._validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# _count_discrete_combinations
# ---------------------------------------------------------------------------

def test_count_discrete_flat_lists():
    cfg = {"n_heads": [1, 2, 4], "n_layers": [2, 4, 8, 16]}
    assert ht._count_discrete_combinations(cfg) == 12


def test_count_discrete_list_of_lists():
    cfg = {"layer_shared_resid_dropout": [[True, False], [False, True], [True, True]]}
    assert ht._count_discrete_combinations(cfg) == 3


def test_count_discrete_choices_dict():
    cfg = {"n_heads": {"choices": [1, 2, 4, 8]}, "dropout": [0.0, 0.1, 0.2]}
    assert ht._count_discrete_combinations(cfg) == 12


def test_count_discrete_int_range_with_step():
    cfg = {"n_layers": {"low": 2, "high": 8, "step": 2}}
    # values: 2, 4, 6, 8 → 4 choices
    assert ht._count_discrete_combinations(cfg) == 4


def test_count_discrete_int_range_default_step():
    cfg = {"n_layers": {"low": 1, "high": 4}}
    # values: 1, 2, 3, 4 → 4 choices
    assert ht._count_discrete_combinations(cfg) == 4


def test_count_discrete_float_range_returns_none():
    cfg = {"lr": {"low": 1e-4, "high": 1e-2, "log": True}}
    assert ht._count_discrete_combinations(cfg) is None


def test_count_discrete_mixed_continuous_returns_none():
    cfg = {"n_heads": [1, 2, 4], "lr": {"low": 1e-4, "high": 1e-2}}
    assert ht._count_discrete_combinations(cfg) is None


def test_count_discrete_skips_reserved_keys():
    cfg = {"objective_metric": "valid_action_accuracy", "n_heads": [1, 2]}
    assert ht._count_discrete_combinations(cfg) == 2


# ---------------------------------------------------------------------------
# _validate_config — discrete saturation check
# ---------------------------------------------------------------------------

def test_validate_config_discrete_saturation_raises_too_many():
    cfg = {"optuna": {"n_heads": [1, 2], "n_layers": [2, 4]}}  # 4 combinations
    with pytest.raises(ValueError, match="n_trials=5 must equal.*4"):
        ht._validate_config(cfg, n_trials=5)


def test_validate_config_discrete_saturation_raises_too_few():
    cfg = {"optuna": {"n_heads": [1, 2], "n_layers": [2, 4]}}  # 4 combinations
    with pytest.raises(ValueError, match="n_trials=3 must equal.*4"):
        ht._validate_config(cfg, n_trials=3)


def test_validate_config_discrete_saturation_exact_passes():
    cfg = {"optuna": {"n_heads": [1, 2], "n_layers": [2, 4]}}  # 4 combinations
    ht._validate_config(cfg, n_trials=4)  # must not raise


def test_validate_config_discrete_saturation_skipped_for_continuous():
    cfg = {"optuna": {"n_heads": [1, 2], "lr": {"low": 1e-4, "high": 1e-2, "log": True}}}
    ht._validate_config(cfg, n_trials=1000)  # continuous param → no check


def test_validate_config_discrete_saturation_skipped_when_n_trials_none():
    cfg = {"optuna": {"n_heads": [1, 2]}}
    ht._validate_config(cfg, n_trials=None)  # must not raise
