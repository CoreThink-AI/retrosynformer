"""Unit tests for scripts/hypertune.py helper functions."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# scripts/ is not a package; add it to the path for import.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import hypertune as ht


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


# ---------------------------------------------------------------------------
# _acquire_lock / _release_lock
# ---------------------------------------------------------------------------

@pytest.fixture
def lock_base(tmp_path, monkeypatch):
    """Redirect RESULTS_BASE to a temp directory for each test."""
    base = str(tmp_path / "hypertune")
    monkeypatch.setattr(ht, "RESULTS_BASE", base)
    return base


def test_acquire_creates_study_dir(lock_base):
    ht._acquire_lock("run1")
    study_dir = lock_base + "-run1"
    assert os.path.isdir(study_dir)


def test_acquire_creates_relative_symlink(lock_base):
    ht._acquire_lock("run1")
    assert os.path.islink(lock_base)
    assert os.readlink(lock_base) == "hypertune-run1"


def test_acquire_symlink_resolves_to_study_dir(lock_base):
    ht._acquire_lock("run1")
    assert os.path.realpath(lock_base) == os.path.realpath(lock_base + "-run1")


def test_acquire_raises_if_symlink_exists(lock_base):
    ht._acquire_lock("run1")
    with pytest.raises(RuntimeError, match="already exists"):
        ht._acquire_lock("run2")


def test_acquire_raises_if_real_dir_exists(lock_base, tmp_path):
    os.makedirs(lock_base)
    with pytest.raises(RuntimeError, match="already exists"):
        ht._acquire_lock("run1")


def test_acquire_error_message_includes_rm_hint(lock_base):
    ht._acquire_lock("run1")
    with pytest.raises(RuntimeError, match=r"rm "):
        ht._acquire_lock("run2")


def test_release_removes_symlink(lock_base):
    ht._acquire_lock("run1")
    ht._release_lock()
    assert not os.path.lexists(lock_base)


def test_release_preserves_study_dir(lock_base):
    ht._acquire_lock("run1")
    ht._release_lock()
    assert os.path.isdir(lock_base + "-run1")


def test_release_is_idempotent(lock_base):
    ht._acquire_lock("run1")
    ht._release_lock()
    ht._release_lock()  # second call must not raise


def test_full_lock_lifecycle(lock_base):
    ht._acquire_lock("study_a")
    assert os.path.islink(lock_base)
    ht._release_lock()
    assert not os.path.lexists(lock_base)
    # can acquire again after release
    ht._acquire_lock("study_b")
    assert os.path.islink(lock_base)
    assert os.readlink(lock_base) == "hypertune-study_b"
    ht._release_lock()
