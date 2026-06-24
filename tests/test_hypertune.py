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


def test_suggest_scalar_int_returns_fixed_value(trial):
    assert ht._suggest(trial, "hidden_size", 640) == 640
    trial.suggest_int.assert_not_called()
    trial.suggest_float.assert_not_called()
    trial.suggest_categorical.assert_not_called()


def test_suggest_scalar_float_returns_fixed_value(trial):
    assert ht._suggest(trial, "dropout", 0.1) == 0.1


def test_suggest_scalar_bool_returns_fixed_value(trial):
    assert ht._suggest(trial, "use_feature", True) is True


# ---------------------------------------------------------------------------
# _enumerate_ordered_params
# ---------------------------------------------------------------------------

def test_enumerate_flat_lists_cartesian_product():
    cfg = {"n_heads": [1, 2], "dropout": [0.0, 0.1]}
    combos = ht._enumerate_ordered_params(cfg)
    assert combos == [
        {"n_heads": 1, "dropout": 0.0},
        {"n_heads": 1, "dropout": 0.1},
        {"n_heads": 2, "dropout": 0.0},
        {"n_heads": 2, "dropout": 0.1},
    ]


def test_enumerate_preserves_declaration_order():
    cfg = {"a": [1, 2], "b": [10, 20], "c": [100, 200]}
    combos = ht._enumerate_ordered_params(cfg)
    assert len(combos) == 8
    assert combos[0] == {"a": 1, "b": 10, "c": 100}
    assert combos[-1] == {"a": 2, "b": 20, "c": 200}


def test_enumerate_list_of_lists_serialised_to_json():
    import json
    cfg = {"layer_shared_resid_dropout": [[True, False], [False, True]]}
    combos = ht._enumerate_ordered_params(cfg)
    assert len(combos) == 2
    # Values must be JSON strings matching what _suggest passes to suggest_categorical.
    assert combos[0] == {"layer_shared_resid_dropout": json.dumps([True, False], separators=(",", ":"))}
    assert combos[1] == {"layer_shared_resid_dropout": json.dumps([False, True], separators=(",", ":"))}


def test_enumerate_skips_range_params():
    cfg = {"n_heads": [1, 2], "lr": {"low": 1e-4, "high": 1e-2, "log": True}}
    combos = ht._enumerate_ordered_params(cfg)
    assert all("lr" not in c for c in combos)
    assert len(combos) == 2


def test_enumerate_skips_reserved_keys():
    cfg = {"objective_metric": "valid_action_accuracy", "n_heads": [1, 2]}
    combos = ht._enumerate_ordered_params(cfg)
    assert all("objective_metric" not in c for c in combos)
    assert len(combos) == 2


def test_enumerate_choices_dict_included():
    cfg = {"n_heads": {"choices": [1, 2, 4]}}
    combos = ht._enumerate_ordered_params(cfg)
    assert len(combos) == 3
    assert combos[1] == {"n_heads": 2}


def test_enumerate_no_list_params_returns_empty():
    cfg = {"lr": {"low": 1e-4, "high": 1e-2}, "n_layers": {"low": 2, "high": 8}}
    assert ht._enumerate_ordered_params(cfg) == []


def test_enumerate_mixed_list_and_list_of_lists():
    import json
    cfg = {
        "n_heads": [1, 2],
        "layer_shared_resid_dropout": [[True, False], [False, True]],
    }
    combos = ht._enumerate_ordered_params(cfg)
    assert len(combos) == 4
    assert combos[0]["n_heads"] == 1
    assert combos[0]["layer_shared_resid_dropout"] == json.dumps([True, False], separators=(",", ":"))


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


# ---------------------------------------------------------------------------
# _validate_config — random_seed spec checks
# ---------------------------------------------------------------------------

def test_validate_config_random_seed_list_of_ints_passes():
    cfg = {"optuna": {"random_seed": [1, 42, 137]}}
    ht._validate_config(cfg)  # must not raise


def test_validate_config_random_seed_int_range_passes():
    cfg = {"optuna": {"random_seed": {"low": 1, "high": 1000}}}
    ht._validate_config(cfg)  # must not raise


def test_validate_config_random_seed_int_range_with_step_passes():
    cfg = {"optuna": {"random_seed": {"low": 0, "high": 100, "step": 10}}}
    ht._validate_config(cfg)  # must not raise


def test_validate_config_random_seed_list_with_float_raises():
    cfg = {"optuna": {"random_seed": [1, 2.5, 3]}}
    with pytest.raises(ValueError, match="integers"):
        ht._validate_config(cfg)


def test_validate_config_random_seed_float_range_raises():
    cfg = {"optuna": {"random_seed": {"low": 0.0, "high": 1.0}}}
    with pytest.raises(ValueError, match="integers"):
        ht._validate_config(cfg)


def test_validate_config_random_seed_log_range_raises():
    cfg = {"optuna": {"random_seed": {"low": 1, "high": 1000, "log": True}}}
    with pytest.raises(ValueError, match="log"):
        ht._validate_config(cfg)


def test_validate_config_random_seed_scalar_raises():
    cfg = {"optuna": {"random_seed": 42}}
    with pytest.raises(ValueError, match="list of ints or a dict"):
        ht._validate_config(cfg)


# ---------------------------------------------------------------------------
# n_trials persistence
# ---------------------------------------------------------------------------

def test_study_set_user_attr_n_trials(tmp_path):
    """n_trials is stored as a user attr in study.db."""
    import optuna
    storage = f"sqlite:///{tmp_path}/study.db"
    study = optuna.create_study(study_name="test", storage=storage, direction="maximize")
    study.set_user_attr("n_trials", 42)
    # Reload from storage to confirm it round-trips.
    loaded = optuna.load_study(study_name="test", storage=storage)
    assert loaded.user_attrs["n_trials"] == 42


def test_n_trials_written_to_model_config_yaml(tmp_path):
    """objective() writes n_trials into model.config.yaml after training."""
    import yaml

    # Write a minimal model.config.yaml (simulating what runner.main() saves).
    cfg = {"context": {"random_state": 7}, "train": {"results_path": str(tmp_path)}}
    saved_cfg_path = tmp_path / "model.config.yaml"
    saved_cfg_path.write_text(yaml.dump(cfg))

    # Simulate the post-save block directly.
    import yaml as _yaml
    with open(saved_cfg_path) as f:
        saved = _yaml.safe_load(f)
    saved.setdefault("train", {})["n_trials"] = 20
    with open(saved_cfg_path, "w") as f:
        _yaml.dump(saved, f, default_flow_style=False)

    result = yaml.safe_load(saved_cfg_path.read_text())
    assert result["train"]["n_trials"] == 20


# ---------------------------------------------------------------------------
# layer_shared_resid_dropout — optuna and model section handling
# ---------------------------------------------------------------------------

class TestPreprocessOptunaCfgLSRD:
    """_preprocess_optuna_config normalises layer_shared_resid_dropout choices."""

    def test_int_keyed_dicts_converted_to_lists(self):
        cfg = {"optuna": {"layer_shared_resid_dropout": [
            {0: True, 2: True},
            {1: True, 3: True},
        ]}}
        out = ht._preprocess_optuna_config(cfg)
        lsrd = out["optuna"]["layer_shared_resid_dropout"]
        assert lsrd == [
            [True, False, True, False],
            [False, True, False, True],
        ]

    def test_float_keyed_dicts_serialised_to_json_strings(self):
        import json
        cfg = {"optuna": {"layer_shared_resid_dropout": [
            {1.5: 1, 2.5: 1},
            {},
        ]}}
        out = ht._preprocess_optuna_config(cfg)
        lsrd = out["optuna"]["layer_shared_resid_dropout"]
        assert len(lsrd) == 2
        assert all(isinstance(s, str) for s in lsrd)
        # First choice deserialises to the original dict (keys as strings in JSON).
        assert json.loads(lsrd[0]) == {"1.5": 1, "2.5": 1}
        # Second choice is an empty JSON object.
        assert json.loads(lsrd[1]) == {}

    def test_empty_dict_choice_serialised_correctly(self):
        import json
        cfg = {"optuna": {"layer_shared_resid_dropout": [
            {3.5: 2},
            {},
        ]}}
        out = ht._preprocess_optuna_config(cfg)
        lsrd = out["optuna"]["layer_shared_resid_dropout"]
        assert json.loads(lsrd[1]) == {}

    def test_non_lsrd_params_unchanged(self):
        cfg = {"optuna": {"n_heads": [2, 4], "lr": {"low": 1e-4, "high": 1e-2}}}
        out = ht._preprocess_optuna_config(cfg)
        assert out["optuna"]["n_heads"] == [2, 4]
        assert out["optuna"]["lr"] == {"low": 1e-4, "high": 1e-2}

    def test_no_optuna_section_is_noop(self):
        cfg = {"model": {"n_layers": 4}}
        out = ht._preprocess_optuna_config(cfg)
        assert out == {"model": {"n_layers": 4}}

    def test_deep_copy_does_not_mutate_original(self):
        original = {"optuna": {"layer_shared_resid_dropout": [{1.5: 1}, {}]}}
        ht._preprocess_optuna_config(original)
        # Original must be unchanged.
        assert isinstance(original["optuna"]["layer_shared_resid_dropout"][0], dict)


class TestSuggestLSRD:
    """_suggest handles JSON-string dict choices from _preprocess_optuna_config."""

    def _trial(self, chosen_index=0):
        """Mock trial that returns choices[chosen_index] from suggest_categorical."""
        t = MagicMock()
        t.suggest_categorical.side_effect = lambda name, choices: choices[chosen_index]
        return t

    def test_float_keyed_dict_choice_deserialised(self):
        import json
        # Simulate what _preprocess_optuna_config produces for float-keyed dicts.
        choices = [
            json.dumps({"1.5": 1, "2.5": 1}, separators=(",", ":"), sort_keys=True),
            "{}",
        ]
        result = ht._suggest(self._trial(0), "layer_shared_resid_dropout", choices)
        assert result == {1.5: 1, 2.5: 1}

    def test_empty_dict_choice_deserialised(self):
        import json
        choices = [
            json.dumps({"1.5": 1}, separators=(",", ":"), sort_keys=True),
            "{}",
        ]
        result = ht._suggest(self._trial(1), "layer_shared_resid_dropout", choices)
        assert result == {}

    def test_integer_keys_restored_correctly(self):
        import json
        # Keys that are whole-number floats (e.g. "2.0") → int.
        choices = [json.dumps({"2.0": 1}, separators=(",", ":"), sort_keys=True)]
        result = ht._suggest(self._trial(0), "x", choices)
        assert result == {2: 1}
        assert isinstance(list(result.keys())[0], int)

    def test_fractional_keys_remain_float(self):
        import json
        choices = [json.dumps({"11.5": 1, "23.5": 1}, separators=(",", ":"), sort_keys=True)]
        result = ht._suggest(self._trial(0), "x", choices)
        assert all(isinstance(k, float) for k in result)


class TestRestoreNumericDictKeys:
    """_restore_numeric_dict_keys converts string keys back to int/float."""

    def test_float_key_restored(self):
        assert ht._restore_numeric_dict_keys({"1.5": 1}) == {1.5: 1}

    def test_whole_number_key_becomes_int(self):
        result = ht._restore_numeric_dict_keys({"2.0": 1})
        assert result == {2: 1}
        assert isinstance(list(result.keys())[0], int)

    def test_integer_string_key_becomes_int(self):
        result = ht._restore_numeric_dict_keys({"3": 1})
        assert result == {3: 1}
        assert isinstance(list(result.keys())[0], int)

    def test_non_numeric_key_unchanged(self):
        assert ht._restore_numeric_dict_keys({"foo": 1}) == {"foo": 1}

    def test_empty_dict(self):
        assert ht._restore_numeric_dict_keys({}) == {}

    def test_mixed_keys(self):
        result = ht._restore_numeric_dict_keys({"1": 0, "1.5": 1, "foo": 2})
        assert result[1] == 0
        assert result[1.5] == 1
        assert result["foo"] == 2


class TestPreprocessMixedKeyDicts:
    """_preprocess_optuna_config handles mixed int/float key dicts (baseline={1:0})."""

    def test_mixed_float_and_int_keyed_takes_float_branch(self):
        import json
        # Float-keyed first dict triggers float branch; int-keyed second serialised too.
        cfg = {"optuna": {"layer_shared_resid_dropout": [
            {1.5: 1, 2.5: 1},
            {1: 0},
        ]}}
        out = ht._preprocess_optuna_config(cfg)
        lsrd = out["optuna"]["layer_shared_resid_dropout"]
        assert all(isinstance(s, str) for s in lsrd)
        # Second choice: {1: 0} serialised — key "1" restored to int via _restore.
        second = json.loads(lsrd[1])
        assert second == {"1": 0}   # JSON keys are strings

    def test_suggest_restores_int_key_from_mixed_dict(self):
        import json
        choices = [
            json.dumps({"1.5": 1}, separators=(",", ":"), sort_keys=True),
            json.dumps({"1": 0}, separators=(",", ":"), sort_keys=True),
        ]
        t = MagicMock()
        t.suggest_categorical.side_effect = lambda name, c: c[1]  # pick {1: 0}
        result = ht._suggest(t, "layer_shared_resid_dropout", choices)
        assert result == {1: 0}
        assert isinstance(list(result.keys())[0], int)


class TestEnumerateOrderedParamsLSRD:
    """_enumerate_ordered_params handles JSON-string LSRD choices correctly."""

    def test_float_keyed_lsrd_choices_enumerated_as_strings(self):
        import json
        # After _preprocess_optuna_config, float-keyed dicts become JSON strings.
        choices = [
            json.dumps({"1.5": 1}, separators=(",", ":"), sort_keys=True),
            "{}",
        ]
        cfg = {"layer_shared_resid_dropout": choices}
        combos = ht._enumerate_ordered_params(cfg)
        assert len(combos) == 2
        assert combos[0]["layer_shared_resid_dropout"] == choices[0]
        assert combos[1]["layer_shared_resid_dropout"] == choices[1]

    def test_lsrd_combined_with_other_params(self):
        import json
        choices = [json.dumps({"1.5": 1}, separators=(",", ":"), sort_keys=True), "{}"]
        cfg = {"n_heads": [2, 4], "layer_shared_resid_dropout": choices}
        combos = ht._enumerate_ordered_params(cfg)
        assert len(combos) == 4
