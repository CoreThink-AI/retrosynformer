"""Tests for the Optuna study.db ORM (retrosynformer.models_optuna).

Exercises the model against three real study.db files with different
characteristics:
  - compare_small_structured_dropout  — few trials, structured dropout params
  - baseline_small_hyperparameter_tuning — 12 trials, FAIL + RUNNING states
  - small-nonuniform-dropout  — complex categorical choices (JSON layer masks)
"""
import json
import os
from pathlib import Path

import pytest

RESULTS_ROOT = Path(__file__).parent.parent / "results"

# Pick three study.db files that cover different shapes of data.
DB_PATHS = {
    "structured_dropout": RESULTS_ROOT / "hypertune-compare_small_structured_dropout" / "study.db",
    "baseline_small":     RESULTS_ROOT / "hypertune-baseline_small_hyperparameter_tuning" / "study.db",
    "nonuniform_dropout": RESULTS_ROOT / "hypertune-small-nonuniform-dropout" / "study.db",
}

# Skip the whole module if the result files are absent (CI without data).
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in DB_PATHS.values()),
    reason="study.db result files not found — skipping ORM tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sessions():
    from retrosynformer.models_optuna import connect
    opened = {k: connect(v) for k, v in DB_PATHS.items()}
    yield opened
    for s in opened.values():
        s.close()


# ---------------------------------------------------------------------------
# VersionInfo / AlembicVersion
# ---------------------------------------------------------------------------

def test_version_info_present(sessions):
    from retrosynformer.models_optuna import VersionInfo
    for name, session in sessions.items():
        vi = session.query(VersionInfo).first()
        assert vi is not None, f"{name}: no version_info row"
        assert isinstance(vi.schema_version, int)
        assert vi.library_version.startswith("4.")  # Optuna 4.x


def test_alembic_version_present(sessions):
    from retrosynformer.models_optuna import AlembicVersion
    for name, session in sessions.items():
        av = session.query(AlembicVersion).first()
        assert av is not None, f"{name}: no alembic_version row"
        assert isinstance(av.version_num, str) and len(av.version_num) > 0


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def test_study_name(sessions):
    from retrosynformer.models_optuna import Study
    expected = {
        "structured_dropout": "compare_small_structured_dropout",
        "baseline_small":     "baseline_small_hyperparameter_tuning",
        "nonuniform_dropout": "small-nonuniform-dropout",
    }
    for name, session in sessions.items():
        studies = session.query(Study).all()
        names = [s.study_name for s in studies]
        assert expected[name] in names, f"{name}: expected study_name not found, got {names}"


def test_study_direction(sessions):
    from retrosynformer.models_optuna import Study
    for name, session in sessions.items():
        study = session.query(Study).first()
        assert study.direction in ("MAXIMIZE", "MINIMIZE"), (
            f"{name}: unexpected direction {study.direction!r}"
        )


def test_study_repr(sessions):
    from retrosynformer.models_optuna import Study
    for name, session in sessions.items():
        study = session.query(Study).first()
        r = repr(study)
        assert study.study_name in r


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------

def test_trials_exist(sessions):
    from retrosynformer.models_optuna import Trial
    for name, session in sessions.items():
        trials = session.query(Trial).all()
        assert len(trials) > 0, f"{name}: no trials"


def test_trial_states(sessions):
    from retrosynformer.models_optuna import Trial
    valid_states = {"COMPLETE", "RUNNING", "FAIL", "WAITING"}
    for name, session in sessions.items():
        for t in session.query(Trial).all():
            assert t.state in valid_states, f"{name} trial {t.number}: bad state {t.state!r}"


def test_trial_number_monotonic(sessions):
    from retrosynformer.models_optuna import Study
    for name, session in sessions.items():
        for study in session.query(Study).all():
            nums = [t.number for t in study.trials]
            assert nums == sorted(nums), f"{name}/{study.study_name}: trial numbers not sorted"


def test_complete_trials_have_datetimes(sessions):
    from retrosynformer.models_optuna import Trial
    for name, session in sessions.items():
        for t in session.query(Trial).filter_by(state="COMPLETE"):
            assert t.datetime_start is not None, f"{name} trial {t.number}: no datetime_start"
            assert t.datetime_complete is not None, f"{name} trial {t.number}: no datetime_complete"
            assert t.datetime_complete >= t.datetime_start


def test_duration_min(sessions):
    from retrosynformer.models_optuna import Trial
    for name, session in sessions.items():
        for t in session.query(Trial).filter_by(state="COMPLETE"):
            dur = t.duration_min
            assert dur is not None and dur >= 0, f"{name} trial {t.number}: bad duration {dur}"


# ---------------------------------------------------------------------------
# TrialParam — decoded values
# ---------------------------------------------------------------------------

def test_params_dict_has_all_params(sessions):
    from retrosynformer.models_optuna import Trial
    for name, session in sessions.items():
        for t in session.query(Trial).filter_by(state="COMPLETE"):
            pd = t.params_dict
            assert isinstance(pd, dict) and len(pd) > 0, (
                f"{name} trial {t.number}: empty params_dict"
            )


def test_categorical_decoded_not_float_index(sessions):
    """Decoded categorical values must be the actual choice, not a raw float."""
    from retrosynformer.models_optuna import TrialParam
    for name, session in sessions.items():
        for p in session.query(TrialParam).all():
            dist = p.distribution
            if dist["name"] == "CategoricalDistribution":
                choices = dist["attributes"]["choices"]
                decoded = p.decoded_value
                assert decoded in choices, (
                    f"{name} param {p.param_name}: decoded {decoded!r} not in choices {choices!r}"
                )
                # Must NOT be a bare float index unless the choice itself is a float
                if not isinstance(choices[0], float):
                    assert not isinstance(decoded, float) or int(decoded) == decoded, (
                        f"{name}: {p.param_name} decoded to raw float index {decoded}"
                    )


def test_float_distribution_decoded_in_bounds(sessions):
    from retrosynformer.models_optuna import TrialParam
    for name, session in sessions.items():
        for p in session.query(TrialParam).all():
            dist = p.distribution
            if dist["name"] == "FloatDistribution":
                lo = dist["attributes"]["low"]
                hi = dist["attributes"]["high"]
                val = p.decoded_value
                assert lo <= val <= hi, (
                    f"{name} {p.param_name}: {val} not in [{lo}, {hi}]"
                )


def test_nonuniform_dropout_complex_choices(sessions):
    """small-nonuniform-dropout has JSON-string layer masks as choices."""
    from retrosynformer.models_optuna import TrialParam
    session = sessions["nonuniform_dropout"]
    layer_params = (
        session.query(TrialParam)
        .filter(TrialParam.param_name == "layer_shared_resid_dropout")
        .all()
    )
    assert len(layer_params) > 0, "Expected layer_shared_resid_dropout params"
    for p in layer_params:
        decoded = p.decoded_value
        # Choices are JSON strings representing lists of booleans
        assert isinstance(decoded, str), f"Expected string choice, got {type(decoded)}: {decoded!r}"
        parsed = json.loads(decoded)
        assert isinstance(parsed, list), f"Expected JSON list, got {type(parsed)}"
        assert all(isinstance(v, bool) for v in parsed)


# ---------------------------------------------------------------------------
# TrialValue — objective score
# ---------------------------------------------------------------------------

def test_complete_trials_have_objective(sessions):
    from retrosynformer.models_optuna import Trial
    for name, session in sessions.items():
        for t in session.query(Trial).filter_by(state="COMPLETE"):
            val = t.objective_value
            assert val is not None, f"{name} trial {t.number}: COMPLETE but no objective_value"
            assert isinstance(val, float)


def test_objective_value_finite(sessions):
    from retrosynformer.models_optuna import Trial, TrialValue
    for name, session in sessions.items():
        for v in session.query(TrialValue).filter_by(value_type="FINITE"):
            assert v.value is not None and abs(v.value) < 1e9, (
                f"{name}: FINITE value {v.value} looks wrong"
            )


# ---------------------------------------------------------------------------
# Study.best_trial
# ---------------------------------------------------------------------------

def test_best_trial_has_max_score(sessions):
    from retrosynformer.models_optuna import Study
    for name, session in sessions.items():
        for study in session.query(Study).all():
            bt = study.best_trial
            if bt is None:
                continue  # no complete trials
            assert bt.state == "COMPLETE"
            if study.direction == "MAXIMIZE":
                assert all(
                    t.objective_value is None or t.objective_value <= bt.objective_value
                    for t in study.complete_trials
                ), f"{name}/{study.study_name}: best_trial is not the maximum"


# ---------------------------------------------------------------------------
# Study.search_space
# ---------------------------------------------------------------------------

def test_search_space_keys_match_params(sessions):
    from retrosynformer.models_optuna import Study
    for name, session in sessions.items():
        for study in session.query(Study).all():
            ss = study.search_space
            all_param_names = {
                p.param_name
                for t in study.complete_trials
                for p in t.params
            }
            assert set(ss.keys()) == all_param_names, (
                f"{name}/{study.study_name}: search_space keys {set(ss.keys())} "
                f"!= actual params {all_param_names}"
            )


def test_search_space_categorical_is_union(sessions):
    """search_space choices must be the union across all trials."""
    from retrosynformer.models_optuna import Study
    for name, session in sessions.items():
        for study in session.query(Study).all():
            ss = study.search_space
            for param_name, dist in ss.items():
                if dist["name"] != "CategoricalDistribution":
                    continue
                merged_choices = set(dist["attributes"]["choices"])
                for t in study.complete_trials:
                    for p in t.params:
                        if p.param_name == param_name:
                            assert p.decoded_value in merged_choices, (
                                f"{name}/{study.study_name}/{param_name}: "
                                f"decoded value {p.decoded_value!r} missing from union"
                            )


# ---------------------------------------------------------------------------
# Merge compatibility check
# ---------------------------------------------------------------------------

def test_check_merge_compatible_same_db(sessions):
    """A database is always compatible with itself."""
    from retrosynformer.models_optuna import check_merge_compatibility
    for name, session in sessions.items():
        check_merge_compatibility(session, session)  # must not raise


def test_check_merge_structured_vs_baseline(sessions):
    """Two MAXIMIZE small-dataset studies should be compatible."""
    from retrosynformer.models_optuna import check_merge_compatibility, MergeConflict
    try:
        check_merge_compatibility(
            sessions["structured_dropout"],
            sessions["baseline_small"],
        )
    except MergeConflict as exc:
        # Both are MAXIMIZE; only failure allowed is distribution-type mismatch
        assert "distribution" in str(exc).lower() or "mismatch" in str(exc).lower(), (
            f"Unexpected MergeConflict: {exc}"
        )


# ---------------------------------------------------------------------------
# merge_search_space
# ---------------------------------------------------------------------------

def test_merge_search_space_contains_all_params(sessions):
    """Merged space must contain every param from either study."""
    from retrosynformer.models_optuna import merge_search_space, TrialParam
    sa = sessions["structured_dropout"]
    sb = sessions["baseline_small"]
    merged = merge_search_space(sa, sb)
    params_a = {p.param_name for p in sa.query(TrialParam).all()}
    params_b = {p.param_name for p in sb.query(TrialParam).all()}
    assert set(merged.keys()) == params_a | params_b


def test_merge_search_space_categorical_union(sessions):
    """Merged CategoricalDistribution choices must be the superset."""
    from retrosynformer.models_optuna import merge_search_space, _collect_param_info
    sa = sessions["structured_dropout"]
    sb = sessions["baseline_small"]
    merged = merge_search_space(sa, sb)
    info_a = _collect_param_info(sa)
    info_b = _collect_param_info(sb)

    for param, rec in merged.items():
        if rec["dist_type"] != "CategoricalDistribution":
            continue
        merged_set = set(rec["choices"])
        if param in info_a and info_a[param]["choices"]:
            assert info_a[param]["choices"] <= merged_set, (
                f"{param}: A choices not subset of merged"
            )
        if param in info_b and info_b[param]["choices"]:
            assert info_b[param]["choices"] <= merged_set, (
                f"{param}: B choices not subset of merged"
            )


def test_merge_search_space_float_bounds_widened(sessions):
    """Merged float bounds must be at least as wide as either source."""
    from retrosynformer.models_optuna import merge_search_space, _collect_param_info
    sa = sessions["structured_dropout"]
    sb = sessions["baseline_small"]
    merged = merge_search_space(sa, sb)
    info_a = _collect_param_info(sa)
    info_b = _collect_param_info(sb)

    for param, rec in merged.items():
        if rec["dist_type"] == "CategoricalDistribution":
            continue
        if param in info_a and info_a[param]["low"] is not None:
            assert rec["low"]  <= info_a[param]["low"]
            assert rec["high"] >= info_a[param]["high"]
        if param in info_b and info_b[param]["low"] is not None:
            assert rec["low"]  <= info_b[param]["low"]
            assert rec["high"] >= info_b[param]["high"]


def test_encode_param_value_categorical(sessions):
    """Re-encoded index must round-trip through the merged choices list."""
    from retrosynformer.models_optuna import (
        merge_search_space, encode_param_value, TrialParam
    )
    sa = sessions["structured_dropout"]
    sb = sessions["baseline_small"]
    merged = merge_search_space(sa, sb)

    for session in (sa, sb):
        for p in session.query(TrialParam).all():
            if p.distribution["name"] != "CategoricalDistribution":
                continue
            if p.param_name not in merged:
                continue
            decoded = p.decoded_value
            new_idx = encode_param_value(decoded, merged[p.param_name])
            # Round-trip: new index must point back to the same value
            assert merged[p.param_name]["choices"][int(new_idx)] == decoded, (
                f"{p.param_name}: round-trip failed for {decoded!r}"
            )


# ---------------------------------------------------------------------------
# plan_merge
# ---------------------------------------------------------------------------

def test_plan_merge_returns_expected_keys(sessions):
    from retrosynformer.models_optuna import plan_merge
    plan = plan_merge(sessions["structured_dropout"], sessions["baseline_small"])
    assert set(plan.keys()) == {"new_study_name", "merged_space", "changes", "warnings"}


def test_plan_merge_study_name(sessions):
    from retrosynformer.models_optuna import plan_merge
    plan = plan_merge(
        sessions["structured_dropout"],
        sessions["baseline_small"],
        new_study_name="custom_merged",
    )
    assert plan["new_study_name"] == "custom_merged"


def test_plan_merge_changes_list(sessions):
    """n_heads has different choices in structured vs baseline — should appear in changes."""
    from retrosynformer.models_optuna import plan_merge
    plan = plan_merge(sessions["structured_dropout"], sessions["baseline_small"])
    changed_params = {c["param"] for c in plan["changes"]}
    # n_layers: structured has [2,3,4,5], baseline has [3,4,8,12,18,26] → widened
    assert "n_layers" in changed_params
    # n_heads: structured has [1,2,3,4], baseline has [1,2,4] → structured is superset,
    # baseline adds nothing new; still a difference (only_in_A non-empty)
    assert "n_heads" in changed_params


def test_plan_merge_warns_about_running_trials(sessions):
    """baseline_small has a RUNNING trial — plan should warn, NOT say convert to FAIL."""
    from retrosynformer.models_optuna import plan_merge
    plan = plan_merge(sessions["structured_dropout"], sessions["baseline_small"])
    warning_text = " ".join(plan["warnings"]).upper()
    assert "RUNNING" in warning_text
    assert "CONVERT" not in warning_text, "Warning must not instruct state conversion"
    assert "DROP" not in warning_text, "Warning must not instruct dropping trials"


# ---------------------------------------------------------------------------
# Config-file augmentation
# ---------------------------------------------------------------------------

def test_trial_config_path_convention():
    from retrosynformer.models_optuna import trial_config_path
    p = trial_config_path(DB_PATHS["structured_dropout"], 2)
    assert p.name == "model.config.yaml"
    assert "trial_002" in str(p)
    assert p.exists(), f"Expected config at {p}"


def test_load_trial_config_returns_dict():
    from retrosynformer.models_optuna import load_trial_config, trial_config_path
    path = trial_config_path(DB_PATHS["structured_dropout"], 2)
    cfg = load_trial_config(path)
    assert isinstance(cfg, dict) and len(cfg) > 0
    # optuna section should be excluded
    assert "optuna" not in cfg
    # must have model and train sections
    assert "model" in cfg
    assert "train" in cfg


def test_load_trial_config_missing_file_returns_empty():
    from retrosynformer.models_optuna import load_trial_config
    cfg = load_trial_config("/nonexistent/path/model.config.yaml")
    assert cfg == {}


def test_study_config_params_has_fixed_keys(sessions):
    """study_config_params should return params NOT searched by Optuna."""
    from retrosynformer.models_optuna import study_config_params
    configs = study_config_params(DB_PATHS["structured_dropout"], sessions["structured_dropout"])
    assert len(configs) > 0, "No configs loaded"
    # All returned dicts should contain fixed parameters
    for trial_num, flat in configs.items():
        assert isinstance(flat, dict)
        # These are fixed across the study — must be present in the config
        assert "train.batch_size" in flat, f"trial {trial_num}: missing train.batch_size"
        assert "train.n_epochs" in flat, f"trial {trial_num}: missing train.n_epochs"
        assert "dataset.action_dim" in flat, f"trial {trial_num}: missing dataset.action_dim"


def test_study_config_params_excludes_searched_params(sessions):
    """Optuna-searched params must not appear in the fixed-params dict."""
    from retrosynformer.models_optuna import study_config_params, TrialParam
    session = sessions["structured_dropout"]
    configs = study_config_params(DB_PATHS["structured_dropout"], session)
    searched_names = {p.param_name for p in session.query(TrialParam).all()}
    for trial_num, flat in configs.items():
        leaf_keys = {k.rsplit(".", 1)[-1] for k in flat}
        overlap = leaf_keys & searched_names
        assert not overlap, (
            f"trial {trial_num}: config returned searched params {overlap}"
        )


def test_study_config_params_no_path_values(sessions):
    """Path-valued keys should be excluded by default."""
    from retrosynformer.models_optuna import study_config_params, _CONFIG_PATH_KEYS
    configs = study_config_params(DB_PATHS["structured_dropout"], sessions["structured_dropout"])
    for trial_num, flat in configs.items():
        assert not (set(flat.keys()) & _CONFIG_PATH_KEYS), (
            f"trial {trial_num}: path keys leaked into fixed params"
        )


def test_fixed_params_diff_same_study(sessions):
    """Comparing a study with itself should yield no differences."""
    from retrosynformer.models_optuna import fixed_params_diff
    diffs = fixed_params_diff(
        DB_PATHS["structured_dropout"], sessions["structured_dropout"],
        DB_PATHS["structured_dropout"], sessions["structured_dropout"],
    )
    assert diffs == {}


def test_fixed_params_diff_different_studies(sessions):
    """structured_dropout vs baseline_small differ in at least n_epochs."""
    from retrosynformer.models_optuna import fixed_params_diff
    diffs = fixed_params_diff(
        DB_PATHS["structured_dropout"], sessions["structured_dropout"],
        DB_PATHS["baseline_small"],    sessions["baseline_small"],
    )
    assert len(diffs) > 0, "Expected differences between two distinct studies"
    # Both use small dataset (action_dim=589) but have different n_epochs
    assert "train.n_epochs" in diffs or "train.early_stopping_patience" in diffs, (
        f"Expected training schedule differences; got: {list(diffs.keys())}"
    )


# ---------------------------------------------------------------------------
# estimate_incomplete_objectives
# ---------------------------------------------------------------------------

# Additional study databases used only for estimation tests.
_LARGE_DB   = "results/hypertune-large-nonuniform-dropout-12-layers/study.db"
_DETAILS_DB = "results/hypertune-standard-v2-dropout-details/study.db"
_LR_DB      = "results/hypertune-standard-v2-lr0005/study.db"


@pytest.fixture(scope="module")
def large_session():
    from retrosynformer.models_optuna import connect
    s = connect(_LARGE_DB)
    yield s
    s.close()


@pytest.fixture(scope="module")
def details_session():
    from retrosynformer.models_optuna import connect
    s = connect(_DETAILS_DB)
    yield s
    s.close()


@pytest.fixture(scope="module")
def lr_session():
    from retrosynformer.models_optuna import connect
    s = connect(_LR_DB)
    yield s
    s.close()


def test_load_jsonl_metric_returns_sorted_pairs():
    """_load_jsonl_metric must return epoch-sorted (epoch, value) pairs."""
    from retrosynformer.models_optuna import _load_jsonl_metric
    pairs = _load_jsonl_metric(
        "results/hypertune-large-nonuniform-dropout-12-layers/trial_002/train_progress.jsonl",
        "valid_route_accuracy",
    )
    assert len(pairs) >= 6, f"Expected ≥6 pairs, got {len(pairs)}"
    epochs = [e for e, _ in pairs]
    assert epochs == sorted(epochs), "Pairs must be sorted by epoch"
    assert all(isinstance(v, float) for _, v in pairs)


def test_load_jsonl_metric_deduplicates_epochs(tmp_path):
    """If the same epoch appears twice, only the last value is kept."""
    from retrosynformer.models_optuna import _load_jsonl_metric
    import json as _json
    jf = tmp_path / "train_progress.jsonl"
    rows = [
        {"epoch": 0, "valid_route_accuracy": 0.1},
        {"epoch": 0, "valid_route_accuracy": 0.2},  # duplicate — keep this
        {"epoch": 1, "valid_route_accuracy": 0.3},
    ]
    jf.write_text("\n".join(_json.dumps(r) for r in rows))
    pairs = _load_jsonl_metric(str(jf), "valid_route_accuracy")
    assert len(pairs) == 2, f"Expected 2 unique epochs, got {len(pairs)}"
    assert pairs[0] == (0, 0.2), "Last value for epoch 0 must survive de-dup"


def test_load_jsonl_metric_missing_file():
    from retrosynformer.models_optuna import _load_jsonl_metric
    pairs = _load_jsonl_metric("/nonexistent/train_progress.jsonl", "valid_route_accuracy")
    assert pairs == []


def test_fit_quadratic_estimate_concave_down():
    """Concave-down parabola should extrapolate to vertex when it precedes target."""
    from retrosynformer.models_optuna import _fit_quadratic_estimate
    import numpy as np
    # Perfect quadratic: value peaks at epoch 50, we've only observed 0..30
    a, b, c = -0.001, 0.1, 0.0
    epochs = list(range(15, 31))
    pairs = [(e, a * e**2 + b * e + c) for e in epochs]
    result = _fit_quadratic_estimate(pairs, target_epoch=99, direction="MAXIMIZE")
    # Vertex at -b/(2a) = -0.1 / (-0.002) = 50
    assert abs(result["target_epoch"] - 50.0) < 1.0, (
        f"Expected extrapolation near vertex (50), got {result['target_epoch']}"
    )
    assert result["r_squared"] > 0.99


def test_fit_quadratic_estimate_concave_up_minimize():
    """Concave-up parabola MINIMIZE: extrapolate to vertex."""
    from retrosynformer.models_optuna import _fit_quadratic_estimate
    # Parabola opens upward, MINIMIZE: minimum at epoch 20
    a, b, c = 0.001, -0.04, 1.0
    epochs = list(range(5, 18))
    pairs = [(e, a * e**2 + b * e + c) for e in epochs]
    result = _fit_quadratic_estimate(pairs, target_epoch=40, direction="MINIMIZE")
    assert abs(result["target_epoch"] - 20.0) < 2.0
    assert result["r_squared"] > 0.99


def test_fit_quadratic_estimate_r_squared_in_range():
    """R² must be in [0, 1] for typical data."""
    from retrosynformer.models_optuna import _fit_quadratic_estimate
    pairs = [(i, 0.1 * i ** 0.5) for i in range(10, 21)]
    result = _fit_quadratic_estimate(pairs, target_epoch=30, direction="MAXIMIZE")
    assert 0.0 <= result["r_squared"] <= 1.0


def test_estimate_incomplete_objectives_large_study(large_session):
    """hypertune-large has a RUNNING trial with 37 epochs — should produce an estimate."""
    from retrosynformer.models_optuna import estimate_incomplete_objectives
    results = estimate_incomplete_objectives(_LARGE_DB, large_session)
    running = {k: v for k, v in results.items() if v["state"] == "RUNNING"}
    assert len(running) >= 1, "Expected at least one RUNNING trial"
    # The trial with 37 epochs must produce a non-skipped estimate
    estimated = {k: v for k, v in running.items() if not v["skipped"]}
    assert len(estimated) >= 1, (
        f"Expected at least one RUNNING trial with ≥6 epochs; got: {running}"
    )
    for trial_num, res in estimated.items():
        assert res["estimated_value"] is not None
        assert isinstance(res["estimated_value"], float)
        assert res["r_squared"] is not None
        assert 0.0 <= res["r_squared"] <= 1.0, f"R² out of range: {res['r_squared']}"
        assert res["n_points_fit"] >= 3, "Second-half must have ≥3 points"
        assert res["poly_coeffs"] is not None and len(res["poly_coeffs"]) == 3


def test_estimate_incomplete_objectives_too_few_epochs(details_session):
    """hypertune-standard-v2-dropout-details trial_5 has only 2 epochs — must skip."""
    from retrosynformer.models_optuna import estimate_incomplete_objectives
    results = estimate_incomplete_objectives(_DETAILS_DB, details_session, min_epochs=6)
    for trial_num, res in results.items():
        if res["n_epochs_observed"] < 6:
            assert res["skipped"] is True
            assert res["estimated_value"] is None
            assert "min" in (res["skip_reason"] or "").lower()


def test_estimate_incomplete_objectives_lr_study(lr_session):
    """hypertune-standard-v2-lr0005 trial_5 has 7 epochs — should not skip."""
    from retrosynformer.models_optuna import estimate_incomplete_objectives
    results = estimate_incomplete_objectives(_LR_DB, lr_session, min_epochs=6)
    not_skipped = {k: v for k, v in results.items() if not v["skipped"]}
    assert len(not_skipped) >= 1, (
        f"Expected ≥1 non-skipped RUNNING trial with 7 epochs; got: {results}"
    )
    for trial_num, res in not_skipped.items():
        assert res["estimated_value"] is not None
        assert isinstance(res["estimated_value"], float)


def test_estimate_incomplete_objectives_metric_override(large_session):
    """Passing metric= should override config-derived metric."""
    from retrosynformer.models_optuna import estimate_incomplete_objectives
    results = estimate_incomplete_objectives(
        _LARGE_DB, large_session, metric="valid_action_accuracy"
    )
    for res in results.values():
        assert res["metric"] == "valid_action_accuracy"


def test_estimate_incomplete_objectives_no_state_mutation(large_session):
    """estimate_incomplete_objectives must not modify any trial states."""
    from retrosynformer.models_optuna import estimate_incomplete_objectives, Trial
    before = {t.number: t.state for t in large_session.query(Trial).all()}
    estimate_incomplete_objectives(_LARGE_DB, large_session)
    after = {t.number: t.state for t in large_session.query(Trial).all()}
    assert before == after, "Trial states must not be modified by estimation"
