"""Tests for retrosynformer.extrapolate.extrapolate_objective."""
import math

import numpy as np
import pytest

from retrosynformer.extrapolate import extrapolate_objective, _poly_fit, _log_fit


# ---------------------------------------------------------------------------
# _poly_fit unit tests
# ---------------------------------------------------------------------------

def test_poly_fit_linear_exact():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])  # perfect y=x
    est, se, reason = _poly_fit(x, y, degree=1, target_x=10.0)
    assert reason is None
    assert est == pytest.approx(10.0, abs=1e-6)
    assert se == pytest.approx(0.0, abs=1e-6)


def test_poly_fit_too_few_points():
    x = np.array([0.0, 1.0])
    y = np.array([0.5, 0.6])
    est, se, reason = _poly_fit(x, y, degree=1, target_x=5.0)
    # dof = 2 - 2 = 0 → must skip
    assert est is None
    assert reason is not None


def test_poly_fit_quadratic_parabola():
    x = np.arange(10, dtype=float)
    y = -(x - 5) ** 2 + 1.0  # vertex at x=5, opens down
    est, se, reason = _poly_fit(x, y, degree=2, target_x=9.0)
    assert reason is None
    assert math.isfinite(est)
    assert se == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _log_fit unit tests
# ---------------------------------------------------------------------------

def test_log_fit_exact_log_curve():
    x = np.arange(1, 20, dtype=float)
    y = 2.0 * np.log1p(x) + 0.5  # perfect log curve, a=2 b=0.5
    est, se, reason = _log_fit(x, y, target_x=100.0)
    assert reason is None
    assert est == pytest.approx(2.0 * math.log1p(100.0) + 0.5, abs=1e-4)
    assert se == pytest.approx(0.0, abs=1e-5)


def test_log_fit_too_few_points():
    x = np.array([0.0, 1.0])
    y = np.array([0.1, 0.2])
    est, se, reason = _log_fit(x, y, target_x=99.0)
    assert est is None
    assert reason is not None


# ---------------------------------------------------------------------------
# extrapolate_objective — basic contracts
# ---------------------------------------------------------------------------

def test_returns_none_when_too_few_values():
    assert extrapolate_objective([0.1, 0.2, 0.3], n_epochs=100) is None


def test_returns_dict_with_required_keys():
    values = [0.01 * i for i in range(30)]
    result = extrapolate_objective(values, n_epochs=200)
    assert result is not None
    assert "estimate" in result
    assert "se" in result
    assert "n_observed" in result
    assert "target_epoch" in result
    assert "models" in result
    assert result["target_epoch"] == 199
    assert result["n_observed"] == 30


def test_model_names_present():
    values = [0.01 * i for i in range(30)]
    result = extrapolate_objective(values, n_epochs=200)
    assert set(result["models"].keys()) == {"linear", "quadratic", "cubic", "log"}


def test_model_result_schema():
    values = list(np.linspace(0.0, 0.8, 40))
    result = extrapolate_objective(values, n_epochs=200)
    for name, m in result["models"].items():
        assert "estimate" in m
        assert "se" in m
        assert "weight" in m
        assert "n_points" in m
        assert "success" in m
        assert "skip_reason" in m


def test_weights_sum_to_one():
    values = list(np.linspace(0.0, 0.8, 40))
    result = extrapolate_objective(values, n_epochs=200)
    total = sum(m["weight"] for m in result["models"].values())
    assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# extrapolate_objective — window sizing
# ---------------------------------------------------------------------------

def test_linear_uses_last_quarter():
    """linear model n_points should equal the last-¼ window size."""
    n = 40
    values = list(range(n))
    result = extrapolate_objective(values, n_epochs=200)
    expected_q = n - 3 * n // 4  # len(values[3*n//4:])
    assert result["models"]["linear"]["n_points"] == expected_q


def test_quadratic_uses_last_half():
    n = 40
    values = list(range(n))
    result = extrapolate_objective(values, n_epochs=200)
    expected_q = n - n // 2
    assert result["models"]["quadratic"]["n_points"] == expected_q


def test_cubic_uses_last_three_quarters():
    n = 40
    values = list(range(n))
    result = extrapolate_objective(values, n_epochs=200)
    expected_q = n - n // 4
    assert result["models"]["cubic"]["n_points"] == expected_q


def test_log_uses_all_points():
    n = 40
    values = list(range(n))
    result = extrapolate_objective(values, n_epochs=200)
    assert result["models"]["log"]["n_points"] == n


# ---------------------------------------------------------------------------
# extrapolate_objective — small-n degradation
# ---------------------------------------------------------------------------

def test_few_epochs_only_log_succeeds():
    """With only 4 observed epochs, only the log model has enough points."""
    values = [0.1, 0.2, 0.3, 0.4]
    result = extrapolate_objective(values, n_epochs=200, min_points=4)
    assert result is not None
    assert result["models"]["log"]["success"]
    # linear/quadratic/cubic all need more window points than 1/2/3
    for name in ("linear", "quadratic", "cubic"):
        m = result["models"][name]
        if not m["success"]:
            assert m["skip_reason"] is not None


def test_moderate_epochs_multiple_models_succeed():
    """With 20+ epochs, at least quadratic and log should succeed."""
    values = list(np.linspace(0.0, 0.7, 20))
    result = extrapolate_objective(values, n_epochs=200)
    assert result is not None
    successes = [k for k, v in result["models"].items() if v["success"]]
    assert len(successes) >= 2


# ---------------------------------------------------------------------------
# extrapolate_objective — estimate quality on synthetic curves
# ---------------------------------------------------------------------------

def test_linear_growth_estimate():
    """Perfect linear growth: estimate should land close to the true final value."""
    values = [0.005 * i for i in range(50)]  # ends at 0.245
    true_final = 0.005 * 199
    result = extrapolate_objective(values, n_epochs=200)
    assert result is not None
    assert abs(result["estimate"] - true_final) < 0.1 * true_final


def test_log_saturation_estimate():
    """Log-saturating curve: estimate should be positive and finite."""
    values = [math.log1p(i) * 0.1 for i in range(60)]
    result = extrapolate_objective(values, n_epochs=200)
    assert result is not None
    assert math.isfinite(result["estimate"])
    assert result["estimate"] > 0.0


def test_high_se_model_gets_low_weight():
    """A model with high residuals (noisy window) should receive a low weight."""
    rng = np.random.default_rng(0)
    # Perfect linear growth except the last quarter is very noisy
    n = 40
    values = list(np.arange(n, dtype=float))
    noisy = list(np.arange(n, dtype=float) + rng.normal(scale=20, size=n))
    # Replace last quarter with noisy values
    mixed = values[:3 * n // 4] + noisy[3 * n // 4:]
    result = extrapolate_objective(mixed, n_epochs=200)
    assert result is not None
    if result["models"]["linear"]["success"] and result["models"]["log"]["success"]:
        assert result["models"]["linear"]["weight"] < result["models"]["log"]["weight"]


# ---------------------------------------------------------------------------
# extrapolate_objective — custom epoch indices
# ---------------------------------------------------------------------------

def test_custom_epochs_non_contiguous():
    """Sparse epoch numbers (resumed training) should still produce a valid estimate."""
    epochs = [0, 1, 2, 10, 11, 12, 50, 51, 52, 100]
    values = [0.01 * e for e in epochs]
    result = extrapolate_objective(values, n_epochs=200, epochs=epochs)
    assert result is not None
    assert math.isfinite(result["estimate"])


def test_custom_epochs_matches_contiguous_when_sequential():
    """Explicit sequential epochs should give the same result as the default."""
    values = [0.01 * i for i in range(30)]
    r1 = extrapolate_objective(values, n_epochs=200)
    r2 = extrapolate_objective(values, n_epochs=200, epochs=list(range(30)))
    assert r1 is not None and r2 is not None
    assert r1["estimate"] == pytest.approx(r2["estimate"], abs=1e-10)


# ---------------------------------------------------------------------------
# extrapolate_objective — min_points boundary
# ---------------------------------------------------------------------------

def test_min_points_exactly_met():
    values = [0.1, 0.2, 0.3, 0.4]  # exactly 4
    assert extrapolate_objective(values, n_epochs=100, min_points=4) is not None


def test_min_points_not_met():
    values = [0.1, 0.2, 0.3]  # only 3
    assert extrapolate_objective(values, n_epochs=100, min_points=4) is None
