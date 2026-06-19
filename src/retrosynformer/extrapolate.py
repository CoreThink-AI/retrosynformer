"""Multi-model objective-metric extrapolation for training curves.

Four models are fit to progressively wider windows of training history and
combined via inverse-variance weighting into a single estimate of the final
metric value at ``n_epochs - 1``.

Quick start
-----------
At the end of each training epoch::

    from retrosynformer.extrapolate import extrapolate_objective

    result = extrapolate_objective(values, n_epochs=200)
    if result:
        print(f"Estimated final score: {result['estimate']:.4f} ± {result['se']:.4f}")
        print(result["models"])  # per-model breakdown

Models
------
+----------+----------+-----------+-----------+
| name     | window   | transform | min pts   |
+==========+==========+===========+===========+
| linear   | last ¼   | none      | 3         |
| quadratic| last ½   | none      | 4         |
| cubic    | last ¾   | none      | 5         |
| log      | all      | log1p(x)  | 3         |
+----------+----------+-----------+-----------+

Each model is fit by ordinary least squares.  Models with too few window
points, non-finite predictions, or zero degrees of freedom are skipped.
The final ``estimate`` is the inverse-variance-weighted mean of successful
models; ``se`` is the combined standard error (``1 / sqrt(Σ weights)``).

Parameters
----------
values:
    Metric values, one per completed epoch (index = relative epoch 0, 1, …).
n_epochs:
    Total planned epochs for the trial.  Extrapolation target is
    ``n_epochs - 1``.
epochs:
    Optional explicit epoch numbers corresponding to *values*.  Useful when
    training was resumed and epoch indices are non-contiguous.  If omitted,
    indices ``0, 1, …, len(values) - 1`` are used.
min_points:
    Minimum number of *observed* epochs required to attempt any fit.
    Returns ``None`` when ``len(values) < min_points``.

Returns
-------
dict or None
    ``None`` when not enough data.  Otherwise::

        {
          "estimate":     float,   # weighted-average final-epoch estimate
          "se":           float,   # combined standard error
          "n_observed":   int,
          "target_epoch": int,     # n_epochs - 1
          "models": {
            "linear":    {"estimate": float|None, "se": float|None,
                          "weight": float, "n_points": int,
                          "success": bool, "skip_reason": str|None},
            "quadratic": {...},
            "cubic":     {...},
            "log":       {...},
          }
        }
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Internal per-model fit helpers
# ---------------------------------------------------------------------------

_EPS = 1e-10  # avoid divide-by-zero in weights


def _poly_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    target_x: float,
) -> tuple[float | None, float | None, str | None]:
    """Fit degree-*degree* polynomial; return (estimate, se, skip_reason)."""
    n = len(x)
    n_params = degree + 1
    dof = n - n_params
    if dof <= 0:
        return None, None, f"dof={dof} (need >{n_params} points for degree {degree})"
    try:
        coeffs = np.polyfit(x, y, degree)
    except np.linalg.LinAlgError as exc:
        return None, None, f"polyfit failed: {exc}"

    estimate = float(np.polyval(coeffs, target_x))
    if not math.isfinite(estimate):
        return None, None, "non-finite estimate"

    residuals = y - np.polyval(coeffs, x)
    rss = float(np.dot(residuals, residuals))
    se = math.sqrt(rss / dof)
    return estimate, se, None


def _log_fit(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
) -> tuple[float | None, float | None, str | None]:
    """Fit y = a·log1p(x) + b; return (estimate, se, skip_reason)."""
    n = len(x)
    dof = n - 2  # two parameters: a, b
    if dof <= 0:
        return None, None, f"dof={dof} (need >2 points for log model)"
    log_x = np.log1p(x)
    try:
        coeffs = np.polyfit(log_x, y, 1)  # degree-1 in log1p space
    except np.linalg.LinAlgError as exc:
        return None, None, f"log polyfit failed: {exc}"

    estimate = float(np.polyval(coeffs, math.log1p(target_x)))
    if not math.isfinite(estimate):
        return None, None, "non-finite estimate"

    residuals = y - np.polyval(coeffs, log_x)
    rss = float(np.dot(residuals, residuals))
    se = math.sqrt(rss / dof)
    return estimate, se, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extrapolate_objective(
    values: Sequence[float],
    n_epochs: int,
    *,
    epochs: Sequence[int] | None = None,
    min_points: int = 4,
) -> dict | None:
    """Extrapolate a training metric to the final epoch using four models.

    See module docstring for full parameter and return-value documentation.
    """
    n_observed = len(values)
    if n_observed < min_points:
        return None

    y_all = np.array(values, dtype=float)
    x_all = (
        np.array(epochs, dtype=float)
        if epochs is not None
        else np.arange(n_observed, dtype=float)
    )
    target_x = float(n_epochs - 1)

    # Slice boundaries (index into the observed array, not epoch numbers).
    q1 = 3 * n_observed // 4   # start of last ¼
    q2 = n_observed // 2       # start of last ½
    q3 = n_observed // 4       # start of last ¾

    model_specs = {
        "linear":    (x_all[q1:], y_all[q1:],  "poly", 1),
        "quadratic": (x_all[q2:], y_all[q2:],  "poly", 2),
        "cubic":     (x_all[q3:], y_all[q3:],  "poly", 3),
        "log":       (x_all,      y_all,        "log",  None),
    }

    model_results: dict[str, dict] = {}
    for name, (mx, my, kind, degree) in model_specs.items():
        n_pts = len(mx)
        if kind == "poly":
            est, se, reason = _poly_fit(mx, my, degree, target_x)
        else:
            est, se, reason = _log_fit(mx, my, target_x)

        model_results[name] = {
            "estimate":    est,
            "se":          se,
            "weight":      0.0,
            "n_points":    n_pts,
            "success":     est is not None,
            "skip_reason": reason,
        }

    # Inverse-variance weighting over successful models.
    good = {k: v for k, v in model_results.items() if v["success"]}
    if not good:
        return None

    raw_weights = {k: 1.0 / (v["se"] ** 2 + _EPS) for k, v in good.items()}
    total_w = sum(raw_weights.values())
    norm_weights = {k: w / total_w for k, w in raw_weights.items()}

    estimate = sum(norm_weights[k] * good[k]["estimate"] for k in good)
    combined_se = 1.0 / math.sqrt(total_w)

    for k, w in norm_weights.items():
        model_results[k]["weight"] = w

    return {
        "estimate":     estimate,
        "se":           combined_se,
        "n_observed":   n_observed,
        "target_epoch": int(target_x),
        "models":       model_results,
    }
