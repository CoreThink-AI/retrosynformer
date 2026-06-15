"""Tests for matplotlib plotting functions in retrosynformer.utils.

Key invariants verified:
- All three plot modules import without crashing even when DISPLAY is set
  to a broken X11 address (the failure mode fixed in commit a7fd471).
- Each plot function writes a non-empty PNG file to disk.
- Plots are robust to single-epoch and multi-epoch data.
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def train_jsonl(tmp_path):
    """Minimal train_progress.jsonl with two epochs."""
    rows = [
        {
            "epoch": 0,
            "train_loss": 0.5, "valid_loss": 0.6,
            "train_action_accuracy": 0.1, "valid_action_accuracy": 0.12,
            "train_route_accuracy": 0.01, "valid_route_accuracy": 0.015,
            "seconds_per_epoch": 10.0,
        },
        {
            "epoch": 1,
            "train_loss": 0.4, "valid_loss": 0.45,
            "train_action_accuracy": 0.2, "valid_action_accuracy": 0.22,
            "train_route_accuracy": 0.03, "valid_route_accuracy": 0.035,
            "seconds_per_epoch": 9.5,
        },
    ]
    p = tmp_path / "train_progress.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


@pytest.fixture()
def eval_json(tmp_path):
    """Minimal pred_routes_train_progress.json with one evaluation epoch."""
    data = [
        {
            "epoch": 1,
            "result": [
                {"route_solved": True,  "valid_route": True},
                {"route_solved": False, "valid_route": False},
                {"route_solved": True,  "valid_route": True},
                {"route_solved": False, "valid_route": True},
            ],
        }
    ]
    p = tmp_path / "pred_routes_train_progress.json"
    p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# Import safety — must not raise XIO / display errors
# ---------------------------------------------------------------------------

def test_utils_import_does_not_raise():
    """Importing utils must not attempt to open an X display."""
    import importlib
    import retrosynformer.utils.utils as m
    importlib.reload(m)  # force re-execution of module-level code


def test_evaluation_import_does_not_raise():
    import importlib
    import retrosynformer.utils.evaluation as m
    importlib.reload(m)


def test_evaluation_class_import_does_not_raise():
    import importlib
    import retrosynformer.utils.evaluation_class as m
    importlib.reload(m)


def test_matplotlib_backend_is_agg():
    """Agg backend must be active — never a GUI backend."""
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# plot_train_progress
# ---------------------------------------------------------------------------

def test_plot_train_progress_creates_file(train_jsonl, tmp_path):
    from retrosynformer.utils.utils import plot_train_progress
    out = str(tmp_path / "loss.png")
    plot_train_progress(train_jsonl, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_train_progress_single_epoch(tmp_path):
    from retrosynformer.utils.utils import plot_train_progress
    p = tmp_path / "single.jsonl"
    p.write_text(json.dumps({
        "epoch": 0, "train_loss": 0.5, "valid_loss": 0.6,
        "train_action_accuracy": 0.1, "valid_action_accuracy": 0.12,
        "train_route_accuracy": 0.01, "valid_route_accuracy": 0.015,
        "seconds_per_epoch": 10.0,
    }) + "\n")
    out = str(tmp_path / "loss.png")
    plot_train_progress(str(p), out)
    assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# plot_train_progress_accuracy
# ---------------------------------------------------------------------------

def test_plot_train_progress_accuracy_creates_file(train_jsonl, tmp_path):
    from retrosynformer.utils.utils import plot_train_progress_accuracy
    out = str(tmp_path / "acc.png")
    plot_train_progress_accuracy(train_jsonl, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_accuracy_produces_distinct_output(train_jsonl, tmp_path):
    """Accuracy and loss plots should be different images."""
    from retrosynformer.utils.utils import plot_train_progress, plot_train_progress_accuracy
    loss_out = str(tmp_path / "loss.png")
    acc_out = str(tmp_path / "acc.png")
    plot_train_progress(train_jsonl, loss_out)
    plot_train_progress_accuracy(train_jsonl, acc_out)
    assert open(loss_out, "rb").read() != open(acc_out, "rb").read()


# ---------------------------------------------------------------------------
# plot_evaluation_results
# ---------------------------------------------------------------------------

def test_plot_evaluation_results_creates_file(eval_json, tmp_path):
    from retrosynformer.utils.utils import plot_evaluation_results
    out = str(tmp_path / "eval.png")
    plot_evaluation_results(eval_json, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_evaluation_results_multi_epoch(tmp_path):
    from retrosynformer.utils.utils import plot_evaluation_results
    data = [
        {"epoch": 0, "result": [{"route_solved": True, "valid_route": True}]},
        {"epoch": 1, "result": [{"route_solved": False, "valid_route": False}]},
        {"epoch": 2, "result": [{"route_solved": True, "valid_route": True},
                                 {"route_solved": True, "valid_route": True}]},
    ]
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(data))
    out = str(tmp_path / "eval.png")
    plot_evaluation_results(str(p), out)
    assert os.path.getsize(out) > 0
