"""hplot — dispatch to RetroSynFormer plot commands.

Usage
-----
    hplot learning [options]      # rs-plot-learning-curves
    hplot training [options]      # rs-plot-learning-curves
    hplot progress [options]      # rs-plot-learning-curves
    hplot learn    [options]      # rs-plot-learning-curves
    hplot train    [options]      # rs-plot-learning-curves
"""
import sys

_LEARNING_CURVE_ALIASES = {
    "learning-curve",
    "learning-curves",
    "learning",
    "learn",
    "progress",
    "training",
    "train",
}

_SUBCOMMANDS = _LEARNING_CURVE_ALIASES


def main() -> None:
    # Strip a recognised subcommand; otherwise treat everything as learning-curves args (default).
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    from retrosynformer.scripts.plot_learning_curves import main as _main
    _main()
