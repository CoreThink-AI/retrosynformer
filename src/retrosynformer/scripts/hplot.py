"""hplot — dispatch to RetroSynFormer plot commands.

On a CPU-only machine, syncs results/ from the remote GPU server (GPU_HOST in
.env) before plotting so that train_progress.jsonl files are up to date.

Usage
-----
    hplot [learning] [options]    # rs-plot-learning-curves (default subcommand)
    hplot training   [options]    # rs-plot-learning-curves
    hplot progress   [options]    # rs-plot-learning-curves
    hplot learn      [options]    # rs-plot-learning-curves
    hplot train      [options]    # rs-plot-learning-curves
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)

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


def _has_gpu() -> bool:
    """Return True if a CUDA or ROCm GPU is available in this environment."""
    try:
        import torch
        return torch.cuda.is_available() or bool(getattr(torch.version, "hip", None))
    except ImportError:
        return False


def _load_dotenv_simple() -> None:
    """Load KEY=VALUE pairs from the nearest .env file into os.environ (no-op if missing)."""
    path = os.path.abspath(".")
    for _ in range(6):  # search up to 6 levels up
        candidate = os.path.join(path, ".env")
        if os.path.isfile(candidate):
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            return
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent


def _configure_logging_from_argv() -> None:
    """Set root log level from --debug / -v / --verbose in sys.argv before parse_args runs."""
    argv = sys.argv[1:]
    if "--debug" in argv:
        level = logging.DEBUG
    elif "-v" in argv or "--verbose" in argv:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _sync_results_from_remote() -> None:
    """Rsync results/ from GPU_HOST (set in .env) into the local results/ dir."""
    _load_dotenv_simple()
    gpu_host = os.environ.get("GPU_HOST", "").strip()
    if not gpu_host:
        logger.info("hplot: GPU_HOST not set in .env — skipping remote sync")
        return

    # GPU_HOST is an SCP-style remote path to the project root, e.g.
    # hobs@taco:code/corethink/retrosynformer/
    if not gpu_host.endswith("/"):
        gpu_host += "/"
    src = f"{gpu_host}results/"
    dst = "results/"

    logger.info("hplot: CPU-only environment — syncing %s → %s", src, dst)
    from retrosynformer.rsync import sync
    rc = sync(src, dst)
    if rc != 0:
        logger.warning("hplot: rsync exited with code %d — proceeding with local data", rc)


def main() -> None:
    # Strip a recognised subcommand; otherwise treat everything as learning-curves args (default).
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Configure logging before the sync so --verbose/-v controls rsync output.
    _configure_logging_from_argv()

    if not _has_gpu():
        _sync_results_from_remote()

    from retrosynformer.scripts.plot_learning_curves import main as _main
    _main()
