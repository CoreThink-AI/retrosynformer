def add_log_args(parser) -> None:
    """Add --debug and -v/--verbose log-level flags to an ArgumentParser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--debug", action="store_true", help="Set log level to DEBUG")
    group.add_argument("-v", "--verbose", action="store_true", help="Set log level to INFO")


def configure_logging(args) -> None:
    """Apply --debug / --verbose to the root logger. Call after parse_args()."""
    import logging
    if getattr(args, "debug", False):
        level = logging.DEBUG
    elif getattr(args, "verbose", False):
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def print_banner() -> None:
    """Print package version and GPU availability at the start of every rs-* command."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        ver = version("retrosynformer")
    except PackageNotFoundError:
        ver = "unknown"

    try:
        import torch
        # torch.version.hip is set when built against ROCm/HIP; None for CUDA builds
        rocm_ver = getattr(torch.version, "hip", None)
        rocm_str = f"ROCm {rocm_ver}" if rocm_ver else (f"CUDA {torch.version.cuda}" if torch.version.cuda else "no ROCm/CUDA")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gpu_str = f"GPU: {name}  |  {rocm_str}"
        else:
            gpu_str = f"GPU: not available (CPU only)  |  {rocm_str}"
    except ImportError:
        gpu_str = "GPU: torch not installed"

    print(f"retrosynformer {ver}  |  {gpu_str}", flush=True)
