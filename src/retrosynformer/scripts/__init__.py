def print_banner() -> None:
    """Print package version and GPU availability at the start of every rs-* command."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        ver = version("retrosynformer")
    except PackageNotFoundError:
        ver = "unknown"

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gpu_str = f"GPU: {name}"
        else:
            gpu_str = "GPU: not available (CPU only)"
    except ImportError:
        gpu_str = "GPU: torch not installed"

    print(f"retrosynformer {ver}  |  {gpu_str}", flush=True)
