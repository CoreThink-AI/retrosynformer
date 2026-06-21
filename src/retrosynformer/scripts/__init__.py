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
