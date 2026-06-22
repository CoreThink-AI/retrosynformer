"""retrosynformer package.

Module-level setup runs here before any submodule (including torch) is imported.
"""
import os

# ---------------------------------------------------------------------------
# ROCm / hipBLASLt compatibility
#
# PyTorch probes hipBLASLt at startup and emits a UserWarning when the GPU
# architecture doesn't support it, then falls back to hipBLAS automatically.
# Setting TORCH_BLAS_PREFER_HIPBLASLT=0 skips the probe entirely.
#
# Detection strategy (in priority order):
#  1. TORCH_BLAS_PREFER_HIPBLASLT already set → respect it, do nothing.
#  2. HSA_OVERRIDE_GFX_VERSION is set → the architecture is non-standard
#     (MI-class cards that support hipBLASLt don't need this override).
#  3. rocminfo → parse the first gfx name; disable if not in the known-good set.
#  4. /opt/rocm absent → not a ROCm system, skip.
# ---------------------------------------------------------------------------

_HIPBLASLT_SUPPORTED_GFX = frozenset({
    "gfx90a",               # Instinct MI200 / MI210 / MI250
    "gfx940", "gfx941", "gfx942",  # Instinct MI300A / MI300X
})


def _disable_hipblaslt_if_unsupported() -> None:
    if "TORCH_BLAS_PREFER_HIPBLASLT" in os.environ:
        return  # user already made an explicit choice

    if not os.path.isdir("/opt/rocm"):
        return  # not a ROCm system

    # HSA_OVERRIDE_GFX_VERSION is only needed for architectures that ROCm
    # doesn't recognise natively (e.g. Strix Halo / RDNA iGPUs).  Those are
    # never in the hipBLASLt supported set, so disable immediately.
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
        return

    # Fall back to rocminfo to identify the GPU architecture.
    try:
        import re
        import subprocess
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=5
        ).stdout
        m = re.search(r"\bgfx\w+\b", out)
        if m and m.group() not in _HIPBLASLT_SUPPORTED_GFX:
            os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    except Exception:
        pass  # rocminfo unavailable or timed out — leave env var unset


_disable_hipblaslt_if_unsupported()
