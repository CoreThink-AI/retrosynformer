"""Compression, serialization format, and dtype utilities for RetroSynFormer model files.

Supported codecs : gz (gzip), bz2 (bzip2), xz (lzma/xz)
Supported formats: pth (PyTorch), safetensors (HuggingFace SafeTensors)
Supported dtypes : fp32 (float32), fp16 (float16), bfloat16
"""
import bz2
import gzip
import io
import lzma
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

CODECS = ("gz", "bz2", "xz")
FORMATS = ("pth", "safetensors")
DTYPES = ("fp32", "fp16", "bfloat16")

_EXT_TO_CODEC: dict[str, str] = {".gz": "gz", ".bz2": "bz2", ".xz": "xz"}
_CODEC_TO_EXT: dict[str, str] = {"gz": ".gz", "bz2": ".bz2", "xz": ".xz"}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Floating dtypes that will be downcast by cast_state_dict.
_FLOAT_DTYPES = frozenset({torch.float32, torch.float64})


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

def codec_extension(codec: str) -> str:
    """Return the file extension for a codec: ``.gz``, ``.bz2``, or ``.xz``."""
    try:
        return _CODEC_TO_EXT[codec]
    except KeyError:
        raise ValueError(f"Unknown codec {codec!r}; expected one of {CODECS}")


def detect_codec(path: "Path | str") -> "str | None":
    """Detect compression codec from file extension; return ``None`` if uncompressed."""
    return _EXT_TO_CODEC.get(Path(path).suffix)


def compress_file(src: Path, dst: Path, codec: str, *, level: "int | None" = None) -> Path:
    """Compress *src* → *dst* using *codec*. Return *dst*. *src* is not modified.

    Default compression levels: gz=1 (fast, suited for network transfer),
    bz2=9 (standard), xz=6 (lzma preset 6).  Pass *level* to override.

    For ``gz``, output is always written with ``mtime=0`` so identical inputs
    produce bit-identical output across separate invocations — required for the
    ``--skip-existing`` MD5 check in ``rs-upload``.
    """
    if codec not in _CODEC_TO_EXT:
        raise ValueError(f"Unknown codec {codec!r}; expected one of {CODECS}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if codec == "gz":
        lvl = level if level is not None else 1
        with open(src, "rb") as f_in, open(dst, "wb") as raw_out:
            # filename="" prevents the destination path from being embedded in the
            # gzip header, keeping output byte-identical across separate invocations
            # (required for --skip-existing MD5 checks to match reliably).
            with gzip.GzipFile(fileobj=raw_out, mode="wb", compresslevel=lvl, mtime=0, filename="") as gz:
                shutil.copyfileobj(f_in, gz)
    elif codec == "bz2":
        lvl = level if level is not None else 9
        with open(src, "rb") as f_in, bz2.open(dst, "wb", compresslevel=lvl) as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:  # xz
        preset = level if level is not None else 6
        with open(src, "rb") as f_in, lzma.open(dst, "wb", preset=preset) as f_out:
            shutil.copyfileobj(f_in, f_out)

    return dst


def decompress_file(src: Path, dst: Path) -> Path:
    """Decompress *src* → *dst*, auto-detecting codec from extension. Return *dst*."""
    codec = detect_codec(src)
    if codec is None:
        raise ValueError(
            f"Cannot detect codec from {src.name!r} — expected one of "
            f"{list(_EXT_TO_CODEC)}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if codec == "gz":
        with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    elif codec == "bz2":
        with bz2.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:  # xz
        with lzma.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return dst


# ---------------------------------------------------------------------------
# Dtype conversion
# ---------------------------------------------------------------------------

def cast_state_dict(state_dict: "dict[str, Any]", dtype: str) -> "dict[str, Any]":
    """Return a new state dict with float32/float64 tensors cast to *dtype*.

    Integer, bool, and already-halved tensors are left unchanged.  The original
    dict is never mutated.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {DTYPES}")
    target = _DTYPE_MAP[dtype]
    return {
        k: v.to(target) if isinstance(v, torch.Tensor) and v.dtype in _FLOAT_DTYPES else v
        for k, v in state_dict.items()
    }


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    """Detect model format from path, stripping any trailing compression extension."""
    p = path
    if p.suffix in _EXT_TO_CODEC:
        p = p.with_suffix("")
    return "safetensors" if p.suffix == ".safetensors" else "pth"


def save_model(
    state_dict: dict,
    path: Path,
    *,
    fmt: str = "pth",
    dtype: "str | None" = None,
    codec: "str | None" = None,
) -> Path:
    """Save *state_dict* to *path*, optionally converting dtype and/or compressing.

    The pipeline is: dtype cast → format serialisation → compression.

    Args:
        state_dict: PyTorch state dict (or any dict serialisable by torch.save).
        path: Destination path **without** a compression extension — the codec
              extension (``.gz``, ``.bz2``, ``.xz``) is appended automatically
              when *codec* is set.  Should end in ``.pth`` or ``.safetensors``.
        fmt: ``"pth"`` (torch.save) or ``"safetensors"``.
        dtype: ``None`` keeps original; ``"fp16"``/``"bfloat16"`` casts float32
               tensors (permanent precision reduction).
        codec: ``None`` for uncompressed; ``"gz"``/``"bz2"``/``"xz"`` to compress.

    Returns:
        The final file path (same as *path* if *codec* is ``None``; otherwise
        *path* with the codec extension appended).
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}; expected one of {FORMATS}")

    if dtype is not None:
        state_dict = cast_state_dict(state_dict, dtype)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file as _st_save
        except ImportError:
            raise ImportError("safetensors not installed; run: pip install safetensors")
        tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        _st_save(tensor_dict, str(path))
    else:
        torch.save(state_dict, path)

    if codec is not None:
        compressed = path.with_suffix(path.suffix + codec_extension(codec))
        compress_file(path, compressed, codec)
        path.unlink()
        return compressed

    return path


def load_model(path: Path, *, map_location=None) -> dict:
    """Load a state dict from *path*, auto-detecting format and decompressing as needed.

    Handles ``.pth``, ``.safetensors``, and any of those with ``.gz``/``.bz2``/``.xz``
    appended.  A compressed file is decompressed to a temporary file and that temp
    file is deleted after loading.
    """
    path = Path(path)
    codec = detect_codec(path)

    if codec is not None:
        inner_suffix = path.with_suffix("").suffix  # e.g. ".pth" from "model.pth.gz"
        with tempfile.NamedTemporaryFile(suffix=inner_suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            decompress_file(path, tmp_path)
            return load_model(tmp_path, map_location=map_location)
        finally:
            tmp_path.unlink(missing_ok=True)

    fmt = _detect_format(path)
    if fmt == "safetensors":
        try:
            from safetensors.torch import load_file as _st_load
        except ImportError:
            raise ImportError("safetensors not installed; run: pip install safetensors")
        device = str(map_location) if map_location is not None else "cpu"
        return _st_load(str(path), device=device)

    return torch.load(path, map_location=map_location, weights_only=True)


def is_valid_model_file(path: "Path | str") -> bool:
    """Return ``True`` if *path* is a readable model checkpoint.

    Supports both PyTorch (``.pth`` — ZIP archive) and SafeTensors formats.
    Reads only headers/metadata, not the full weight tensors.
    """
    import zipfile
    path = Path(path)
    fmt = _detect_format(path)
    try:
        if fmt == "safetensors":
            try:
                from safetensors import safe_open
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    _ = list(f.keys())
                return True
            except ImportError:
                # safetensors not installed: check magic bytes manually.
                # A valid safetensors file starts with an 8-byte little-endian
                # uint64 giving the JSON header length, then a JSON object.
                with open(path, "rb") as f:
                    header_len = int.from_bytes(f.read(8), "little")
                return 0 < header_len < 100_000_000
        else:
            with zipfile.ZipFile(path, "r") as z:
                z.namelist()
            return True
    except Exception as exc:
        print(f"  Model validation failed for {path}: {exc}", flush=True)
        return False
