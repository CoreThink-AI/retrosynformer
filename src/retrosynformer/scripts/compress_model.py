"""rs-compress: offline compression and dtype/format conversion for RetroSynFormer models.

Compresses a local model file so rs-upload can pick it up and infer the codec
automatically from the output filename, without needing --codec at upload time.

The output path is derived from the input by applying format and codec extensions:
  model.pth --codec gz             → model.pth.gz
  model.pth --codec bz2            → model.pth.bz2
  model.pth --format safetensors   → model.safetensors
  model.pth --dtype fp16 --codec xz → model.pth.xz  (fp16 weights inside)
  model.pth --format safetensors --dtype fp16 --codec gz → model.safetensors.gz

Usage::

    rs-compress results/hypertune-large-23-layer/trial_003/model.pth --codec bz2
    rs-compress results/.../model.pth --format safetensors --dtype fp16 --codec gz
    rs-compress results/.../model.pth --codec xz --output /other/path/model.pth.xz
"""
import argparse
import sys
import time
from pathlib import Path

from retrosynformer.compression import (
    CODECS,
    DTYPES,
    FORMATS,
    codec_extension,
    detect_codec,
    save_model,
)


def _derived_output(src: Path, fmt: str, codec: "str | None") -> Path:
    """Compute the default output path from src, format, and codec."""
    # Strip any existing codec extension from the source name first.
    name = src.name
    existing_codec = detect_codec(src)
    if existing_codec is not None:
        name = name[: -len(codec_extension(existing_codec))]

    # Change the format extension if needed.
    stem, _ext = name.rsplit(".", 1) if "." in name else (name, "")
    new_ext = ".safetensors" if fmt == "safetensors" else f".{_ext}" if _ext else ".pth"
    new_name = stem + new_ext

    # Append codec extension.
    if codec is not None:
        new_name += codec_extension(codec)

    return src.parent / new_name


def main():
    parser = argparse.ArgumentParser(
        description="Compress (and optionally convert) a RetroSynFormer model file offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output filename is derived automatically from the source:\n"
            "  model.pth --codec gz             → model.pth.gz\n"
            "  model.pth --codec bz2            → model.pth.bz2\n"
            "  model.pth --codec xz             → model.pth.xz\n"
            "  model.pth --format safetensors   → model.safetensors\n"
            "  model.pth --dtype fp16 --codec gz → model.pth.gz  (fp16 weights)\n"
            "  model.pth --format safetensors --dtype fp16 --codec gz → model.safetensors.gz\n\n"
            "After compression, upload with:\n"
            "  rs-upload model.pth.bz2 gs://bucket/models/my-model/\n"
            "(codec and filename are inferred automatically)\n"
        ),
    )
    parser.add_argument("local_path", type=Path, help="Source model file (.pth or .safetensors)")
    parser.add_argument(
        "--codec", choices=list(CODECS), metavar="CODEC",
        help="Compression algorithm: gz (fast), bz2 (better ratio), xz (best ratio/slowest)",
    )
    parser.add_argument(
        "--format", dest="fmt", choices=list(FORMATS), default=None,
        help="Output format: pth (PyTorch, default) or safetensors",
    )
    parser.add_argument(
        "--dtype", choices=list(DTYPES),
        help="Cast float32 weights: fp16 or bfloat16 halve file size (irreversible)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, metavar="PATH",
        help="Explicit output path (default: derived from source filename)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output file if it already exists",
    )
    args = parser.parse_args()

    if not args.local_path.exists():
        sys.exit(f"Source file not found: {args.local_path}")

    # Resolve format: default to same as source.
    src_fmt = "safetensors" if args.local_path.name.replace(
        *([codec_extension(c) for c in CODECS if args.local_path.name.endswith(codec_extension(c))] or ["", ""])
    ).endswith(".safetensors") else "pth"
    fmt = args.fmt or src_fmt

    # Derive or validate output path.
    if args.output is not None:
        out_path = args.output
        # Infer codec from output path if --codec not given.
        codec = args.codec or detect_codec(out_path)
    else:
        codec = args.codec
        out_path = _derived_output(args.local_path, fmt, codec)

    if out_path.exists() and not args.overwrite:
        sys.exit(
            f"Output already exists: {out_path}\n"
            f"Use --overwrite to replace it."
        )

    if args.dtype is None and codec is None and fmt == src_fmt:
        sys.exit(
            "Nothing to do — specify at least one of --codec, --dtype, or --format."
        )

    src_size = args.local_path.stat().st_size
    print(f"Source : {args.local_path}  ({src_size / 1e9:.3f} GB)", flush=True)
    if args.dtype:
        print(f"Dtype  : {args.dtype} (float32 weights will be downcast)", flush=True)
    if fmt != src_fmt:
        print(f"Format : {src_fmt} → {fmt}", flush=True)
    if codec:
        print(f"Codec  : {codec}", flush=True)
    print(f"Output : {out_path}", flush=True)

    import torch
    t0 = time.perf_counter()
    print("Loading model …", flush=True)
    state_dict = torch.load(args.local_path, map_location="cpu", weights_only=True)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    print("Saving …", flush=True)
    # save_model handles: dtype cast → format serialisation → compression.
    # Pass the path without the codec extension; save_model appends it and returns final path.
    base_path = out_path
    if codec is not None and out_path.suffix == codec_extension(codec):
        base_path = out_path.with_suffix("")  # strip codec ext; save_model re-adds it
    final = save_model(state_dict, base_path, fmt=fmt, dtype=args.dtype, codec=codec)
    elapsed = time.perf_counter() - t1
    out_size = final.stat().st_size
    ratio = src_size / out_size if out_size else 0.0
    print(
        f"  done in {elapsed:.1f}s  "
        f"{src_size / 1e9:.3f} GB → {out_size / 1e9:.3f} GB  "
        f"(ratio {ratio:.2f}×)",
        flush=True,
    )
    print(f"Written: {final}", flush=True)
