"""Tests for retrosynformer.compression."""
import tempfile
from pathlib import Path

import pytest
import torch

from retrosynformer.compression import (
    CODECS,
    DTYPES,
    FORMATS,
    cast_state_dict,
    codec_extension,
    compress_file,
    decompress_file,
    detect_codec,
    is_valid_model_file,
    load_model,
    save_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def small_state_dict():
    """Tiny state dict covering the dtype variety present in real model checkpoints."""
    return {
        "weight": torch.randn(8, 4, dtype=torch.float32),
        "bias": torch.randn(8, dtype=torch.float32),
        "large_weight": torch.randn(32, 16, dtype=torch.float64),
        "running_mean": torch.zeros(8, dtype=torch.float32),
        "num_batches_tracked": torch.tensor(42, dtype=torch.long),
        "mask": torch.ones(4, dtype=torch.bool),
        "fp16_param": torch.randn(4, dtype=torch.float16),
    }


# ---------------------------------------------------------------------------
# detect_codec / codec_extension
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("model.pth.gz", "gz"),
    ("model.pth.bz2", "bz2"),
    ("model.pth.xz", "xz"),
    ("model.pth", None),
    ("model.safetensors", None),
    ("model.safetensors.gz", "gz"),
    ("model.safetensors.bz2", "bz2"),
    ("model.safetensors.xz", "xz"),
])
def test_detect_codec(filename, expected):
    assert detect_codec(Path(filename)) == expected


@pytest.mark.parametrize("codec,ext", [("gz", ".gz"), ("bz2", ".bz2"), ("xz", ".xz")])
def test_codec_extension(codec, ext):
    assert codec_extension(codec) == ext


def test_codec_extension_invalid():
    with pytest.raises(ValueError, match="Unknown codec"):
        codec_extension("zstd")


# ---------------------------------------------------------------------------
# compress_file / decompress_file — round-trip for each codec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codec", CODECS)
def test_compress_decompress_roundtrip(tmp_dir, codec):
    data = b"hello retrosynformer " * 5_000
    src = tmp_dir / "source.bin"
    src.write_bytes(data)
    compressed = tmp_dir / f"source.bin{codec_extension(codec)}"
    dst = tmp_dir / "restored.bin"

    compress_file(src, compressed, codec)
    assert compressed.exists() and compressed.stat().st_size > 0
    assert src.exists(), "compress_file must not remove the source"

    decompress_file(compressed, dst)
    assert dst.read_bytes() == data


@pytest.mark.parametrize("codec", CODECS)
def test_compress_reduces_size_for_repetitive_data(tmp_dir, codec):
    data = b"aaaa" * 200_000
    src = tmp_dir / "rep.bin"
    src.write_bytes(data)
    compressed = tmp_dir / f"rep.bin{codec_extension(codec)}"
    compress_file(src, compressed, codec)
    assert compressed.stat().st_size < src.stat().st_size // 4


def test_gz_output_is_deterministic(tmp_dir):
    """Identical input → identical gz bytes (mtime=0 is set)."""
    data = b"deterministic " * 10_000
    src = tmp_dir / "data.bin"
    src.write_bytes(data)
    out1 = tmp_dir / "a.gz"
    out2 = tmp_dir / "b.gz"
    compress_file(src, out1, "gz")
    compress_file(src, out2, "gz")
    assert out1.read_bytes() == out2.read_bytes()


def test_compress_custom_level(tmp_dir):
    data = b"level test " * 50_000
    src = tmp_dir / "src.bin"
    src.write_bytes(data)
    # level 1 (fast) and level 9 (slow) should both decompress to original
    for lvl in (1, 9):
        compressed = tmp_dir / f"src_{lvl}.gz"
        dst = tmp_dir / f"dst_{lvl}.bin"
        compress_file(src, compressed, "gz", level=lvl)
        decompress_file(compressed, dst)
        assert dst.read_bytes() == data


def test_decompress_unknown_extension_raises(tmp_dir):
    f = tmp_dir / "model.pth"
    f.write_bytes(b"data")
    with pytest.raises(ValueError, match="Cannot detect codec"):
        decompress_file(f, tmp_dir / "out.pth")


# ---------------------------------------------------------------------------
# cast_state_dict
# ---------------------------------------------------------------------------

def test_cast_to_fp16_casts_float32_and_float64(small_state_dict):
    out = cast_state_dict(small_state_dict, "fp16")
    assert out["weight"].dtype == torch.float16
    assert out["bias"].dtype == torch.float16
    assert out["large_weight"].dtype == torch.float16


def test_cast_to_bfloat16(small_state_dict):
    out = cast_state_dict(small_state_dict, "bfloat16")
    assert out["weight"].dtype == torch.bfloat16
    assert out["large_weight"].dtype == torch.bfloat16


def test_cast_leaves_non_float_tensors_unchanged(small_state_dict):
    out = cast_state_dict(small_state_dict, "fp16")
    assert out["num_batches_tracked"].dtype == torch.long
    assert out["mask"].dtype == torch.bool


def test_cast_leaves_already_halved_tensors_unchanged(small_state_dict):
    out = cast_state_dict(small_state_dict, "fp16")
    assert out["fp16_param"].dtype == torch.float16


def test_cast_fp16_values_within_tolerance(small_state_dict):
    out = cast_state_dict(small_state_dict, "fp16")
    orig = small_state_dict["weight"].float()
    back = out["weight"].float()
    assert torch.allclose(orig, back, atol=1e-2), "fp16 precision drift too large"


def test_cast_bfloat16_values_within_tolerance(small_state_dict):
    out = cast_state_dict(small_state_dict, "bfloat16")
    orig = small_state_dict["weight"].float()
    back = out["weight"].float()
    assert torch.allclose(orig, back, atol=1e-2)


def test_cast_fp32_is_noop(small_state_dict):
    out = cast_state_dict(small_state_dict, "fp32")
    assert torch.equal(out["weight"], small_state_dict["weight"])


def test_cast_does_not_mutate_original(small_state_dict):
    orig_dtype = small_state_dict["weight"].dtype
    cast_state_dict(small_state_dict, "fp16")
    assert small_state_dict["weight"].dtype == orig_dtype


def test_cast_invalid_dtype_raises(small_state_dict):
    with pytest.raises(ValueError, match="Unknown dtype"):
        cast_state_dict(small_state_dict, "int8")


# ---------------------------------------------------------------------------
# save_model / load_model — PyTorch .pth format
# ---------------------------------------------------------------------------

def test_save_load_pth_roundtrip(tmp_dir, small_state_dict):
    path = tmp_dir / "model.pth"
    final = save_model(small_state_dict, path, fmt="pth")
    assert final == path
    loaded = load_model(path)
    assert set(loaded.keys()) == set(small_state_dict.keys())
    assert torch.equal(loaded["weight"], small_state_dict["weight"])
    assert torch.equal(loaded["num_batches_tracked"], small_state_dict["num_batches_tracked"])


def test_save_pth_with_fp16(tmp_dir, small_state_dict):
    path = tmp_dir / "model.pth"
    save_model(small_state_dict, path, fmt="pth", dtype="fp16")
    loaded = load_model(path)
    assert loaded["weight"].dtype == torch.float16
    assert loaded["num_batches_tracked"].dtype == torch.long


@pytest.mark.parametrize("codec", CODECS)
def test_save_load_pth_compressed(tmp_dir, small_state_dict, codec):
    path = tmp_dir / "model.pth"
    final = save_model(small_state_dict, path, fmt="pth", codec=codec)
    assert final.suffix == codec_extension(codec)
    assert not path.exists(), "intermediate .pth should be cleaned up after compression"
    loaded = load_model(final)
    assert torch.equal(loaded["weight"], small_state_dict["weight"])


@pytest.mark.parametrize("codec", CODECS)
def test_save_load_pth_fp16_compressed(tmp_dir, small_state_dict, codec):
    path = tmp_dir / "model.pth"
    final = save_model(small_state_dict, path, fmt="pth", dtype="fp16", codec=codec)
    loaded = load_model(final)
    assert loaded["weight"].dtype == torch.float16
    assert loaded["num_batches_tracked"].dtype == torch.long


# ---------------------------------------------------------------------------
# save_model / load_model — SafeTensors format
# ---------------------------------------------------------------------------

safetensors = pytest.importorskip("safetensors", reason="safetensors not installed")


def test_save_load_safetensors_roundtrip(tmp_dir, small_state_dict):
    path = tmp_dir / "model.safetensors"
    final = save_model(small_state_dict, path, fmt="safetensors")
    assert final == path
    loaded = load_model(path)
    # Non-tensor entries (Python scalars, etc.) are not stored in safetensors
    for key in ("weight", "bias", "running_mean", "fp16_param"):
        assert torch.equal(loaded[key], small_state_dict[key])


def test_save_load_safetensors_fp16(tmp_dir, small_state_dict):
    path = tmp_dir / "model.safetensors"
    save_model(small_state_dict, path, fmt="safetensors", dtype="fp16")
    loaded = load_model(path)
    assert loaded["weight"].dtype == torch.float16
    assert loaded["mask"].dtype == torch.bool


@pytest.mark.parametrize("codec", CODECS)
def test_save_load_safetensors_compressed(tmp_dir, small_state_dict, codec):
    path = tmp_dir / "model.safetensors"
    final = save_model(small_state_dict, path, fmt="safetensors", codec=codec)
    assert final.suffix == codec_extension(codec)
    assert not path.exists()
    loaded = load_model(final)
    assert torch.equal(loaded["weight"], small_state_dict["weight"])


# ---------------------------------------------------------------------------
# load_model — map_location
# ---------------------------------------------------------------------------

def test_load_model_map_location_cpu(tmp_dir, small_state_dict):
    path = tmp_dir / "model.pth"
    save_model(small_state_dict, path)
    loaded = load_model(path, map_location="cpu")
    assert loaded["weight"].device.type == "cpu"


# ---------------------------------------------------------------------------
# is_valid_model_file
# ---------------------------------------------------------------------------

def test_is_valid_pth(tmp_dir, small_state_dict):
    path = tmp_dir / "model.pth"
    save_model(small_state_dict, path)
    assert is_valid_model_file(path)


def test_is_valid_safetensors(tmp_dir, small_state_dict):
    path = tmp_dir / "model.safetensors"
    save_model(small_state_dict, path, fmt="safetensors")
    assert is_valid_model_file(path)


def test_is_valid_corrupt_pth(tmp_dir):
    path = tmp_dir / "corrupt.pth"
    path.write_bytes(b"this is not a zip")
    assert not is_valid_model_file(path)


def test_is_valid_corrupt_safetensors(tmp_dir):
    path = tmp_dir / "corrupt.safetensors"
    path.write_bytes(b"\x00" * 8 + b"not json")  # header_len=0 → invalid
    assert not is_valid_model_file(path)


def test_is_valid_missing_file(tmp_dir):
    assert not is_valid_model_file(tmp_dir / "missing.pth")
