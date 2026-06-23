"""Download model artifacts from GCS to local disk before server startup.

Reads pairs of env vars:
  - *_GCS  : gs:// URI of the source file
  - *_PATH : local destination path

Skips files that already exist (safe for container restarts when /tmp persists).

Chunked uploads: if a manifest file exists alongside the target blob
(<gcs_uri>.manifest), the file was uploaded as parallel 50 MB chunks by
scripts/upload_model_to_gcs.py.  This script downloads all chunks in parallel,
reassembles them, and verifies the SHA-256.  No change to *_GCS env vars needed.

GCS URIs ending in `.gz` are decompressed after download; the decompressed
file is written to the local path specified by *_PATH (without the .gz suffix).

Fallback: if the primary model weights are corrupt or missing, the script
automatically downloads from FALLBACK_MODEL_WEIGHTS_GCS / FALLBACK_MODEL_CONFIG_GCS
and overwrites the primary local paths so the app loads a known-good model.
"""
import gzip
import hashlib
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Use compression module when retrosynformer is installed (container / editable install).
# Fall back to stdlib gzip-only for standalone / test invocations.
try:
    from retrosynformer.compression import (
        detect_codec as _detect_codec,
        decompress_file as _decompress_file,
        is_valid_model_file as _is_valid_model_file,
    )
    _COMPRESSION_AVAILABLE = True
except ImportError:
    _COMPRESSION_AVAILABLE = False

    def _detect_codec(path):  # type: ignore[misc]
        return "gz" if str(path).endswith(".gz") else None

    def _decompress_file(src, dst):  # type: ignore[misc]
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        return dst

    def _is_valid_model_file(path):  # type: ignore[misc]
        try:
            with zipfile.ZipFile(path, "r") as z:
                z.namelist()
            return True
        except Exception as exc:
            print(f"  Model validation failed for {path}: {exc}", flush=True)
            return False


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("google-cloud-storage is not installed; cannot fetch artifacts from GCS")
    return storage.Client()


def _has_manifest(bucket, blob_name: str) -> bool:
    return bucket.blob(f"{blob_name}.manifest").exists()


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _download_chunked(bucket, blob_name: str, local_path: Path) -> str:
    """Download a file that was uploaded as chunks. Returns verified SHA-256 hex string."""
    manifest_text = bucket.blob(f"{blob_name}.manifest").download_as_text()
    parts = manifest_text.strip().split()
    n_chunks, total_bytes, expected_sha = int(parts[0]), int(parts[1]), parts[2]
    print(f"  Chunked download: {n_chunks} chunks, {total_bytes/1e9:.3f} GB total", flush=True)

    local_path.parent.mkdir(parents=True, exist_ok=True)

    def _fetch(idx: int) -> tuple[int, bytes]:
        data = bucket.blob(f"{blob_name}.chunk.{idx:04d}").download_as_bytes()
        print(f"  chunk {idx:04d}: {len(data)/1e6:.1f} MB", flush=True)
        return idx, data

    max_workers = min(n_chunks, 16)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, i): i for i in range(n_chunks)}
        chunks: dict[int, bytes] = {}
        for fut in as_completed(futures):
            idx, data = fut.result()
            chunks[idx] = data

    print(f"  Reassembling {n_chunks} chunks → {local_path} …", flush=True)
    sha = hashlib.sha256()
    with open(local_path, "wb") as f:
        for i in range(n_chunks):
            f.write(chunks[i])
            sha.update(chunks[i])

    actual_sha = sha.hexdigest()
    if actual_sha != expected_sha:
        local_path.unlink(missing_ok=True)
        sys.exit(f"SHA-256 mismatch after reassembly: expected {expected_sha}, got {actual_sha}")

    print(f"  SHA-256 verified. ({local_path.stat().st_size/1e6:.1f} MB)", flush=True)
    return actual_sha


def _write_metadata_sidecar(
    local_path: Path,
    gcs_uri: str,
    blob,
    sha256_hash: str | None = None,
) -> None:
    """Write a .metadata.json sidecar alongside local_path with GCS object metadata.

    The sidecar is read by ModelPredictor to populate the /health endpoint with
    the GCS source URI, upload timestamp, sha256 hash, and file size.
    """
    try:
        stat = local_path.stat()
        meta = {
            "gcs_uri": gcs_uri,
            "gcs_time_created": blob.time_created.isoformat() if blob.time_created else None,
            "gcs_updated": blob.updated.isoformat() if blob.updated else None,
            "gcs_size_bytes": blob.size,
            "sha256": sha256_hash,
            "file_size_bytes": stat.st_size,
        }
        sidecar = Path(str(local_path) + ".metadata.json")
        sidecar.write_text(json.dumps(meta, indent=2))
        print(f"  Metadata sidecar → {sidecar.name}", flush=True)
    except Exception as exc:
        print(f"  Warning: could not write metadata sidecar: {exc}", flush=True)


def _download(gcs_uri: str, local_path: str) -> None:
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri!r}"
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    codec = _detect_codec(gcs_uri)

    # --- Chunked path (rs-upload wrote a manifest) ---
    if _has_manifest(bucket, blob_name):
        if codec is not None:
            compressed_ext = Path(gcs_uri).suffix  # e.g. ".gz", ".bz2", ".xz"
            compressed_path = path.with_suffix(path.suffix + compressed_ext)
            chunk_sha = _download_chunked(bucket, blob_name, compressed_path)
            print(f"  Decompressing {compressed_path.name} ({codec}) → {path.name} …", flush=True)
            _decompress_file(compressed_path, path)
            compressed_path.unlink()
            print(f"  Done ({path.stat().st_size/1e6:.1f} MB decompressed)", flush=True)
            print(f"  Computing SHA-256 of decompressed model …", flush=True)
            sha256 = _sha256_file(path)
            print(f"  SHA-256: {sha256}", flush=True)
        else:
            sha256 = _download_chunked(bucket, blob_name, path)
        # Use manifest blob's timeCreated as the canonical model release timestamp.
        manifest_blob = bucket.blob(f"{blob_name}.manifest")
        manifest_blob.reload()
        _write_metadata_sidecar(path, gcs_uri, manifest_blob, sha256_hash=sha256)
        return

    # --- Single-file path ---
    blob = bucket.blob(blob_name)
    blob.reload()
    size_mb = blob.size / 1e6 if blob.size else 0.0

    if codec is not None:
        compressed_ext = Path(gcs_uri).suffix
        compressed_path = path.with_suffix(path.suffix + compressed_ext)
        print(f"  {gcs_uri} → {compressed_path.name} ({size_mb:.1f} MB compressed) ...", flush=True)
        blob.download_to_filename(str(compressed_path))
        print(f"  Decompressing ({codec}) → {path.name} ...", flush=True)
        _decompress_file(compressed_path, path)
        compressed_path.unlink()
        print(f"  Done ({path.stat().st_size/1e6:.1f} MB decompressed)", flush=True)
    else:
        print(f"  {gcs_uri} → {local_path} ({size_mb:.1f} MB) ...", flush=True)
        blob.download_to_filename(local_path)
        print(f"  Done ({path.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"  Computing SHA-256 …", flush=True)
    sha256 = _sha256_file(path)
    print(f"  SHA-256: {sha256}", flush=True)
    _write_metadata_sidecar(path, gcs_uri, blob, sha256_hash=sha256)


def _download_if_missing(gcs_uri: str, local_path: str) -> None:
    if Path(local_path).exists():
        print(f"  {local_path}: already present, skipping", flush=True)
        return
    _download(gcs_uri, local_path)


def _is_valid_model(local_path: str) -> bool:
    """Return True if local_path is a readable model file (PyTorch or SafeTensors)."""
    return _is_valid_model_file(local_path)


ARTIFACT_PAIRS = [
    ("MODEL_WEIGHTS_GCS", "MODEL_WEIGHTS_PATH"),
    ("MODEL_CONFIG_GCS", "MODEL_CONFIG_PATH"),
    ("BUILDING_BLOCKS_GCS", "BUILDING_BLOCKS_PATH"),
    ("TEMPLATES_GCS", "TEMPLATES_PATH"),
]

FALLBACK_PAIRS = [
    ("FALLBACK_MODEL_WEIGHTS_GCS", "MODEL_WEIGHTS_PATH"),
    ("FALLBACK_MODEL_CONFIG_GCS", "MODEL_CONFIG_PATH"),
]


if __name__ == "__main__":
    print("==> Downloading model artifacts from GCS...", flush=True)
    for gcs_var, path_var in ARTIFACT_PAIRS:
        gcs_uri = os.environ.get(gcs_var)
        local_path = os.environ.get(path_var)
        if not gcs_uri:
            print(f"  {gcs_var} not set, skipping", flush=True)
            continue
        if not local_path:
            sys.exit(f"  {gcs_var} is set but {path_var} is not — cannot determine local destination")
        _download_if_missing(gcs_uri, local_path)
    print("==> Artifact download complete.", flush=True)

    # Validate model weights; fall back to the previous known-good model if corrupt.
    model_path = os.environ.get("MODEL_WEIGHTS_PATH")
    if model_path and not _is_valid_model(model_path):
        fallback_weights = os.environ.get("FALLBACK_MODEL_WEIGHTS_GCS")
        fallback_config = os.environ.get("FALLBACK_MODEL_CONFIG_GCS")
        if not fallback_weights or not fallback_config:
            sys.exit("Primary model is invalid and no fallback configured (set FALLBACK_MODEL_WEIGHTS_GCS / FALLBACK_MODEL_CONFIG_GCS)")

        print("==> Primary model is invalid — falling back to:", flush=True)
        print(f"    weights: {fallback_weights}", flush=True)
        print(f"    config:  {fallback_config}", flush=True)

        # Remove the corrupt files so _download won't skip them.
        Path(model_path).unlink(missing_ok=True)
        config_path = os.environ.get("MODEL_CONFIG_PATH")
        if config_path:
            Path(config_path).unlink(missing_ok=True)

        _download(fallback_weights, model_path)
        if config_path:
            _download(fallback_config, config_path)

        if not _is_valid_model(model_path):
            sys.exit("Fallback model is also invalid — cannot start server")
        print("==> Fallback model loaded successfully.", flush=True)
