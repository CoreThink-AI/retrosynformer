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
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("google-cloud-storage is not installed; cannot fetch artifacts from GCS")
    return storage.Client()


def _has_manifest(bucket, blob_name: str) -> bool:
    return bucket.blob(f"{blob_name}.manifest").exists()


def _download_chunked(bucket, blob_name: str, local_path: Path) -> None:
    """Download a file that was uploaded as chunks by upload_model_to_gcs.py."""
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


def _download(gcs_uri: str, local_path: str) -> None:
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri!r}"
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    # --- Chunked path (upload_model_to_gcs.py wrote a manifest) ---
    if _has_manifest(bucket, blob_name):
        if gcs_uri.endswith(".gz"):
            gz_path = path.with_suffix(path.suffix + ".gz")
            _download_chunked(bucket, blob_name, gz_path)
            print(f"  Decompressing {gz_path} → {path} …", flush=True)
            with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            print(f"  Done ({path.stat().st_size/1e6:.1f} MB decompressed)", flush=True)
        else:
            _download_chunked(bucket, blob_name, path)
        return

    # --- Single-file path ---
    blob = bucket.blob(blob_name)
    blob.reload()
    size_mb = blob.size / 1e6 if blob.size else 0.0

    if gcs_uri.endswith(".gz"):
        gz_path = path.with_suffix(path.suffix + ".gz")
        print(f"  {gcs_uri} → {gz_path} ({size_mb:.1f} MB compressed) ...", flush=True)
        blob.download_to_filename(str(gz_path))
        print(f"  Decompressing {gz_path} → {path} ...", flush=True)
        with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        print(f"  Done ({path.stat().st_size/1e6:.1f} MB decompressed)", flush=True)
    else:
        print(f"  {gcs_uri} → {local_path} ({size_mb:.1f} MB) ...", flush=True)
        blob.download_to_filename(local_path)
        print(f"  Done ({path.stat().st_size/1e6:.1f} MB)", flush=True)


def _download_if_missing(gcs_uri: str, local_path: str) -> None:
    if Path(local_path).exists():
        print(f"  {local_path}: already present, skipping", flush=True)
        return
    _download(gcs_uri, local_path)


def _is_valid_model(local_path: str) -> bool:
    """Return True if local_path is a readable PyTorch checkpoint (ZIP archive).

    Reads only the ZIP central directory — fast even for large .pth files.
    """
    try:
        with zipfile.ZipFile(local_path, "r") as z:
            z.namelist()
        return True
    except Exception as e:
        print(f"  Model validation failed for {local_path}: {e}", flush=True)
        return False


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
