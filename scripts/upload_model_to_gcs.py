#!/usr/bin/env python3
"""Upload a model file to GCS as parallel 50 MB chunks.

Chunks are stored as:
    <gcs_uri>.chunk.0000, .chunk.0001, …
A manifest is written alongside:
    <gcs_uri>.manifest   (text: "<n_chunks>\\n<total_bytes>\\n<sha256>\\n")

gcs_download.py detects the manifest and reassembles chunks in parallel at
container startup — no change to MODEL_WEIGHTS_GCS env var needed.

Usage (from repo root):
    python scripts/upload_model_to_gcs.py \\
        results/hypertune-large-emma-24-26_layer/trial_000/model.pth \\
        gs://biochem-db-by-hobs/retrosynformer/models/large_emma_24layers_trial000/model.pth.gz \\
        [--chunk-mb 50] [--workers 8] [--compress] [--skip-existing]
"""
import argparse
import gzip
import hashlib
import io
import math
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CHUNK_SIZE_DEFAULT = 50 * 1024 * 1024  # 50 MB


def _gcs_client(project: str | None = None):
    """Return an authenticated GCS client using gcloud user token (bypasses stale ADC)."""
    import subprocess
    from google.cloud import storage
    from google.oauth2.credentials import Credentials
    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], stderr=subprocess.PIPE
        ).decode().strip()
        creds = Credentials(token=token)
        return storage.Client(credentials=creds, project=project)
    except Exception as e:
        sys.exit(f"Cannot get gcloud token (run `gcloud auth login`): {e}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _upload_chunk(bucket, blob_prefix: str, idx: int, data: bytes, skip_existing: bool) -> tuple[int, int]:
    blob = bucket.blob(f"{blob_prefix}.chunk.{idx:04d}")
    if skip_existing:
        try:
            blob.reload()
            if blob.size == len(data):
                print(f"  chunk {idx:04d}: already present ({len(data)/1e6:.1f} MB), skipping", flush=True)
                return idx, len(data)
        except Exception:
            pass
    blob.upload_from_string(data, timeout=300, retry=_retry_policy())
    print(f"  chunk {idx:04d}: uploaded {len(data)/1e6:.1f} MB", flush=True)
    return idx, len(data)


def _retry_policy():
    from google.api_core import retry as api_retry
    return api_retry.Retry(
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=600.0,
    )


def upload_chunked(
    local_path: Path,
    gcs_uri: str,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    max_workers: int = 8,
    compress: bool = False,
    skip_existing: bool = False,
) -> None:
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got {gcs_uri!r}"
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)

    try:
        from google.cloud import storage  # noqa: F401 (just verify installed)
    except ImportError:
        sys.exit("google-cloud-storage not installed")

    # Optionally gzip the source file first
    if compress and not str(local_path).endswith(".gz"):
        print(f"Compressing {local_path} …", flush=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".gz", delete=False)
        with open(local_path, "rb") as src, gzip.open(tmp.name, "wb", compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        source_path = Path(tmp.name)
        print(f"  compressed: {local_path.stat().st_size/1e9:.2f} GB → {source_path.stat().st_size/1e9:.2f} GB", flush=True)
    else:
        source_path = local_path

    total_bytes = source_path.stat().st_size
    n_chunks = math.ceil(total_bytes / chunk_size)
    print(f"Source: {source_path}  ({total_bytes/1e9:.3f} GB)", flush=True)
    print(f"Target: {gcs_uri}", flush=True)
    print(f"Chunks: {n_chunks} × {chunk_size//1e6:.0f} MB  workers={max_workers}", flush=True)

    sha = _sha256(source_path)
    print(f"SHA256: {sha}", flush=True)

    client = _gcs_client(project=bucket_name)
    bucket = client.bucket(bucket_name)

    # Read entire file into memory in chunks and upload in parallel.
    # For a 1.1 GB file this is fine; for larger models stream from disk.
    chunk_data = []
    with open(source_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunk_data.append(block)

    assert len(chunk_data) == n_chunks

    print(f"Uploading {n_chunks} chunks …", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_upload_chunk, bucket, blob_name, i, data, skip_existing): i
            for i, data in enumerate(chunk_data)
        }
        failed = []
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                idx = futures[fut]
                print(f"  chunk {idx:04d}: FAILED — {exc}", flush=True)
                failed.append(idx)

    if failed:
        sys.exit(f"Upload failed for chunks: {failed}")

    # Write manifest last — its presence signals a complete upload.
    manifest = f"{n_chunks}\n{total_bytes}\n{sha}\n"
    bucket.blob(f"{blob_name}.manifest").upload_from_string(manifest, timeout=60)
    print(f"Manifest written: {blob_name}.manifest", flush=True)
    print(f"Done. {n_chunks} chunks, {total_bytes/1e9:.3f} GB total.", flush=True)

    # Clean up temp gz file if we created one
    if compress and source_path != local_path:
        source_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("local_path", type=Path, help="Local file to upload")
    parser.add_argument("gcs_uri", help="Destination gs:// URI (e.g. gs://bucket/path/model.pth.gz)")
    parser.add_argument("--chunk-mb", type=int, default=50, help="Chunk size in MB (default: 50)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel upload workers (default: 8)")
    parser.add_argument("--compress", action="store_true", help="gzip the file before chunking (for .pth → .pth.gz)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip chunks already in GCS with matching size")
    args = parser.parse_args()

    upload_chunked(
        local_path=args.local_path,
        gcs_uri=args.gcs_uri,
        chunk_size=args.chunk_mb * 1024 * 1024,
        max_workers=args.workers,
        compress=args.compress,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
