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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK_SIZE_DEFAULT = 50 * 1024 * 1024  # 50 MB

_thread_local = threading.local()


def _gcs_client(project: str | None = None):
    """Return a per-thread GCS client using gcloud user token.

    Creates one client per thread so credential state is never shared across
    concurrent uploads (sharing a Credentials(token=…) object across threads
    triggers a refresh that fails because the object has no refresh_token).
    """
    import subprocess

    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    key = f"client_{project}"
    client = getattr(_thread_local, key, None)
    if client is not None:
        return client
    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], stderr=subprocess.PIPE
        ).decode().strip()
        creds = Credentials(token=token)
        client = storage.Client(credentials=creds, project=project)
        setattr(_thread_local, key, client)
        return client
    except Exception as e:
        sys.exit(f"Cannot get gcloud token (run `gcloud auth login`): {e}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_PROGRESS_CHUNK = 4 * 1024 * 1024  # GCS resumable-upload increment; controls progress granularity


class _ProgressReader:
    """File-like wrapper that prints per-chunk upload progress and rate as the GCS client reads it."""

    def __init__(self, data: bytes, idx: int):
        self._buf = io.BytesIO(data)
        self._total = len(data)
        self._idx = idx
        self._sent = 0
        self._t0 = time.perf_counter()

    def read(self, n=-1):
        block = self._buf.read(n)
        self._sent += len(block)
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        rate = self._sent / elapsed / 1e6
        pct = 100 * self._sent / self._total
        print(f"    chunk {self._idx:04d}  {pct:5.1f}%  {self._sent/1e6:.0f}/{self._total/1e6:.0f} MB  {rate:.3f} MB/s", flush=True)
        return block

    def seek(self, pos, whence=0):
        return self._buf.seek(pos, whence)

    def tell(self):
        return self._buf.tell()


def _upload_chunk(bucket_name: str, blob_prefix: str, idx: int, data: bytes, skip_existing: bool) -> tuple[int, int]:
    client = _gcs_client(project=bucket_name)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{blob_prefix}.chunk.{idx:04d}")
    if skip_existing:
        try:
            blob.reload()
            if blob.size == len(data):
                print(f"  chunk {idx:04d}: already present ({len(data)/1e6:.1f} MB), skipping", flush=True)
                return idx, len(data)
        except Exception:
            pass
    blob.chunk_size = _PROGRESS_CHUNK  # forces resumable upload in 4 MB increments
    reader = _ProgressReader(data, idx)
    blob.upload_from_file(reader, size=len(data), timeout=1800, retry=_retry_policy())
    elapsed = time.perf_counter() - reader._t0
    rate = len(data) / elapsed / 1e6
    print(f"  chunk {idx:04d}: done  {len(data)/1e6:.1f} MB  avg {rate:.3f} MB/s", flush=True)
    return idx, len(data)


def _retry_policy():
    from google.api_core import retry as api_retry
    return api_retry.Retry(
        initial=5.0,
        maximum=120.0,
        multiplier=2.0,
        deadline=7200.0,
    )


def upload_chunked(
    local_path: Path,
    gcs_uri: str,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    max_workers: int = 1,
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
    done_count = 0
    bytes_uploaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_upload_chunk, bucket_name, blob_name, i, data, skip_existing): i
            for i, data in enumerate(chunk_data)
        }
        print(f"  {len(futures)} futures submitted, waiting …", flush=True)
        failed = []
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _, nbytes = fut.result()
                done_count += 1
                bytes_uploaded += nbytes
                pct = 100 * done_count / n_chunks
                print(f"  [{done_count}/{n_chunks}  {pct:5.1f}%  {bytes_uploaded/1e9:.3f} GB]  chunk {idx:04d}: done", flush=True)
            except Exception as exc:
                done_count += 1
                print(f"  [{done_count}/{n_chunks}]  chunk {idx:04d}: FAILED — {exc}", flush=True)
                failed.append(idx)

    if failed:
        sys.exit(f"Upload failed for chunks: {failed}")

    # Write manifest last — its presence signals a complete upload.
    manifest = f"{n_chunks}\n{total_bytes}\n{sha}\n"
    client = _gcs_client(project=bucket_name)
    client.bucket(bucket_name).blob(f"{blob_name}.manifest").upload_from_string(manifest, timeout=60)
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
    parser.add_argument("--workers", type=int, default=1, help="Upload workers (default: 1 — network is always the bottleneck)")
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
