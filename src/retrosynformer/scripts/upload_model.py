"""rs-upload: chunked GCS upload for RetroSynFormer model files.

Splits a local model file into 50 MB chunks and uploads them to GCS in
parallel, writing a manifest so gcs_download.py can reassemble on container
startup without any change to env vars.

Usage::

    rs-upload results/hypertune-large-23-layer/trial_000/model.pth \\
              gs://biochem-db-by-hobs/retrosynformer/models/large_emma_23layers_trial000/model.pth.gz \\
              --compress
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
_PROGRESS_CHUNK = 4 * 1024 * 1024

_thread_local = threading.local()
_client_lock = threading.Lock()


def _gcs_client(project: str | None = None, force_refresh: bool = False):
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    key = f"client_{project}"
    if not force_refresh:
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


def _refresh_gcs_client(project: str | None = None) -> None:
    """Discard the cached GCS client for this thread so the next call re-auths."""
    key = f"client_{project}"
    try:
        delattr(_thread_local, key)
    except AttributeError:
        pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _md5_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def _retry_policy():
    from google.api_core import retry as api_retry
    return api_retry.Retry(initial=5.0, maximum=120.0, multiplier=2.0, deadline=7200.0)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as h:mm:ss or m:ss."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class _GlobalProgress:
    """Thread-safe tracker for overall upload progress and timing."""

    def __init__(self, total_bytes: int):
        self.total_bytes = total_bytes
        self.bytes_done = 0
        self.start_time = time.perf_counter()
        self._lock = threading.Lock()

    def add(self, n: int) -> tuple[int, float, float, str]:
        """Add n bytes; return (bytes_done, overall_rate_MBs, pct, eta_str)."""
        with self._lock:
            self.bytes_done += n
            done = self.bytes_done
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        rate = done / elapsed / 1e6
        pct = 100 * done / self.total_bytes if self.total_bytes else 0.0
        remaining = self.total_bytes - done
        eta = _fmt_duration(remaining / (done / elapsed)) if done else "—"
        return done, rate, pct, eta, _fmt_duration(elapsed)


class _ProgressReader:
    def __init__(self, data: bytes, idx: int, global_progress: _GlobalProgress):
        self._buf = io.BytesIO(data)
        self._total = len(data)
        self._idx = idx
        self._sent = 0
        self._t0 = time.perf_counter()
        self._gp = global_progress

    def read(self, n=-1):
        block = self._buf.read(n)
        self._sent += len(block)
        self._gp.add(len(block))
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        chunk_rate = self._sent / elapsed / 1e6
        chunk_pct = 100 * self._sent / self._total
        g_done, g_rate, g_pct, eta, elapsed_str = self._gp.add(0)
        print(
            f"    chunk {self._idx:04d}  {chunk_pct:5.1f}%  "
            f"{self._sent/1e6:.0f}/{self._total/1e6:.0f} MB  {chunk_rate:.3f} MB/s  |  "
            f"overall {g_pct:.1f}%  {g_rate:.3f} MB/s  "
            f"elapsed {elapsed_str}  ETA {eta}",
            flush=True,
        )
        return block

    def seek(self, pos, whence=0):
        return self._buf.seek(pos, whence)

    def tell(self):
        return self._buf.tell()


_AUTH_ERROR_PHRASES = (
    "credentials do not contain",
    "invalid_grant",
    "token has been expired",
    "could not refresh access token",
    "unauthorized",
    "401",
)


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _AUTH_ERROR_PHRASES)


def _upload_chunk(
    bucket_name: str, blob_prefix: str, idx: int, data: bytes,
    skip_existing: bool, gp: _GlobalProgress,
    _attempt: int = 0,
) -> tuple[int, int]:
    try:
        client = _gcs_client(project=bucket_name)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{blob_prefix}.chunk.{idx:04d}")
        if skip_existing:
            try:
                blob.reload()
                if blob.size == len(data) and blob.md5_hash == _md5_b64(data):
                    gp.add(len(data))
                    _, g_rate, g_pct, eta, elapsed_str = gp.add(0)
                    print(
                        f"  chunk {idx:04d}: already verified, skipping  |  "
                        f"overall {g_pct:.1f}%  {g_rate:.3f} MB/s  elapsed {elapsed_str}  ETA {eta}",
                        flush=True,
                    )
                    return idx, len(data)
                elif blob.size is not None:
                    print(f"  chunk {idx:04d}: exists but content differs, re-uploading", flush=True)
            except Exception:
                pass
        blob.chunk_size = _PROGRESS_CHUNK
        reader = _ProgressReader(data, idx, gp)
        blob.upload_from_file(reader, size=len(data), timeout=1800, retry=_retry_policy())
        elapsed = time.perf_counter() - reader._t0
        print(f"  chunk {idx:04d}: done  {len(data)/1e6:.1f} MB  avg {len(data)/elapsed/1e6:.3f} MB/s", flush=True)
        return idx, len(data)
    except Exception as exc:
        if _is_auth_error(exc) and _attempt == 0:
            print(f"  chunk {idx:04d}: auth error — refreshing token and retrying …", flush=True)
            _refresh_gcs_client(bucket_name)
            return _upload_chunk(bucket_name, blob_prefix, idx, data, skip_existing, gp, _attempt=1)
        raise


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
        from google.cloud import storage  # noqa: F401
    except ImportError:
        sys.exit("google-cloud-storage not installed; run: pip install google-cloud-storage")

    if compress and not str(local_path).endswith(".gz"):
        print(f"Compressing {local_path} …", flush=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".gz", delete=False)
        # mtime=0 makes gzip output deterministic for identical input →
        # --skip-existing MD5 checks match across separate invocations.
        with open(local_path, "rb") as src, open(tmp.name, "wb") as raw_out:
            with gzip.GzipFile(fileobj=raw_out, mode="wb", compresslevel=1, mtime=0) as gz:
                shutil.copyfileobj(src, gz)
        source_path = Path(tmp.name)
        print(f"  {local_path.stat().st_size/1e9:.2f} GB → {source_path.stat().st_size/1e9:.2f} GB compressed", flush=True)
    else:
        source_path = local_path

    total_bytes = source_path.stat().st_size
    n_chunks = math.ceil(total_bytes / chunk_size)
    sha = _sha256(source_path)

    print(f"Source : {source_path}  ({total_bytes/1e9:.3f} GB)", flush=True)
    print(f"Target : {gcs_uri}", flush=True)
    print(f"Chunks : {n_chunks} × {chunk_size//1_000_000:.0f} MB  workers={max_workers}", flush=True)
    print(f"SHA256 : {sha}", flush=True)

    chunk_data = []
    with open(source_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunk_data.append(block)

    gp = _GlobalProgress(total_bytes)
    print(f"Uploading {n_chunks} chunks …", flush=True)
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_upload_chunk, bucket_name, blob_name, i, data, skip_existing, gp): i
            for i, data in enumerate(chunk_data)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _, nb = fut.result()
                done += 1
                _, g_rate, g_pct, eta, elapsed_str = gp.add(0)
                print(
                    f"  [{done}/{n_chunks}  {g_pct:.1f}%  {gp.bytes_done/1e9:.3f} GB]  "
                    f"chunk {idx:04d} done  {g_rate:.3f} MB/s  elapsed {elapsed_str}  ETA {eta}",
                    flush=True,
                )
            except Exception as exc:
                done += 1
                print(f"  [{done}/{n_chunks}]  chunk {idx:04d} FAILED — {exc}", flush=True)
                failed.append(idx)

    if failed:
        print(
            f"\n  {len(failed)} chunk(s) failed: {failed}\n"
            f"  Refreshing auth and retrying with --skip-existing …",
            flush=True,
        )
        _refresh_gcs_client(bucket_name)
        retry_total = sum(len(chunk_data[i]) for i in failed)
        retry_gp = _GlobalProgress(retry_total)
        still_failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            retry_futures = {
                executor.submit(_upload_chunk, bucket_name, blob_name, i, chunk_data[i], True, retry_gp): i
                for i in failed
            }
            for fut in as_completed(retry_futures):
                i = retry_futures[fut]
                try:
                    fut.result()
                    print(f"  chunk {i:04d}: retry succeeded", flush=True)
                except Exception as exc:
                    still_failed.append(i)
                    print(f"  chunk {i:04d}: retry FAILED — {exc}", flush=True)
        if still_failed:
            sys.exit(
                f"Upload failed after retry for chunks: {still_failed}\n"
                f"Re-run with:  rs-upload ... --skip-existing"
            )
        print("  All previously-failed chunks recovered on retry.", flush=True)

    manifest = f"{n_chunks}\n{total_bytes}\n{sha}\n"
    _gcs_client(project=bucket_name).bucket(bucket_name).blob(f"{blob_name}.manifest").upload_from_string(manifest, timeout=60)
    print(f"Manifest written → {blob_name}.manifest", flush=True)
    print(f"Done. {n_chunks} chunks, {total_bytes/1e9:.3f} GB.", flush=True)

    if compress and source_path != local_path:
        source_path.unlink()


def deploy_to_cloud_run(
    gcs_uri: str,
    service: str,
    region: str = "us-central1",
    config_gcs_uri: str | None = None,
) -> None:
    """Update MODEL_WEIGHTS_GCS (and optionally MODEL_CONFIG_GCS) on a Cloud Run service."""
    env_updates = [f"MODEL_WEIGHTS_GCS={gcs_uri}"]
    if config_gcs_uri:
        env_updates.append(f"MODEL_CONFIG_GCS={config_gcs_uri}")

    cmd = [
        "gcloud", "run", "services", "update", service,
        "--region", region,
        f"--update-env-vars={','.join(env_updates)}",
    ]
    print(f"\nDeploying to Cloud Run service '{service}' (region: {region}) …", flush=True)
    print(f"  MODEL_WEIGHTS_GCS → {gcs_uri}", flush=True)
    if config_gcs_uri:
        print(f"  MODEL_CONFIG_GCS  → {config_gcs_uri}", flush=True)
    print(f"  $ {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit(f"gcloud deploy failed (exit {result.returncode})")
    print(f"Deploy complete. New revision is live at: {service}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Upload a model file to GCS as parallel 50 MB chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rs-upload results/hypertune-large-23-layer/trial_000/model.pth \\\n"
            "            gs://biochem-db-by-hobs/retrosynformer/models/large-23-layer-trial000/model.pth.gz \\\n"
            "            --compress\n\n"
            "  # Upload and immediately redeploy the inference endpoint:\n"
            "  rs-upload results/.../model.pth gs://bucket/models/new-model/model.pth.gz \\\n"
            "            --compress --deploy retrosynformer-inference-v3\n"
        ),
    )
    parser.add_argument("local_path", type=Path, help="Local model file to upload")
    parser.add_argument("gcs_uri", help="Destination gs:// URI (suffix with .gz to enable decompression at startup)")
    parser.add_argument("--chunk-mb", type=int, default=50, help="Chunk size in MB (default: 50)")
    parser.add_argument("--workers", type=int, default=1, help="Upload workers (default: 1 — network is the bottleneck)")
    parser.add_argument("--compress", action="store_true", help="gzip the file before chunking (.pth → .pth.gz)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip chunks already in GCS with matching size+MD5")
    parser.add_argument(
        "--deploy", metavar="SERVICE",
        help="Cloud Run service to update after upload (sets MODEL_WEIGHTS_GCS env var)",
    )
    parser.add_argument(
        "--deploy-region", default="us-central1", metavar="REGION",
        help="Cloud Run region for --deploy (default: us-central1)",
    )
    parser.add_argument(
        "--deploy-config-gcs", metavar="GCS_URI",
        help="Also update MODEL_CONFIG_GCS on the service (optional)",
    )
    args = parser.parse_args()

    upload_chunked(
        local_path=args.local_path,
        gcs_uri=args.gcs_uri,
        chunk_size=args.chunk_mb * 1024 * 1024,
        max_workers=args.workers,
        compress=args.compress,
        skip_existing=args.skip_existing,
    )

    if args.deploy:
        deploy_to_cloud_run(
            gcs_uri=args.gcs_uri,
            service=args.deploy,
            region=args.deploy_region,
            config_gcs_uri=args.deploy_config_gcs,
        )


if __name__ == "__main__":
    main()
