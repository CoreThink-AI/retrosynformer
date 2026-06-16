"""Download model artifacts from GCS to local disk before server startup.

Reads pairs of env vars:
  - *_GCS  : gs:// URI of the source file
  - *_PATH : local destination path

Skips files that already exist (safe for container restarts when /tmp persists).
"""
import os
import sys
from pathlib import Path


def _download(gcs_uri: str, local_path: str) -> None:
    path = Path(local_path)
    if path.exists():
        print(f"  {local_path}: already present, skipping", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("google-cloud-storage is not installed; cannot fetch artifacts from GCS")

    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri!r}"
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.reload()  # fetch metadata so blob.size is populated
    size_mb = blob.size / 1e6 if blob.size else 0.0
    print(f"  {gcs_uri} → {local_path} ({size_mb:.1f} MB) ...", flush=True)
    blob.download_to_filename(local_path)
    print(f"  Done ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


ARTIFACT_PAIRS = [
    ("MODEL_WEIGHTS_GCS", "MODEL_WEIGHTS_PATH"),
    ("MODEL_CONFIG_GCS", "MODEL_CONFIG_PATH"),
    ("BUILDING_BLOCKS_GCS", "BUILDING_BLOCKS_PATH"),
    ("TEMPLATES_GCS", "TEMPLATES_PATH"),
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
        _download(gcs_uri, local_path)
    print("==> Artifact download complete.", flush=True)
