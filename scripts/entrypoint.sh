#!/bin/bash
# Container entrypoint: download model artifacts from GCS, then start the server.
set -euo pipefail

python /app/scripts/gcs_download.py
exec rs-serve --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
