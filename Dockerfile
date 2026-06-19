# ── Stage 1: build the retrosynformer wheel ───────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# gcc/g++ needed by rdchiral's C extension; cmake for any indirect dep
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /dist

# ── Stage 2: runtime (PyTorch CUDA) ──────────────────────────────────────────
# ~8–9 GB compressed. Switch to python:3.12-slim + torch[cpu] for local testing.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Runtime deps for RDKit / rdchiral
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# Install retrosynformer wheel + inference-only Python deps.
# rdchiral is installed from the CoreThink-AI fork via git (not the local path
# used in development).
# google-cloud-storage is used by scripts/gcs_download.py to fetch model
# artifacts from GCS at container startup.
COPY --from=builder /dist/*.whl /tmp/

RUN pip install --no-cache-dir \
        /tmp/retrosynformer-*.whl \
        "fastapi>=0.115,<1" \
        "uvicorn[standard]>=0.30,<1" \
        "rdchiral @ git+https://github.com/CoreThink-AI/rdchiral" \
        "reaction-utils==1.9.3" \
        "transformers>=4.35.0,<5" \
        "pydantic>=2.0,<3" \
        rdkit \
        pandas \
        httpx \
        google-cloud-storage

# Startup script: downloads model artifacts from GCS, then launches rs-serve.
COPY scripts/gcs_download.py /app/scripts/gcs_download.py
COPY scripts/entrypoint.sh   /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# GCS source URIs (overridable at deploy time via --set-env-vars).
# Local destination paths are where the app reads the files after download.
ENV MODEL_WEIGHTS_GCS=gs://biochem-db-by-hobs/retrosynformer/models/large_nonuniform_trial000/model.pth \
    MODEL_CONFIG_GCS=gs://biochem-db-by-hobs/retrosynformer/models/large_nonuniform_trial000/config.yaml \
    BUILDING_BLOCKS_GCS=gs://biochem-db-by-hobs/retrosynformer/data/large_building_blocks.csv \
    TEMPLATES_GCS=gs://biochem-db-by-hobs/retrosynformer/data/large_reaction_templates.pickle \
    MODEL_WEIGHTS_PATH=/tmp/model/model.pth \
    MODEL_CONFIG_PATH=/tmp/model/config.yaml \
    BUILDING_BLOCKS_PATH=/tmp/data/large_building_blocks.csv \
    TEMPLATES_PATH=/tmp/data/large_reaction_templates.pickle \
    FALLBACK_MODEL_WEIGHTS_GCS=gs://biochem-db-by-hobs/retrosynformer/models/large_nonuniform_trial000/model.pth \
    FALLBACK_MODEL_CONFIG_GCS=gs://biochem-db-by-hobs/retrosynformer/models/large_nonuniform_trial000/config.yaml \
    PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
