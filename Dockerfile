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
        httpx

# Bake in small data files (< 3 MB total).
# model.pth and config.yaml are NOT baked in — mount from GCS at runtime.
COPY data/standard_building_blocks.csv       /app/data/standard_building_blocks.csv
COPY data/standard_reaction_templates.pickle /app/data/standard_reaction_templates.pickle

ENV MODEL_CONFIG_PATH=/app/model/config.yaml \
    MODEL_WEIGHTS_PATH=/app/model/model.pth \
    BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv \
    TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# --workers 1 required when using a single GPU; the asyncio.Semaphore in app.py
# serialises concurrent requests within the one process.
CMD ["retrosynformer-serve", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
