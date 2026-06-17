# PLAN: FastAPI Inference Endpoint for Google Cloud

*Branch: `feature-structured-dropout` — June 2026*

---

## What we're serving

`RoutePredictor.predict_route(smiles, beam_width)` in `src/retrosynformer/inference.py` — beam-search over a Decision Transformer (HuggingFace `transformers`) + RDKit/rdchiral chemistry engine. Each call is synchronous, CPU/GPU-bound, and takes O(1–30 s) depending on `beam_width`. The model, building-blocks CSV, and templates pickle must be loaded once at startup (~2 GB in memory), not per request.

---

## API surface

```
POST /predict
Content-Type: application/json

Body:
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "beam_width": 50,
  "target_reward": 0.5
}

200 Response:
{
  "route_solved": true,
  "reaction_list": ["reactants>>product", "..."],
  "leafs": ["CC", "OC(=O)c1ccccc1"],
  "trajectory_prob": 0.42,
  "n_reactions": 3,
  "n_dead_ends": 0,
  "time_seconds": 1.8
}

422: invalid SMILES string (caught before any GPU work)
503: model not yet loaded (startup in progress)

GET /health  →  { "status": "ok", "model_loaded": true }
GET /config  →  echo active config keys (omits file paths)
```

---

## Concurrency model

`predict_route` is sequential internally — each beam-expansion step waits for the previous transformer forward pass and rdchiral template application. True batching across concurrent requests would require significant refactoring. Instead, we scale horizontally:

**CPU deployment** (initial):  
Gunicorn with `N` `UvicornWorker` processes (`N` ≈ CPU count or 4). Each process loads its own model copy independently. Cloud Run `--concurrency 4` + `--min-instances 1` achieves the same with simpler ops — Cloud Run handles the process model.

**GPU deployment** (when latency budget demands it):  
Single GPU per container instance. One `asyncio.Semaphore(1)` serializes GPU calls inside the process; `run_in_executor` keeps the event loop responsive while the thread holds the semaphore. PyTorch releases the GIL during CUDA kernels, so 1–2 threads can overlap Python bookkeeping. Cloud Run GPU (NVIDIA L4 or T4) supports this directly.

**Recommendation**: start with Cloud Run CPU. Graduate to GPU when P95 latency exceeds the SLA.

---

## Files to create

```
src/retrosynformer/api/
    __init__.py
    app.py          ← FastAPI lifespan, /predict, /health, /config routes
    schemas.py      ← PredictRequest, PredictResponse (Pydantic v2)
    predictor.py    ← ModelPredictor singleton, loads model at startup

Dockerfile          ← multi-stage; builds rdchiral C extension then copies app
.dockerignore
scripts/deploy_cloud_run.sh   ← gcloud run deploy one-liner + env var wiring
```

`pyproject.toml` gets a new `[project.optional-dependencies]` group `api`:
`fastapi`, `uvicorn[standard]`, `python-multipart`.

Add CLI entry point to `[project.scripts]`:
`retrosynformer-api = "retrosynformer.api.app:run"`

---

## Module designs

### `schemas.py`

```python
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    smiles: str
    beam_width: int = Field(default=50, ge=1, le=200)
    target_reward: float = Field(default=0.5, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    route_solved: bool
    reaction_list: list[str]
    leafs: list[str]
    trajectory_prob: float
    n_reactions: int
    n_dead_ends: int
    time_seconds: float
```

### `predictor.py`

```python
class ModelPredictor:
    """Loaded once at startup; thread-safe for concurrent reads (torch no_grad)."""

    def __init__(self, config_path: str, model_path: str):
        config = read_config(config_path)
        model = init_model(config)
        model.load_state_dict(torch.load(model_path, map_location=get_device()))
        self._predictor = RoutePredictor(model, config)
        self._gpu_sem = asyncio.Semaphore(1)  # only used on GPU builds

    def predict_sync(self, smiles, beam_width, target_reward) -> dict:
        """Called from a thread-pool executor; returns a plain dict."""
        t0 = time.monotonic()
        beam = self._predictor.predict_route(smiles, beam_width, target_reward)
        elapsed = time.monotonic() - t0
        if beam is None:
            return {"route_solved": False, "reaction_list": [], "leafs": [],
                    "trajectory_prob": 0.0, "n_reactions": 0,
                    "n_dead_ends": 0, "time_seconds": elapsed}
        return {
            "route_solved": beam.route_solved,
            "reaction_list": beam.reaction_list,
            "leafs": beam.env.leafs,
            "trajectory_prob": float(beam.trajectory_prob),
            "n_reactions": len(beam.reaction_list),
            "n_dead_ends": len(beam.env.dead_ends),
            "time_seconds": elapsed,
        }
```

### `app.py`

```python
from contextlib import asynccontextmanager
import asyncio, os
from fastapi import FastAPI, HTTPException
from rdkit import Chem
from .predictor import ModelPredictor
from .schemas import PredictRequest, PredictResponse

_predictor: ModelPredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = ModelPredictor(
        config_path=os.environ["RSF_CONFIG"],
        model_path=os.environ["RSF_MODEL"],
    )
    yield
    _predictor = None

app = FastAPI(title="RetroSynFormer API", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _predictor is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    if Chem.MolFromSmiles(req.smiles) is None:
        raise HTTPException(422, f"Invalid SMILES: {req.smiles!r}")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _predictor.predict_sync, req.smiles, req.beam_width, req.target_reward
    )
    return result

def run():
    import uvicorn
    uvicorn.run("retrosynformer.api.app:app", host="0.0.0.0", port=8080)
```

---

## Dockerfile

Multi-stage build so the final image doesn't carry build tools (gcc, cmake) needed by rdchiral's C extension.

```dockerfile
# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ cmake libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
# Install into an explicit venv so we can copy it cleanly
RUN uv venv /venv && \
    uv sync --extra cpu --extra api --no-install-project --python /venv/bin/python

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY src/ src/
RUN pip install --no-cache-dir -e . --no-deps

# Data files (model.pth, config.yaml, building_blocks.csv, templates.pickle)
# are mounted at runtime via GCS download script or Cloud Run volume.
# RSF_CONFIG and RSF_MODEL env vars point to their paths inside the container.

EXPOSE 8080
CMD ["uvicorn", "retrosynformer.api.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

---

## `.dockerignore`

```
.venv/
__pycache__/
*.pyc
data/
results/
*.pth
*.db
*.jsonl
.git/
tests/
docs/
```

---

## Cloud Run deployment (`scripts/deploy_cloud_run.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT=${GCP_PROJECT:?set GCP_PROJECT}
REGION=${GCP_REGION:-us-central1}
IMAGE="gcr.io/$PROJECT/retrosynformer-api:latest"

echo "==> Building and pushing image"
gcloud builds submit --tag "$IMAGE" .

echo "==> Deploying to Cloud Run"
gcloud run deploy retrosynformer-api \
  --image "$IMAGE" \
  --region "$REGION" \
  --memory 8Gi \
  --cpu 4 \
  --concurrency 4 \
  --min-instances 1 \
  --max-instances 20 \
  --timeout 120 \
  --set-env-vars "RSF_CONFIG=/data/config.yaml,RSF_MODEL=/data/model.pth" \
  --no-allow-unauthenticated

echo "==> Done. Service URL:"
gcloud run services describe retrosynformer-api \
  --region "$REGION" --format "value(status.url)"
```

**Model file strategy**: on container startup, a small init script downloads model files from GCS before uvicorn starts. Alternatively, use Cloud Run volume mounts (GA as of 2024) backed by a GCS bucket — simpler and avoids baking large binaries into the image.

---

## Environment variables

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `RSF_CONFIG` | yes | `/data/config.yaml` | Path to config YAML inside container |
| `RSF_MODEL` | yes | `/data/model.pth` | Path to trained model checkpoint |
| `RSF_LOG_LEVEL` | no | `INFO` | Python logging level |
| `PORT` | no | `8080` | Override listen port (Cloud Run sets this) |

---

## Scale-up path

| Request rate | Approach |
|-------------|----------|
| < 5 req/s | Cloud Run CPU, `--concurrency 4`, `--min-instances 1` |
| 5–20 req/s | Increase `--max-instances`; tune `--concurrency` |
| 20–50 req/s | Cloud Run GPU (L4), `--concurrency 8` + `Semaphore(1)` in predictor |
| > 50 req/s | GKE with GPU node pool + Horizontal Pod Autoscaler; consider batched inference |

---

## Implementation order

1. `src/retrosynformer/api/schemas.py` — Pydantic models, no dependencies
2. `src/retrosynformer/api/predictor.py` — wraps `RoutePredictor`, unit-testable standalone
3. `src/retrosynformer/api/app.py` — FastAPI app wiring
4. `src/retrosynformer/api/__init__.py` — empty
5. Update `pyproject.toml` — add `[api]` extra deps and CLI entry point
6. `Dockerfile` + `.dockerignore`
7. `scripts/deploy_cloud_run.sh`
8. Integration test: `pytest tests/test_api.py` with `TestClient`

---

## Open questions

- **Auth**: Cloud Run `--no-allow-unauthenticated` + Google IAM is sufficient for internal use. If the API is public, add an API key header checked in a FastAPI middleware.
- **Model versioning**: should the `RSF_MODEL` env var point to a GCS URI and be downloaded at startup, or should we bake a specific checkpoint into the image tag?
- **Beam width cap**: `beam_width` is capped at 200 in the schema; tune this based on observed P99 latency in staging.
- **SMILES validation depth**: currently just `Chem.MolFromSmiles()`; consider also checking for a minimum atom count to reject trivial inputs early.
