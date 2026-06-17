# FastAPI Inference Service on Google Cloud Run (GPU)

## Context

The trained RetroSynFormer model predicts retrosynthesis routes for a target molecule via beam search. Currently this is only accessible via CLI (`predict.py`). This plan deploys a FastAPI service to Google Cloud Run with an NVIDIA GPU so that API clients can obtain multiple routes for a target SMILES in under 30 seconds.

The single-molecule path is `RoutePredictor.predict_route(target_smiles, beam_width, target_reward)` which returns a `Beam` namedtuple. All routes up to `beam_width` are retained in the beam and returned. Model, building blocks, and templates are loaded once at startup via FastAPI's lifespan context manager.

**Runtime files needed** (small — no 1.4 GB routes JSON):
- `model.pth` (~16–50 MB)
- `config.yaml` (2 KB)
- `standard_building_blocks.csv` (~2 MB, InChI keys)
- `standard_reaction_templates.pickle` (~323 KB, SMARTS)

**No existing Dockerfile or cloud infrastructure** in the repo.

---

## 1. FastAPI Application (`src/retrosynformer/serve/app.py`)

New sub-package `src/retrosynformer/serve/`.

### Pydantic schemas (`serve/schemas.py`)

```python
class PredictRequest(BaseModel):
    smiles: str                          # Target molecule SMILES string
    beam_width: int = Field(default=10, ge=1, le=50)
    target_reward: float = Field(default=0.5, ge=0.0, le=1.0)
    sort_on: Literal["trajectory_prob", "total_reward"] = "trajectory_prob"

class ReactionStep(BaseModel):
    reaction_smarts: str                 # "R1.R2>>P" format
    template_index: int                  # Index into template vocab
    reward: float

class Route(BaseModel):
    route_solved: bool                   # All leaves are purchasable building blocks
    trajectory_prob: float               # Product of per-step action probabilities
    n_steps: int
    reactions: list[ReactionStep]
    leaf_smiles: list[str]               # Terminal building block SMILES
    dead_ends: list[str]                 # SMILES that hit no applicable template

class PredictResponse(BaseModel):
    smiles: str                          # Echo input
    n_routes: int                        # Number of routes found (≤ beam_width)
    n_solved: int                        # Routes where all leaves are building blocks
    routes: list[Route]                  # Sorted by sort_on descending
    elapsed_s: float                     # Wall-clock seconds for beam search

class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str                          # "cuda:0" or "cpu"
    beam_width_default: int
    action_dim: int
```

### Application factory (`serve/app.py`)

```python
import time, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
import torch

from retrosynformer.runner import init_model, read_config
from retrosynformer.inference import RoutePredictor
from .schemas import PredictRequest, PredictResponse, Route, ReactionStep, HealthResponse

_state: dict = {}   # module-level, populated in lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data once at container startup."""
    config_path = os.environ["MODEL_CONFIG_PATH"]    # e.g. /app/config.yaml
    model_path  = os.environ["MODEL_WEIGHTS_PATH"]   # e.g. /app/model.pth

    config = read_config(config_path)
    # Override data paths from env so container doesn't need code changes
    config["context"]["building_blocks"] = os.environ.get(
        "BUILDING_BLOCKS_PATH", config["context"]["building_blocks"])
    config["context"]["templates_path"]  = os.environ.get(
        "TEMPLATES_PATH", config["context"]["templates_path"])

    model = init_model(config, model_path=model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _state["predictor"] = RoutePredictor(model, config)
    _state["config"]    = config
    _state["device"]    = device
    yield
    _state.clear()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

def _verify_key(key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.environ.get("API_KEY", "")
    if not expected or key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return key

app = FastAPI(title="RetroSynFormer Inference API", version="1.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if "predictor" in _state else "loading",
        model_loaded="predictor" in _state,
        device=_state.get("device", "unknown"),
        beam_width_default=_state.get("config", {}).get("evaluation", {}).get("beam_width", 10),
        action_dim=_state.get("config", {}).get("dataset", {}).get("action_dim", 0),
    )

@app.post("/predict", response_model=PredictResponse, dependencies=[Security(_verify_key)])
def predict(req: PredictRequest):
    predictor: RoutePredictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.perf_counter()

    # Run beam search — predict_route returns the best single beam.
    # To return all beams, call the internal expand_beam loop directly.
    beams = _run_beam_search(predictor, req.smiles, req.beam_width, req.target_reward)

    elapsed = time.perf_counter() - t0

    key = "trajectory_prob" if req.sort_on == "trajectory_prob" else "total_reward"
    beams.sort(key=lambda b: getattr(b, key), reverse=True)

    routes = []
    for beam in beams:
        steps = [
            ReactionStep(reaction_smarts=r, template_index=a, reward=float(rw))
            for r, a, rw in zip(
                beam.reaction_list,
                beam.predicted_actions,
                beam.rewards.squeeze().tolist() if beam.rewards.numel() > 1
                    else [beam.rewards.item()],
            )
        ]
        routes.append(Route(
            route_solved=beam.route_solved,
            trajectory_prob=beam.trajectory_prob,
            n_steps=len(steps),
            reactions=steps,
            leaf_smiles=list(beam.env.leafs),
            dead_ends=list(beam.env.dead_ends),
        ))

    return PredictResponse(
        smiles=req.smiles,
        n_routes=len(routes),
        n_solved=sum(r.route_solved for r in routes),
        routes=routes,
        elapsed_s=round(elapsed, 3),
    )
```

`_run_beam_search()` wraps `predictor.expand_beam()` to collect all terminal beams (both `route_done` and `route_solved`) rather than only the single best. It reuses the existing beam-expansion logic from `inference.py` without copying it — confirm the exact loop against `inference.py` during implementation.

### CLI entry point (`scripts/serve.py`)

```
rs-serve [--host 0.0.0.0] [--port 8080] [--workers 1]
```

Uses `uvicorn.run("retrosynformer.serve.app:app", ...)`. Registered in `pyproject.toml`.

---

## 2. Container Image (`Dockerfile`)

Multi-stage build. Base is official PyTorch CUDA image.

```dockerfile
# ── Stage 1: build wheel ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install build && python -m build --wheel --outdir /dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime is ~7 GB compressed
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app

# OS-level deps for RDKit and rdchiral
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# Install the retrosynformer wheel + inference-only deps (no training deps)
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir \
        /tmp/retrosynformer-*.whl \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        rdkit \
        "rdchiral @ git+https://github.com/CoreThink-AI/rdchiral" \
        "reaction-utils==1.9.3" \
        "transformers>=4.35.0" \
        pandas \
        "pydantic>=2.0"

# Copy model artefacts (baked into image for simplicity;
# alternatively mount from Cloud Storage via FUSE at runtime)
COPY data/standard_building_blocks.csv        /app/data/
COPY data/standard_reaction_templates.pickle  /app/data/
COPY model.pth                                /app/model.pth
COPY results/config/standard.yaml             /app/config.yaml

ENV MODEL_CONFIG_PATH=/app/config.yaml
ENV MODEL_WEIGHTS_PATH=/app/model.pth
ENV BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv
ENV TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle
ENV PORT=8080

EXPOSE 8080
CMD ["rs-serve", "--host", "0.0.0.0", "--port", "8080"]
```

**Image size estimate:** ~8–9 GB (PyTorch CUDA base ~7 GB + deps ~1 GB + data ~3 MB + model ~50 MB).

---

## 3. Google Cloud Run Deployment

### GPU selection

| GPU | VRAM | vCPU | RAM | Price (us-central1) | Recommended for |
|-----|------|------|-----|----------------------|-----------------|
| T4  | 16 GB | 4 | 16 GB | ~$0.35/hr | beam_width ≤ 20 |
| L4  | 24 GB | 8 | 32 GB | ~$0.70/hr | beam_width ≤ 50, < 30s target |

**Recommendation: L4** — the DecisionTransformer (4 heads × 256 dim × 26 layers) fits comfortably in 24 GB; L4's 30 TFLOPS fp16 throughput handles 50-beam search in well under 30s.

### `clouddeploy.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT=${GCP_PROJECT:?Set GCP_PROJECT}
REGION=${GCP_REGION:-us-central1}
SERVICE=retrosynformer-inference
IMAGE=gcr.io/$PROJECT/$SERVICE:$(git rev-parse --short HEAD)

# Build and push via Cloud Build
gcloud builds submit \
    --tag "$IMAGE" \
    --timeout=20m \
    --machine-type=e2-highcpu-8 \
    .

# Deploy to Cloud Run with GPU
gcloud run deploy "$SERVICE" \
    --image "$IMAGE" \
    --region "$REGION" \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=4 \
    --min-instances=1 \
    --max-instances=4 \
    --timeout=120 \
    --set-env-vars="MODEL_CONFIG_PATH=/app/config.yaml,MODEL_WEIGHTS_PATH=/app/model.pth,BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv,TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle" \
    --set-secrets="API_KEY=retrosynformer-api-key:latest" \
    --no-allow-unauthenticated
```

**Key flags:**
- `--min-instances=1` — keeps one warm instance; model loading takes ~30–60s so cold starts must be avoided
- `--concurrency=4` — up to 4 concurrent requests per instance; beam search is GPU-bound so this is safe
- `--timeout=120` — 120s request timeout, well above the 30s inference target
- `--no-allow-unauthenticated` — requires Google IAM token; `X-API-Key` header adds application-layer auth

### Secret management

```bash
# Create the API key secret (one-time setup)
echo -n "$(openssl rand -hex 32)" | \
    gcloud secrets create retrosynformer-api-key \
        --data-file=- \
        --replication-policy=automatic
```

### `cloudbuild.yaml`

CI/CD: build and deploy on each push to `main`.

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "-t", "$_IMAGE", "."]
  - name: gcr.io/cloud-builders/docker
    args: ["push", "$_IMAGE"]
  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: gcloud
    args:
      - run
      - deploy
      - retrosynformer-inference
      - --image=$_IMAGE
      - --region=$_REGION
      - --gpu=1
      - --gpu-type=nvidia-l4
      - --cpu=8
      - --memory=32Gi
      - --min-instances=1
      - --concurrency=4
      - --timeout=120
      - --set-secrets=API_KEY=retrosynformer-api-key:latest
      - --no-allow-unauthenticated
substitutions:
  _IMAGE: gcr.io/$PROJECT_ID/retrosynformer-inference:$SHORT_SHA
  _REGION: us-central1
```

---

## 4. New Dependencies (`pyproject.toml`)

```toml
[project.optional-dependencies]
serve = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
]

# CUDA extra for NVIDIA GPU training/serving (Cloud Run, Colab, etc.)
cuda = [
    "torch>=2.0.0",
]
```

```toml
[project.scripts]
rs-serve = "retrosynformer.scripts.serve:main"
```

```toml
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-rocm",  extra = "rocm"   },
    { index = "pytorch-rocm",  extra = "amdgpu" },
    { index = "pytorch-cpu",   extra = "cpu"    },
    { index = "pytorch-cuda",  extra = "cuda"   },
]
```

---

## 5. Files Created / Modified

| Action | Path |
|--------|------|
| Create | `src/retrosynformer/serve/__init__.py` |
| Create | `src/retrosynformer/serve/app.py` |
| Create | `src/retrosynformer/serve/schemas.py` |
| Create | `src/retrosynformer/scripts/serve.py` |
| Create | `Dockerfile` |
| Create | `cloudbuild.yaml` |
| Create | `clouddeploy.sh` |
| Modify | `pyproject.toml` (`[serve]` + `[cuda]` extras, `rs-serve` CLI) |
| No change | All existing CLI scripts, trainer, runner, inference.py |

---

## 6. Verification

```bash
# Local smoke test (CPU — no GPU required)
MODEL_CONFIG_PATH=results/config/standard.yaml \
MODEL_WEIGHTS_PATH=results/hypertune-compare2_small_structured_dropout/trial_001/model.pth \
BUILDING_BLOCKS_PATH=data/standard_building_blocks.csv \
TEMPLATES_PATH=data/standard_reaction_templates.pickle \
API_KEY=test-key \
  rs-serve --host 127.0.0.1 --port 8080

# Health check
curl http://localhost:8080/health
# → {"status":"ok","model_loaded":true,"device":"cpu","beam_width_default":1,"action_dim":589}

# Predict aspirin (known-solvable target)
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "beam_width": 5}' \
  | python -m json.tool
# → {"smiles":"CC(=O)...","n_routes":5,"n_solved":≥1,"routes":[...],"elapsed_s":≤30}

# Invalid SMILES → expect error detail
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "not-valid-smiles", "beam_width": 3}' | python -m json.tool

# Auth check — should return 403
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)O", "beam_width": 1}'

# Docker build test
docker build -t retrosynformer-inference:local .
docker run --rm -p 8080:8080 \
  -e MODEL_CONFIG_PATH=/app/config.yaml \
  -e MODEL_WEIGHTS_PATH=/app/model.pth \
  -e BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv \
  -e TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle \
  -e API_KEY=test-key \
  retrosynformer-inference:local

# Cloud Run deployment and integration test
bash clouddeploy.sh
SERVICE_URL=$(gcloud run services describe retrosynformer-inference \
    --region=us-central1 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
curl -X POST "$SERVICE_URL/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "beam_width": 50}'
# elapsed_s should be < 30 on L4 GPU
```
