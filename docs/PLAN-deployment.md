# Unified Deployment Plan: RetroSynFormer on Google Cloud + AiZynthFinder Integration

*Branch: `feature-structured-dropout` — June 2026*

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Drug Discovery Web App                    │
│               (React/Next.js, Cloud Run or GCS)              │
└──────────────┬──────────────────────────────┬───────────────┘
               │  POST /routes                │  POST /routes
               ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│  Route Fusion Gateway    │   │   (direct, optional bypass)  │
│  Cloud Run — CPU only    │   │                              │
│  GET /health             │   │                              │
│  POST /routes            │◄──┤                              │
└─────────┬────────────────┘   └──────────────────────────────┘
          │ parallel fan-out
    ┌─────┴─────┐
    ▼           ▼
┌──────────┐  ┌─────────────────────────────┐
│ RSF      │  │ AiZynthFinder MCTS Service  │
│ Inference│  │ Cloud Run / GKE             │
│ Service  │  │ POST /api/api/find-routes   │
│ Cloud Run│  │ (existing or new deployment)│
│ GPU (L4) │  └─────────────────────────────┘
└──────────┘

Internal tooling (not customer-facing):
┌──────────────────────────────────────────────────────────────┐
│  Dashboard (Flask-Admin + REST v1)                           │
│  Monitors Optuna hypertune studies, trial curves, enqueue    │
│  See PLAN-flask-admin.md + PLAN-rest-api.md                  │
└──────────────────────────────────────────────────────────────┘
```

**Three customer-facing Cloud Run services:**

| Service | Image | GPU | RAM | Purpose |
|---------|-------|-----|-----|---------|
| `retrosynformer-inference` | `gcr.io/$PROJECT/rsf-inference` | L4 | 32 Gi | RetroSynFormer beam search |
| `aizynthfinder-mcts` | `gcr.io/$PROJECT/aizynthfinder` | none | 8 Gi | AiZynthFinder MCTS (existing or containerised) |
| `route-fusion-gateway` | `gcr.io/$PROJECT/route-fusion` | none | 2 Gi | Fan-out, merge, rank, serve the web app |

---

## 2. RetroSynFormer Inference Service

### 2.1 API surface

```
POST /predict
X-API-Key: <secret>
Content-Type: application/json

Body:
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "beam_width": 50,
  "target_reward": 0.5,
  "sort_on": "trajectory_prob"   // or "total_reward"
}

200 Response:
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "n_routes": 50,
  "n_solved": 12,
  "routes": [
    {
      "route_solved": true,
      "trajectory_prob": 0.42,
      "n_steps": 3,
      "reactions": [
        {"reaction_smarts": "R1.R2>>P", "template_index": 137, "reward": 0.8}
      ],
      "leaf_smiles": ["CC", "OC(=O)c1ccccc1"],
      "dead_ends": []
    }
  ],
  "elapsed_s": 4.2
}

GET /health → {"status": "ok", "model_loaded": true, "device": "cuda:0",
               "beam_width_default": 10, "action_dim": 1573}
422: invalid SMILES (checked before any GPU work via Chem.MolFromSmiles)
503: model still loading (startup takes ~30–60s)
403: missing or wrong X-API-Key
```

### 2.2 Pydantic schemas (`src/retrosynformer/serve/schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    smiles: str
    beam_width: int = Field(default=10, ge=1, le=50)
    target_reward: float = Field(default=0.5, ge=0.0, le=1.0)
    sort_on: Literal["trajectory_prob", "total_reward"] = "trajectory_prob"

class ReactionStep(BaseModel):
    reaction_smarts: str      # "R1.R2>>P" SMARTS format
    template_index: int
    reward: float

class Route(BaseModel):
    route_solved: bool
    trajectory_prob: float
    n_steps: int
    reactions: list[ReactionStep]
    leaf_smiles: list[str]
    dead_ends: list[str]

class PredictResponse(BaseModel):
    smiles: str
    n_routes: int
    n_solved: int
    routes: list[Route]
    elapsed_s: float

class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str
    beam_width_default: int
    action_dim: int
```

### 2.3 Predictor singleton (`src/retrosynformer/serve/predictor.py`)

```python
import time, asyncio, torch
from retrosynformer.runner import init_model, read_config
from retrosynformer.inference import RoutePredictor

class ModelPredictor:
    """Loaded once at startup; thread-safe for concurrent reads (torch.no_grad)."""

    def __init__(self, config_path: str, model_path: str,
                 building_blocks_path: str, templates_path: str):
        config = read_config(config_path)
        config["context"]["building_blocks"] = building_blocks_path
        config["context"]["templates_path"]  = templates_path
        model = init_model(config, model_path=model_path)
        model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        self._predictor = RoutePredictor(model, config)
        self._config = config

    def predict_sync(self, smiles: str, beam_width: int,
                     target_reward: float, sort_on: str) -> dict:
        """Runs in a thread-pool executor; returns a plain dict."""
        t0 = time.perf_counter()
        beams = _run_all_beams(self._predictor, smiles, beam_width, target_reward)
        elapsed = time.perf_counter() - t0

        key = "trajectory_prob" if sort_on == "trajectory_prob" else "total_reward"
        beams.sort(key=lambda b: getattr(b, key, 0.0), reverse=True)

        routes = []
        for beam in beams:
            steps = [
                {"reaction_smarts": r, "template_index": int(a), "reward": float(rw)}
                for r, a, rw in zip(
                    beam.reaction_list,
                    beam.predicted_actions,
                    beam.rewards.squeeze().tolist()
                    if beam.rewards.numel() > 1 else [beam.rewards.item()],
                )
            ]
            routes.append({
                "route_solved": beam.route_solved,
                "trajectory_prob": float(beam.trajectory_prob),
                "n_steps": len(steps),
                "reactions": steps,
                "leaf_smiles": list(beam.env.leafs),
                "dead_ends": list(beam.env.dead_ends),
            })

        return {
            "n_routes": len(routes),
            "n_solved": sum(r["route_solved"] for r in routes),
            "routes": routes,
            "elapsed_s": round(elapsed, 3),
        }
```

`_run_all_beams()` wraps the internal beam-expansion loop from `inference.py` to collect all terminal beams (both solved and dead-end), not just the single best. Confirm the exact loop interface against `inference.py` before implementing.

### 2.4 FastAPI application (`src/retrosynformer/serve/app.py`)

```python
import asyncio, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from rdkit import Chem

from .predictor import ModelPredictor
from .schemas import PredictRequest, PredictResponse, HealthResponse

_state: dict = {}
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

def _verify_key(key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.environ.get("API_KEY", "")
    if not expected or key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return key

@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["predictor"] = ModelPredictor(
        config_path=os.environ["MODEL_CONFIG_PATH"],
        model_path=os.environ["MODEL_WEIGHTS_PATH"],
        building_blocks_path=os.environ["BUILDING_BLOCKS_PATH"],
        templates_path=os.environ["TEMPLATES_PATH"],
    )
    yield
    _state.clear()

app = FastAPI(title="RetroSynFormer Inference API", version="1.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health():
    p = _state.get("predictor")
    return HealthResponse(
        status="ok" if p else "loading",
        model_loaded=bool(p),
        device=p.device if p else "unknown",
        beam_width_default=10,
        action_dim=p._config["dataset"]["action_dim"] if p else 0,
    )

@app.post("/predict", response_model=PredictResponse,
          dependencies=[Security(_verify_key)])
async def predict(req: PredictRequest):
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(503, "Model not loaded yet")
    if Chem.MolFromSmiles(req.smiles) is None:
        raise HTTPException(422, f"Invalid SMILES: {req.smiles!r}")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, predictor.predict_sync,
        req.smiles, req.beam_width, req.target_reward, req.sort_on,
    )
    return {"smiles": req.smiles, **result}
```

### 2.5 CLI entry point (`src/retrosynformer/scripts/serve.py`)

```python
def main():
    import uvicorn, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    uvicorn.run("retrosynformer.serve.app:app",
                host=args.host, port=args.port, workers=args.workers)
```

### 2.6 Files

| Action | Path |
|--------|------|
| Create | `src/retrosynformer/serve/__init__.py` |
| Create | `src/retrosynformer/serve/app.py` |
| Create | `src/retrosynformer/serve/schemas.py` |
| Create | `src/retrosynformer/serve/predictor.py` |
| Create | `src/retrosynformer/scripts/serve.py` |
| Modify | `pyproject.toml` — add `[serve]` extra + `retrosynformer-serve` CLI |

---

## 3. AiZynthFinder Integration

### 3.1 Integration strategy

RetroSynFormer and AiZynthFinder are complementary rather than competing:

| Property | RetroSynFormer | AiZynthFinder |
|----------|----------------|---------------|
| Search algorithm | Beam search (Decision Transformer) | MCTS (policy + rollout) |
| Trained on | PaRoutes dataset | USPTO + Reaxys (AZ proprietary) |
| Strength | End-to-end sequence model, captures long-range route structure | Broader template coverage, mature tooling |

**Recommended integration: parallel query + route fusion** (Phase 1).  
Both services run independently; the Route Fusion Gateway calls them in parallel, deduplicates, merges, and ranks routes. This avoids coupling their release cycles.

**Optional Phase 2: RSF as AiZynthFinder expansion policy.**  
AiZynthFinder supports pluggable policy networks via `aizynthfinder.context.policy.ExpansionPolicy`. RetroSynFormer's per-step template probability could replace or augment the default policy, letting AiZynthFinder's MCTS use RSF's transformer scores. This requires RSF to expose a `score_templates(smiles, templates) → list[float]` endpoint — add to Phase 2 only if MCTS integration is prioritised over latency.

### 3.2 AiZynthFinder service contract

The gateway expects AiZynthFinder to expose:

```
POST /api/api/find-routes
Content-Type: application/json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "max_transforms": 5,
  "iteration_limit": 500
}

200 Response:
{
  "routes": [
    {
      "fraction_in_stock": 1.0,
      "score": 0.95,
      "number_of_steps": 2,
      "reactions": [...]
    }
  ],
  "search_time": 12.3
}
```

If deploying AiZynthFinder from scratch, containerise with `aizynthfinder[rest]` and mount policy/stock files from GCS. Expose the REST API on port 5000 in a separate Cloud Run service (`aizynthfinder-mcts`).

### 3.3 Route Fusion Gateway (`src/route_fusion/`)

New lightweight FastAPI service, separate from `retrosynformer`:

```python
# POST /routes
# Fans out to both backends in parallel, merges, deduplicates, ranks.

class RouteRequest(BaseModel):
    smiles: str
    beam_width: int = 20
    max_transforms: int = 5
    backends: list[Literal["rsf", "aizynthfinder"]] = ["rsf", "aizynthfinder"]
    timeout_s: float = 60.0

class FusedRoute(BaseModel):
    source: str                  # "rsf" | "aizynthfinder" | "both"
    route_solved: bool
    score: float                 # unified score (RSF: trajectory_prob; AZ: score)
    n_steps: int
    reactions: list[dict]        # union of both schemas, normalised
    leaf_smiles: list[str]

class RouteResponse(BaseModel):
    smiles: str
    routes: list[FusedRoute]     # sorted by score descending, deduplicated
    rsf_elapsed_s: float | None
    aizynthfinder_elapsed_s: float | None
    total_elapsed_s: float
```

Deduplication: two routes are considered identical if their reaction SMARTS sequences are canonically identical (sort reactants within each step, compare).

Ranking: routes are ranked by `score`; RSF `trajectory_prob` and AiZynthFinder `fraction_in_stock × score` are normalised to [0, 1] before merging.

```
GET /health     → aggregate health from both backends
GET /backends   → show URL + status of each backend
```

---

## 4. Web Application

### 4.1 Purpose

A browser UI for medicinal chemists to:
1. Enter a target molecule (draw with JSME or paste SMILES)
2. Receive ranked synthesis routes from RSF + AiZynthFinder
3. View step-by-step reaction trees with structure images (RDKit SVG)
4. Compare and annotate routes; export to PDF

### 4.2 Architecture

```
Next.js (App Router) → deployed on Cloud Run (or Firebase Hosting + Cloud Run API)
  → calls Route Fusion Gateway (/routes) via fetch()
  → authenticated via Google OAuth (Identity Platform) for end users
  → backend-to-gateway calls authenticated via service account token
```

### 4.3 Key pages

| Route | Content |
|-------|---------|
| `/` | Molecule input (JSME sketcher + SMILES text field) |
| `/routes/[jobId]` | Ranked route list, expandable reaction trees |
| `/routes/[jobId]/compare` | Side-by-side RSF vs AiZynthFinder routes |
| `/admin` | Link to internal dashboard (separate domain) |

### 4.4 Molecule structure rendering

Use RDKit-js (WebAssembly) in the browser to render SMILES as SVG with no server round-trip. Fallback: request SVG from a `/render?smiles=...` endpoint backed by RDKit Python.

### 4.5 Async job pattern

Beam search (50 beams) can take 5–30 s. Use an async job pattern to avoid browser timeouts:

```
POST /routes → {job_id: "abc123"}        # returns immediately
GET  /routes/{job_id}/status → {status: "running" | "done" | "error"}
GET  /routes/{job_id}/result → RouteResponse
```

The fusion gateway stores results in Cloud Memorystore (Redis) with a 1-hour TTL. The web app polls `/status` every 2 s until done.

---

## 5. Google Cloud Deployment

### 5.1 Dockerfile — RetroSynFormer Inference

Multi-stage build; base is official PyTorch CUDA image (~8–9 GB total):

```dockerfile
# ── Stage 1: build wheel ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN pip install build && python -m build --wheel --outdir /dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir \
        /tmp/retrosynformer-*.whl \
        "fastapi>=0.115,<1" \
        "uvicorn[standard]>=0.30,<1" \
        rdkit \
        "rdchiral @ git+https://github.com/CoreThink-AI/rdchiral" \
        "reaction-utils==1.9.3" \
        "transformers>=4.35.0,<5" \
        pandas \
        "pydantic>=2.0,<3"

# Model artefacts: mounted at runtime from GCS (see below), not baked in.
# Small data files can be baked in to avoid cold-start GCS download latency:
COPY data/standard_building_blocks.csv       /app/data/
COPY data/standard_reaction_templates.pickle /app/data/

ENV MODEL_CONFIG_PATH=/app/config.yaml \
    MODEL_WEIGHTS_PATH=/app/model.pth \
    BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv \
    TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["retrosynformer-serve", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

`.dockerignore`:
```
.venv/
__pycache__/
*.pyc
*.pth
*.db
*.jsonl
results/
.git/
tests/
docs/
```

### 5.2 GCS model artifact strategy

Do not bake `model.pth` or `config.yaml` into the image — they change with each training run. Two options:

**Option A — Cloud Run volume mount (recommended):**  
Attach a GCS bucket as a FUSE volume at `/app/model/`. Container reads `model.pth` and `config.yaml` from there at startup. No image rebuild on model update; update the bucket object and restart the service.

```bash
gcloud run services update retrosynformer-inference \
  --add-volume=name=model-vol,type=cloud-storage,bucket=rsf-models \
  --add-volume-mount=volume=model-vol,mount-path=/app/model \
  --update-env-vars="MODEL_CONFIG_PATH=/app/model/config.yaml,MODEL_WEIGHTS_PATH=/app/model/model.pth"
```

**Option B — init container download:**  
A startup script runs `gcloud storage cp gs://rsf-models/... /app/model/` before uvicorn starts. Simpler but adds ~5–10s cold-start latency and requires wider IAM permissions in the container.

### 5.3 Cloud Run service — inference

```bash
gcloud run deploy retrosynformer-inference \
  --image "gcr.io/$PROJECT/rsf-inference:$SHA" \
  --region "$REGION" \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --cpu=8 \
  --memory=32Gi \
  --concurrency=4 \
  --min-instances=1 \
  --max-instances=4 \
  --timeout=120 \
  --set-env-vars="MODEL_CONFIG_PATH=/app/model/config.yaml,MODEL_WEIGHTS_PATH=/app/model/model.pth,BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv,TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle" \
  --set-secrets="API_KEY=retrosynformer-api-key:latest" \
  --no-allow-unauthenticated
```

Key flags:
- `--min-instances=1` — keeps one warm instance; model loading takes 30–60s so cold starts degrade UX
- `--concurrency=4` — beam search is GPU-bound; 4 concurrent requests share the GPU safely
- `--timeout=120` — well above the 30s P95 inference target
- `--no-allow-unauthenticated` — IAM token required; `X-API-Key` header is a second application-layer check

### 5.4 Cloud Run service — Route Fusion Gateway

```bash
gcloud run deploy route-fusion-gateway \
  --image "gcr.io/$PROJECT/route-fusion:$SHA" \
  --region "$REGION" \
  --cpu=2 \
  --memory=2Gi \
  --concurrency=80 \
  --min-instances=1 \
  --max-instances=10 \
  --timeout=90 \
  --set-env-vars="RSF_URL=https://retrosynformer-inference-xxx-uc.a.run.app,AIZYNTHFINDER_URL=https://aizynthfinder-mcts-xxx-uc.a.run.app,REDIS_URL=redis://10.x.x.x:6379" \
  --set-secrets="RSF_API_KEY=retrosynformer-api-key:latest" \
  --no-allow-unauthenticated
```

### 5.5 IAM wiring

```
Web App service account  →  Route Fusion Gateway (roles/run.invoker)
Fusion Gateway SA        →  RSF Inference (roles/run.invoker)
Fusion Gateway SA        →  AiZynthFinder MCTS (roles/run.invoker)
RSF Inference SA         →  GCS bucket rsf-models (roles/storage.objectViewer)
```

Service-to-service calls use `google.auth.transport.requests` to attach `Authorization: Bearer <identity-token>` from the instance metadata server — no credentials in env vars.

### 5.6 Secret management

```bash
# One-time setup
echo -n "$(openssl rand -hex 32)" | \
  gcloud secrets create retrosynformer-api-key \
    --data-file=- \
    --replication-policy=automatic

# Grant Cloud Run SA access
gcloud secrets add-iam-policy-binding retrosynformer-api-key \
  --role=roles/secretmanager.secretAccessor \
  --member="serviceAccount:rsf-inference-sa@$PROJECT.iam.gserviceaccount.com"
```

### 5.7 CI/CD (`cloudbuild.yaml`)

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    id: build-rsf
    args: ["build", "-t", "$_RSF_IMAGE", "-f", "Dockerfile", "."]

  - name: gcr.io/cloud-builders/docker
    id: push-rsf
    args: ["push", "$_RSF_IMAGE"]
    waitFor: ["build-rsf"]

  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    id: deploy-rsf
    entrypoint: gcloud
    args:
      - run
      - deploy
      - retrosynformer-inference
      - --image=$_RSF_IMAGE
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
    waitFor: ["push-rsf"]

substitutions:
  _RSF_IMAGE: gcr.io/$PROJECT_ID/rsf-inference:$SHORT_SHA
  _REGION: us-central1
options:
  machineType: E2_HIGHCPU_8
  substitution_option: ALLOW_LOOSE
```

### 5.8 Deployment script (`clouddeploy.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT=${GCP_PROJECT:?set GCP_PROJECT}
REGION=${GCP_REGION:-us-central1}
SHA=$(git rev-parse --short HEAD)
RSF_IMAGE="gcr.io/$PROJECT/rsf-inference:$SHA"

echo "==> Building and pushing RSF inference image"
gcloud builds submit --tag "$RSF_IMAGE" --timeout=20m --machine-type=e2-highcpu-8 .

echo "==> Deploying RSF inference service"
gcloud run deploy retrosynformer-inference \
  --image "$RSF_IMAGE" \
  --region "$REGION" \
  --gpu=1 --gpu-type=nvidia-l4 \
  --cpu=8 --memory=32Gi \
  --concurrency=4 --min-instances=1 --max-instances=4 \
  --timeout=120 \
  --set-secrets="API_KEY=retrosynformer-api-key:latest" \
  --no-allow-unauthenticated

echo "==> Service URL:"
gcloud run services describe retrosynformer-inference \
  --region "$REGION" --format "value(status.url)"
```

### 5.9 Scale-up path

| Request rate | Approach |
|-------------|----------|
| < 5 req/s | Cloud Run L4, `--concurrency 4`, `--min-instances 1` |
| 5–20 req/s | Increase `--max-instances`; tune `--concurrency` |
| 20–50 req/s | Second L4 instance; consider `beam_width` cap enforcement |
| > 50 req/s | GKE GPU node pool + Horizontal Pod Autoscaler; batched inference |

---

## 6. Environment Variables

### RSF Inference Service

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `MODEL_CONFIG_PATH` | yes | `/app/model/config.yaml` | Config YAML path inside container |
| `MODEL_WEIGHTS_PATH` | yes | `/app/model/model.pth` | Trained checkpoint path |
| `BUILDING_BLOCKS_PATH` | yes | `/app/data/standard_building_blocks.csv` | Building blocks CSV |
| `TEMPLATES_PATH` | yes | `/app/data/standard_reaction_templates.pickle` | Reaction templates |
| `API_KEY` | yes | (from Secret Manager) | Application-layer auth header |
| `PORT` | no | `8080` | Overridden by Cloud Run |

### Route Fusion Gateway

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `RSF_URL` | yes | `https://retrosynformer-inference-xxx.run.app` | RSF inference service URL |
| `AIZYNTHFINDER_URL` | yes | `https://aizynthfinder-mcts-xxx.run.app` | AiZynthFinder service URL |
| `RSF_API_KEY` | yes | (from Secret Manager) | API key to attach to RSF calls |
| `REDIS_URL` | yes | `redis://10.x.x.x:6379` | Cloud Memorystore for async job results |

---

## 7. `pyproject.toml` additions

```toml
[project.optional-dependencies]
serve = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
]

cuda = [
    "torch>=2.0.0",
]

[project.scripts]
retrosynformer-serve = "retrosynformer.scripts.serve:main"

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

## 8. Implementation Order

### Phase 1 — RSF Inference Service (2–3 days)

1. `src/retrosynformer/serve/schemas.py` — Pydantic models, no deps
2. `src/retrosynformer/serve/predictor.py` — wrap `RoutePredictor`; verify `_run_all_beams` loop against `inference.py`
3. `src/retrosynformer/serve/app.py` — FastAPI lifespan, `/predict`, `/health`
4. `src/retrosynformer/serve/__init__.py` — empty
5. `src/retrosynformer/scripts/serve.py` — CLI shim
6. `pyproject.toml` — `[serve]` + `[cuda]` extras, `retrosynformer-serve` script
7. Local smoke test (CPU): `retrosynformer-serve --port 8080` + curl aspirin
8. `Dockerfile` + `.dockerignore`
9. Docker build + local container test

### Phase 2 — Cloud Run Deployment (1–2 days)

10. GCS bucket `rsf-models`; upload `model.pth` + `config.yaml`
11. Service accounts + IAM bindings
12. Secret Manager — create `retrosynformer-api-key`
13. `clouddeploy.sh` + `cloudbuild.yaml`
14. `gcloud builds submit` → Cloud Run deploy
15. Integration test against live service URL

### Phase 3 — Route Fusion Gateway (2–3 days)

16. `src/route_fusion/` — new FastAPI service (separate repo or `services/route-fusion/`)
17. Parallel fan-out to RSF + AiZynthFinder using `asyncio.gather`
18. Deduplication and unified ranking
19. Async job pattern with Redis
20. Deploy to Cloud Run (`route-fusion-gateway`)

### Phase 4 — Web Application (3–5 days)

21. Next.js project with JSME molecule sketcher
22. `/routes` page — calls gateway, polls job status, renders routes
23. RDKit-js for client-side structure rendering
24. Google OAuth (Identity Platform) for end-user auth
25. Deploy to Cloud Run or Firebase Hosting

### Phase 5 — AiZynthFinder Policy Integration (optional, 3–5 days)

26. Expose `POST /score-templates` on RSF inference service
27. Implement `RetroSynFormerExpansionPolicy` plugin for AiZynthFinder
28. A/B test MCTS routes vs beam-search routes in production

---

## 9. Verification

```bash
# Local RSF service (CPU)
MODEL_CONFIG_PATH=results/config/standard.yaml \
MODEL_WEIGHTS_PATH=results/hypertune-compare2_small_structured_dropout/trial_001/model.pth \
BUILDING_BLOCKS_PATH=data/standard_building_blocks.csv \
TEMPLATES_PATH=data/standard_reaction_templates.pickle \
API_KEY=test-key \
  retrosynformer-serve --host 127.0.0.1 --port 8080

# Health
curl http://localhost:8080/health
# → {"status":"ok","model_loaded":true,"device":"cpu",...}

# Aspirin (known-solvable)
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "beam_width": 5}' \
  | python -m json.tool
# → n_solved >= 1, elapsed_s reasonable for CPU

# Invalid SMILES → 422
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "not-valid", "beam_width": 1}'

# Missing key → 403
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)O", "beam_width": 1}'

# Docker build
docker build -t rsf-inference:local .
docker run --rm -p 8080:8080 \
  -e MODEL_CONFIG_PATH=/app/config.yaml \
  -e MODEL_WEIGHTS_PATH=/app/model.pth \
  -e BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv \
  -e TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle \
  -e API_KEY=test-key \
  rsf-inference:local

# Cloud Run integration test
bash clouddeploy.sh
SERVICE_URL=$(gcloud run services describe retrosynformer-inference \
    --region=us-central1 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
API_KEY=$(gcloud secrets versions access latest --secret=retrosynformer-api-key)
curl -X POST "$SERVICE_URL/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "beam_width": 50}' \
  | python -m json.tool
# elapsed_s should be < 30 on L4 GPU
```

---

## 10. Open Questions

- **AiZynthFinder containerisation**: does a deployable image already exist, or does it need to be built from scratch? The policy/stock files required by AiZynthFinder can be large (10–40 GB) — plan for GCS-mounted volumes rather than baking into image.
- **Beam width cap**: `beam_width` is capped at 50 in Phase 1; raise to 200 after measuring P95 latency on GPU in staging.
- **Async vs sync gateway**: if 90% of requests complete in < 10s, drop the Redis job store and return synchronously to simplify the web app.
- **Model versioning in GCS**: use versioned object paths `gs://rsf-models/v1.2/model.pth` rather than `latest` so rollbacks are trivial.
- **Auth for the web app**: Google OAuth is the simplest. If the app needs external (non-Google) users, swap for Cognito or Auth0.

---

*Supersedes `PLAN-inference-endpoint.md`, `PLAN-cloud-run.md`.  
Internal dashboard and REST API spec remain in `PLAN-flask-admin.md` and `PLAN-rest-api.md`.*
