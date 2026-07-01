# Deploying rsgpt and rsgpt-embed to Cloud Run

Source repo: `../RSGPT/`  
GCP project: `biochem-db-by-hobs`  
Region: `us-central1`

---

## Architecture

Each service runs two processes inside the same container:

```
                ┌─────────────────────────────────────┐
  HTTP :8080    │  FastAPI / uvicorn  (app.py)         │
 ───────────►  │    tokenize SMILES                    │
                │    ▼                                  │
                │  llama-server :8181  (llama.cpp)     │
                │    /completion  (rsgpt)               │
                │    /embedding   (rsgpt-embed)         │
                │                                       │
                │  /models/  ← GCS volume (read-only)  │
                └─────────────────────────────────────┘
```

Models are served from a GCS bucket mounted as a read-only Cloud Run volume — no download at startup.

---

## Services

| Service | Image | Model | Endpoint |
|---|---|---|---|
| `rsgpt` | `gcr.io/biochem-db-by-hobs/rsgpt:latest` | `rsgpt-q4_k_m.gguf` (Q4_K_M, ~974 MB) | `POST /predict` |
| `rsgpt-embed` | `gcr.io/biochem-db-by-hobs/rsgpt-embed:latest` | `rsgpt.gguf` (F16, ~3.2 GB) | `POST /embed` |

Both models live in GCS bucket `gs://biochem-db-by-hobs-rsgpt-models/`.

---

## Build

### rsgpt

```bash
cd ../RSGPT
gcloud builds submit \
  --tag gcr.io/biochem-db-by-hobs/rsgpt:latest \
  --machine-type e2-highcpu-32 \
  --timeout 30m \
  .
```

Uses `Dockerfile` (Ubuntu 24.04 base, builds llama.cpp from source, installs Python deps from inline pip install).

### rsgpt-embed

```bash
cd ../RSGPT
gcloud builds submit \
  --config cloudbuild-embed.yaml \
  .
```

Uses `Dockerfile.embed` — identical to `Dockerfile` except it copies `embed_app.py` and `deploy/start-embed.sh`.

---

## Deploy

### rsgpt

```bash
cd ../RSGPT
gcloud run services replace deploy/cloudrun.yaml --region us-central1
```

### rsgpt-embed

```bash
cd ../RSGPT
gcloud run services replace deploy/cloudrun-embed.yaml --region us-central1
```

---

## Cloud Run configuration

| Setting | rsgpt | rsgpt-embed |
|---|---|---|
| CPU | 4 vCPU | 4 vCPU |
| Memory | 4 GiB | 6 GiB (larger model) |
| Min instances | 1 | 1 |
| Max instances | 3 | 3 |
| Concurrency | 1 | 1 |
| Request timeout | 1200 s | 120 s |
| Startup probe | `GET /health`, 60 s delay, 18 retries × 10 s | `GET /health`, 120 s delay, 30 retries × 10 s |
| Auth | unauthenticated (public) | unauthenticated (public) |
| GCS volume mount | `biochem-db-by-hobs-rsgpt-models` → `/models` | same |

---

## Container startup sequence

Both services follow the same pattern via their entrypoint scripts:

**`rsgpt`** — `deploy/start.sh`:
1. Wait for `/models/rsgpt-q4_k_m.gguf` to appear (GCS mount can take a few seconds)
2. Start `llama-server` on port 8181 (`-t $N_THREADS`, `--ctx-size $N_CTX`)
3. Poll `localhost:8181/health` until ready
4. Start `uvicorn app:app --port 8080 --workers 1`

**`rsgpt-embed`** — `deploy/start-embed.sh`:
1. Wait for `/models/rsgpt.gguf`
2. Start `llama-server --embedding --pooling mean` on port 8181
3. Poll health
4. Start `uvicorn embed_app:app --port 8080 --workers 1`

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/rsgpt-q4_k_m.gguf` | GGUF model file path |
| `VOCAB_PATH` | `/app/vocab.json` | HuggingFace tokenizer vocab |
| `LLAMA_PORT` | `8181` | Internal llama-server port |
| `N_CTX` | `512` | Context window (tokens) |
| `N_THREADS` | `4` | llama.cpp CPU threads |
| `PORT` | `8080` | External FastAPI port |

---

## API

### rsgpt — `POST /predict`

```json
// Request
{"smiles": "CCO", "beam_size": 5, "max_new_tokens": 80}

// Response
{
  "smiles": "CCO",
  "canonical_smiles": "CCO",
  "reactions": ["CC.O>>CCO", "..."]
}
```

Inference flow: canonicalize SMILES → encode to token IDs → call llama-server `/completion` `beam_size` times (first pass greedy temp=0, rest temp=0.7) → decode fragment tokens → assemble `reactants>>product` strings.

Prompt format: `<s><Isyn><O>{canonical_smiles}<F1>`

### rsgpt-embed — `POST /embed`

```json
// Request
{"smiles": "CCO"}

// Response
{
  "smiles": "CCO",
  "canonical_smiles": "CCO",
  "embedding": [0.123, -0.456, ...],   // 2048-dimensional
  "n_tokens": 7
}
```

Prompt format: `<s><O>{canonical_smiles}`

---

## Key files in `../RSGPT/`

| File | Purpose |
|---|---|
| `Dockerfile` | rsgpt image (llama.cpp + FastAPI) |
| `Dockerfile.embed` | rsgpt-embed image |
| `app.py` | FastAPI retrosynthesis app |
| `embed_app.py` | FastAPI embedding app |
| `vocab.json` | SMILES tokenizer vocabulary |
| `deploy/start.sh` | Container entrypoint for rsgpt |
| `deploy/start-embed.sh` | Container entrypoint for rsgpt-embed |
| `deploy/cloudrun.yaml` | Cloud Run service manifest for rsgpt |
| `deploy/cloudrun-embed.yaml` | Cloud Run service manifest for rsgpt-embed |
| `cloudbuild-embed.yaml` | Cloud Build config for rsgpt-embed |
| `deploy/README.md` | Prerequisites and IAM setup notes |

> **Note:** `deploy.sh` in the repo root is outdated — it references a GPU (NVIDIA L4) and PyTorch checkpoint from an earlier iteration. The current deployment is CPU-only llama.cpp; use `cloudbuild-embed.yaml` and the `gcloud builds submit` command above instead.

---

## Prerequisites

```bash
# Auth
gcloud auth login
gcloud config set project biochem-db-by-hobs
gcloud auth configure-docker

# Enable APIs (one-time)
gcloud services enable run.googleapis.com storage.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com

# Grant compute SA access to the model bucket (one-time)
gcloud storage buckets add-iam-policy-binding gs://biochem-db-by-hobs-rsgpt-models \
  --member="serviceAccount:125069248164-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```
