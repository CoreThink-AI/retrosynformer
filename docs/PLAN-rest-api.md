# REST API Layer for RetroSynFormer Dashboard

## Context

The Flask-Admin dashboard (see `PLAN-flask-admin.md`) has ad-hoc API endpoints (`/api/sync`, `/api/launch`, etc.) bolted onto the admin UI. This plan designs a proper versioned REST API blueprint (`/api/v1/`) that:

- Provides a stable, typed interface for external tooling (CI scripts, notebooks, the `rs-sync-results` CLI, future UIs)
- Adds OpenAPI documentation auto-generated from Pydantic schemas
- Lives in the same Flask app so it shares the SQLAlchemy session and sync layer from Plan A

No new framework is introduced — the API is a Flask Blueprint with Pydantic for request/response validation. Pydantic is already a project dependency.

---

## 1. Pydantic Response Schemas (`dashboard/schemas.py`)

All responses are wrapped in a standard envelope:

```python
class Envelope(BaseModel, Generic[T]):
    ok: bool = True
    data: T
    error: str | None = None
```

Error responses: `{"ok": false, "data": null, "error": "message"}` with appropriate HTTP status.

### Resource schemas

```python
class TrialSchema(BaseModel):
    trial_number: int
    state: str                          # COMPLETE | RUNNING | FAIL | WAITING
    epoch_count: int
    duration_min: float | None
    optuna_score: float | None
    valid_loss: float | None
    valid_action_accuracy: float | None
    valid_route_accuracy: float | None
    fraction_targets_solved: float | None
    params: dict[str, Any]              # decoded from params_json
    trial_dir: str | None
    datetime_start: datetime | None
    datetime_complete: datetime | None
    synced_at: datetime

class StudySchema(BaseModel):
    study_name: str
    status: str
    objective_metric: str | None
    direction: str
    n_trials: int
    n_complete: int
    n_running: int
    n_failed: int
    best_score: float | None
    mean_score: float | None
    std_score: float | None
    best_trial_number: int | None
    param_importance: dict[str, float] | None   # decoded from param_importance_json
    last_synced_at: datetime | None
    db_path: str
    config_path: str | None
    pid: int | None

class EpochSchema(BaseModel):
    epoch: int
    train_loss: float
    train_action_accuracy: float
    train_route_accuracy: float
    valid_loss: float
    valid_action_accuracy: float
    valid_route_accuracy: float
    seconds_per_epoch: float

class StudyEventSchema(BaseModel):
    id: int
    event: str
    trial_number: int | None
    ts: datetime | None
    payload: dict[str, Any]             # decoded from payload_json

class ProcessSchema(BaseModel):
    id: int
    study_name: str
    pid: int
    command: list[str]                  # decoded from command_json
    started_at: datetime
    stopped_at: datetime | None
    exit_code: int | None
    status: str
```

### Request schemas

```python
class LaunchRequest(BaseModel):
    study_name: str
    config_path: str
    n_trials: int = 20
    n_epochs: int = 200
    dataset: Literal["small", "standard", "large"] = "small"

class EnqueueRequest(BaseModel):
    params: dict[str, Any]             # {n_heads: 4, lr: 0.001, …}
    skip_if_exists: bool = True

class SyncRemoteRequest(BaseModel):
    host: str = "taco"
    remote_path: str = "code/corethink/retrosynformer/results/"
    dry_run: bool = False
```

---

## 2. Endpoint Design

All endpoints live under the `/api/v1/` prefix. Registered as a Flask Blueprint named `api_v1`.

### Studies

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/studies` | List all studies; supports `?status=active`, `?min_score=0.3`, `?study=substr` query params |
| `GET` | `/api/v1/studies/{study_name}` | Single study with full stats |
| `POST` | `/api/v1/studies/{study_name}/sync` | Re-read Optuna files for this study → update meta-DB |
| `POST` | `/api/v1/studies/{study_name}/launch` | Body: `LaunchRequest` → starts subprocess, returns PID |
| `POST` | `/api/v1/studies/{study_name}/stop` | Send SIGTERM to `Study.pid`; 404 if no PID |
| `DELETE` | `/api/v1/studies/{study_name}` | Remove Study + Trials + Events from meta-DB (does not delete files) |

### Trials

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/studies/{study_name}/trials` | List trials; `?state=RUNNING`, `?min_score=0.3`, `?sort=optuna_score` |
| `GET` | `/api/v1/studies/{study_name}/trials/{trial_num}` | Single trial detail |
| `GET` | `/api/v1/studies/{study_name}/trials/{trial_num}/epochs` | Per-epoch metrics from JSONL (on-demand read, not from DB) |
| `POST` | `/api/v1/studies/{study_name}/trials/enqueue` | Body: `EnqueueRequest` → `optuna.load_study().enqueue_trial()` |

### Events & Processes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/studies/{study_name}/events` | Study event log; `?event=trial_end`, `?trial=3` |
| `GET` | `/api/v1/processes` | All tracked subprocesses with live status check |

### Global Actions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sync` | Sync all discovered studies |
| `POST` | `/api/v1/sync-remote` | Body: `SyncRemoteRequest` → rsync from remote host |
| `GET` | `/api/v1/openapi.json` | Auto-generated OpenAPI 3.1 spec |
| `GET` | `/api/v1/docs` | Swagger UI (served via `flask-swagger-ui`) |

---

## 3. Implementation Pattern

Each endpoint follows this pattern (explicit validate+serialize, no `flask-pydantic` magic):

```python
# dashboard/api.py
from flask import Blueprint, jsonify, request, current_app
from .models import Study, Trial, db
from .schemas import StudySchema
from .sync import sync_study

api = Blueprint("api_v1", __name__, url_prefix="/api/v1")

@api.get("/studies")
def list_studies():
    q = db.session.query(Study)
    status = request.args.get("status")
    if status:
        q = q.filter(Study.status == status)
    min_score = request.args.get("min_score", type=float)
    if min_score is not None:
        q = q.filter(Study.best_score >= min_score)
    study_filter = request.args.get("study")
    if study_filter:
        q = q.filter(Study.study_name.contains(study_filter))
    rows = q.order_by(Study.last_synced_at.desc()).all()
    data = [StudySchema.model_validate(s, from_attributes=True).model_dump() for s in rows]
    return jsonify({"ok": True, "data": data})

@api.get("/studies/<study_name>/trials/<int:trial_num>/epochs")
def trial_epochs(study_name: str, trial_num: int):
    """Read train_progress.jsonl on-demand — not from meta-DB."""
    trial = (
        db.session.query(Trial)
        .join(Study)
        .filter(Study.study_name == study_name, Trial.trial_number == trial_num)
        .first_or_404()
    )
    from retrosynformer.scripts.plot_learning_curves import _load_jsonl
    import os
    jsonl_path = os.path.join(trial.trial_dir, "train_progress.jsonl")
    df = _load_jsonl(jsonl_path)
    return jsonify({"ok": True, "data": df.to_dict(orient="records")})
```

Error handling via registered `@api.errorhandler`:

```python
@api.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "data": None, "error": str(e)}), 404

@api.errorhandler(Exception)
def generic_error(e):
    return jsonify({"ok": False, "data": None, "error": str(e)}), 500
```

---

## 4. OpenAPI Documentation

Use `flask-swagger-ui` to serve Swagger UI at `/api/v1/docs`. The OpenAPI spec is hand-authored in `dashboard/openapi.yaml`:

```python
from flask_swagger_ui import get_swaggerui_blueprint
swaggerui_bp = get_swaggerui_blueprint(
    "/api/v1/docs",
    "/api/v1/openapi.json",
    config={"app_name": "RetroSynFormer API"},
)
app.register_blueprint(swaggerui_bp)
```

---

## 5. Process Status Live-Check

`GET /api/v1/processes` calls `os.kill(pid, 0)` for each `Process` row to check liveness and auto-updates `status` + `stopped_at` for any that have exited:

```python
for proc in db.session.query(Process).filter_by(status="running").all():
    try:
        os.kill(proc.pid, 0)   # 0 = existence check, no signal sent
    except ProcessLookupError:
        proc.status = "completed"
        proc.stopped_at = datetime.utcnow()
db.session.commit()
```

---

## 6. New Dependencies (added to `[dashboard]` extra)

```toml
dashboard = [
    "flask>=3.0,<4",
    "flask-admin>=1.6,<2",
    "flask-sqlalchemy>=3.1,<4",
    "wtforms>=3.1,<4",
    "flask-swagger-ui>=4.11,<5",
]
```

No additional deps: Pydantic is already in base deps.

---

## 7. Files Modified / Created

| Action | Path |
|--------|------|
| Create | `src/retrosynformer/dashboard/schemas.py` |
| Create | `src/retrosynformer/dashboard/api.py` |
| Create | `src/retrosynformer/dashboard/openapi.yaml` |
| Modify | `src/retrosynformer/dashboard/__init__.py` (register `api_v1` blueprint + swagger) |
| Modify | `pyproject.toml` (add `flask-swagger-ui` to `[dashboard]`) |
| Modify | `src/retrosynformer/dashboard/views.py` (replace ad-hoc `/api/*` routes with redirects to `/api/v1/`) |

The Plan A ad-hoc endpoints (`/api/sync`, `/api/launch`, `/api/stop/<name>`, `/api/enqueue/<name>`, `/api/sync-remote`) are replaced by their `/api/v1/` equivalents. The Flask-Admin action buttons are updated to call `/api/v1/` URLs.

---

## 8. Verification

```bash
# OpenAPI spec is valid JSON
curl http://localhost:5050/api/v1/openapi.json | python -m json.tool

# Swagger UI loads
open http://localhost:5050/api/v1/docs

# List studies
curl http://localhost:5050/api/v1/studies | python -m json.tool

# Filter by status
curl "http://localhost:5050/api/v1/studies?status=active"

# Trial epoch data
curl http://localhost:5050/api/v1/studies/compare2_small_structured_dropout/trials/1/epochs \
  | python -m json.tool

# Enqueue
curl -X POST http://localhost:5050/api/v1/studies/compare2_small_structured_dropout/trials/enqueue \
  -H "Content-Type: application/json" \
  -d '{"params": {"n_heads": 2, "n_layers": 3, "head_dim": 128, "lr": 0.001, "dropout": 0.1}}'

# Process status check
curl http://localhost:5050/api/v1/processes | python -m json.tool

# Error response shape
curl http://localhost:5050/api/v1/studies/nonexistent | python -m json.tool
# → {"ok": false, "data": null, "error": "404 Not Found: ..."}
```
