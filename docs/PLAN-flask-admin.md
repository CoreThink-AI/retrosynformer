# Flask-Admin Dashboard for RetroSynFormer Optuna Experiments

## Context

RetroSynFormer runs multi-trial Optuna hyperparameter searches that produce scattered artefacts: `results/hypertune-{name}/study.db` (Optuna SQLite), `run.jsonl` (study events), and `trial_NNN/train_progress.jsonl` (per-epoch metrics). The only monitoring today is a set of CLI scripts. This plan adds a web dashboard for monitoring and controlling experiments without touching any existing CLI scripts.

No Flask/Flask-Admin/SQLAlchemy is currently in the project. `sqlalchemy>=2.0` is already a transitive dep via `alembic`.

---

## What We're Building

`src/retrosynformer/dashboard/` — a new optional sub-package:

```
src/retrosynformer/dashboard/
    __init__.py          # create_app() factory
    models.py            # SQLAlchemy ORM models (dashboard.db)
    sync.py              # Populate meta-DB from Optuna files
    views.py             # Flask-Admin ModelViews + custom Blueprint
    templates/dashboard/
        index.html       # Overview page
        trial_curves.html  # Chart.js learning curves
src/retrosynformer/scripts/dashboard.py  # CLI: retrosynformer-dashboard
```

---

## 1. SQLAlchemy ORM Models (`models.py`)

Meta-DB lives at `results/dashboard.db`. All five models use SQLAlchemy 2.0 `DeclarativeBase` / `Mapped` style.

### `Experiment`
Groups studies sharing a config family.

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| name | String(256) unique | e.g. "small-structured-dropout" |
| config_path | String(512) | Path to base YAML |
| description | Text | |
| tags | String(512) | Comma-separated |
| created_at / updated_at | DateTime | |

Relationship: `studies` → `Study` (one-to-many)

### `Study`
One row per `results/hypertune-{name}/` directory.

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| experiment_id | FK → experiments.id nullable | |
| study_name | String(256) unique | Optuna study name |
| db_path | String(512) | Absolute path to study.db |
| run_jsonl_path | String(512) nullable | Absolute path to run.jsonl |
| config_path | String(512) nullable | YAML used for this study |
| objective_metric | String(64) | e.g. "valid_route_accuracy" |
| direction | String(8) | "maximize" |
| status | String(16) | "active" \| "complete" \| "failed" \| "unknown" |
| n_trials / n_complete / n_running / n_failed | Integer | Denormalised counts |
| best_score / mean_score / std_score | Float nullable | Denormalised stats |
| best_trial_number | Integer nullable | |
| param_importance_json | Text nullable | JSON from `optuna.importance` (cached) |
| pid | Integer nullable | PID of running hypertune process on this host |
| last_synced_at | DateTime nullable | |
| created_at | DateTime | |

Relationships: `experiment` (many-to-one), `trials` (one-to-many), `events` (one-to-many)

### `Trial`
Denormalised summary — one row per Optuna trial. Hyperparams stored as JSON to avoid schema churn when YAML search spaces change.

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| study_id | FK → studies.id | |
| trial_number | Integer | 0-based; maps to `trial_NNN` directory |
| optuna_trial_id | Integer | From Optuna `trials.trial_id`; used for incremental sync |
| state | String(8) | COMPLETE / RUNNING / FAIL / WAITING |
| datetime_start / datetime_complete | DateTime nullable | |
| duration_min | Float nullable | |
| params_json | Text nullable | `{"n_heads": 4, "lr": 0.001, …}` (decoded) |
| optuna_score | Float nullable | From `trial_values.value` |
| valid_loss | Float nullable | From `run.jsonl trial_end.results.accuracy` |
| valid_action_accuracy | Float nullable | |
| valid_route_accuracy | Float nullable | |
| fraction_targets_solved | Float nullable | |
| epoch_count | Integer | Line count of `train_progress.jsonl` (last contiguous run) |
| trial_dir | String(512) nullable | Absolute path to `trial_NNN/` |
| synced_at | DateTime | |

Unique constraint: `(study_id, trial_number)`

### `StudyEvent`
Append-only log from `run.jsonl`. `file_offset` enables incremental reads.

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| study_id | FK → studies.id | |
| event | String(32) | study_start / trial_start / trial_end / trial_complete / trial_fail / study_end / log |
| trial_number | Integer nullable | null for study-level events |
| ts | DateTime nullable | From event's `"ts"` field |
| payload_json | Text | Full event dict as JSON |
| file_offset | Integer | Byte offset in run.jsonl (for incremental read) |

### `Process`
Tracks subprocesses launched from the dashboard.

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| study_name | String(256) | |
| pid | Integer | OS PID |
| command_json | Text | `argv` list as JSON |
| started_at | DateTime | |
| stopped_at / exit_code | DateTime / Integer nullable | |
| status | String(16) | running / stopped / completed / killed |

---

## 2. Sync Layer (`sync.py`)

**Key principle:** Optuna's `study.db` and the JSONL files are authoritative. The meta-DB only stores derived/summary data and is rebuilt by `sync_all()`.

**Reuse existing utilities:**
- `from retrosynformer.study import to_dfs, dfs_to_trials_df` — decode Optuna params
- `from retrosynformer.scripts.plot_learning_curves import _load_jsonl, _jsonl_stats` — JSONL reading with epoch-reset detection

### `discover_study_dbs(root) → list[str]`
`glob("results/**/study.db", recursive=True)`, dedup by `os.path.realpath()` to skip the `results/hypertune` symlink alias.

### `sync_all(db_session, root, force=False) → dict`
Calls `sync_study()` for each discovered path. Returns `{synced, skipped, errors}`.

### `sync_study(db_session, db_path, root)`
1. `to_dfs(db_path)` — load Optuna tables into DataFrames
2. Upsert `Study` row (insert if new; skip if `study.db` mtime < `last_synced_at` and not `force`)
3. Call `sync_trials()`; call `sync_events()`
4. Recompute Study aggregates (`n_complete`, `best_score`, etc.)
5. Compute `param_importance_json` if ≥ 3 COMPLETE trials (via `optuna.importance.get_param_importances`)

### `sync_trials(db_session, study_row, dfs)`
Incremental algorithm:
```
existing = {t.optuna_trial_id: t for t in study_row.trials}

for each row in dfs["trials"]:
    if row.trial_id NOT in existing:
        params = decode via dfs["trial_params"]
        score  = from dfs["trial_values"]
        metrics = from run.jsonl trial_end event for this trial_num
        epoch_count = _jsonl_stats(trial_dir/train_progress.jsonl)["n_epochs"]
        INSERT Trial(...)

    elif row.state == "RUNNING" and existing[id].state == "RUNNING":
        # cheap update — only re-read epoch count
        UPDATE Trial SET epoch_count = _jsonl_stats(...)["n_epochs"], synced_at = now

    elif row.state in {"COMPLETE","FAIL"} and existing[id].state == "RUNNING":
        # terminal transition — fill in metrics
        score, metrics, epoch_count = (full read)
        UPDATE Trial SET state=…, score=…, metrics=…, epoch_count=…

    # else COMPLETE/FAIL unchanged — skip
```

### `sync_events(db_session, study_row)`
Read `run.jsonl` starting from `MAX(file_offset)` of existing `StudyEvent` rows for this study. Append new events with current byte offset recorded.

---

## 3. Flask-Admin ModelViews (`views.py`)

### `ExperimentAdmin(ModelView)`
- `column_list`: name, config_path, tags, studies count, created_at
- `column_searchable_list`: name, tags, description
- `form_columns`: name, config_path, description, tags

### `StudyAdmin(ModelView)`
- `column_list`: study_name, status badge, n_trials, n_complete, n_running, best_score, mean±std, objective_metric, last_synced_at
- `column_filters`: status, best_score >=, n_trials <=
- Custom actions: **Sync** (calls `sync_study` for selected rows), **Stop** (sends SIGTERM to `study.pid`)

### `TrialAdmin(ModelView)`
- `column_list`: study name, trial_number, state, epoch_count, duration_min, optuna_score, valid_route_accuracy, fraction_targets_solved, params summary (first 3 keys), 📈 curves link
- `column_filters`: state, optuna_score >=, epoch_count >=, fraction_targets_solved >=
- `can_create = False`, `can_delete = False`
- Custom action: **Enqueue** — calls `optuna.load_study().enqueue_trial(params)` on selected trial's study

### `StudyEventAdmin(ModelView)`
- `column_list`: study name, event, trial_number, ts, payload summary (first 80 chars)
- `can_create = can_edit = can_delete = False`

---

## 4. Custom Flask Blueprint (`views.py`, continued)

### `GET /` — Dashboard Index
Renders `index.html`: study summary table + running-trial live panel.  
Uses `<meta http-equiv="refresh" content="30">` for auto-refresh (no SSE/WebSocket needed).

### `GET /trial/<study_name>/<trial_num>/curves`
Reads `trial_NNN/train_progress.jsonl` via `_load_jsonl()` → returns `trial_curves.html` with Chart.js JSON payload.  
**No matplotlib / no X11** — chart rendered in-browser from JSON data.

### `POST /api/sync`
Calls `sync_all(db.session, app.config["RESULTS_ROOT"])`. Returns `{synced, skipped, errors}` JSON.

### `POST /api/launch`
Body: `{study_name, config_path, n_trials, n_epochs, dataset}`  
Runs `rs-hypertune …` via `subprocess.Popen`, saves PID to `Process` table and `Study.pid`.

### `POST /api/stop/<study_name>`
Sends `SIGTERM` to `Study.pid`; clears the PID column.

### `POST /api/enqueue/<study_name>`
Body: `{params: {…}}`  
Opens `optuna.load_study(storage=f"sqlite:///{db_path}")` and calls `study.enqueue_trial(params, skip_if_exists=True)`.

### `POST /api/sync-remote`
Body: `{host, remote_path, dry_run}`  
Delegates to `retrosynformer.rsync.sync()`.

---

## 5. Application Factory (`__init__.py`)

```python
def create_app(results_root=None, db_url=None, initial_sync=True) -> Flask:
    # resolve paths, configure SQLALCHEMY_DATABASE_URI
    # db.init_app(app); Base.metadata.create_all(...)
    # register Flask-Admin with bootstrap4 template_mode
    # register custom Blueprint
    # if initial_sync: sync_all(db.session, results_root)
    return app
```

`DashboardIndexView(AdminIndexView)` overrides Flask-Admin's `/admin/` to redirect to the custom `/` overview.

---

## 6. New Dependencies (`pyproject.toml`)

```toml
[project.optional-dependencies]
dashboard = [
    "flask>=3.0,<4",
    "flask-admin>=1.6,<2",
    "flask-sqlalchemy>=3.1,<4",
    "wtforms>=3.1,<4",
]

[project.scripts]
retrosynformer-dashboard = "retrosynformer.scripts.dashboard:main"
```

Install: `uv sync --extra dashboard`

---

## 7. CLI (`scripts/dashboard.py`)

```
retrosynformer-dashboard [--host 127.0.0.1] [--port 5050]
                         [--results results/] [--db sqlite:///...]
                         [--debug] [--no-sync]
```

Graceful ImportError if `[dashboard]` extra not installed.

---

## 8. Files Modified / Created

| Action | Path |
|--------|------|
| Create | `src/retrosynformer/dashboard/__init__.py` |
| Create | `src/retrosynformer/dashboard/models.py` |
| Create | `src/retrosynformer/dashboard/sync.py` |
| Create | `src/retrosynformer/dashboard/views.py` |
| Create | `src/retrosynformer/dashboard/templates/dashboard/index.html` |
| Create | `src/retrosynformer/dashboard/templates/dashboard/trial_curves.html` |
| Create | `src/retrosynformer/scripts/dashboard.py` |
| Modify | `pyproject.toml` (add `[dashboard]` extra + CLI entry) |
| No change | All existing CLI scripts, trainer.py, runner.py, study.py |

---

## 9. Verification

```bash
# Install dashboard extra
uv sync --extra dashboard

# Start with no initial sync (fast)
retrosynformer-dashboard --port 5050 --no-sync --debug

# Trigger sync via API
curl -X POST http://localhost:5050/api/sync

# Check admin pages load
open http://localhost:5050/admin/study/
open http://localhost:5050/admin/trial/

# Check curves page for an existing trial
open http://localhost:5050/trial/compare2_small_structured_dropout/1/curves

# Enqueue a trial
curl -X POST http://localhost:5050/api/enqueue/compare2_small_structured_dropout \
  -H "Content-Type: application/json" \
  -d '{"params": {"n_heads": 2, "n_layers": 3, "head_dim": 128, "lr": 0.001, "dropout": 0.1, "structured_dropout_bottleneck": 64}}'

# Verify existing CLIs are unaffected
rs-show-study results/hypertune-compare2_small_structured_dropout/study.db
rs-plot-learning-curves --yscale linear --study structured

# Verify incremental sync doesn't duplicate StudyEvent rows
curl -X POST http://localhost:5050/api/sync
curl -X POST http://localhost:5050/api/sync
# Row count should be same after second sync
```
