# RetroSynFormer Dashboard — Quick-Start Guide

Flask-Admin web UI + REST API for monitoring Optuna hyper-tuning studies and
training trials.  Reads directly from `results/` — no extra setup beyond
`uv sync --extra dashboard`.

## Live on taco

**<https://taco.tail9f615d.ts.net/>** — HTTPS, browser-trusted Let's Encrypt
certificate via Tailscale Serve (tailnet only; connect to Tailscale first).

Login: `admin` / see `.env.dashboard` on taco for the password.

## Install

```bash
uv sync --extra dashboard
source .venv/bin/activate
```

## Start

```bash
rs-dashboard                              # default: http://127.0.0.1:5050/
rs-dashboard --port 5051 --no-sync        # different port, skip startup sync
rs-dashboard --results /path/to/results   # point at a non-default results dir
rs-dashboard --debug                      # Flask debug mode (auto-reload)
```

The server syncs all `study.db` files found under `--results` on startup, then
auto-refreshes the index page every 30 seconds.

## Pages

| URL | What you see |
|-----|--------------|
| `/` | Summary cards + studies table + active-trial live rows |
| `/admin/study_admin/` | Flask-Admin table: filter/search/sort studies |
| `/admin/trial_admin/` | Flask-Admin table: filter/search/sort trials |
| `/trial/<study>/<N>/curves` | Per-trial learning curves (loss, accuracy, frac_solved) |

## REST API

All responses are `{"ok": true, "data": ...}`.

```bash
# List all studies
curl http://127.0.0.1:5050/api/v1/studies

# Filter by status or score
curl "http://127.0.0.1:5050/api/v1/studies?status=complete&min_score=0.3"

# Single study
curl http://127.0.0.1:5050/api/v1/studies/compare_small_standard_dropout_baseline

# Trials for a study
curl http://127.0.0.1:5050/api/v1/studies/compare_small_standard_dropout_baseline/trials
curl "http://127.0.0.1:5050/api/v1/studies/.../trials?state=COMPLETE&min_score=0.25"

# Single trial
curl http://127.0.0.1:5050/api/v1/studies/compare_small_standard_dropout_baseline/trials/2

# Per-epoch JSONL data for a trial
curl http://127.0.0.1:5050/api/v1/studies/.../trials/2/epochs

# Trigger a manual sync (re-reads all study.db files)
curl -X POST http://127.0.0.1:5050/api/v1/sync

# Sync a single study
curl -X POST http://127.0.0.1:5050/api/v1/studies/compare_small_standard_dropout_baseline/sync

# Rsync from taco then re-sync (requires rs-sync-results SSH access)
curl -X POST http://127.0.0.1:5050/api/v1/sync-remote \
     -H 'Content-Type: application/json' \
     -d '{"host": "taco", "remote_path": "code/corethink/retrosynformer/results/"}'
```

## Cloud Run monitoring

Pass `--cloud-run-url` to show a live health panel on the index page:

```bash
rs-dashboard --cloud-run-url https://retrosynformer-inference-xxxx.run.app
# or
CLOUD_RUN_URL=https://... rs-dashboard
```

## How sync works

On startup (and on every `POST /api/v1/sync`) the dashboard walks `results/`
for `study.db` files, reads each Optuna database, and upserts a lightweight
meta-DB at `results/dashboard.db` (SQLite).  The meta-DB is a derived cache —
deleting it and restarting rebuilds it from scratch in a few seconds.

When the same Optuna `study_name` appears in two different `study.db` files
(e.g. an rsync'd copy and the canonical hypertune directory), the dashboard
keeps whichever path is directly under `results/` and silently ignores the
duplicate.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RESULTS_ROOT` | `results/` | Root directory to scan |
| `DASHBOARD_DB_URL` | `sqlite:///<results>/dashboard.db` | SQLAlchemy DB URL |
| `CLOUD_RUN_URL` | _(empty)_ | Cloud Run service URL for health panel |
| `SECRET_KEY` | `dev-key-change-in-prod` | Flask session secret (change in prod) |
