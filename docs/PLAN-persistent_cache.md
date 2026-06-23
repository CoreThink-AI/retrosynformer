# Plan: PostgreSQL Persistent Cache for Retrosynthesis Routes

## Goal

Replace (or back) the in-memory LRU cache in `serve/app.py` with a Cloud SQL
(PostgreSQL) persistent cache that:

- **Retains every entry indefinitely** — rows are never deleted or expired.
- Scopes entries to a specific model version (SHA-256 hash + human label).
- Records when each entry was computed, last served, and how many times.
- Supports a **flexible text tag array** for provenance, review status,
  pipeline state, and classification — any code or person can add/remove tags
  without a schema migration.
- Serves as the **staging area** before routes/reactions are ported to the
  ORD-style relational database in `../biochem-db/`.

---

## Schema

```sql
-- migrations/001_route_cache.sql

CREATE TABLE route_cache (
    id               BIGSERIAL PRIMARY KEY,

    -- Cache key (unique per model version)
    canonical_smiles TEXT        NOT NULL,
    max_routes       SMALLINT    NOT NULL,
    max_steps        SMALLINT    NOT NULL,
    model_id         TEXT        NOT NULL,   -- human label, e.g. "large-23-layer-trial003"
    model_sha256     TEXT        NOT NULL,   -- SHA-256 of model.pth

    -- Cached payload
    route_dicts      JSONB       NOT NULL,   -- list[RouteResponse dict]
    n_routes         SMALLINT    NOT NULL,
    n_solved         SMALLINT    NOT NULL,

    -- Timestamps and access tracking
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count     INT         NOT NULL DEFAULT 0,

    -- Flexible tag array — no schema migration needed to add new tag types.
    -- Reserved tag conventions (see Tag conventions section below):
    --   'invalid'             — skip this entry during cache lookup
    --   'model:<model_id>'    — provenance; added automatically on insert
    --   'hobson-validated'    — reviewed and approved by Hobson
    --   'hobson-flagged'      — flagged for follow-up by Hobson
    --   'reviewed'            — any chemist review complete
    --   'ord-queued'          — selected for ORD import pipeline
    --   'ord-imported'        — successfully written to biochem-db ORD tables
    tags             TEXT[]      NOT NULL DEFAULT '{}',

    UNIQUE (canonical_smiles, max_routes, max_steps, model_sha256)
);

-- GIN index: fast containment queries — WHERE 'hobson-validated' = ANY(tags)
CREATE INDEX idx_rc_tags ON route_cache USING GIN (tags);

-- Partial lookup index: cache hits exclude entries tagged 'invalid'
CREATE INDEX idx_rc_lookup ON route_cache
    (canonical_smiles, max_routes, max_steps, model_sha256)
    WHERE NOT ('invalid' = ANY(tags));

CREATE INDEX idx_rc_model     ON route_cache (model_sha256);
CREATE INDEX idx_rc_model_id  ON route_cache (model_id);
CREATE INDEX idx_rc_created   ON route_cache (created_at);
CREATE INDEX idx_rc_accessed  ON route_cache (last_accessed_at);
CREATE INDEX idx_rc_solved    ON route_cache (n_solved) WHERE n_solved > 0;
```

---

## Tag conventions

Tags are free-form strings. The table below documents the reserved conventions;
nothing enforces them at the DB level, which keeps the schema stable as new
pipeline stages are added.

| Tag | Set by | Meaning |
|-----|--------|---------|
| `invalid` | app / script | Skip during cache lookup; entry stays in DB |
| `model:<model_id>` | app on insert | E.g. `model:large-23-layer-trial003` |
| `hobson-validated` | chemist script / UI | Route reviewed and approved |
| `hobson-flagged` | chemist script / UI | Route needs follow-up |
| `reviewed` | any reviewer | Chemist review complete (any reviewer) |
| `cyclic` | post-processing | Route contains a cyclic disconnection |
| `unsolvable` | post-processing | All routes failed to reach building blocks |
| `ord-queued` | ORD pipeline | Selected for import to `biochem-db` |
| `ord-imported` | ORD pipeline | Written to ORD reaction/route tables |
| `classification:<label>` | ML pipeline | E.g. `classification:high-confidence` |

Add new conventions here as the pipeline grows; no migration required.

---

## Cache key design

| Field | Source | Why |
|---|---|---|
| `canonical_smiles` | `Chem.MolToSmiles(mol)` — RDKit canonical | Normalises any SMILES variant |
| `max_routes` | request field | Different beam widths → different route sets |
| `max_steps` | request field | Different depth limits → different route sets |
| `model_sha256` | `predictor.model_sha256_hash` | Scopes to exact weights; survives re-labelling |

`model_id` is stored for readability but is not part of the unique key.

---

## Tag management queries

### Add a tag to one entry

```sql
UPDATE route_cache
SET    tags = array_append(tags, $1)
WHERE  id = $2
  AND  NOT ($1 = ANY(tags));   -- idempotent
```

### Remove a tag

```sql
UPDATE route_cache
SET    tags = array_remove(tags, $1)
WHERE  id = $2;
```

### Mark a route invalid (still persists, just skipped by cache lookup)

```sql
UPDATE route_cache
SET    tags = array_append(tags, 'invalid')
WHERE  canonical_smiles = $1 AND model_sha256 = $2
  AND  NOT ('invalid' = ANY(tags));
```

### Bulk-tag all entries for a model

```sql
-- E.g. mark all trial003 entries as provenance-tagged
UPDATE route_cache
SET    tags = array_append(tags, 'model:large-23-layer-trial003')
WHERE  model_sha256 = $1
  AND  NOT ('model:large-23-layer-trial003' = ANY(tags));
```

### Invalidate all entries for a superseded model

```sql
UPDATE route_cache
SET    tags = array_append(tags, 'invalid')
WHERE  model_sha256 = $old_sha256
  AND  NOT ('invalid' = ANY(tags));
```

### Find entries ready for ORD import

```sql
SELECT id, canonical_smiles, model_id, route_dicts, tags
FROM   route_cache
WHERE  'hobson-validated' = ANY(tags)
  AND  NOT ('ord-imported' = ANY(tags))
ORDER  BY created_at;
```

### Find unreviewed solved routes

```sql
SELECT id, canonical_smiles, n_solved, model_id, created_at
FROM   route_cache
WHERE  n_solved > 0
  AND  NOT ('reviewed'         = ANY(tags))
  AND  NOT ('hobson-validated' = ANY(tags))
  AND  NOT ('hobson-flagged'   = ANY(tags))
ORDER  BY n_solved DESC, created_at DESC;
```

---

## Python integration

### Dependencies

```toml
# pyproject.toml
"asyncpg>=0.29",
"cloud-sql-python-connector[asyncpg]>=1.9",
```

### Connection pool (lifespan)

```python
from google.cloud.sql.connector import AsyncConnector, IPTypes
import asyncpg

_db_pool: asyncpg.Pool | None = None

async def _create_db_pool() -> asyncpg.Pool:
    connector = AsyncConnector()

    async def _connect(conn_name: str, **kwargs) -> asyncpg.Connection:
        return await connector.connect_async(
            conn_name, "asyncpg",
            user=os.environ["DB_USER"],
            password=os.environ.get("DB_PASSWORD"),
            db=os.environ["DB_NAME"],
            ip_type=IPTypes.PRIVATE,
            **kwargs,
        )

    return await asyncpg.create_pool(
        dsn=None,
        connect=lambda: _connect(os.environ["CLOUD_SQL_INSTANCE"]),
        min_size=1,
        max_size=5,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_pool
    _state["sem"] = asyncio.Semaphore(1)
    _state["predictor"] = ModelPredictor(...)
    if os.environ.get("CLOUD_SQL_INSTANCE"):
        _db_pool = await _create_db_pool()
    yield
    if _db_pool:
        await _db_pool.close()
    _state.clear()
```

### Cache lookup (excludes 'invalid'-tagged entries)

```python
async def _db_cache_get(
    pool: asyncpg.Pool,
    canon: str,
    max_routes: int,
    max_steps: int,
    model_sha256: str,
) -> list | None:
    row = await pool.fetchrow(
        """
        UPDATE route_cache
        SET    last_accessed_at = NOW(),
               access_count     = access_count + 1
        WHERE  canonical_smiles = $1
          AND  max_routes       = $2
          AND  max_steps        = $3
          AND  model_sha256     = $4
          AND  NOT ('invalid' = ANY(tags))
        RETURNING route_dicts
        """,
        canon, max_routes, max_steps, model_sha256,
    )
    return list(row["route_dicts"]) if row else None
```

### Cache write (auto-tags with model provenance)

```python
async def _db_cache_put(
    pool: asyncpg.Pool,
    canon: str,
    max_routes: int,
    max_steps: int,
    model_id: str,
    model_sha256: str,
    route_dicts: list,
) -> None:
    n_solved = sum(1 for r in route_dicts if r.get("all_leaves_purchasable"))
    model_tag = f"model:{model_id}"
    await pool.execute(
        """
        INSERT INTO route_cache
            (canonical_smiles, max_routes, max_steps,
             model_id, model_sha256, route_dicts,
             n_routes, n_solved, tags)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, ARRAY[$9])
        ON CONFLICT (canonical_smiles, max_routes, max_steps, model_sha256)
        DO UPDATE SET
            last_accessed_at = NOW(),
            access_count     = route_cache.access_count + 1
        -- Never overwrite route_dicts or tags on conflict —
        -- the original computation is the canonical record.
        """,
        canon, max_routes, max_steps,
        model_id, model_sha256,
        route_dicts,
        len(route_dicts), n_solved,
        model_tag,
    )
```

### Updated `/retrosynthesis` (three-tier lookup)

```python
@app.post("/retrosynthesis", ...)
async def retrosynthesis(request: Request, req: RetrosynthesisRequest):
    ...
    canon = Chem.MolToSmiles(mol)
    model_sha256 = predictor.model_sha256_hash or ""

    # 1. In-memory LRU (microseconds)
    mem_key = (canon, req.max_routes, req.max_steps, model_sha256)
    cached = _cache_get(mem_key)
    if cached is not None:
        return _build_response(req.smiles, canon, cached)

    # 2. PostgreSQL (milliseconds)
    if _db_pool:
        cached = await _db_cache_get(
            _db_pool, canon, req.max_routes, req.max_steps, model_sha256
        )
        if cached is not None:
            _cache_put(mem_key, cached)
            return _build_response(req.smiles, canon, cached)

    # 3. Inference (seconds to minutes)
    async with _state["sem"]:
        ...
        route_dicts = future.result()

    _cache_put(mem_key, route_dicts)
    if _db_pool:
        asyncio.create_task(
            _db_cache_put(
                _db_pool, canon, req.max_routes, req.max_steps,
                predictor.model_id, model_sha256, route_dicts,
            )
        )  # fire-and-forget; does not block the response

    return _build_response(req.smiles, canon, route_dicts)
```

Note: `model_id` (e.g. `"large-23-layer-trial003"`) needs to be added as an
attribute on `ModelPredictor` — read from `train.results_path` in the config
YAML or a dedicated `model_id` key.

---

## ORD pipeline integration

The route cache is the **upstream staging area** for the ORD-style reaction
database in `../biochem-db/`. The pipeline flow is:

```
route_cache (retrosynformer DB)
    │  tag: 'hobson-validated'
    ▼
ORD extraction script  (reads route_dicts JSONB, extracts reaction steps)
    │  tags added: 'ord-queued'
    ▼
biochem-db ORD tables  (reactions, routes, route_steps, molecules, …)
    │  tags updated: 'ord-imported'
    ▼
route_cache row        (persists forever; now tagged 'ord-imported')
```

### Extraction query

Each `RouteResponse` dict in `route_dicts` contains a `steps` list. Each step
has `target`, `reactants`, `reagents`, `reaction_id`, `confidence`, `source`.
Unnesting to individual reaction rows:

```sql
-- Expand route_dicts JSONB into one row per reaction step
SELECT
    rc.id                                       AS cache_id,
    rc.canonical_smiles                         AS target_smiles,
    rc.model_id,
    rc.created_at,
    route.ordinality                            AS route_index,
    route.value ->> 'score'                     AS route_score,
    route.value ->> 'depth'                     AS route_depth,
    route.value ->> 'all_leaves_purchasable'    AS solved,
    step.ordinality                             AS step_index,
    step.value ->> 'target'                     AS step_target,
    step.value ->> 'reaction_id'                AS reaction_id,
    step.value ->> 'confidence'                 AS confidence,
    step.value -> 'reactants'                   AS reactants,
    step.value -> 'reagents'                    AS reagents
FROM route_cache rc,
     LATERAL jsonb_array_elements(rc.route_dicts) WITH ORDINALITY AS route,
     LATERAL jsonb_array_elements(route.value -> 'steps') WITH ORDINALITY AS step
WHERE 'hobson-validated' = ANY(rc.tags)
  AND NOT ('ord-imported' = ANY(rc.tags))
ORDER BY rc.id, route.ordinality, step.ordinality;
```

### Mark as imported (after successful write to biochem-db)

```sql
UPDATE route_cache
SET    tags = array_append(tags, 'ord-imported')
WHERE  id = ANY($1::bigint[])   -- pass array of IDs from the extraction batch
  AND  NOT ('ord-imported' = ANY(tags));
```

### Suggested biochem-db target tables (ORD-compatible)

```sql
-- In ../biochem-db — minimal sketch; align with existing ORD schema

CREATE TABLE reactions (
    id               BIGSERIAL PRIMARY KEY,
    reaction_smiles  TEXT,          -- reactants>>products
    reaction_id      TEXT,          -- template/reaction_id from RetroSynFormer
    confidence       FLOAT,
    source           TEXT,          -- "retrosynformer:large-23-layer-trial003"
    cache_id         BIGINT,        -- FK → route_cache.id (cross-DB reference, stored as plain int)
    route_index      SMALLINT,
    step_index       SMALLINT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE routes (
    id               BIGSERIAL PRIMARY KEY,
    target_smiles    TEXT NOT NULL,
    score            FLOAT,
    depth            SMALLINT,
    solved           BOOLEAN,
    cache_id         BIGINT,
    model_id         TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE route_reactions (
    route_id     BIGINT REFERENCES routes(id),
    reaction_id  BIGINT REFERENCES reactions(id),
    step_index   SMALLINT,
    PRIMARY KEY (route_id, step_index)
);
```

---

## Cloud SQL setup (GCP)

### Existing instance

An instance already exists in the same project and region — no new instance
needed:

```
Instance:  biochem-db-by-hobs:us-central1:biochem-db-pubchem
Version:   PostgreSQL 16
Tier:      db-g1-small  (0.6 GB RAM — upgrade to db-n1-standard-1 if latency
                          degrades under load)
Public IP: 136.113.250.47
Databases: postgres, biochem-db-pubchem
```

### Create the retrosynformer database and user

```bash
# Add a dedicated database to the existing instance
gcloud sql databases create retrosynformer \
  --instance biochem-db-pubchem

# Create a least-privilege user (separate from the postgres superuser)
gcloud sql users create retrosynformer_app \
  --instance biochem-db-pubchem \
  --password "$(openssl rand -base64 32)"

# Store password in Secret Manager
gcloud secrets create retrosynformer-db-password \
  --data-file=<(echo -n "$RETROSYNFORMER_DB_PASSWORD")
```

Then connect and run `migrations/001_route_cache.sql`:

```bash
# Via proxy (already runs as a systemd service on localhost, port 5431)
psql "host=127.0.0.1 port=5431 dbname=retrosynformer user=retrosynformer_app" \
  -f migrations/001_route_cache.sql
```

### Connect Cloud Run to Cloud SQL

```bash
gcloud run services update retrosynformer-inference-v3 \
  --region us-central1 \
  --add-cloudsql-instances biochem-db-by-hobs:us-central1:biochem-db-pubchem \
  --update-env-vars \
    CLOUD_SQL_INSTANCE=biochem-db-by-hobs:us-central1:biochem-db-pubchem,\
    DB_USER=retrosynformer_app,\
    DB_NAME=retrosynformer \
  --update-secrets DB_PASSWORD=retrosynformer-db-password:latest
```

### IAM permissions

```bash
gcloud projects add-iam-policy-binding biochem-db-by-hobs \
  --member serviceAccount:$(gcloud run services describe retrosynformer-inference-v3 \
    --region us-central1 --format='value(spec.template.spec.serviceAccountName)') \
  --role roles/cloudsql.client
```

---

## Environment variables added to Cloud Run

| Variable | Value | Secret? |
|---|---|---|
| `CLOUD_SQL_INSTANCE` | `biochem-db-by-hobs:us-central1:biochem-db-pubchem` | No |
| `DB_USER` | `retrosynformer_app` | No |
| `DB_NAME` | `retrosynformer` | No |
| `DB_PASSWORD` | — | Yes — Secret Manager |

If `CLOUD_SQL_INSTANCE` is unset the app runs without the DB tier (falls back
to in-memory LRU only) — useful for local development.

---

## Migration path from in-memory LRU

1. Create `retrosynformer` database on `biochem-db-pubchem` instance.
2. Run `migrations/001_route_cache.sql`.
3. Add `asyncpg` and `cloud-sql-python-connector[asyncpg]` to `pyproject.toml`.
4. Update `serve/app.py` with pool setup and three-tier lookup (above).
5. Add `model_id` attribute to `ModelPredictor` (read from config YAML).
6. Update the in-memory cache key to include `model_sha256` — prevents stale
   in-memory hits across Cloud Run revisions during traffic shifts.
7. Set Cloud Run env vars and redeploy.
8. Bulk-tag existing DB entries with `model:<model_id>` after first populate.

---

## Evaluation run tracking

### Goal

Every execution of `rs-evaluate` writes a row to `eval_runs` and one row per
molecule to `eval_run_molecules`.  This gives a permanent, queryable record of
accuracy and latency across every benchmark run — comparable to experiment
tracking in MLflow but living in the same Postgres instance as the route cache.

### Schema

```sql
-- migrations/002_eval_runs.sql

CREATE TABLE eval_runs (
    id            BIGSERIAL PRIMARY KEY,

    -- Identification
    run_name      TEXT NOT NULL,       -- auto-generated: "{study_name}-trial{trial_num}-{date}"
    study_name    TEXT,                -- e.g. "v3-cpu-endpoint"
    trial_num     TEXT,                -- e.g. "000"
    mode          TEXT NOT NULL,       -- "endpoint" | "local"
    endpoint_url  TEXT,                -- if mode=endpoint
    model_id      TEXT,                -- human label from predictor / config
    model_sha256  TEXT,                -- SHA-256 of model.pth if known

    -- Run configuration
    pass_config   JSONB,               -- [{max_routes, max_steps, timeout_s, label}, ...]
    top_routes    SMALLINT,
    source_file   TEXT,                -- path to test_molecules.yml used

    -- Aggregate results
    n_molecules   SMALLINT NOT NULL,
    n_solved      SMALLINT NOT NULL,
    n_cyclic      SMALLINT NOT NULL DEFAULT 0,
    n_skipped     SMALLINT NOT NULL DEFAULT 0,

    -- Timing
    started_at    TIMESTAMPTZ NOT NULL,
    completed_at  TIMESTAMPTZ,

    -- Free-form notes and tags (same conventions as route_cache)
    notes         TEXT,
    tags          TEXT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_er_study     ON eval_runs (study_name);
CREATE INDEX idx_er_model     ON eval_runs (model_sha256);
CREATE INDEX idx_er_started   ON eval_runs (started_at);
CREATE INDEX idx_er_tags      ON eval_runs USING GIN (tags);


CREATE TABLE eval_run_molecules (
    id               BIGSERIAL PRIMARY KEY,

    eval_run_id      BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    canonical_smiles TEXT    NOT NULL,
    query_name       TEXT,            -- human name from test_molecules.yml

    -- Link to cached result (null if the molecule errored before reaching cache)
    route_cache_id   BIGINT REFERENCES route_cache(id),

    -- Per-molecule outcome
    solved           BOOLEAN,
    solved_on_pass   SMALLINT,        -- 1 | 2 | 3 | NULL
    depth            SMALLINT,
    is_cyclic        BOOLEAN,
    n_routes         SMALLINT,
    latency_s        FLOAT,           -- wall-clock seconds (sum across passes)
    error            TEXT,            -- non-null when skipped or API error

    UNIQUE (eval_run_id, canonical_smiles)
);

CREATE INDEX idx_erm_run      ON eval_run_molecules (eval_run_id);
CREATE INDEX idx_erm_smiles   ON eval_run_molecules (canonical_smiles);
CREATE INDEX idx_erm_solved   ON eval_run_molecules (solved);
```

### Python integration (rs-evaluate)

Add `--db` flag (or read `CLOUD_SQL_INSTANCE`) in `evaluate.py`. After the
evaluation loop completes, write results in two steps:

```python
async def record_eval_run(pool, meta: dict, results: list[dict]) -> int:
    run_id = await pool.fetchval(
        """
        INSERT INTO eval_runs
            (run_name, study_name, trial_num, mode, endpoint_url,
             model_id, pass_config, top_routes, source_file,
             n_molecules, n_solved, n_cyclic, n_skipped, started_at, completed_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13,$14,NOW())
        RETURNING id
        """,
        meta["run_name"], meta["study_name"], meta["trial_num"],
        meta["mode"], meta.get("endpoint_url"), meta.get("model_id"),
        json.dumps(meta["pass_config"]), meta["top_routes"], meta["source_file"],
        meta["n_molecules"], meta["n_solved"], meta["n_cyclic"], meta["n_skipped"],
        meta["started_at"],
    )
    rows = [
        (run_id, m["canonical_smiles"], m.get("query_name"),
         m.get("route_cache_id"), m.get("solved"), m.get("solved_on_pass"),
         m.get("depth"), m.get("is_cyclic"), m.get("n_routes"),
         m.get("latency_s"), m.get("error"))
        for m in results
    ]
    await pool.executemany(
        """
        INSERT INTO eval_run_molecules
            (eval_run_id, canonical_smiles, query_name, route_cache_id,
             solved, solved_on_pass, depth, is_cyclic, n_routes, latency_s, error)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        ON CONFLICT (eval_run_id, canonical_smiles) DO NOTHING
        """,
        rows,
    )
    return run_id
```

### Useful queries

**Solve rate over time by model:**
```sql
SELECT study_name, trial_num, started_at::date,
       n_solved, n_molecules,
       ROUND(100.0 * n_solved / NULLIF(n_molecules, 0), 1) AS pct_solved
FROM   eval_runs
ORDER  BY started_at;
```

**Molecules that consistently fail across all runs:**
```sql
SELECT canonical_smiles, query_name,
       COUNT(*)                            AS total_runs,
       SUM(CASE WHEN solved THEN 1 END)    AS times_solved
FROM   eval_run_molecules
GROUP  BY canonical_smiles, query_name
HAVING SUM(CASE WHEN solved THEN 1 END) = 0
ORDER  BY total_runs DESC;
```

**Compare two eval runs (solved in B but not A):**
```sql
SELECT b.query_name, b.depth, b.latency_s
FROM   eval_run_molecules a
JOIN   eval_run_molecules b USING (canonical_smiles)
WHERE  a.eval_run_id = $run_a
  AND  b.eval_run_id = $run_b
  AND  NOT a.solved
  AND  b.solved;
```

**Avg latency per pass, latest run:**
```sql
SELECT solved_on_pass, ROUND(AVG(latency_s)::numeric, 1) AS avg_s, COUNT(*) AS n
FROM   eval_run_molecules
WHERE  eval_run_id = (SELECT MAX(id) FROM eval_runs)
GROUP  BY solved_on_pass
ORDER  BY solved_on_pass NULLS LAST;
```

---

## Molecule access log

### Goal

An append-only log of **every individual molecule request** to the API and eval
runner — recording the cache tier that served it, the model used, and the
latency. Unlike `route_cache.access_count` (a running total), this table gives
the full time-series: per-molecule hit/miss rates, cold-start patterns, and
cache efficiency over any window.

### Schema

```sql
-- migrations/003_access_log.sql

CREATE TABLE molecule_access_log (
    id               BIGSERIAL PRIMARY KEY,
    canonical_smiles TEXT        NOT NULL,
    accessed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Who/what triggered this request
    source           TEXT        NOT NULL,  -- 'api' | 'eval_run' | 'cache_warm'
    eval_run_id      BIGINT REFERENCES eval_runs(id),  -- set when source='eval_run'

    -- Request parameters
    max_routes       SMALLINT,
    max_steps        SMALLINT,

    -- How it was served
    cache_tier       TEXT,   -- 'lru' | 'postgres' | 'inference' | 'error'
    model_id         TEXT,
    model_sha256     TEXT,

    -- Result summary
    n_routes         SMALLINT,
    n_solved         SMALLINT,
    latency_ms       INT,    -- total wall-clock milliseconds for this request

    -- Cloud Run instance that handled the request (for cold-start debugging)
    instance_id      TEXT
);

CREATE INDEX idx_mal_smiles    ON molecule_access_log (canonical_smiles);
CREATE INDEX idx_mal_accessed  ON molecule_access_log (accessed_at DESC);
CREATE INDEX idx_mal_source    ON molecule_access_log (source);
CREATE INDEX idx_mal_tier      ON molecule_access_log (cache_tier);
CREATE INDEX idx_mal_run       ON molecule_access_log (eval_run_id)
    WHERE eval_run_id IS NOT NULL;
```

### app.py integration

Fire-and-forget log write in `_build_response` (or at the end of the
`/retrosynthesis` handler) so it never blocks the response:

```python
async def _log_access(
    pool: asyncpg.Pool,
    canon: str,
    max_routes: int,
    max_steps: int,
    cache_tier: str,       # 'lru' | 'postgres' | 'inference' | 'error'
    model_id: str,
    model_sha256: str,
    n_routes: int,
    n_solved: int,
    latency_ms: int,
    source: str = "api",
    eval_run_id: int | None = None,
) -> None:
    instance_id = os.environ.get("K_REVISION")   # Cloud Run revision name
    await pool.execute(
        """
        INSERT INTO molecule_access_log
            (canonical_smiles, max_routes, max_steps,
             source, eval_run_id,
             cache_tier, model_id, model_sha256,
             n_routes, n_solved, latency_ms, instance_id)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        """,
        canon, max_routes, max_steps,
        source, eval_run_id,
        cache_tier, model_id, model_sha256,
        n_routes, n_solved, latency_ms, instance_id,
    )

# In /retrosynthesis, replace the fire-and-forget db_cache_put call:
if _db_pool:
    asyncio.create_task(_db_cache_put(...))
    asyncio.create_task(_log_access(
        _db_pool, canon, req.max_routes, req.max_steps,
        cache_tier,            # set to 'lru', 'postgres', or 'inference'
        predictor.model_id, model_sha256,
        len(route_dicts), n_solved, latency_ms,
    ))
```

### Useful queries

**Cache hit rate (last 7 days):**
```sql
SELECT cache_tier,
       COUNT(*)                              AS requests,
       ROUND(100.0 * COUNT(*) /
             SUM(COUNT(*)) OVER (), 1)       AS pct
FROM   molecule_access_log
WHERE  accessed_at > NOW() - INTERVAL '7 days'
GROUP  BY cache_tier
ORDER  BY requests DESC;
```

**Per-molecule access count + last seen + cache tier breakdown:**
```sql
SELECT canonical_smiles,
       COUNT(*)                                       AS total_requests,
       MAX(accessed_at)                               AS last_seen,
       SUM(CASE WHEN cache_tier = 'lru'       THEN 1 END) AS lru_hits,
       SUM(CASE WHEN cache_tier = 'postgres'  THEN 1 END) AS pg_hits,
       SUM(CASE WHEN cache_tier = 'inference' THEN 1 END) AS inferences
FROM   molecule_access_log
GROUP  BY canonical_smiles
ORDER  BY total_requests DESC
LIMIT  50;
```

**Latency distribution by cache tier:**
```sql
SELECT cache_tier,
       PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms
FROM   molecule_access_log
WHERE  accessed_at > NOW() - INTERVAL '30 days'
GROUP  BY cache_tier;
```

**Cold-start requests (first hit per instance_id):**
```sql
SELECT instance_id,
       MIN(accessed_at) AS first_request,
       MIN(latency_ms)  AS first_latency_ms
FROM   molecule_access_log
WHERE  cache_tier = 'inference'
GROUP  BY instance_id
ORDER  BY first_request DESC;
```

---

## Open questions / future work

- **`/admin/tag` endpoint**: `POST /admin/tag` (admin-key protected) that
  accepts `{ids: [...], add: [...], remove: [...]}` so the webapp or a Jupyter
  notebook can tag routes without direct DB access.
- **pgvector similarity search**: Store Morgan fingerprints as `vector(2048)`
  to serve approximate cache hits for structurally similar molecules.
- **Duplicate reaction dedup in biochem-db**: The same reaction template can
  appear in routes for many different target molecules — deduplicate on
  `reaction_id + reactants` before inserting into the ORD `reactions` table.
- **Cache warming**: On model deployment, run the `test_molecules.yml` set
  through the endpoint so the first real user request never hits cold inference.
