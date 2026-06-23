# Changelog

All notable changes to RetroSynFormer are documented here.
Full release notes are in [`docs/`](docs/).

---

## [0.1.45] — 2026-06-23

**In-memory LRU cache for `/retrosynthesis`; Cloud Run switched to CPU-only (8 vCPU / 32 GiB).**

- `serve/app.py`: 256-entry LRU cache keyed on `(canonical_smiles, max_routes, max_steps)`; cache hits bypass the inference semaphore entirely. Uses `collections.OrderedDict` + `threading.Lock` — thread-safe, no extra dependencies.
- `serve/schemas.py`: `HealthResponse` exposes `route_cache_size: int` so callers can observe cache utilisation.
- Cloud Run `retrosynformer-inference-v3`: removed NVIDIA L4 GPU (`--gpu 0`); kept `--cpu 8 --memory 32Gi`. PyTorch auto-detects CPU via `torch.cuda.is_available()` in `predictor.py`.

## [0.1.44] — 2026-06-23

**Reduce `max_routes` server cap to prevent OOM on 27-layer model.**

- `serve/schemas.py`: `RetrosynthesisRequest.max_routes` upper bound lowered from 50 → 15; the 799MB 27-layer model OOMs on Cloud Run (NVIDIA L4 24GB) when beam width exceeds ~20 routes.
- `scripts/evaluate.py`: `_RETRY_PASSES` updated to use `max_routes=15` for passes 2 and 3 (was 30 and 50); inline string literals and docstrings updated to match.

## [0.1.43] — 2026-06-22

**`compression.load_model` wired into `runner.init_model`; `prevent_cyclic_routes` beam pruning; test coverage report.**

- `compression.py` (new): format/codec utilities — `save_model`/`load_model` with auto-detection of `.pth`, `.safetensors`, `.gz`/`.bz2`/`.xz`; `cast_state_dict` for fp16/bfloat16 dtype conversion; `is_valid_model_file` for header-only validation.
- `scripts/compress_model.py` (new): `rs-compress` CLI — convert and/or compress a local model file.
- `runner.py`: replaced bare `torch.load()` in `init_model` with `compression.load_model`, which auto-detects format and compression. Added `_CHECKPOINT_CANDIDATES` tuple and `_find_model_checkpoint()` so `--resume` probes all supported filenames rather than hardcoding `model.pth`. Raises `FileNotFoundError` with candidate list when nothing is found. Backward compatible: `model.pth` remains first in the probe order.
- `scripts/upload_model.py`: extended `rs-upload` to support `--format safetensors`, `--dtype fp16/bfloat16`, and all three codecs; `--deploy` auto-uploads config alongside weights.
- `scripts/gcs_download.py`: extended to support bz2/xz decompression (was gz-only); uses `compression.decompress_file` when available.
- `environment.py`: `prevent_cyclic_routes` parameter (default `True`) — prunes beam branches that would revisit an already-decomposed molecule; stored as `_decomposed_molecules` set per branch. Set to `false` to reproduce pre-v0.1.43 behaviour.
- `results/config/baseline_{small,standard,large}.yaml`: `prevent_cyclic_routes: true`; `baseline_large.yaml` added.
- `tests/test_compression.py` (new): 49 tests covering all format/codec/dtype paths.
- `docs/coverage.md` (new): test coverage report (298 tests, 20% overall; compression 92%, dropout 96%, extrapolate 90%).

---

## [0.1.41] — 2026-06-22

**Unified `layer_shared_resid_dropout`: intra- and inter-layer residual mask tying in one parameter.**

- `dropout.py`: `apply_shared_resid_dropout(model, p, spec)` — new unified entry point for `layer_shared_resid_dropout`. Accepts a list/bool (intra-layer, backward-compatible), a dict with integer keys N (1-based, intra-layer), float keys N.5 (inter-layer boundary with group-ID value), or any mix of the two. Internally delegates to `apply_layer_shared_resid_dropout` and `apply_interlayer_tied_dropout`.
- `runner.py`: collapsed the separate `layer_shared_resid_dropout` and `tied_resid_drop` blocks in `init_model` into a single unified block; removed `_validate_tied_resid_drop`; extended `_validate_layer_shared_resid_dropout` to validate both list and dict formats.
- `tests/test_dropout.py`: 10 new `TestUnifiedAPI` tests covering all spec formats (list, bool, dict intra-only, dict inter-only, combined overlapping, combined non-overlapping, zero-p); 23 tests total.
- `results/config/baseline_{small,standard}.yaml`: consolidated comments under a single `layer_shared_resid_dropout` heading documenting all four usage forms.

---

## [0.1.40] — 2026-06-22

**hplot column legend; `UNABBREV` dict and `unabbreviate()` in `names.py`.**

- `names.py`: added `UNABBREV` (exact inversion of `ABBREV`) and `unabbreviate()` (reverses `abbreviate()` via UNABBREV lookup then systematic prefix/suffix rules). Added round-trip doctests covering all 16 cases.
- `dataframes.py`: `print_and_save_trials_table` now prints a `Column legend:` block before the table mapping each abbreviated header to its full name, skipping columns whose abbreviation is identical to the full name.

---

## [0.1.39] — 2026-06-22

**hplot table: dataset column populated for all trials; `dataset_name` abbreviation added.**

- `names.py`: added `"dataset_name": "dataset"` to `ABBREV` so `abbreviate("dataset_name")` returns `"dataset"` for consistent column labelling.
- `dataframes.py`: `load_run_params` now reads fixed study-level params (e.g. `dataset_name`) from `model.config.yaml` in any trial subdirectory and merges them into every trial's param dict — including trials whose `trial_start` event was never written to `run.jsonl`. This fills the `dataset` column for all trials rather than only those with a `trial_start` log entry.

---

## [0.1.38] — 2026-06-22

**Server-side disconnect detection; evaluate retry on timeout; pass 1 timeout raised.**

- `serve/app.py`: `/predict` and `/retrosynthesis` endpoints now poll `request.is_disconnected()` every 10 s while the beam-search executor thread runs. On disconnect, the semaphore is released immediately so the next request can start — prevents cascading timeouts from blocking the entire evaluation queue when a long computation outlasts the client timeout.
- `evaluate.py`: pass 1 client timeout raised 240 s → 360 s to accommodate CUDA JIT warmup (~250 s) on the first request after a container restart.
- `evaluate.py`: on a timeout exception the pass loop now `continue`s to the next pass (higher `max_routes`/`max_steps`/timeout) rather than `break`ing. Timed-out molecules are retried up to 3 passes before being marked as errors.

---

## [0.1.37] — 2026-06-21

**Add `retrosynformer_version` to `/health` response.**

- `serve/app.py`, `serve/schemas.py`: `/health` now returns `retrosynformer_version` (string, nullable) populated via `importlib.metadata.version("retrosynformer")` at import time. Reflects the version installed at container build time; `PackageNotFoundError`-safe.

---

## [0.1.36] — 2026-06-21

**`rs-evaluate` progressive retry; `/health` endpoint model provenance; GCS metadata sidecar.**

- `evaluate.py`: `rs-evaluate` now reads `RETROSYNFORMER_URL` and `RETROSYNFORMER_API_KEY` from `.env` (repo root or cwd) so `--endpoint` and `--api-key` can be omitted when running against the default deployment.
- `evaluate.py`: progressive retry — molecules without a complete route (`all_leaves_purchasable=False`) are retried up to twice: pass 2 uses `max_routes=30, max_steps=10`; pass 3 uses `max_routes=50, max_steps=15`. Endpoint mode only (local model runs single pass). The YAML output and markdown report record `retrosynformer_solved_on_pass` and break out solved counts per pass.
- `evaluate.py`: fix `/retrosynthesis` endpoint parameter names — request body now correctly uses `max_routes`/`max_steps` (was `beam_width` which the schema does not recognise).
- `evaluate.py`: client-side request timeout now scales with search depth — 240s / 600s / 1500s for passes 1/2/3 (was hardcoded 180s).
- Cloud Run `retrosynformer-inference-v3`: request timeout raised from 600s to 1800s to support deep beam searches in pass 3.
- `serve/app.py`, `serve/predictor.py`, `serve/schemas.py`: `/health` endpoint now returns `model_path` (GCS URI from sidecar, not container path), `model_released_at` (GCS `timeCreated` from sidecar), `model_sha256_hash` (SHA-256 of decompressed local file), and `model_file_size_bytes` (decompressed file size).
- `scripts/gcs_download.py`: after each artifact download, writes a `.metadata.json` sidecar alongside the local file containing GCS URI, `timeCreated`, `updated`, `gcs_size_bytes`, `sha256` (computed locally from the decompressed file), and `file_size_bytes`. SHA-256 computation is ~0.5-1 s for a 1-2 GB model — negligible vs. download time.

---

## [0.1.35] — 2026-06-22

**`rs-upload --deploy` auto-uploads config; fixes subprocess import and flaky doctest.**

- `upload_model.py`: `--deploy` now auto-detects `model.config.yaml` (or `config.yaml`) beside the local model, uploads it to `<gcs_dir>/config.yaml` via `gsutil cp`, and passes it as `MODEL_CONFIG_GCS` in the same `gcloud run services update` call. Prevents architecture mismatch crash when weights and config have different `hidden_size`/`n_layers`.
- `upload_model.py`: add missing top-level `import subprocess` (was dropped from `_gcs_client` during auth-retry refactor).
- `structured_dropout.py`: fix flaky `get_mask` doctest — pin `net[0].weight` to 0.5 so zeros vs ones inputs reliably diverge after ReLU regardless of random init.

---

## [0.1.34] — 2026-06-21

**`rs-evaluate`, `rs-upload --deploy`, and upload robustness fixes.**

- `scripts/evaluate.py` + `src/retrosynformer/scripts/evaluate.py`: new `rs-evaluate` CLI command. Runs retrosynthesis evaluation against all test molecules (reads `test_molecules.yml`), supports both HTTP endpoint mode (`--endpoint URL`) and local model mode (`--model path/to/model.pth`). Auto-detects study name and trial number from model path. Fills null fields (SMILES, InChI, InChIKey, CID) via PubChem API. Saves `data/test_molecules_retrosynformer_{study}-trial{trial}-routes.yml` and a markdown report.
- `upload_model.py`: add `--deploy SERVICE` flag — after a successful upload, runs `gcloud run services update SERVICE --update-env-vars MODEL_WEIGHTS_GCS=<gcs_uri>` to redeploy the Cloud Run inference endpoint with the new model. `--deploy-config-gcs` also updates `MODEL_CONFIG_GCS`.
- `upload_model.py`: auth-error retry — `_upload_chunk` catches credential expiry errors, refreshes the `gcloud auth print-access-token` token (clears per-thread cache), and retries the chunk once automatically.
- `upload_model.py`: auto-retry failed chunks — after the main upload loop, any failed chunks are retried with `skip_existing=True` and a fresh auth token.
- `upload_model.py`: deterministic gzip compression — use `gzip.GzipFile(mtime=0)` so repeated `--compress` runs on the same file produce identical bytes; fixes `--skip-existing` MD5 mismatches across invocations.

---

## [0.1.33] — 2026-06-21

**`rs-hypertune --resume` now respects `--n-epochs`.**

- `hypertune.py`: the `train()` call in the `--resume` block was missing `n_epochs=args.n_epochs`, so the saved `model.config.yaml` value was used instead of the CLI argument. Fixed by passing `n_epochs=args.n_epochs` through.

---

## [0.1.32] — 2026-06-21

**Auto-disable hipBLASLt on unsupported ROCm architectures.**

- `retrosynformer/__init__.py`: new `_disable_hipblaslt_if_unsupported()` runs at package import time (before any submodule loads torch). Sets `TORCH_BLAS_PREFER_HIPBLASLT=0` when `HSA_OVERRIDE_GFX_VERSION` is present (non-standard arch such as Strix Halo) or when `rocminfo` reports a gfx name outside `{gfx90a, gfx940, gfx941, gfx942}`. No-ops if the env var is already set or `/opt/rocm` is absent.

---

## [0.1.31] — 2026-06-21

**Capture rsync output into `logger.info()` so `hplot` is silent by default.**

- `rsync.sync()`: capture subprocess stdout/stderr with `PIPE`; emit each line via `logger.info()` / `logger.warning()` so rsync file lists are silent at WARNING (default) and visible with `-v`.
- `hplot`: add `_configure_logging_from_argv()` to pre-parse `--debug`/`-v`/`--verbose` before `_sync_results_from_remote()` runs; convert status prints to `logger.info/warning`.

---

## [0.1.30] — 2026-06-21

**Logger migration across core library; `--debug`/`-v` flags on all `rs-*` commands.**

- `scripts/__init__.py`: `add_log_args(parser)` and `configure_logging(args)` helpers; all 13 `rs-*` CLIs now accept `--debug` (DEBUG) and `-v/--verbose` (INFO); default remains WARNING.
- `trainer.py`: replaced 13 `[debug] print()` calls with `logger.debug()`; moved Ctrl-C, init, save-folder, early-stopping, training-range, and route-eval-trigger prints to `logger.info/debug`.
- `runner.py`: moved 36 status prints (overrides, milestones, data-split stats, resume messages) to `logger.info/debug`.
- `inference.py`, `data.py`, `environment.py`, `dropout.py`, `rsync.py`, `utils/utils.py`, `utils/evaluation.py`, `dashboard/sync.py`: remaining status prints converted; new module loggers added where missing.
- `plot_learning_curves.py`: filter empty DataFrames before `pd.concat` to fix `FutureWarning` about all-NA column dtype inference.
- `rsync.py`: `verbose` default changed `True → False` in `build_cmd()` and `sync()` so rsync no longer prints every transferred filename by default.

---

## [0.1.29] — 2026-06-21

**Resume guard for `rs-train --resume` and `rs-hypertune --resume TRIAL` flag.**

- `runner.py`: when `--resume` is passed and the checkpoint's epoch count already equals `n_epochs`, print a warning and interactively offer to increase `n_epochs` by 1.5× in the source config file before starting; exits cleanly if the user declines.
- `trainer.py`: initialize `epoch = start_epoch - 1` before the training loop so the post-loop route-eval block never raises `UnboundLocalError` when the loop body never executes.
- `scripts/hypertune.py`: new `--resume [TRIAL]` flag resumes a partially-trained trial (via `runner.main(..., resume=True, eval_routes_at_end=True)`) before the study's normal `optimize()` loop; auto-detects the highest `trial_NNN` directory on disk when no number is given; calls `_backfill_failed_trials()` afterwards so Optuna's TPE uses the resumed run's history.

---

## [0.1.28] — 2026-06-21

**New `retrosynformer.dataframes` module consolidating trial/study DataFrame utilities.**

- `src/retrosynformer/dataframes.py`: new shared module extracted from `plot_learning_curves` and deduplicated from `show_study` / `show_all_studies`.
  - `load_jsonl`, `jsonl_rank_stats` — JSONL progress loading and per-trial metric extraction
  - `jsonl_path`, `find_trial_base`, `load_run_params`, `trials_df_from_db` — trial directory helpers and enriched DB loader
  - `build_trials_df`, `print_and_save_trials_table` — summary table construction and display
  - `fmt_trial_value` — cell formatter (previously duplicated as `_fmt_value` in both show scripts)
  - `SCIENTIFIC_COLS`, `SCORE_COLS`, `HIGHER_IS_BETTER`, `METRIC_COLS` — display constants (previously duplicated across show scripts)
- `plot_learning_curves.py`: dropped ~130 lines; now imports from `dataframes`.
- `show_study.py`, `show_all_studies.py`: dropped duplicate `_fmt_value` and display constants; import from `dataframes`.

---

## [0.1.18] — 2026-06-18

**Async metric extrapolation co-scheduled with route evaluation.**

- `async_eval.py`: `_extrapolation_worker()` extrapolates all 5 training metrics (`train_loss`, `train_action_accuracy`, `train_route_accuracy`, `valid_action_accuracy`, `valid_route_accuracy`) to 4 time horizons (1.0×, 1.1×, 1.5×, 2.0× `n_epochs`) using `extrapolate_objective()`; runs in a `ThreadPoolExecutor` (pure numpy — no process overhead needed). `AsyncRouteEvalPool` gains `submit_extrapolation()`, `collect_extrapolation_if_ready()`, `collect_extrapolation_blocking()`.
- `trainer.py`: extrapolation submitted alongside every route eval (async and sync paths); result merged into the same `pred_routes_train_progress.json` entry under an `"extrapolation"` key; `_extrap_buffer` coordinates the faster thread result with the slower beam-search result.

---

## [0.1.17] — 2026-06-18

**Multi-model objective extrapolation module (`extrapolate.py`).**

- New `src/retrosynformer/extrapolate.py`: `extrapolate_objective(values, n_epochs)` fits four models to progressively wider windows of training history and combines them via inverse-variance weighting into a single estimate with standard error.
  - `linear` (degree 1) — last ¼ of epochs
  - `quadratic` (degree 2) — last ½ of epochs
  - `cubic` (degree 3) — last ¾ of epochs
  - `log` (`y = a·log1p(x) + b`) — all epochs
  - Returns `None` when `n_observed < min_points` (default 4); accepts optional `epochs=` list for non-contiguous resumed training.
- `models_optuna.py`: `estimate_incomplete_objectives()` now delegates to `extrapolate_objective()` replacing the single quadratic fit; result dict gains `se` and `models` keys.
- 23 new tests in `tests/test_extrapolate.py`.

---

## [0.1.16] — 2026-06-18

**Async CPU route evaluation: beam search on idle CPU cores while GPU trains.**

- New `src/retrosynformer/async_eval.py`: `AsyncRouteEvalPool` backed by `ProcessPoolExecutor`; compounds sharded across `eval_n_workers` CPU processes; `submit()` is non-blocking, `collect_if_ready()` returns `None` while running and the merged result dict when done, `collect_blocking()` for end-of-training.
- `_eval_worker_chunk()`: module-level picklable worker that reconstructs the model on CPU from serialised `state_dict` bytes, patches `get_device()` to force CPU, and runs beam search with the full `RoutePredictor` pipeline (including TED scoring).
- `trainer.py`: evaluation block branches on `evaluation.async_route_eval`; async path submits at eval epoch and collects at the top of each subsequent epoch; synchronous path unchanged; pool shut down after training.
- `results/config.yaml`: `async_route_eval: false` and `eval_n_workers: 24` added to `evaluation:` section (opt-in, default off).

---

## [0.1.15] — 2026-06-18

**EpochLogger singleton: configurable per-epoch JSONL accumulator.**

- New `src/retrosynformer/epoch_logger.py`: module-level singleton (all classmethods) for accumulating training metrics from any depth of the call stack and flushing them to `train_progress.jsonl` at epoch end.
- Three-tier state: `_persistent` (constant for the whole run), `_state` (reset each epoch), `_providers` (zero-arg callables evaluated at flush time, e.g. `elapsed_seconds`, `timestamp`).
- Field filtering via `logging.fields` in `model.config.yaml`; omit the key to write every accumulated field (backward-compatible).
- `trainer.py`: wired `EpochLogger.configure/set_persistent/begin_epoch/update_many/flush` into `_train`; `train_one_epoch` now calls `EpochLogger.update("gradient_norm", …)` as a side effect; removed manual `json.dumps` file write.
- `results/config.yaml`: added `logging:` section documenting all 19 available fields in grouped comments.

---

## [0.1.14] — 2026-06-18

**Quadratic objective estimation for incomplete trials; large-study merge; 11 new jsonl fields per epoch.**

- `models_optuna.py`: `estimate_incomplete_objectives()` fits a degree-2 polynomial to the second half of each RUNNING/WAITING trial's `train_progress.jsonl` and extrapolates to the vertex or `n_epochs`; helpers `_load_jsonl_metric` (epoch de-dup for append-on-restart files), `_trial_objective_metric`, `_fit_quadratic_estimate`.
- `scripts/merge_large_studies.py`: merges all `hypertune-*large*` study databases into `results/hypermerge/large-merged/`; COMPLETE trials carry objective values, RUNNING/WAITING/FAIL stored as FAIL; relative symlinks to original `trial_NNN/` directories; `SOURCES.txt` provenance; fixed params that vary across studies (`hidden_size`, `head_dim`, `batch_size`, `early_stopping_patience`) promoted to `CategoricalDistribution` dimensions so the TPE surrogate can learn their effect.
- `trainer.py`: 11 new fields appended to `train_progress.jsonl` each epoch: `learning_rate`, `elapsed_seconds`, `is_best`, `epochs_without_improvement`, `timestamp`, `gradient_norm`, `n_lr_reductions`, `best_valid_route_accuracy`, `study_name`, `trial_number`, `config_hash`; removed dead `time.time()` call.
- 11 new tests in `test_models_optuna.py` for the polynomial estimation logic.

---

## [0.1.13] — 2026-06-17

**`rs-plot-learning-curves`: multiple `--metric` flags; fixed params in trial table; Optuna params marked with `*`.**

- `--metric` now accepts multiple values (repeat the flag); trials ranked by first metric; each metric drawn with a distinct linestyle; two-part legend (trial colors + metric linestyles) when multiple metrics are active.
- `--also-train` extended: inserts `train_*` counterparts for each `valid_*` metric not already listed.
- `hypertune.py`: `trial_start` record in `run.jsonl` now includes `all_params` (full resolved config minus path/control metadata) and `optuna_keys` (params actually suggested by Optuna).
- Plot script reads `run.jsonl` to supplement Optuna trial params with fixed architecture values (`n_heads`, `n_layers`, `head_dim`, `lr`, etc.); Optuna-searched column headers are suffixed with `*`.
- Long list params (e.g. `layer_shared_resid_dropout`) summarised as `[first ... last]` in the trial table.

[Full notes](docs/release-notes-0.1.13.md)

---

## [0.1.12] — 2026-06-17

**Removed rs-bump command and anthropic API dependency.**

- Deleted `scripts/bump.py` and `src/retrosynformer/scripts/bump.py`.
- Removed `rs-bump` and `bump` entry points from `[project.scripts]`.
- Removed `anthropic>=0.25` from the `[dev]` extra.

---

## [0.1.11] — 2026-06-17

**Fixed verbose training column alignment; header reprints every 10 epochs.**

- `study_name` was prepended to each row (before `epoch`) but the header printed it as a suffix (after `note`) — both now use suffix position.
- Replaced tab-separated columns with fixed-width space-aligned format (`>5` epoch/trial, `>7` loss/accuracy, `>6` s/ep, `<4` note).
- Header row reprints automatically every 10 epochs relative to `start_epoch`, keeping labels visible during long runs.
- Extracted header print into a `_print_header()` closure so the format is defined once.

[Full notes](docs/release-notes-0.1.11.md)

---

## [0.1.10] — 2026-06-17

**Graceful Ctrl-C: route eval and study.db write before stopping.**

- First Ctrl-C finishes the current epoch, runs route evaluation, returns normally so Optuna records the trial result to `study.db`, then raises `KeyboardInterrupt`.
- Second Ctrl-C within 1 second raises `KeyboardInterrupt` immediately.
- `trainer.py`: `_handle_sigint`, `set/clear_interrupt_callback`, `is_interrupted`; handler installed/restored in `train()` via `try/finally`; interrupt flag checked at end of epoch loop, forces `eval_routes_at_end=True`.
- `hypertune.py`: `_objective_with_interrupt` wrapper registers `study.stop` as callback; raises `KeyboardInterrupt` after `study.optimize()` if interrupted.
- `runner.py`: raises `KeyboardInterrupt` after `train()` if interrupted (`rs-train`).

[Full notes](docs/release-notes-0.1.10.md)

---

## [0.1.9] — 2026-06-17

**Layer-shared residual dropout; atomic per-epoch checkpointing.**

- New `src/retrosynformer/dropout.py`: `SharedResidMaskDropout` + `apply_layer_shared_resid_dropout` — ties the attention and MLP residual masks within each transformer block via a forward pre-hook; no new parameters, `load_state_dict` unaffected.
- Config key `model.layer_shared_resid_dropout`: scalar `bool` for uniform application or `list[bool]` for per-layer control; `0`/`1` accepted as aliases throughout.
- Optuna list-of-lists support: specify multiple `layer_shared_resid_dropout` patterns as discrete categorical choices; inner lists JSON-serialised for Optuna storage compatibility.
- Pre-flight validation in `runner` and `hypertune`: non-jagged check, length ≥ max `n_layers`, valid-value check.
- New `results/config/small_nonuniform_dropout.yaml`: fixed best-trial architecture, four dropout-pattern choices, small dataset, 50 epochs.
- Trainer: `model.last.pth` written atomically after every epoch; `model.pth` copied from it (not re-serialised) when a new best-loss is achieved; eliminates partial-read risk during concurrent rsync.

[Full notes](docs/release-notes-0.1.9.md)

---

## [0.1.8] — 2026-06-16

**Training dashboard, trial status CLI, ntfy.sh alerter, ROCm pin fixes.**

- New `rs-dashboard`: Flask-Admin web app over a local SQLite mirror of Optuna trial data; login-gated, deployed to `taco` via Tailscale user-mode systemd.
- New `rs-status`: syncs remote results and prints a ranked Pandas table of trials with score, accuracy, `frac_solved`, LR, architecture dims, and last-modified time; `--top N` flag.
- New `rs-monitor` (`monitor_training.py`): background ntfy.sh alerter that fires a push notification whenever `valid_action_accuracy` improves.
- Fixed ROCm deps: reverted `uv.lock` to ROCm 6.2 / torch 2.5.1 (6.4 crashes taco iGPU); removed the `amdgpu` alias extra from `pyproject.toml`.
- Fixed dashboard Trials table showing `Study` repr instead of `study_name`.

[Full notes](docs/release-notes-0.1.8.md)

---

## [0.1.7] — 2026-06-15

**Learning-curve CLI, rsync utility, configurable Optuna objective.**

- New `rs-plot-learning-curves` command: aggregates all `study.db` files, ranks completed Optuna trials by a chosen metric, and overlays per-epoch learning curves from `train_progress.jsonl`; handles epoch-reset restarts, rsync-nested duplicates, and multi-study SQLite files.
- New `rs-sync-results` command and `retrosynformer.rsync` module: pulls only `study.db` + `train_progress.jsonl` from a remote host via rsync while preserving directory structure.
- Trainer: added `eval_routes_at_end` flag so a full route evaluation always runs at least once per hypertune trial, preventing spurious 0.0 Optuna scores when early stopping fires before the first `eval_routes_frequency` epoch.
- Hypertune: Optuna objective changed from `fraction_targets_solved` → `max(valid_route_accuracy)` — always non-zero, better reflects training progress.
- Hypertune: `objective_metric` key in the `optuna:` config section makes the objective configurable without editing Python (`valid_action_accuracy` | `valid_route_accuracy` | `fraction_solved`); validated by `_validate_config`.
- All six configs in `results/config/` updated with documented `objective_metric` key.
- New `docs/README-config.md`: full config reference and hyperparameter tuning walkthrough.

[Full notes](docs/release-notes-0.1.7.md)

---

## [0.1.6] — 2026-06-14

**PyTorch extras, early-stopping crash fix, tighter hypertune ranges.**

- Removed torch from base deps; added `[cpu]`, `[rocm]` (canonical), and kept `[amdgpu]` (alias) extras with correct index routing.
- Fixed phantom `torch>=2.12.0` version pin (PyTorch tops out at 2.5.x on the ROCm 6.2 index); changed to `>=2.0.0`.
- Fixed `KeyError: 'epoch'` crash in `trainer.train()` when early stopping fires before the first route-evaluation epoch.
- Tightened small-dataset hypertune search space: `n_heads` → `[1,2,3,4]`, `n_layers` → `[2,3,4,5]`; increased `early_stopping_patience` from 4 → 6.

[Full notes](docs/release-notes-0.1.6.md)

---

## [0.1.5] — 2026-06-14

**Scripts installable as CLI entry points; `hypertune_verbose.py` removed.**

- Added `[build-system]` and `[project.scripts]` to `pyproject.toml`; six `retrosynformer-*` commands now installed to `PATH` on `pip install`.
- Moved script logic from `scripts/*.py` into the installable package at `src/retrosynformer/scripts/`; `scripts/*.py` become 4-line shims.
- Shell scripts (`monitor_train_progress.sh`, `train_structured_dropout_comparison.sh`) get shebangs and `chmod +x`.
- Deleted `scripts/hypertune_verbose.py` cruft.

[Full notes](docs/release-notes-0.1.5.md)

---

## [0.1.4] — 2026-06-14

**Structured dropout, Optuna study tooling, training robustness, AMD ROCm support.**

- Added `MoleculeConditionedMaskGenerator` and `StructuredDropoutDecisionTransformer` — molecule-conditioned per-channel masking of the Decision Transformer hidden state.
- New `retrosynformer.study` module with `to_trials_df`, `concat`, `concat_all`; new `show_study.py` and `show_all_studies.py` scripts.
- Hypertune improvements: study-directory locking, interactive resume, verbose JSONL event logging.
- Training: early-stopping patience, `--start-epoch` / `--batch-size` / `--num-workers` CLI flags, resume fix for mixed-dataset JSONL history.
- `RemoteTrialMonitor` for polling a remote training server via rsync.
- Fixed matplotlib X11 crash on headless SSH servers (`matplotlib.use("Agg")`).
- AMD ROCm support via `pip install retrosynformer[amdgpu]` with `[tool.uv.sources]` index routing.
- 700+ lines of new tests across 4 test files.

[Full notes](docs/release-notes-0.1.4.md)
