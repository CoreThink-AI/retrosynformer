# Test Coverage Report

Generated: 2026-06-22  
Python:  · pytest-cov 7.14.3  
Command: `pytest tests/ --cov=src/retrosynformer`

## Summary

| Metric | Value |
|--------|-------|
| Statements | 7,505 |
| Missing | 6,003 |
| Covered | 1,502 |
| **Total coverage** | **20.0%** |
| Tests collected | 298 |
| Tests passed | 298 |

## Per-module Coverage

| Module | Stmts | Miss | Cover | Bar |
|--------|------:|-----:|------:|-----|
| `dashboard/__init__.py` | 71 | 71 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `dashboard/extensions.py` | 3 | 3 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `dashboard/models.py` | 123 | 123 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `dashboard/plots.py` | 89 | 89 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `dashboard/sync.py` | 391 | 391 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `dashboard/views.py` | 235 | 235 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `dataframes.py` | 309 | 309 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `fit.py` | 21 | 21 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `names.py` | 10 | 10 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `rsync.py` | 50 | 50 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/cleanup_study.py` | 166 | 166 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/compress_model.py` | 63 | 63 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/dashboard.py` | 27 | 27 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/evaluate.py` | 437 | 437 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/hplot.py` | 63 | 63 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/mark_stopped.py` | 102 | 102 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/monitor_jsonl.py` | 25 | 25 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/monitor_progress.py` | 36 | 36 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/plot_learning_curves.py` | 232 | 232 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/serve.py` | 19 | 19 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/show_all_studies.py` | 73 | 73 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/show_study.py` | 69 | 69 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/status.py` | 110 | 110 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/sync_results.py` | 32 | 32 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/train.py` | 28 | 28 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `scripts/upload_model.py` | 283 | 283 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `study.py` | 181 | 181 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `utils/evaluation_average.py` | 157 | 157 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `utils/evaluation_compare_aizynth.py` | 24 | 24 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| `inference.py` | 280 | 258 | 7.9% | ██░░░░░░░░░░░░░░░░░░ |
| `trainer.py` | 344 | 311 | 9.6% | ██░░░░░░░░░░░░░░░░░░ |
| `runner.py` | 307 | 277 | 9.8% | ██░░░░░░░░░░░░░░░░░░ |
| `scripts/__init__.py` | 29 | 26 | 10.3% | ██░░░░░░░░░░░░░░░░░░ |
| `utils/evaluation_class.py` | 346 | 308 | 11.0% | ██░░░░░░░░░░░░░░░░░░ |
| `async_eval.py` | 221 | 192 | 13.1% | ███░░░░░░░░░░░░░░░░░ |
| `utils/evaluation.py` | 217 | 184 | 15.2% | ███░░░░░░░░░░░░░░░░░ |
| `training_display.py` | 44 | 36 | 18.2% | ████░░░░░░░░░░░░░░░░ |
| `environment.py` | 222 | 180 | 18.9% | ████░░░░░░░░░░░░░░░░ |
| `monitor.py` | 100 | 81 | 19.0% | ████░░░░░░░░░░░░░░░░ |
| `data.py` | 165 | 126 | 23.6% | █████░░░░░░░░░░░░░░░ |
| `__init__.py` | 20 | 14 | 30.0% | ██████░░░░░░░░░░░░░░ |
| `serve/predictor.py` | 93 | 63 | 32.3% | ██████░░░░░░░░░░░░░░ |
| `scripts/hypertune.py` | 289 | 158 | 45.3% | █████████░░░░░░░░░░░ |
| `epoch_logger.py` | 56 | 29 | 48.2% | ██████████░░░░░░░░░░ |
| `structured_dropout.py` | 50 | 25 | 50.0% | ██████████░░░░░░░░░░ |
| `utils/utils.py` | 269 | 119 | 55.8% | ███████████░░░░░░░░░ |
| `serve/app.py` | 72 | 25 | 65.3% | █████████████░░░░░░░ |
| `models_optuna.py` | 563 | 141 | 75.0% | ███████████████░░░░░ |
| `extrapolate.py` | 68 | 7 | 89.7% | ██████████████████░░ |
| `compression.py` | 126 | 10 | 92.1% | ██████████████████░░ |
| `dropout.py` | 101 | 4 | 96.0% | ███████████████████░ |
| `etl.py` | 5 | 0 | 100.0% | ████████████████████ |
| `serve/__init__.py` | 0 | 0 | 100.0% | ████████████████████ |
| `serve/schemas.py` | 68 | 0 | 100.0% | ████████████████████ |
| `utils/__init__.py` | 0 | 0 | 100.0% | ████████████████████ |
| `utils/reward_functions.py` | 21 | 0 | 100.0% | ████████████████████ |

## Coverage Tiers

| Tier | Modules |
|------|---------|
| ≥ 90% | `compression.py`, `dropout.py`, `etl.py`, `serve/__init__.py`, `serve/schemas.py`, `utils/__init__.py`, `utils/reward_functions.py` |
| 75–89% | `extrapolate.py`, `models_optuna.py` |
| 50–74% | `serve/app.py`, `structured_dropout.py`, `utils/utils.py` |
| 25–49% | `__init__.py`, `epoch_logger.py`, `scripts/hypertune.py`, `serve/predictor.py` |
| < 25% | `async_eval.py`, `dashboard/__init__.py`, `dashboard/extensions.py`, `dashboard/models.py`, `dashboard/plots.py`, `dashboard/sync.py`, `dashboard/views.py`, `data.py`, `dataframes.py`, `environment.py`, `fit.py`, `inference.py`, `monitor.py`, `names.py`, `rsync.py`, `runner.py`, `scripts/__init__.py`, `scripts/cleanup_study.py`, `scripts/compress_model.py`, `scripts/dashboard.py`, `scripts/evaluate.py`, `scripts/hplot.py`, `scripts/mark_stopped.py`, `scripts/monitor_jsonl.py`, `scripts/monitor_progress.py`, `scripts/plot_learning_curves.py`, `scripts/serve.py`, `scripts/show_all_studies.py`, `scripts/show_study.py`, `scripts/status.py`, `scripts/sync_results.py`, `scripts/train.py`, `scripts/upload_model.py`, `study.py`, `trainer.py`, `training_display.py`, `utils/evaluation.py`, `utils/evaluation_average.py`, `utils/evaluation_class.py`, `utils/evaluation_compare_aizynth.py` |

## Notes

- **Dashboard, scripts, study, dataframes**: 0% — CLI entry points and GCS/Cloud Run tooling
  not exercised by unit tests (require network/GCS/Optuna DB).
- **runner, trainer**: ~10% — training loop is integration-level; tested indirectly via
  `test_hypertune.py` which exercises `runner.main()` end-to-end.
- **compression**: 92% — all format/codec paths covered; uncovered lines are
  import-error guards for optional `safetensors` dependency.
- **dropout, extrapolate, reward_functions, serve/schemas, etl**: ≥ 90%.
