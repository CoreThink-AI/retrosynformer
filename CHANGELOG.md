# Changelog

All notable changes to RetroSynFormer are documented here.
Full release notes are in [`docs/`](docs/).

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
