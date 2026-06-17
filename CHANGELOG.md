# Changelog

All notable changes to RetroSynFormer are documented here.
Full release notes are in [`docs/`](docs/).

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
