# RetroSynFormer 0.1.8 Release Notes

*Branch: `feature-structured-dropout` — June 2026*

---

## Summary

Operational monitoring tooling: a web dashboard for inspecting Optuna trials and
training progress, a CLI table command for quick per-trial status, and a background
ntfy.sh alerter for long-running GPU runs. Also fixes ROCm dependency pins and a
dashboard display regression.

---

## Changes

### 1. `rs-dashboard` — Flask-Admin training dashboard

New `src/retrosynformer/dashboard/` package and `rs-dashboard` CLI entry point.

- Flask-Admin views over a local SQLite file (`results/dashboard.db`) that mirrors
  Optuna trial data: Trials table, Studies table, and a summary page.
- **Trials table** shows `study_name`, trial number, state, score, and all
  hyperparameters in sortable/filterable columns.
- Login-gated: HTTP Basic Auth protects all views (credentials in config or env).
- Deployed to the `taco` GPU server via Tailscale; accessible at the server's
  Tailscale HTTPS address.
- `deploy/taco-dashboard.service` — user-mode systemd unit that starts the
  dashboard on boot under `.venv-rocm`.

### 2. `rs-status` — trial summary table CLI

New `src/retrosynformer/scripts/status.py` and `rs-status` entry point.

- Syncs `study.db` and `train_progress.jsonl` from the remote host (same rsync
  logic as `rs-sync-results`), then prints a ranked Pandas table:

  | Column | Notes |
  |--------|-------|
  | trial | trial number within study |
  | study | study name |
  | score | Optuna objective value |
  | valid_action_accuracy | best across epochs |
  | frac_solved | fraction of targets solved |
  | lr | learning rate (scientific notation) |
  | n_heads / n_layers | architecture dims |
  | mtime | last-modified time of `train_progress.jsonl` |

- `--top N` limits output to the N highest-scoring trials (default 20).
- Fixed `frac_solved` parsing: reads `route_solved` from the nested
  `pred_routes` structure produced by the current evaluator format.

### 3. `monitor_training.py` — ntfy.sh background alerter

New `monitor_training.py` script (installed as `rs-monitor`).

- Polls `train_progress.jsonl` for new epoch rows and pushes a push notification
  via `ntfy.sh` whenever `valid_action_accuracy` improves.
- Runs in the background; safe to leave running over an SSH session.
- ntfy topic name kept out of git via `.env` (`NTFY_TOPIC`).

### 4. ROCm / torch dependency fixes

- `uv.lock`: reverted from ROCm 6.4 / torch 2.9.1 back to ROCm 6.2 / torch 2.5.1
  — the 6.4 index crashes on `taco`'s Strix Halo iGPU.
- `pyproject.toml`: removed the `amdgpu` alias extra; only `rocm` is canonical.
  The alias caused `uv` to resolve conflicting torch versions when both extras
  were active.

### 5. Dashboard display fix

- Trials table was showing the Python `repr()` of the SQLAlchemy `Study` object
  instead of its `study_name` string. Fixed by referencing the relationship
  attribute directly.

---

## Upgrade notes

- `rs-dashboard`, `rs-status`, and `rs-monitor` require `uv sync` / `pip install -e .`
  to register the new entry points.
- ROCm users: run `uv sync --extra rocm` (not `--extra amdgpu`) after pulling;
  the alias is gone.
- `results/dashboard.db` is generated on first `rs-dashboard` launch; it is
  git-ignored and should not be committed.
