# RetroSynFormer 0.1.5 Release Notes

*Branch: `feature-structured-dropout` — June 2026*

---

## Summary

Scripts are now installable as first-class CLI commands via `pip install` / `uv sync`.  The `hypertune_verbose.py` cruft file has been deleted.

---

## Changes

### Packaging — `pyproject.toml`

- **`[build-system]`** added: `setuptools>=68` is now the declared build backend, enabling both `[project.scripts]` console-script entry points and `[tool.setuptools]` shell-script installation.

- **`[project.scripts]`** — six Python CLI commands installed to `PATH` on `pip install retrosynformer`:

  | Command | Entry point |
  |---------|-------------|
  | `rs-train` | `retrosynformer.scripts.train:main` |
  | `rs-hypertune` | `retrosynformer.scripts.hypertune:main` |
  | `rs-show-study` | `retrosynformer.scripts.show_study:main` |
  | `rs-show-all-studies` | `retrosynformer.scripts.show_all_studies:main` |
  | `rs-monitor-jsonl` | `retrosynformer.scripts.monitor_jsonl:main` |
  | `rs-monitor-progress` | `retrosynformer.scripts.monitor_progress:main` |

- **`[tool.setuptools.script-files]`** — two shell scripts installed as executables:
  - `monitor_train_progress.sh`
  - `train_structured_dropout_comparison.sh`

- **`[tool.setuptools.packages.find]`** `where = ["src"]` — explicit src-layout declaration.

### New package: `src/retrosynformer/scripts/`

Logic for each script has moved from `scripts/*.py` into the installable package at `src/retrosynformer/scripts/`.  Changes per file:

| File | Changes vs `scripts/` original |
|------|-------------------------------|
| `train.py` | argparse block wrapped in `def main()` |
| `hypertune.py` | identical to `scripts/hypertune.py` |
| `show_study.py` | `sys.path.insert` dev hack removed |
| `show_all_studies.py` | `sys.path.insert` dev hack removed |
| `monitor_jsonl.py` | module-level code wrapped in `def main()`; path now an argparse positional arg |
| `monitor_progress.py` | module-level code wrapped in `def main()`; path now an argparse positional arg |

### `scripts/*.py` — thin shims

All `scripts/*.py` files are now 4-line shims that delegate to the package:

```python
from retrosynformer.scripts.X import main
if __name__ == "__main__":
    main()
```

`python scripts/train.py -c ...` continues to work unchanged for developers running from the repo root.

### Shell scripts

- `monitor_train_progress.sh` and `train_structured_dropout_comparison.sh` — added `#!/usr/bin/env bash` shebang and `chmod +x`.
- `train_structured_dropout_comparison.sh` — updated to call `rs-hypertune` instead of `python scripts/hypertune.py`, so it works after `pip install`.

### Deleted

- `scripts/hypertune_verbose.py` — removed as cruft; `rs-hypertune` (`scripts/hypertune.py`) is the canonical implementation.

---

## Upgrade notes

After `pip install -e .` or `uv sync`, the six `retrosynformer-*` commands are available on `PATH`.  Existing `python scripts/*.py` invocations are unaffected.
