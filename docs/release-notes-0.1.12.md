# RetroSynFormer 0.1.12 Release Notes

*Branch: `feature-nonuniform-dropout` — June 2026*

---

## Summary

Removed the `rs-bump` / `bump` CLI commands and the `anthropic` SDK dependency
that backed them. Version bumps are now done manually.

---

## Changes

- Deleted `scripts/bump.py` and `src/retrosynformer/scripts/bump.py`.
- Removed `rs-bump` and `bump` entry points from `[project.scripts]` in `pyproject.toml`.
- Removed `anthropic>=0.25` from the `[dev]` extra in `pyproject.toml`.
