#!/usr/bin/env python
"""Sync results from taco and print a summary table of all active trials.

Usage
-----
    rs-status                          # sync all studies, print table
    rs-status --no-sync                # skip rsync, use local files only
    rs-status --study standard-v2-lr0005
    rs-status --host gpu-box --results results/
"""
import argparse
import json
import sys
from pathlib import Path

from retrosynformer.rsync import sync

EXTRA_INCLUDES = [
    "pred_routes_train_progress.json",
    "model.config.yaml",
]

DEFAULT_HOST = "taco"
DEFAULT_REMOTE = "code/corethink/retrosynformer/results/"
DEFAULT_LOCAL = "results/"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _last_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    return last


def _best_fraction_solved(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None
    values = [r["fraction_solved"] for r in records if r.get("fraction_solved") is not None]
    return max(values) if values else None


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _parse_trial(trial_dir: Path) -> dict | None:
    last = _last_jsonl(trial_dir / "train_progress.jsonl")
    if last is None:
        return None
    cfg = _load_config(trial_dir / "model.config.yaml")
    m = cfg.get("model", {})
    o = cfg.get("optimizer", {})
    t = cfg.get("train", {})
    n_epochs = t.get("n_epochs", "?")
    epoch = last.get("epoch", "?")
    spe = last.get("seconds_per_epoch")
    if isinstance(epoch, int) and isinstance(n_epochs, int) and spe:
        remaining_s = (n_epochs - epoch - 1) * spe
        eta = f"{remaining_s / 3600:.1f}h" if remaining_s > 60 else "done"
    else:
        eta = "?"
    frac = _best_fraction_solved(trial_dir / "pred_routes_train_progress.json")
    return {
        "study":      trial_dir.parent.name.removeprefix("hypertune-"),
        "trial":      trial_dir.name,
        "epoch":      f"{epoch}/{n_epochs}",
        "valid_loss": last.get("valid_loss"),
        "act_acc":    last.get("valid_action_accuracy"),
        "route_acc":  last.get("valid_route_accuracy"),
        "frac_solved": frac,
        "eta":        eta,
        "heads":      m.get("n_heads", "?"),
        "layers":     m.get("n_layers", "?"),
        "dim":        m.get("head_dim", "?"),
        "lr":         o.get("lr", "?"),
    }


def collect_all_trials(results_root: Path, study_filter: str | None) -> list[dict]:
    rows = []
    pattern = f"hypertune-{study_filter}" if study_filter else "hypertune-*"
    for study_dir in sorted(results_root.glob(pattern)):
        for trial_dir in sorted(study_dir.glob("trial_*/")):
            row = _parse_trial(trial_dir)
            if row:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".4f") -> str:
    if val is None:
        return "—"
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


COLUMNS = [
    ("study",       "Study",       "{}",    20),
    ("trial",       "Trial",       "{}",    10),
    ("epoch",       "Epoch",       "{}",     8),
    ("valid_loss",  "Loss",        ".4f",    8),
    ("act_acc",     "ActAcc",      ".4f",    8),
    ("route_acc",   "RouteAcc",    ".4f",    9),
    ("frac_solved", "FracSolved",  ".4f",   10),
    ("eta",         "ETA",         "{}",     6),
    ("heads",       "H",           "{}",     3),
    ("layers",      "L",           "{}",     3),
    ("dim",         "Dim",         "{}",     4),
    ("lr",          "LR",          "{}",    10),
]


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No trial data found.")
        return
    header = "  ".join(f"{h:<{w}}" for _, h, _, w in COLUMNS)
    sep = "  ".join("-" * w for _, _, _, w in COLUMNS)
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for key, _, fmt, width in COLUMNS:
            val = row.get(key)
            if fmt == "{}":
                s = str(val) if val is not None else "—"
            else:
                s = _fmt(val, fmt)
            cells.append(f"{s:<{width}}")
        print("  ".join(cells))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--study", default=None,
                        help="Filter to one study (without 'hypertune-' prefix)")
    parser.add_argument("--no-sync", action="store_true", dest="no_sync",
                        help="Skip rsync, use local files only")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Remote SSH host (default: {DEFAULT_HOST})")
    parser.add_argument("--results", default=DEFAULT_LOCAL, dest="results",
                        help=f"Local results directory (default: {DEFAULT_LOCAL})")
    args = parser.parse_args()

    if not args.no_sync:
        from retrosynformer.rsync import DEFAULT_INCLUDES
        remote = f"{args.host}:{DEFAULT_REMOTE}"
        rc = sync(
            src=remote,
            dst=args.results,
            includes=list(DEFAULT_INCLUDES) + EXTRA_INCLUDES,
            verbose=False,
        )
        if rc != 0:
            print(f"[warning] rsync exited {rc}", file=sys.stderr)

    rows = collect_all_trials(Path(args.results), args.study)
    print_table(rows)


if __name__ == "__main__":
    main()
