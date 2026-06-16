#!/usr/bin/env python3
"""Monitor RetroSynFormer training progress and send periodic alerts.

Rsyncs fresh progress files from taco every --interval seconds, parses the
latest stats from all active trials in a study, and sends a summary via
ntfy.sh (push to phone/browser) and notify-send (desktop).

Setup (one-time):
    Subscribe in browser: https://ntfy.sh/<your-topic>
    Or install the ntfy app and subscribe to the same topic.

Usage:
    python scripts/monitor_training.py --study standard-v2-lr0005
    python scripts/monitor_training.py --study standard-v2-lr0005 --interval 300
    python scripts/monitor_training.py --study standard-v2-lr0005 --ntfy-topic my-secret-topic
    nohup python scripts/monitor_training.py --study standard-v2-lr0005 > monitor.log 2>&1 &
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Rsync
# ---------------------------------------------------------------------------

RSYNC_CMD = [
    "rsync", "-a",
    "--include=*/",
    "--include=study.db",
    "--include=*.yaml",
    "--include=pred_routes_train_progress.json",
    "--include=train_progress.jsonl",
    "--include=*config*",
    "--include=*.log",
    "--exclude=*",
]


def rsync_from_taco(study: str, results_root: Path) -> bool:
    remote = f"taco:code/corethink/retrosynformer/results/hypertune-{study}/"
    local = str(results_root / f"hypertune-{study}") + "/"
    Path(local).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        RSYNC_CMD + [remote, local],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[rsync error] {result.stderr.strip()}", flush=True)
        return False
    changed = [l for l in result.stdout.splitlines() if not l.startswith("sent") and l.strip()]
    if changed:
        print(f"[rsync] updated: {changed}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Progress parsing
# ---------------------------------------------------------------------------

def _last_jsonl(path: Path) -> dict | None:
    """Return the last valid JSON object from a .jsonl file."""
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
    """Return the best fraction_solved from pred_routes_train_progress.json."""
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None
    if not records:
        return None
    values = [r.get("fraction_solved") for r in records if r.get("fraction_solved") is not None]
    return max(values) if values else None


def parse_trial(trial_dir: Path) -> dict | None:
    """Return a summary dict for one trial directory."""
    last = _last_jsonl(trial_dir / "train_progress.jsonl")
    if last is None:
        return None
    cfg_path = trial_dir / "model.config.yaml"
    params: dict = {}
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            m = cfg.get("model", {})
            o = cfg.get("optimizer", {})
            t = cfg.get("train", {})
            params = {
                "n_heads": m.get("n_heads"),
                "n_layers": m.get("n_layers"),
                "head_dim": m.get("head_dim"),
                "lr": o.get("lr"),
                "n_epochs": t.get("n_epochs"),
            }
        except Exception:
            pass

    epoch = last.get("epoch", "?")
    n_epochs = params.get("n_epochs") or "?"
    frac = _best_fraction_solved(trial_dir / "pred_routes_train_progress.json")

    return {
        "trial": trial_dir.name,
        "epoch": epoch,
        "n_epochs": n_epochs,
        "valid_loss": last.get("valid_loss"),
        "valid_action_acc": last.get("valid_action_accuracy"),
        "valid_route_acc": last.get("valid_route_accuracy"),
        "best_fraction_solved": frac,
        "params": params,
        "secs_per_epoch": last.get("seconds_per_epoch"),
    }


def collect_study_stats(study: str, results_root: Path) -> list[dict]:
    study_dir = results_root / f"hypertune-{study}"
    if not study_dir.exists():
        return []
    trials = sorted(study_dir.glob("trial_*/"), key=lambda p: p.name)
    summaries = []
    for t in trials:
        s = parse_trial(t)
        if s:
            summaries.append(s)
    return summaries


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_message(study: str, summaries: list[dict]) -> str:
    if not summaries:
        return f"[{study}] No trial data yet."

    lines = [f"RetroSynFormer — {study}"]
    for s in summaries:
        p = s["params"]
        epoch_str = f"epoch {s['epoch']}/{s['n_epochs']}"
        eta = ""
        if isinstance(s["epoch"], int) and isinstance(s["n_epochs"], int) and s["secs_per_epoch"]:
            remaining = (s["n_epochs"] - s["epoch"] - 1) * s["secs_per_epoch"]
            eta = f"  ETA {remaining/3600:.1f}h"

        lines.append(
            f"\n{s['trial']}  {epoch_str}{eta}"
        )
        if p:
            lines.append(
                f"  params: heads={p.get('n_heads')} layers={p.get('n_layers')} "
                f"dim={p.get('head_dim')} lr={p.get('lr')}"
            )
        loss = f"{s['valid_loss']:.4f}" if s["valid_loss"] is not None else "—"
        acc  = f"{s['valid_action_acc']:.4f}" if s["valid_action_acc"] is not None else "—"
        racc = f"{s['valid_route_acc']:.4f}" if s["valid_route_acc"] is not None else "—"
        frac = f"{s['best_fraction_solved']:.4f}" if s["best_fraction_solved"] is not None else "—"
        lines.append(f"  loss={loss}  act_acc={acc}  route_acc={racc}  frac_solved={frac}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def notify_ntfy(topic: str, title: str, body: str) -> None:
    try:
        subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"https://ntfy.sh/{topic}",
             "-H", f"Title: {title}",
             "-H", "Priority: default",
             "-d", body],
            capture_output=True, timeout=10,
        )
    except Exception as e:
        print(f"[ntfy] failed: {e}", flush=True)


def notify_desktop(title: str, body: str) -> None:
    try:
        subprocess.run(
            ["notify-send", "-t", "30000", title, body],
            capture_output=True, timeout=5,
        )
    except Exception as e:
        print(f"[notify-send] failed: {e}", flush=True)


def send_alert(study: str, summaries: list[dict], ntfy_topic: str | None, desktop: bool) -> None:
    msg = format_message(study, summaries)
    title = f"RetroSynFormer {study}"
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)
    if ntfy_topic:
        notify_ntfy(ntfy_topic, title, msg)
    if desktop:
        notify_desktop(title, msg)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True,
                        help="Study name (without 'hypertune-' prefix), e.g. standard-v2-lr0005")
    parser.add_argument("--interval", type=int, default=600,
                        help="Seconds between checks (default: 600 = 10 min)")
    parser.add_argument("--ntfy-topic", default=None, dest="ntfy_topic",
                        help="ntfy.sh topic to push alerts to (subscribe at https://ntfy.sh/<topic>)")
    parser.add_argument("--no-desktop", action="store_true",
                        help="Disable notify-send desktop notifications")
    parser.add_argument("--results-root", default="results", dest="results_root",
                        help="Local results directory (default: results/)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no loop)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    desktop = not args.no_desktop

    pid_file = Path(f".monitor_{args.study}.pid")
    pid_file.write_text(f"PID: {os.getpid()}\nkill: kill {os.getpid()}\ntopic: {args.ntfy_topic or 'none'}\n")
    print(f"PID {os.getpid()} saved to {pid_file}", flush=True)

    if args.ntfy_topic:
        print(f"Subscribe at: https://ntfy.sh/{args.ntfy_topic}", flush=True)

    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}] Syncing from taco...", flush=True)
        rsync_from_taco(args.study, results_root)
        summaries = collect_study_stats(args.study, results_root)
        send_alert(args.study, summaries, args.ntfy_topic, desktop)

        if args.once:
            break
        print(f"Next check in {args.interval // 60}m {args.interval % 60}s...", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
