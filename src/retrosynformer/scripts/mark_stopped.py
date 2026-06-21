#!/usr/bin/env python
"""Mark zombie RUNNING Optuna trials as STOPPED.

A trial is considered a zombie when its RUNNING state is stale — i.e. the
training process was killed without Optuna recording a terminal state.

For each zombie the script:
  • Sets the Optuna study.db state to FAIL (the closest valid Optuna state).
  • Sets the dashboard.db trial state to STOPPED.
  • Recomputes study-level aggregates (n_running, status) in dashboard.db.

Usage
-----
    rs-mark-stopped                    # default: RUNNING trials older than 6h
    rs-mark-stopped --threshold 12     # only trials older than 12 hours
    rs-mark-stopped --all              # every RUNNING trial regardless of age
    rs-mark-stopped --dry-run          # preview without writing
    rs-mark-stopped --study sunday-optuna-12trials-200epochs-4patience
"""
import argparse
import glob
import os
import sqlite3
import sys
from datetime import datetime, timezone
from retrosynformer.scripts import add_log_args, configure_logging, print_banner


RESULTS_ROOT = "results"
DASHBOARD_DB = os.path.join(RESULTS_ROOT, "dashboard.db")


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _age_hours(dt_start_str: str) -> float:
    if not dt_start_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(dt_start_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except ValueError:
        return 0.0


def _find_zombies(threshold_hours: float, study_filter: str | None) -> list[dict]:
    """Return list of zombie trial dicts from all study.db files."""
    zombies = []
    for db_path in sorted(glob.glob(f"{RESULTS_ROOT}/**/study.db", recursive=True)):
        study_dir = os.path.dirname(db_path)
        folder = os.path.basename(study_dir)
        try:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT t.trial_id, t.number, t.state, t.datetime_start, "
                "       s.study_name "
                "FROM trials t JOIN studies s ON t.study_id = s.study_id "
                "WHERE t.state = 'RUNNING'"
            ).fetchall()
            con.close()
        except Exception as e:
            print(f"  [skip] {folder}: {e}", file=sys.stderr)
            continue

        for row in rows:
            sname = row["study_name"]
            if study_filter and study_filter not in (sname, folder):
                continue
            age = _age_hours(row["datetime_start"])
            if age >= threshold_hours:
                zombies.append({
                    "db_path": db_path,
                    "folder": folder,
                    "study_name": sname,
                    "trial_id": row["trial_id"],
                    "trial_number": row["number"],
                    "datetime_start": row["datetime_start"],
                    "age_hours": age,
                })
    return zombies


def _mark_optuna_fail(db_path: str, trial_id: int, now_str: str, dry_run: bool) -> None:
    if dry_run:
        return
    con = sqlite3.connect(db_path)
    con.execute(
        "UPDATE trials SET state='FAIL', datetime_complete=? WHERE trial_id=?",
        (now_str, trial_id),
    )
    con.commit()
    con.close()


def _mark_dashboard_stopped(study_name: str, trial_number: int, now_str: str, dry_run: bool) -> None:
    if dry_run or not os.path.exists(DASHBOARD_DB):
        return
    con = sqlite3.connect(DASHBOARD_DB)
    # Find the study id
    row = con.execute(
        "SELECT id FROM studies WHERE study_name=?", (study_name,)
    ).fetchone()
    if row is None:
        con.close()
        return
    study_id = row[0]
    # Upsert: update if exists, insert if not (e.g. dashboard not yet synced for this trial)
    existing = con.execute(
        "SELECT id FROM trials WHERE study_id=? AND trial_number=?",
        (study_id, trial_number),
    ).fetchone()
    if existing:
        con.execute(
            "UPDATE trials SET state='STOPPED', datetime_complete=? "
            "WHERE study_id=? AND trial_number=?",
            (now_str, study_id, trial_number),
        )
    else:
        con.execute(
            "INSERT INTO trials (study_id, trial_number, state, datetime_complete, synced_at) "
            "VALUES (?, ?, 'STOPPED', ?, ?)",
            (study_id, trial_number, now_str, now_str),
        )
    con.commit()
    # Recompute study aggregates
    stats = con.execute(
        "SELECT "
        "  COUNT(*) AS n_total, "
        "  SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) AS n_complete, "
        "  SUM(CASE WHEN state='RUNNING'  THEN 1 ELSE 0 END) AS n_running, "
        "  SUM(CASE WHEN state IN ('FAIL','FAILED') THEN 1 ELSE 0 END) AS n_failed "
        "FROM trials WHERE study_id=?",
        (study_id,),
    ).fetchone()
    n_total, n_complete, n_running, n_failed = stats
    n_stopped = con.execute(
        "SELECT COUNT(*) FROM trials WHERE study_id=? AND state='STOPPED'",
        (study_id,),
    ).fetchone()[0]

    if n_running and n_running > 0:
        status = "active"
    elif n_complete and n_complete > 0:
        status = "complete"
    elif n_stopped and n_stopped > 0:
        status = "stopped"
    else:
        status = "unknown"

    con.execute(
        "UPDATE studies SET n_trials=?, n_complete=?, n_running=?, n_failed=?, status=? "
        "WHERE id=?",
        (n_total, n_complete or 0, n_running or 0, n_failed or 0, status, study_id),
    )
    con.commit()
    con.close()


def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threshold", type=float, default=6.0, metavar="HOURS",
        help="Mark trials that have been RUNNING for at least this many hours (default: 6)",
    )
    parser.add_argument(
        "--all", dest="mark_all", action="store_true",
        help="Mark every RUNNING trial regardless of age",
    )
    parser.add_argument(
        "--study", metavar="NAME",
        help="Restrict to a specific study name or folder name",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be changed without writing anything",
    )
    add_log_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    threshold = 0.0 if args.mark_all else args.threshold
    zombies = _find_zombies(threshold, args.study)

    if not zombies:
        print("No zombie RUNNING trials found.")
        return

    now_str = _utcnow_str()
    verb = "Would mark" if args.dry_run else "Marking"

    for z in zombies:
        print(
            f"{verb} STOPPED: {z['study_name']} / trial #{z['trial_number']}"
            f"  (started {z['datetime_start']}, {z['age_hours']:.1f}h ago)"
        )
        _mark_optuna_fail(z["db_path"], z["trial_id"], now_str, args.dry_run)
        _mark_dashboard_stopped(z["study_name"], z["trial_number"], now_str, args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] {len(zombies)} trial(s) would be marked stopped.")
    else:
        print(f"\nDone. {len(zombies)} trial(s) marked stopped.")


if __name__ == "__main__":
    main()
