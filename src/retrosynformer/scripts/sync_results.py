#!/usr/bin/env python
"""Sync Optuna study results from a remote host.

Transfers only study.db and train_progress.jsonl files (plus any extra
files passed via --include) while preserving the remote directory structure.

After a successful sync, stale RUNNING/FAILED/CANCELED trials (those with no
objective value recorded) are automatically repaired by extrapolating their
training curves and marking them COMPLETE.  This is idempotent: trials that
already have an objective value are left untouched, so a genuinely completed
trial on the remote is never overwritten.  Pass --no-cleanup-stale to skip.

Usage
-----
    rs-sync-results
    rs-sync-results --host taco --dry-run
    rs-sync-results --include run.jsonl --include "*.png"
    rs-sync-results --remote-path code/corethink/retrosynformer/results/
    rs-sync-results --no-cleanup-stale
"""
import argparse
import logging
import sys

from retrosynformer.rsync import DEFAULT_INCLUDES, cleanup_stale, sync
from retrosynformer.scripts import add_log_args, configure_logging, print_banner

logger = logging.getLogger(__name__)

DEFAULT_HOST = "taco"
DEFAULT_REMOTE_PATH = "code/corethink/retrosynformer/results/"
DEFAULT_LOCAL_PATH = "results/"


def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Remote SSH host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--remote-path", default=DEFAULT_REMOTE_PATH, dest="remote_path",
        help=f"Path on the remote host (default: {DEFAULT_REMOTE_PATH})",
    )
    parser.add_argument(
        "--local-path", default=DEFAULT_LOCAL_PATH, dest="local_path",
        help=f"Local destination directory (default: {DEFAULT_LOCAL_PATH})",
    )
    parser.add_argument(
        "--include", metavar="PATTERN", action="append", default=None,
        help="Extra filename or rsync pattern to include (repeat for multiple). "
             f"Always included: {DEFAULT_INCLUDES}",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be transferred without actually doing it",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress rsync file-by-file output",
    )
    parser.add_argument(
        "--no-cleanup-stale", dest="cleanup_stale", action="store_false", default=True,
        help="Skip post-sync stale-trial cleanup (default: cleanup is enabled)",
    )
    add_log_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    includes = list(DEFAULT_INCLUDES)
    if args.include:
        includes += args.include

    src = f"{args.host}:{args.remote_path}"
    rc = sync(
        src=src,
        dst=args.local_path,
        includes=includes,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )

    if rc == 0 and args.cleanup_stale:
        cleanup_stale(args.local_path, dry_run=args.dry_run)

    sys.exit(rc)


if __name__ == "__main__":
    main()
