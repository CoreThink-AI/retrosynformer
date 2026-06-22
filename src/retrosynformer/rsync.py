"""Utilities for rsyncing Optuna study results from a remote host.

Public API
----------
DEFAULT_INCLUDES        list[str]   filenames included by default
build_cmd(...)          -> list[str]   build the rsync argv list
sync(...)               -> int          run rsync, return exit code
cleanup_stale(path)     -> None         mark stale RUNNING trials COMPLETE after sync
"""
import glob
import logging
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_INCLUDES: list[str] = [
    "study.db",
    "train_progress.jsonl",
]


def build_cmd(
    src: str,
    dst: str,
    includes: Sequence[str] = DEFAULT_INCLUDES,
    *,
    archive: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Return the rsync argv list for syncing *includes* files under *src* → *dst*.

    The filter ordering follows the rsync rule: include all directories first
    (so rsync recurses the tree), then include the named files, then exclude
    everything else.

    Parameters
    ----------
    src:
        Source path, e.g. ``"taco:code/corethink/retrosynformer/results/"``.
        A trailing slash is preserved as-is (rsync semantics: sync *contents*).
    dst:
        Local destination directory, e.g. ``"results/"``.
    includes:
        Filenames (or rsync patterns) to transfer.  Directories are always
        included automatically so the remote tree is traversed.
    archive:
        Pass ``-a`` (preserves permissions, timestamps, symlinks).
    verbose:
        Pass ``-v``.
    dry_run:
        Pass ``-n`` (show what would be transferred without doing it).
    """
    flags = "-a" if archive else "-r"
    if verbose:
        flags += "v"
    if dry_run:
        flags += "n"

    cmd = ["rsync", flags]
    cmd += ["--include=*/"]          # recurse into all subdirectories
    for pattern in includes:
        cmd += [f"--include={pattern}"]
    cmd += ["--exclude=*"]           # drop everything not explicitly included
    cmd += [src, dst]
    return cmd


def sync(
    src: str,
    dst: str,
    includes: Sequence[str] = DEFAULT_INCLUDES,
    *,
    archive: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
) -> int:
    """Run rsync to transfer *includes* files from *src* to *dst*.

    stdout is captured and emitted via logger.info() so it is silent at the
    default WARNING level and visible when the caller sets INFO or DEBUG.

    Returns the rsync exit code (0 = success).
    """
    cmd = build_cmd(src, dst, includes, archive=archive, verbose=verbose, dry_run=dry_run)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if line.strip():
            logger.info("%s", line)
    for line in result.stderr.splitlines():
        if line.strip():
            logger.warning("%s", line)
    return result.returncode


def cleanup_stale(local_path: str, *, dry_run: bool = False) -> None:
    """Mark stale RUNNING/FAILED/CANCELED trials COMPLETE in every study.db under *local_path*.

    Called automatically after a successful sync.  Safe to run repeatedly —
    trials that already have an objective value are skipped.
    """
    try:
        from retrosynformer.models_optuna import complete_stale_trials
    except Exception as exc:
        logger.warning("Stale-trial cleanup unavailable: %s", exc)
        return

    dbs = glob.glob(str(Path(local_path) / "**" / "study.db"), recursive=True)
    for db in sorted(dbs):
        try:
            results = complete_stale_trials(db, dry_run=dry_run)
        except Exception as exc:
            logger.warning("complete_stale_trials failed for %s: %s", db, exc)
            continue

        study_name = Path(db).parent.name
        for trial_num, r in sorted(results.items()):
            if r["action"] == "skipped":
                logger.info(
                    "[%s] trial %03d: skipped (%s)",
                    study_name, trial_num, r["skip_reason"],
                )
            else:
                verb = "would complete" if dry_run else "completed"
                logger.info(
                    "[%s] trial %03d: %s  metric=%s  estimate=%.4f ± %.4f  n_obs=%d",
                    study_name, trial_num, verb,
                    r["metric_used"], r["estimated_value"], r["se"] or 0.0,
                    r["n_epochs_observed"],
                )
