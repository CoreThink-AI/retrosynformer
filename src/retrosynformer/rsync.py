"""Utilities for rsyncing Optuna study results from a remote host.

Public API
----------
DEFAULT_INCLUDES   list[str]   filenames included by default
build_cmd(...)     -> list[str]   build the rsync argv list
sync(...)          -> int          run rsync, return exit code
"""
import subprocess
from typing import Sequence

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
    verbose: bool = True,
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
    verbose: bool = True,
    dry_run: bool = False,
) -> int:
    """Run rsync to transfer *includes* files from *src* to *dst*.

    Returns the rsync exit code (0 = success).
    """
    cmd = build_cmd(src, dst, includes, archive=archive, verbose=verbose, dry_run=dry_run)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode
