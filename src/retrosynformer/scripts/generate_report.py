#!/usr/bin/env python
"""rs-report — generate report.yaml for one or more hypertune trials.

Duplicates hyperparameters/score from study.db and the training curve from
train_progress.jsonl, and adds learned-parameter counts plus entropy/
complexity estimates of the trained weights (see retrosynformer.model_stats).

Usage
-----
    rs-report results/hypertune-large-24-26-layer/trial_003
    rs-report results/hypertune-large-24-26-layer --all       # every trial_NNN/
    rs-report results/hypertune-large-24-26-layer/trial_003 --codec xz --per-tensor
"""
import argparse
import sys
from pathlib import Path

from retrosynformer.report import generate_trial_report
from retrosynformer.scripts import add_log_args, configure_logging, print_banner


def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="A trial_NNN/ directory, or a study directory with --all")
    parser.add_argument(
        "--all", action="store_true",
        help="Generate reports for every trial_NNN/ directory found under PATH",
    )
    parser.add_argument(
        "--codec", default="bz2", choices=["gz", "bz2", "xz"],
        help="Compression codec for the whole-file complexity estimate (default: bz2)",
    )
    parser.add_argument(
        "--per-tensor", action="store_true",
        help="Include a full per-tensor complexity breakdown (verbose)",
    )
    add_log_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    base = Path(args.path)
    if args.all:
        trial_dirs = sorted(base.glob("trial_[0-9][0-9][0-9]"))
        if not trial_dirs:
            sys.exit(f"No trial_NNN directories found under {base}")
    else:
        trial_dirs = [base]

    n_written = 0
    for trial_dir in trial_dirs:
        try:
            report = generate_trial_report(
                trial_dir,
                compression_codec=args.codec,
                include_per_tensor_stats=args.per_tensor,
            )
        except FileNotFoundError as exc:
            print(f"  {trial_dir}: skipped ({exc})", file=sys.stderr)
            continue

        n_params = report["model"]["parameters"]["total_parameters"]
        rank = report["model"]["complexity"]["overall"].get("mean_effective_rank")
        rank_str = f", mean effective rank {rank:.1f}" if rank else ""
        print(f"  {report['_report_path']}  —  {n_params:,} params{rank_str}")
        n_written += 1

    print(f"\nDone. {n_written}/{len(trial_dirs)} report(s) written.")


if __name__ == "__main__":
    main()
