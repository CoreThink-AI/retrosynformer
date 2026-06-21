#!/usr/bin/env python3
"""Tail a train_progress.jsonl file and print a formatted progress table."""
import argparse
import json
import time
from retrosynformer.scripts import add_log_args, configure_logging, print_banner


def main():
    print_banner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl", nargs="?", default="results/train_progress.jsonl",
        help="Path to train_progress.jsonl (default: results/train_progress.jsonl)",
    )
    add_log_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    path = args.jsonl
    print(f"{'ep':>4}  {'train':>7}  {'valid':>7}  {'route':>7}  {'loss':>10}  {'secs':>5}")
    print("-" * 52)

    seen = 0
    while True:
        with open(path) as f:
            lines = f.readlines()
        for line in lines[seen:]:
            r = json.loads(line)
            print(
                f"{r['epoch']:>4}  "
                f"{r['train_action_accuracy']*100:>6.2f}%  "
                f"{r['valid_action_accuracy']*100:>6.2f}%  "
                f"{r['valid_route_accuracy']*100:>6.2f}%  "
                f"{r['valid_loss']:>10.6f}  "
                f"{r['seconds_per_epoch']:>4.0f}s"
            )
        seen = len(lines)
        time.sleep(10)


if __name__ == "__main__":
    main()
