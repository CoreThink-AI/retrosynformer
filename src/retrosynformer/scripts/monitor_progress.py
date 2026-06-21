#!/usr/bin/env python
"""Live animated plot of RetroSynFormer training progress.

Reads train_progress.jsonl and refreshes the plot every 30 seconds.
Requires an interactive display (X11 / macOS / Windows).
"""
import argparse
import json

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from retrosynformer.scripts import print_banner


def main():
    print_banner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl", nargs="?", default="results/train_progress.jsonl",
        help="Path to train_progress.jsonl (default: results/train_progress.jsonl)",
    )
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    def update(_):
        rows = [json.loads(line) for line in open(args.jsonl)]
        idx = max(
            (i for i in range(1, len(rows)) if rows[i]["epoch"] < rows[i - 1]["epoch"]),
            default=0,
        )
        df = pd.DataFrame(rows[idx:])
        for ax in (ax1, ax2):
            ax.cla()
        ax1.plot(df.epoch, df.train_action_accuracy * 100, label="train acc")
        ax1.plot(df.epoch, df.valid_action_accuracy * 100, label="valid acc")
        ax1.set_ylabel("Action accuracy %")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(df.epoch, df.valid_loss, label="valid loss", color="red")
        ax2.set_ylabel("Valid loss")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True)
        fig.suptitle(
            f"Epoch {df.epoch.iloc[-1]} — valid {df.valid_action_accuracy.iloc[-1]*100:.1f}%"
        )

    ani = animation.FuncAnimation(fig, update, interval=30_000)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
