#!/usr/bin/env python
"""Train RetroSynFormer. Requires the retrosynformer package to be installed."""
import argparse

from retrosynformer.runner import main as runner_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load model.pth from results_path and continue from last epoch",
    )
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=None,
        help="Override n_epochs from config",
    )
    parser.add_argument(
        "-d", "--dataset", choices=["small", "standard", "large"], default=None,
        help="Override dataset paths and action_dim in config (small=589, standard=1573, large=2957 templates)",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=None, dest="start_epoch",
        help="Override the starting epoch (useful when resuming across mixed-dataset runs)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=None, dest="batch_size",
        help="Override train and eval batch size from config",
    )
    parser.add_argument(
        "--n-heads", type=int, default=None, dest="n_heads",
        help="Override model.n_heads from config",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None, dest="n_layers",
        help="Override model.n_layers from config",
    )
    parser.add_argument(
        "--seed", type=int, default=None, dest="seed",
        help="Override context.random_state from config",
    )
    parser.add_argument(
        "--head-dim", type=int, default=None, dest="head_dim",
        help="Override model.head_dim from config (hidden_size = n_heads * head_dim)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, dest="lr",
        help="Override optimizer.lr from config",
    )
    parser.add_argument(
        "--dropout", type=float, default=None, dest="dropout",
        help="Override all dropout rates (attn_pdrop, embd_pdrop, resid_pdrop) from config",
    )
    parser.add_argument(
        "--momentum", type=float, default=None, dest="momentum",
        help="Override optimizer.momentum from config",
    )
    parser.add_argument(
        "--structured-dropout-rate", type=float, default=None, dest="structured_dropout_rate",
        help="Global multiplier on learned structured-dropout probabilities (0=off, 1=default)",
    )
    parser.add_argument(
        "--eval-n-batches", type=int, default=None, dest="eval_n_batches",
        help="Override evaluation.eval_n_batches from config (reduce for faster runs)",
    )
    parser.add_argument(
        "--results-path", type=str, default=None, dest="results_path",
        help="Override train.results_path from config",
    )
    args = parser.parse_args()
    runner_main(
        config_path=args.config_path,
        resume=args.resume,
        n_epochs=args.n_epochs,
        dataset=args.dataset,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        seed=args.seed,
        head_dim=args.head_dim,
        lr=args.lr,
        momentum=args.momentum,
        eval_n_batches=args.eval_n_batches,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
