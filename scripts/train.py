#!/usr/bin/env python
"""Train RetroSynFormer. Requires the retrosynformer package to be installed."""
import argparse

from retrosynformer.runner import main

DATASET_CONFIGS = {
    "small":    {"routes": "data/small_routes.json",    "building_blocks": "data/small_building_blocks.csv",    "templates": "data/small_reaction_templates.pickle",    "action_dim": 589},
    "standard": {"routes": "data/standard_routes.json", "building_blocks": "data/standard_building_blocks.csv", "templates": "data/standard_reaction_templates.pickle", "action_dim": 1573},
    "large":    {"routes": "data/large_routes.json",    "building_blocks": "data/large_building_blocks.csv",    "templates": "data/large_reaction_templates.pickle",    "action_dim": 2957},
}

if __name__ == "__main__":
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
    args = parser.parse_args()
    main(
        config_path=args.config_path,
        resume=args.resume,
        n_epochs=args.n_epochs,
        dataset=args.dataset,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
    )
