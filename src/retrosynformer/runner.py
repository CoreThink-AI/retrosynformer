import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from transformers import DecisionTransformerConfig, DecisionTransformerModel

from .data import RouteDatasetTorch, collate_fn
from .trainer import RetroTrainer
from .utils import evaluation, reward_functions, utils

from .utils.utils import get_device
DEVICE = get_device()


def read_config(config_path):
    return utils.read_config(config_path)


def init_data(config):
    # Read routes from file
    routes_data = pd.read_json(config["dataset"]["routes_path"])
    if len(routes_data["target"].tolist()[0]) == 1:
        # TODO short term solution - fix routes data
        routes_data["target"] = [i[0] for i in routes_data["target"]]
    if config["dataset"]["synthetic_routes_path"]:
        synthetic_routes_data = pd.read_json(config["dataset"]["synthetic_routes_path"])
        synthetic_routes_data["target"] = [
            i[0] for i in synthetic_routes_data["target"]
        ]
        routes_data = pd.concat((routes_data, synthetic_routes_data))
    if config["dataset"]["shuffle"]:
        routes_data = routes_data.sample(
            frac=1, random_state=config["context"]["random_state"]
        )
    # Converting from general reward to specific according to reward_mapping in config
    reward_mapping = reward_functions.get_reward_mapping(config)
    routes_data["reward_specific"] = reward_functions.get_specific_rewards(
        routes_data["reward_general"].values,
        routes_data["depth"].values,
        reward_mapping,
    )
    print("len(routes_data)", len(routes_data))

    # Split data into train, valid and test
    test_frac = config["evaluation"]["test_frac"]

    n_test = int(len(routes_data.drop_duplicates(subset=["target"])) * test_frac)

    if config["dataset"]["valid_set"] == "random_split":
        test_valid_data = routes_data.drop_duplicates(subset=["target"]).sample(
            n=int(2 * n_test), random_state=config["context"]["random_state"]
        )
        train_data = routes_data.drop(test_valid_data.index)
        test_valid_targets = list(test_valid_data["target"].values)
        train_data = train_data[~train_data["target"].isin(test_valid_targets)]
        valid_data = test_valid_data.sample(
            n=n_test, random_state=config["context"]["random_state"]
        )
        valid_data = valid_data.drop_duplicates(subset=["target"])
        test_data = test_valid_data.drop(valid_data.index)
        test_data = test_data.drop_duplicates(subset=["target"])

    elif config["dataset"]["valid_set"] == "n1+n5":
        print("n1+n5")
        test_data = routes_data[
            (routes_data["n1_target"]) | (routes_data["n5_target"])
        ]
        print(
            "len(test_data)",
            len(test_data),
            len(test_data.drop_duplicates(subset=["target"])),
        )
        train_valid_data = routes_data[
            ~routes_data["target"].isin(test_data["target"].tolist())
        ]
        valid_data = train_valid_data.drop_duplicates(subset=["target"]).sample(
            n=n_test, random_state=config["context"]["random_state"]
        )
        print("len(valid_data)", len(valid_data))
        train_data = train_valid_data[
            ~train_valid_data["target"].isin(valid_data["target"].tolist())
        ]

    print("len train data", len(train_data))
    if (
        "drop_duplicates" in config["dataset"].keys()
        and config["dataset"]["drop_duplicates"]
    ):
        train_data = train_data.drop_duplicates(subset=["target"])
        print("drop_duplicates done: len train data", len(train_data))
    if (
        "train_fraction" in config["dataset"].keys()
        and config["dataset"]["train_fraction"] < 1
    ):
        n = int(len(train_data) * config["dataset"]["train_fraction"])
        train_data = train_data.sample(n=n, random_state=config["context"]["random_state"])
        print(
            f'Training on fraction: {config["dataset"]["train_fraction"]}. Len train data: {len(train_data)}'
        )

    train_dataset = RouteDatasetTorch(
        train_data,
        "reward_specific",
        config["dataset"]["action_dim"],
        config["dataset"]["fp_dim"],
        drop_duplicates=False,
    )
    valid_dataset = RouteDatasetTorch(
        valid_data,
        "reward_specific",
        config["dataset"]["action_dim"],
        config["dataset"]["fp_dim"],
        drop_duplicates=False,
    )
    test_dataset = RouteDatasetTorch(
        test_data,
        "reward_specific",
        config["dataset"]["action_dim"],
        config["dataset"]["fp_dim"],
        drop_duplicates=False,
    )

    print(f"Train dataset has {len(train_dataset)} routes.")
    print(f"Valid dataset has {len(valid_dataset)} routes.")
    print(f"Test dataset has {len(test_dataset)} routes.")

    return train_dataset, valid_dataset, test_dataset


def create_dataloaders_n1_n5(datasets, config, shuffle=False):
    train_dataset, valid_dataset, test_dataset = datasets
    n1_data = test_dataset.data[test_dataset.data["n1_target"]].drop_duplicates(
        subset=["target"]
    )
    n5_data = test_dataset.data[test_dataset.data["n5_target"]].drop_duplicates(
        subset=["target"]
    )

    n1_dataset = RouteDatasetTorch(
        n1_data,
        "reward_specific",
        config["dataset"]["action_dim"],
        config["dataset"]["fp_dim"],
        drop_duplicates=True,
    )
    n5_dataset = RouteDatasetTorch(
        n5_data,
        "reward_specific",
        config["dataset"]["action_dim"],
        config["dataset"]["fp_dim"],
        drop_duplicates=True,
    )

    batch_size = config["evaluation"]["batch_size"]
    n1_dataloader = DataLoader(n1_dataset, batch_size, collate_fn=collate_fn)
    n5_dataloader = DataLoader(n5_dataset, batch_size, collate_fn=collate_fn)

    return n1_dataloader, n5_dataloader


def create_dataloaders(datasets, config, shuffle=False, batch_size=None):
    train_dataset, valid_dataset, test_dataset = datasets

    train_batch_size = batch_size or config["train"]["batch_size"]
    eval_batch_size = batch_size or config["evaluation"]["batch_size"]
    num_workers = config["train"].get("num_workers", 0)
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset, train_batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    valid_dataloader = DataLoader(valid_dataset, eval_batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, eval_batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataloader, valid_dataloader, test_dataloader


def init_model(config, model_path=None):

    dt_config = DecisionTransformerConfig(bos_token_id=None, eos_token_id=None)
    dt_config.act_dim = config["dataset"]["action_dim"]
    state_dim = int(config["dataset"]["fp_dim"] * config["dataset"]["n_in_state"])
    dt_config.state_dim = state_dim
    dt_config.max_ep_len = config["model"]["max_ep_len"]
    dt_config.hidden_size = config["model"]["hidden_size"]  # derived: n_heads * head_dim
    dt_config.n_layer = config["model"]["n_layers"]
    dt_config.n_head = config["model"]["n_heads"]
    dt_config.activation_function = config["model"]["activation_function"]
    dt_config.action_tanh = config["model"]["action_tanh"]
    dt_config.attn_pdrop = config["model"]["attn_pdrop"]
    dt_config.embd_pdrop = config["model"]["embd_pdrop"]
    dt_config.resid_pdrop = config["model"]["resid_pdrop"]

    if config["model"].get("use_structured_dropout", False):
        from .structured_dropout import StructuredDropoutDecisionTransformer
        fp_dim = config["dataset"]["fp_dim"]
        bottleneck = config["model"].get("structured_dropout_bottleneck", 128)
        rate = config["model"].get("structured_dropout_rate", 1.0)
        model = StructuredDropoutDecisionTransformer(dt_config, fp_dim, bottleneck, rate=rate)
        print(f"Using StructuredDropoutDecisionTransformer (fp_dim={fp_dim}, bottleneck={bottleneck}, rate={rate})")
    else:
        model = DecisionTransformerModel(dt_config)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    return model


DATASET_CONFIGS = {
    "small":    {"routes": "data/small_routes.json",    "building_blocks": "data/small_building_blocks.csv",    "templates": "data/small_reaction_templates.pickle",    "action_dim": 589},
    "standard": {"routes": "data/standard_routes.json", "building_blocks": "data/standard_building_blocks.csv", "templates": "data/standard_reaction_templates.pickle", "action_dim": 1573},
    "large":    {"routes": "data/large_routes.json",    "building_blocks": "data/large_building_blocks.csv",    "templates": "data/large_reaction_templates.pickle",    "action_dim": 2957},
}


def main(config_path, resume=False, n_epochs=None, dataset=None, start_epoch=None, batch_size=None, n_heads=None, n_layers=None, seed=None, head_dim=None, results_path=None, lr=None, dropout=None, momentum=None, eval_n_batches=None, structured_dropout_bottleneck=None, structured_dropout_rate=None, eval_routes_at_end=False):
    start_time = time.time()
    print("Initiate training.")
    config = read_config(config_path)
    if n_epochs is not None:
        config["train"]["n_epochs"] = n_epochs
    if dataset is not None:
        ds = DATASET_CONFIGS[dataset]
        config["dataset"]["routes_path"] = ds["routes"]
        config["context"]["building_blocks"] = ds["building_blocks"]
        config["context"]["templates_path"] = ds["templates"]
        config["dataset"]["action_dim"] = ds["action_dim"]
        print(f"Dataset override: {dataset} ({ds['action_dim']} templates)")
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
        config["evaluation"]["batch_size"] = batch_size
        print(f"Batch size override: {batch_size}")
    if n_heads is not None:
        config["model"]["n_heads"] = n_heads
        print(f"n_heads override: {n_heads}")
    if n_layers is not None:
        config["model"]["n_layers"] = n_layers
        print(f"n_layers override: {n_layers}")
    if seed is not None:
        config["context"]["random_state"] = seed
        print(f"seed override: {seed}")
    if head_dim is not None:
        config["model"]["head_dim"] = head_dim
        print(f"head_dim override: {head_dim}")
    if results_path is not None:
        config["train"]["results_path"] = results_path
        print(f"results_path override: {results_path}")
    if lr is not None:
        config["optimizer"]["lr"] = lr
        print(f"lr override: {lr}")
    if momentum is not None:
        config["optimizer"]["momentum"] = momentum
        print(f"momentum override: {momentum}")
    if dropout is not None:
        config["model"]["attn_pdrop"] = dropout
        config["model"]["embd_pdrop"] = dropout
        config["model"]["resid_pdrop"] = dropout
        print(f"dropout override: {dropout} (attn/embd/resid)")
    if eval_n_batches is not None:
        config["evaluation"]["eval_n_batches"] = eval_n_batches
        print(f"eval_n_batches override: {eval_n_batches}")
    if structured_dropout_bottleneck is not None:
        config["model"]["structured_dropout_bottleneck"] = int(structured_dropout_bottleneck)
        print(f"structured_dropout_bottleneck override: {structured_dropout_bottleneck}")
    if structured_dropout_rate is not None:
        config["model"]["structured_dropout_rate"] = float(structured_dropout_rate)
        print(f"structured_dropout_rate override: {structured_dropout_rate}")
    # Derive hidden_size from n_heads * head_dim whenever head_dim is present in config.
    if "head_dim" in config["model"]:
        config["model"]["hidden_size"] = config["model"]["n_heads"] * config["model"]["head_dim"]
    model_path = None
    derived_start_epoch = 0
    if resume:
        results_path = config["train"]["results_path"].rstrip("/")
        model_path = results_path + "/model.pth"
        progress_path = results_path + "/train_progress.jsonl"
        if pd.io.common.file_exists(progress_path):
            prev = pd.read_json(progress_path, lines=True)
            derived_start_epoch = int(prev["epoch"].max()) + 1
        print(f"Resuming from checkpoint: {model_path}, starting at epoch {derived_start_epoch}")
    if start_epoch is not None:
        derived_start_epoch = start_epoch
        print(f"Start epoch overridden to {derived_start_epoch}")
    start_epoch = derived_start_epoch

    # Write all effective values back into config before saving so that
    # model.config.yaml always reflects what was actually used.
    config["train"]["start_epoch"] = derived_start_epoch
    config["train"]["resume"] = resume
    if dataset is not None:
        config["dataset"]["dataset_name"] = dataset

    results_path = config["train"]["results_path"].rstrip("/")
    effective_config_path = results_path + "/model.config.yaml"
    with open(effective_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Effective config saved to {effective_config_path}")

    seed = config["context"]["random_state"]
    torch.manual_seed(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed + 3)

    model = init_model(config, model_path=model_path)
    print("Model is loaded!")
    datasets = init_data(config)
    print("Dataset loaded!")
    dataloaders = create_dataloaders(datasets, config, batch_size=batch_size)
    begin_train_time = time.time()
    print("Begin training after: ", (begin_train_time - start_time) / 60, " minutes.")
    trainer = RetroTrainer(dataloaders, model, config)
    (
        validation_loss,
        validation_accuracy,
        validation_route_accuracy,
        fraction_targets_solved,
    ) = trainer.train(start_epoch=derived_start_epoch, eval_routes_at_end=eval_routes_at_end)
    print("Training is completed.")
    end_time = time.time()
    print("Training took: ", (end_time - begin_train_time) / (60 * 60), " hours.")
    result_dir = config["train"]["results_path"]
    # Only run post-hoc evaluation when predicted_routes.json exists (written by
    # predict.py). pred_routes_train_progress.json is always written by the trainer
    # but has a different format that evaluation.main() cannot consume.
    _pred = result_dir.rstrip("/") + "/predicted_routes.json"
    if os.path.exists(_pred):
        try:
            evaluation.main(result_dir=result_dir)
        except Exception as exc:
            logger.error("Post-training evaluation failed (non-fatal): %s", exc)

    return (
        validation_loss,
        validation_accuracy,
        validation_route_accuracy,
        fraction_targets_solved,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load model.pth from results_path before training",
    )
    parser.add_argument(
        "-n",
        "--n_epochs",
        type=int,
        default=None,
        help="Override n_epochs from config",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["small", "standard", "large"],
        default=None,
        help="Override dataset paths and action_dim in config",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        dest="start_epoch",
        help="Override the starting epoch (useful when resuming across mixed-dataset runs)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override train and eval batch size from config",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        dest="n_heads",
        help="Override model.n_heads from config",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        dest="n_layers",
        help="Override model.n_layers from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        dest="seed",
        help="Override context.random_state from config",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        dest="head_dim",
        help="Override model.head_dim from config (hidden_size = n_heads * head_dim)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        dest="lr",
        help="Override optimizer.lr from config",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        dest="dropout",
        help="Override all dropout rates (attn_pdrop, embd_pdrop, resid_pdrop) from config",
    )
    parser.add_argument(
        "--structured-dropout-rate",
        type=float,
        default=None,
        dest="structured_dropout_rate",
        help="Global multiplier on learned structured-dropout probabilities (0=off, 1=default, >1=amplify)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        dest="momentum",
        help="Override optimizer.momentum from config",
    )

    args = parser.parse_args()
    main(
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
        dropout=args.dropout,
        momentum=args.momentum,
        structured_dropout_rate=args.structured_dropout_rate,
    )
