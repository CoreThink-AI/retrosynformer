import argparse
import json
import os
import time

import yaml

from retrosynformer.inference import RoutePredictor
from retrosynformer.runner import (
    create_dataloaders,
    create_dataloaders_n1_n5,
    init_data,
    init_model,
    read_config,
)


def main(
    model_dir,
    beam_width,
    sort_on="trajectory_prob",
    n1=False,
    n5=False,
    test=False,
    n_batches=None,
    batch_size=None,
):
    model_config_path = model_dir + "/config.yaml"
    if not os.path.exists(model_config_path):
        model_config_path = model_dir + "/model.config.yaml"
    model_config = read_config(model_config_path)
    model_path = model_config["train"]["results_path"] + "/model.pth"

    model_config["evaluation"]["beam_width"] = beam_width
    model_config["evaluation"]["sort_on"] = sort_on
    if batch_size:
        model_config["evaluation"]["batch_size"] = batch_size
    if n_batches:
        model_config["evaluation"]["eval_n_batches"] = n_batches

    # Define the directory path you want to create
    predictions_path = (
        f"{model_dir}/bw{beam_width}_sorton_{sort_on}_n1{n1}_n5{n5}"
    )
    # Create the directory if it does not exist
    os.makedirs(
        predictions_path, exist_ok=True
    )  # `exist_ok=True` avoids an error if the directory already exists
    print(f"Prediction directory '{predictions_path}' created successfully.")

    new_config_path = predictions_path + "/config.yaml"

    with open(new_config_path, "w") as file:
        yaml.dump(model_config, file)

    time.time()
    print("Initiate training.")
    model = init_model(model_config, model_path=model_path)
    print("Model is loaded!")
    datasets = init_data(model_config)
    print("Dataset loaded!")
    if n1 or n5:
        n1_dataloader, n5_dataloader = create_dataloaders_n1_n5(datasets, model_config)
    else:
        train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
            datasets, model_config
        )
    eval_start_time = time.time()
    print("Evaluation started!")
    route_predictor = RoutePredictor(model, model_config)
    if n1:
        print("Predicting routes for n1 targets.")
        pred_routes = route_predictor.eval_predicted_routes(n1_dataloader)
        with open(predictions_path + "/predicted_routes_n1.json", "w") as results:
            json.dump(pred_routes, results)
    elif n5:
        print("Predicting routes for n5 targets.")
        pred_routes = route_predictor.eval_predicted_routes(n5_dataloader)
        with open(predictions_path + "/predicted_routes_n5.json", "w") as results:
            json.dump(pred_routes, results)
    else:
        dataloader = test_dataloader if test else valid_dataloader
        print(f'Predict routes for {"test" if test else "valid"} targets.')
        pred_routes = route_predictor.eval_predicted_routes(dataloader)
        with open(predictions_path + "/predicted_routes.json", "w") as results:
            json.dump(pred_routes, results)

    print("Evaluation took: ", (time.time() - eval_start_time) / 60, "min")
    print(
        f"This is an average of {(time.time() - eval_start_time) / len(pred_routes)} seconds per target."
    )
    print(f"Evaluating {len(pred_routes)} routes.")

    return pred_routes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments

    parser.add_argument(
        "-d",
        "--model_dir",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--beam_width",
        type=int,
    )
    parser.add_argument("-s", "--sort_on", type=str, default="trajectory_prob")
    parser.add_argument(
        "--n1",
        action="store_true",  # If the argument is present, n1 will be True, otherwise False
        help="Set n1 to True",
    )
    parser.add_argument(
        "--n5",
        action="store_true",  # If the argument is present, n5 will be True, otherwise False
        help="Set n5 to True",
    )
    parser.add_argument("-t", "--testset", type=bool, default=False)
    parser.add_argument(
        "-n",
        "--n_batches",
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
    )

    args = parser.parse_args()
    main(
        model_dir=args.model_dir,
        beam_width=args.beam_width,
        sort_on=args.sort_on,
        n1=args.n1,
        n5=args.n5,
        test=args.testset,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
    )
