import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from inference import RoutePredictor

from .utils import utils


class RetroTrainer:
    def __init__(self, dataloaders, model, config):
        print("Initiating the trainer")

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = dataloaders
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_df = pd.DataFrame({})
        self.results_eval = []
        self.state_dim = int(
            self.config["dataset"]["fp_dim"] * self.config["dataset"]["n_in_state"]
        )

        lr = self.config["optimizer"]["lr"]
        momentum = self.config["optimizer"]["momentum"]
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min"
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def unpack_data(self, data):

        (
            (states, actions, rewards, timesteps, attention_mask),
            action_labels,
            target_routes,
        ) = data
        batch_size, episode_length = states.size(0), states.size(1)

        states = states[:, :, : self.state_dim].to(
            device=self.device, dtype=torch.float32
        )  # (batch_size, episode_length, state_dim)

        actions = actions.to(
            device=self.device, dtype=torch.float32
        )  # (batch_size, episode_length, state_dim)

        rewards = rewards.to(device=self.device, dtype=torch.float32)

        # Calculate the return to go from the rewards
        cumulative_reward = torch.cumsum(rewards.squeeze(-1), dim=1)  # new
        rtgs = torch.sum(rewards, dim=1).repeat(1, episode_length) - cumulative_reward
        rtgs = rtgs.unsqueeze(-1).to(
            device=self.device, dtype=torch.float32
        )  # (batch_size, episode_length, 1

        timesteps = torch.cat(
            [
                torch.arange(episode_length).reshape(1, episode_length)
                for _ in range(batch_size)
            ],
            dim=0,
        ).to(device=self.device, dtype=torch.long)

        attention_mask = attention_mask.squeeze(-1).to(
            device=self.device, dtype=torch.long
        )
        return (
            states,
            actions,
            action_labels,
            rewards,
            timesteps,
            attention_mask,
            rtgs,
        ), target_routes

    def train_one_epoch(self):
        total_loss = 0
        actions_id_batch, actions_id_pred_batch, action_preds_batch = [], [], []

        for i, data in enumerate(self.train_dataloader):

            (
                states,
                actions,
                actions_id,
                rewards,
                timesteps,
                attention_mask,
                rtgs,
            ), target_routes = self.unpack_data(data)

            self.model.train(True)
            self.optimizer.zero_grad()

            _, action_preds, _ = self.model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtgs,
                timesteps=timesteps,
                return_dict=False,
                attention_mask=attention_mask,
            )

            actions_id_pred = action_preds.argmax(dim=-1)

            loss = self.loss_fn(action_preds, actions)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            actions_id_batch.extend(actions_id)
            filtered_actions_id_pred = [
                a[: len(b)].tolist() for a, b in zip(actions_id_pred, actions_id)
            ]
            actions_id_pred_batch.extend(filtered_actions_id_pred)

        flat_actions_id_batch = utils.flatten(actions_id_batch)
        flat_actions_id_pred_batch = utils.flatten(actions_id_pred_batch)

        route_accuracy = sum(
            [
                a_pred == a_target
                for a_pred, a_target in zip(actions_id_batch, actions_id_pred_batch)
            ]
        ) / len(actions_id_batch)
        print("train route accuracy: ", route_accuracy)
        action_accuracy = accuracy_score(
            flat_actions_id_batch, flat_actions_id_pred_batch
        )
        print("train action accuracy: ", action_accuracy)

        return total_loss / len(self.train_dataloader), action_accuracy, route_accuracy

    def eval(self, dataloader=None):
        if not dataloader:
            dataloader = self.valid_dataloader
        total_loss = 0
        self.model.eval()
        actions_id_batch, actions_id_pred_batch, actions_pred_batch = (
            [],
            [],
            [],
        )
        actions_id_batch, actions_id_pred_batch = [], []
        with torch.no_grad():
            for _, data in enumerate(dataloader):

                (
                    states,
                    actions,
                    actions_id,
                    rewards,
                    timesteps,
                    attention_mask,
                    rtgs,
                ), target_routes = self.unpack_data(data)

                _, action_preds, _ = self.model(
                    states=states,
                    actions=actions,
                    returns_to_go=rtgs,
                    rewards=rewards,
                    timesteps=timesteps,
                    return_dict=False,
                    attention_mask=attention_mask,
                )
                actions_id_pred = action_preds.argmax(dim=-1)
                actions_id_batch.extend(actions_id)
                filtered_actions_id_pred = [
                    a[: len(b)].tolist() for a, b in zip(actions_id_pred, actions_id)
                ]
                actions_id_pred_batch.extend(filtered_actions_id_pred)

                loss = self.loss_fn(action_preds, actions)
                total_loss += loss.item()

            flat_actions_id_batch = utils.flatten(actions_id_batch)
            flat_actions_id_pred_batch = utils.flatten(actions_id_pred_batch)

            route_accuracy = sum(
                [
                    a_pred == a_target
                    for a_pred, a_target in zip(actions_id_batch, actions_id_pred_batch)
                ]
            ) / len(actions_id_batch)
            print("valid route accuracy: ", route_accuracy)
            action_accuracy = accuracy_score(
                flat_actions_id_batch, flat_actions_id_pred_batch
            )
            print("valid action accuracy: ", action_accuracy)

        return (
            total_loss / len(dataloader),
            action_accuracy,
            route_accuracy,
            actions_id_pred_batch,
            actions_id_batch,
        )

    def train(self, verbose=True):
        start_time = time.time()
        route_predictor = RoutePredictor(
            self.model, self.config, beam_width=self.config["evaluation"]["beam_width"]
        )

        n_epochs = self.config["train"]["n_epochs"]
        save_folder = self.config["train"]["results_path"]

        current_datetime = datetime.now()
        if save_folder:
            save_folder = self.config["train"]["results_path"]
        else:
            save_folder = f"results/{current_datetime.strftime('%Y-%m-%d-%H:%M:%S')}/"

        print("Save result at: ", save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print("Directory created successfully.")

        lowest_valid_loss = 1000

        training_loss, validation_loss = [], []
        (
            training_accuracy,
            validation_accuracy,
        ) = (
            [],
            [],
        )
        training_route_accuracy, validation_route_accuracy = [], []

        # import cProfile
        # profiler = cProfile.Profile()
        # profiler.enable()
        for epoch in range(n_epochs):

            if verbose:
                print("epoch: ", epoch)
            train_loss, train_action_accuracy, train_route_accuracy = (
                self.train_one_epoch()
            )
            training_loss.append(train_loss)
            training_accuracy.append(train_action_accuracy)
            training_route_accuracy.append(train_route_accuracy)

            valid_loss, valid_action_accuracy, valid_route_accuracy, _, _ = self.eval()
            self.scheduler.step(valid_loss)
            validation_loss.append(valid_loss)
            validation_accuracy.append(valid_action_accuracy)
            validation_route_accuracy.append(valid_route_accuracy)

            eval_routes_frequency = self.config["evaluation"]["eval_routes_frequency"]
            if (
                # epoch == 0 or
                (epoch % eval_routes_frequency == 0 and epoch > 0)
                or epoch == n_epochs - 1
            ):
                # import cProfile
                # profiler = cProfile.Profile()
                # profiler.enable()
                eval_start_time = time.time()
                print("Evaluation for epoch ", epoch, " has started!")
                route_predictor.set_model(self.model)
                pred_routes = route_predictor.eval_predicted_routes(
                    self.valid_dataloader
                )
                self.results_eval.append({"epoch": epoch, "result": pred_routes})
                print("Evaluation took: ", (time.time() - eval_start_time) / 60, "min")
                print(
                    f"This is an average of {(time.time() - eval_start_time) / len(pred_routes)} seconds per target."
                )
                print(f"Evaluating {len(pred_routes)} routes.")
                # profiler.dump_stats(os.path.join(save_folder, 'emmas_predict_3.cprofile'))

                solved_routes = [r["route_solved"] for r in pred_routes]
                fraction_targets_solved = sum(solved_routes) / len(solved_routes)
                print("fraction solved targets: ", fraction_targets_solved)

                valid_routes = [r["valid_route"] for r in pred_routes]
                fraction_valid_routes = sum(valid_routes) / len(valid_routes)
                print("fraction valid_routes: ", fraction_valid_routes)
            else:
                fraction_targets_solved = None

            results_df_epoch = pd.DataFrame(
                {
                    "epoch": [epoch],
                    "train_loss": [train_loss],
                    "train_action_accuracy": [train_action_accuracy],
                    "train_route_accuracy": [train_route_accuracy],
                    "valid_loss": [valid_loss],
                    "valid_action_accuracy": [valid_action_accuracy],
                    "valid_route_accuracy": [valid_route_accuracy],
                }
            )

            self.result_df = pd.concat(
                [self.result_df, results_df_epoch], ignore_index=True
            )

            if verbose:
                print("Train loss: ", train_loss, "Valid loss: ", valid_loss)
                print(
                    "Train accuracy: ",
                    train_action_accuracy,
                    "Valid accuracy: ",
                    valid_action_accuracy,
                )
                print(
                    "Train route accuracy: ",
                    train_route_accuracy,
                    "Valid route accuracy: ",
                    valid_route_accuracy,
                )

            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss
                torch.save(self.model.state_dict(), save_folder + "/model.pth")
            print("total loss:")

            self.result_df.to_csv(save_folder + "/train_progress.csv")
            with open(save_folder + "/pred_routes_train_progress.json", "w") as results:
                json.dump(self.results_eval, results)

        # profiler.dump_stats(os.path.join(save_folder, 'emmas.cprofile'))
        utils.plot_train_progress(
            save_folder + "/train_progress.csv",
            save_folder + "/train_progress_loss.png",
        )
        utils.plot_train_progress_accuracy(
            save_folder + "/train_progress.csv",
            save_folder + "/train_progress_accuracy.png",
        )
        utils.plot_evaluation_results(
            save_folder + "/pred_routes_train_progress.json",
            save_folder + "/evaluation_target_solved.png",
        )

        return (
            validation_loss,
            validation_accuracy,
            validation_route_accuracy,
            fraction_targets_solved,
        )
