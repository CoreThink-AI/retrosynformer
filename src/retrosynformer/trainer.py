import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from .inference import RoutePredictor
from .utils import utils


class RetroTrainer:
    def __init__(self, dataloaders, model, config):
        print("Initiating the trainer")

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = dataloaders
        self.config = config
        self.device = utils.get_device()
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
        lr_scheduler_patience = self.config["train"].get("lr_scheduler_patience", 10)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=lr_scheduler_patience
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

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

        # RTG[t] = sum(r[t], r[t+1], ..., r[T]) — inclusive of current step.
        # Reverse cumsum over the time dimension gives this directly.
        r = rewards.squeeze(-1)  # (batch, episode_length)
        rtgs = torch.flip(torch.cumsum(torch.flip(r, dims=[1]), dim=1), dims=[1])
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
        actions_id_batch, actions_id_pred_batch, _action_preds_batch = [], [], []

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

            loss = self.loss_fn(action_preds, actions) / attention_mask.sum()
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
        action_accuracy = accuracy_score(
            flat_actions_id_batch, flat_actions_id_pred_batch
        )

        return total_loss / len(self.train_dataloader), action_accuracy, route_accuracy

    def eval(self, dataloader=None):
        if not dataloader:
            dataloader = self.valid_dataloader
        total_loss = 0
        self.model.eval()
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

                loss = self.loss_fn(action_preds, actions) / attention_mask.sum()
                total_loss += loss.item()

            flat_actions_id_batch = utils.flatten(actions_id_batch)
            flat_actions_id_pred_batch = utils.flatten(actions_id_pred_batch)

            route_accuracy = sum(
                [
                    a_pred == a_target
                    for a_pred, a_target in zip(actions_id_batch, actions_id_pred_batch)
                ]
            ) / len(actions_id_batch)
            action_accuracy = accuracy_score(
                flat_actions_id_batch, flat_actions_id_pred_batch
            )

        return (
            total_loss / len(dataloader),
            action_accuracy,
            route_accuracy,
            actions_id_pred_batch,
            actions_id_batch,
        )

    def train(self, verbose=True, start_epoch=0, eval_routes_at_end=False,
              trial_number=None, study_name=None):
        time.time()
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
        patience = self.config["train"].get("early_stopping_patience", 0)
        epochs_no_improve = 0
        fraction_targets_solved = None

        training_loss, validation_loss = [], []
        (
            training_accuracy,
            validation_accuracy,
        ) = (
            [],
            [],
        )
        training_route_accuracy, validation_route_accuracy = [], []

        progress_path = save_folder.rstrip("/") + "/train_progress.jsonl"
        eval_path = save_folder.rstrip("/") + "/pred_routes_train_progress.json"

        if os.path.exists(progress_path):
            self.result_df = pd.read_json(progress_path, lines=True)
            # Detect the most recent contiguous run (epoch counter may restart at 0
            # when a trial is re-run on the same directory after a crash).
            all_epochs = self.result_df["epoch"].tolist()
            restart_idx = 0
            for i in range(1, len(all_epochs)):
                if all_epochs[i] < all_epochs[i - 1]:
                    restart_idx = i
            current_run = self.result_df.iloc[restart_idx:]
            current_run = current_run[current_run["epoch"] < start_epoch] if start_epoch > 0 else current_run
            lowest_valid_loss = current_run["valid_loss"].min() if len(current_run) > 0 else 1000
            print(f"Restored training history ({len(self.result_df)} epochs). Best valid loss (current run): {lowest_valid_loss:.6f}")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                self.results_eval = json.load(f)

        print(f"Training epochs {start_epoch} to {n_epochs - 1}.")
        if verbose:
            _hdr_prefix = []
            if trial_number is not None:
                _hdr_prefix.append("trial")
            _hdr_suffix = ["study"] if study_name is not None else []
            print("\t".join(_hdr_prefix + ["epoch", "t_loss", "t_acc", "t_racc",
                                           "v_loss", "v_acc", "v_racc", "s/ep", "note"] + _hdr_suffix),
                  flush=True)

        for epoch in range(start_epoch, n_epochs):
            epoch_start = time.time()

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

            seconds_this_epoch = time.time() - epoch_start

            # Save model before printing so the note column reflects this epoch's outcome.
            if valid_loss < lowest_valid_loss:
                model_path = save_folder + "/model.pth"
                lowest_valid_loss = valid_loss
                torch.save(self.model.state_dict(), model_path)
                epochs_no_improve = 0
                note = "*"
            else:
                epochs_no_improve += 1
                note = ""

            eval_routes_frequency = self.config["evaluation"]["eval_routes_frequency"]
            if (
                (epoch % eval_routes_frequency == 0 and epoch > 0)
                or epoch == n_epochs - 1
            ):
                eval_start_time = time.time()
                print(f"Epoch {epoch}: running route evaluation …", flush=True)
                route_predictor.set_model(self.model)
                pred_routes = route_predictor.eval_predicted_routes(
                    self.valid_dataloader
                )
                self.results_eval.append({"epoch": epoch, "result": pred_routes})
                solved_routes = [r["route_solved"] for r in pred_routes]
                fraction_targets_solved = sum(solved_routes) / len(solved_routes)
                valid_routes = [r["valid_route"] for r in pred_routes]
                fraction_valid_routes = sum(valid_routes) / len(valid_routes)
                print(f"  solved={fraction_targets_solved:.5f}  valid={fraction_valid_routes:.5f}"
                      f"  ({len(pred_routes)} routes, {(time.time()-eval_start_time)/60:.1f} min)",
                      flush=True)
            else:
                fraction_targets_solved = None

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_action_accuracy": train_action_accuracy,
                "train_route_accuracy": train_route_accuracy,
                "valid_loss": valid_loss,
                "valid_action_accuracy": valid_action_accuracy,
                "valid_route_accuracy": valid_route_accuracy,
                "seconds_per_epoch": seconds_this_epoch,
            }
            self.result_df = pd.concat(
                [self.result_df, pd.DataFrame([record])], ignore_index=True
            )

            if verbose:
                _row_prefix = []
                if trial_number is not None:
                    _row_prefix.append(str(trial_number))
                if study_name is not None:
                    _row_prefix.append(str(study_name))
                print("\t".join(_row_prefix + [
                    str(epoch),
                    f"{train_loss:.5f}",
                    f"{train_action_accuracy:.5f}",
                    f"{train_route_accuracy:.5f}",
                    f"{valid_loss:.5f}",
                    f"{valid_action_accuracy:.5f}",
                    f"{valid_route_accuracy:.5f}",
                    f"{seconds_this_epoch:.1f}",
                    note,
                ]), flush=True)

            with open(progress_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            with open(eval_path, "w") as results:
                json.dump(self.results_eval, results)

            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping: valid_loss has not improved for {patience} consecutive epochs.")
                break

        # Run final route evaluation when explicitly requested (hypertune) or
        # when early stopping skipped the last scheduled eval epoch.
        if eval_routes_at_end or fraction_targets_solved is None:
            print("Running final route evaluation …")
            eval_start_time = time.time()
            route_predictor.set_model(self.model)
            pred_routes = route_predictor.eval_predicted_routes(self.valid_dataloader)
            self.results_eval.append({"epoch": epoch, "result": pred_routes})
            with open(eval_path, "w") as results:
                json.dump(self.results_eval, results)
            solved_routes = [r["route_solved"] for r in pred_routes]
            fraction_targets_solved = sum(solved_routes) / len(solved_routes) if solved_routes else 0.0
            print(f"Final route eval: {fraction_targets_solved:.4f} fraction solved "
                  f"({sum(solved_routes)}/{len(solved_routes)}) "
                  f"in {(time.time() - eval_start_time) / 60:.1f} min")

        # profiler.dump_stats(os.path.join(save_folder, 'emmas.cprofile'))
        utils.plot_train_progress(
            save_folder + "/train_progress.jsonl",
            save_folder + "/train_progress_loss.png",
        )
        utils.plot_train_progress_accuracy(
            save_folder + "/train_progress.jsonl",
            save_folder + "/train_progress_accuracy.png",
        )
        eval_results_path = save_folder + "/pred_routes_train_progress.json"
        if os.path.exists(eval_results_path) and os.path.getsize(eval_results_path) > 2:
            utils.plot_evaluation_results(
                eval_results_path,
                save_folder + "/evaluation_target_solved.png",
            )

        return (
            validation_loss,
            validation_accuracy,
            validation_route_accuracy,
            fraction_targets_solved,
        )
