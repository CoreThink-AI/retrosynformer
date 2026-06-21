import hashlib
import json
import logging
import os
import shutil
import signal
import tempfile
import time
from datetime import datetime

logger = logging.getLogger(__name__)

import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from .async_eval import AsyncRouteEvalPool
from .epoch_logger import EpochLogger
from .inference import RoutePredictor
from .utils import utils

# ---------------------------------------------------------------------------
# Graceful Ctrl-C handling
#
# On the first SIGINT the handler sets _interrupted and (optionally) calls
# _stop_study_callback so Optuna does not start another trial.  The training
# loop detects the flag at the end of each epoch, breaks, runs route eval,
# and returns normally — allowing Optuna to record the trial result.
#
# A second SIGINT arriving within 1 second of the first restores the original
# handler and re-raises immediately (no waiting for the epoch to finish).
# ---------------------------------------------------------------------------

_interrupted: bool = False
_last_interrupt_time: float = 0.0
_stop_study_callback = None   # optional callable, e.g. study.stop
_original_sigint = signal.SIG_DFL


def set_interrupt_callback(fn) -> None:
    """Register a zero-argument callable invoked on the first Ctrl-C (e.g. study.stop)."""
    global _stop_study_callback
    _stop_study_callback = fn


def clear_interrupt_callback() -> None:
    global _stop_study_callback
    _stop_study_callback = None


def is_interrupted() -> bool:
    """True if training was stopped by Ctrl-C (check after train() returns)."""
    return _interrupted


def _handle_sigint(signum, frame):
    global _interrupted, _last_interrupt_time, _original_sigint
    now = time.time()
    if now - _last_interrupt_time < 1.0:
        # Second Ctrl-C within 1 s — restore original handler and raise immediately.
        signal.signal(signal.SIGINT, _original_sigint)
        raise KeyboardInterrupt
    _last_interrupt_time = now
    _interrupted = True
    print(
        "\n[trainer] Ctrl-C caught — finishing current epoch and running route "
        "evaluation before stopping. Hit Ctrl-C again within 1 s to abort immediately.",
        flush=True,
    )
    if _stop_study_callback is not None:
        try:
            _stop_study_callback()
        except Exception:
            pass


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
        logger.debug("trainer device=%s  model on %s", self.device, next(self.model.parameters()).device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum
        )
        lr_scheduler_patience = self.config["train"].get("lr_scheduler_patience", 5)
        lr_scheduler_factor = self.config["train"].get("lr_scheduler_factor", 0.75)
        self._lr_metric = self.config["train"].get("lr_scheduler_metric", "valid_loss")
        _lr_mode = "min" if "loss" in self._lr_metric else "max"
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, _lr_mode, patience=lr_scheduler_patience, factor=lr_scheduler_factor
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self._n_lr_reductions = 0
        self._best_valid_route_accuracy = 0.0

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
        total_grad_norm = 0.0
        actions_id_batch, actions_id_pred_batch, _action_preds_batch = [], [], []

        for i, data in enumerate(self.train_dataloader):
            if i == 0:
                logger.debug("train_one_epoch: first batch received")

            (
                states,

                actions,
                actions_id,
                rewards,
                timesteps,
                attention_mask,
                rtgs,
            ), target_routes = self.unpack_data(data)

            if i == 0:
                logger.debug("batch 0 tensors: states=%s actions=%s rtgs=%s", states.device, actions.device, rtgs.device)

            self.model.train(True)
            self.optimizer.zero_grad()

            if i == 0:
                logger.debug("batch 0: starting forward pass")
            _, action_preds, _ = self.model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtgs,
                timesteps=timesteps,
                return_dict=False,
                attention_mask=attention_mask,
            )

            if i == 0:
                logger.debug("batch 0: forward done, starting backward")
            actions_id_pred = action_preds.argmax(dim=-1)

            loss = self.loss_fn(action_preds, actions) / attention_mask.sum()
            loss.backward()
            if i == 0:
                logger.debug("batch 0: backward done")
            # Compute gradient norm before the optimizer step clears the graph.
            # clip_grad_norm_ with inf max_norm returns the norm without clipping.
            total_grad_norm += torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float("inf")
            ).item()
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

        mean_grad_norm = total_grad_norm / len(self.train_dataloader)
        EpochLogger.update("gradient_norm", mean_grad_norm)
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
        global _interrupted, _last_interrupt_time, _original_sigint
        _interrupted = False
        _last_interrupt_time = 0.0
        _original_sigint = signal.signal(signal.SIGINT, _handle_sigint)

        try:
            return self._train(
                verbose=verbose,
                start_epoch=start_epoch,
                eval_routes_at_end=eval_routes_at_end,
                trial_number=trial_number,
                study_name=study_name,
            )
        finally:
            signal.signal(signal.SIGINT, _original_sigint)

    def _train(self, verbose=True, start_epoch=0, eval_routes_at_end=False,
               trial_number=None, study_name=None):
        eval_cfg = self.config["evaluation"]
        use_async = eval_cfg.get("async_route_eval", False)
        route_predictor = None if use_async else RoutePredictor(
            self.model, self.config, beam_width=eval_cfg["beam_width"]
        )
        async_pool = AsyncRouteEvalPool(
            n_workers=eval_cfg.get("eval_n_workers", 24)
        ) if use_async else None

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

        _es_metric = self.config["train"].get("early_stopping_metric", "valid_action_accuracy")
        _es_minimize = "loss" in _es_metric
        _es_best = float("inf") if _es_minimize else 0.0
        patience = self.config["train"].get("early_stopping_patience", 0)
        epochs_no_improve = 0
        fraction_targets_solved = None
        _extrap_buffer: dict | None = None  # holds extrapolation result until route eval lands

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
            if len(current_run) > 0 and _es_metric in current_run.columns:
                _es_best = current_run[_es_metric].min() if _es_minimize else current_run[_es_metric].max()
            print(f"Restored training history ({len(self.result_df)} epochs). Best {_es_metric} (current run): {_es_best:.6f}")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                self.results_eval = json.load(f)

        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        train_start_time = time.time()

        EpochLogger.configure(
            logging_config=self.config.get("logging", {}),
            jsonl_path=progress_path,
            train_start_time=train_start_time,
        )
        EpochLogger.set_persistent(
            study_name=study_name,
            trial_number=trial_number,
            config_hash=config_hash,
            n_epochs=n_epochs,
        )

        print(f"Training epochs {start_epoch} to {n_epochs - 1}.")
        if verbose:
            _hdr_prefix = [f"{'trial':>5}"] if trial_number is not None else []
            _hdr_suffix = ["study"] if study_name is not None else []

            def _print_header():
                print("  ".join(_hdr_prefix + [
                    f"{'epoch':>5}", f"{'t_loss':>7}", f"{'t_acc':>7}", f"{'t_racc':>7}",
                    f"{'v_loss':>7}", f"{'v_acc':>7}", f"{'v_racc':>7}", f"{'s/ep':>6}", f"{'note':<4}",
                ] + _hdr_suffix), flush=True)

            _print_header()

        logger.debug("entering epoch loop: start=%d end=%d patience=%d _es_metric=%s _es_best=%.6f", start_epoch, n_epochs - 1, patience, _es_metric, _es_best)
        epoch = start_epoch - 1  # defined even when the loop body never executes
        for epoch in range(start_epoch, n_epochs):
            epoch_start = time.time()
            logger.debug("epoch %d start", epoch)

            EpochLogger.begin_epoch(epoch)
            logger.debug("epoch %d train_one_epoch ...", epoch)
            train_loss, train_action_accuracy, train_route_accuracy = (
                self.train_one_epoch()
            )
            logger.debug("epoch %d train done: loss=%.5f acc=%.4f", epoch, train_loss, train_action_accuracy)
            training_loss.append(train_loss)
            training_accuracy.append(train_action_accuracy)
            training_route_accuracy.append(train_route_accuracy)

            logger.debug("epoch %d eval ...", epoch)
            valid_loss, valid_action_accuracy, valid_route_accuracy, _, _ = self.eval()
            logger.debug("epoch %d eval done: v_loss=%.5f v_acc=%.4f", epoch, valid_loss, valid_action_accuracy)
            _metric_values = {
                "valid_loss": valid_loss,
                "valid_action_accuracy": valid_action_accuracy,
                "valid_route_accuracy": valid_route_accuracy,
            }
            lr_before_step = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(_metric_values.get(self._lr_metric, valid_loss))
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr < lr_before_step:
                self._n_lr_reductions += 1
            validation_loss.append(valid_loss)
            validation_accuracy.append(valid_action_accuracy)
            validation_route_accuracy.append(valid_route_accuracy)

            seconds_this_epoch = time.time() - epoch_start

            # Always write model.last.pth at the end of every epoch so the
            # final weights are available regardless of which epoch was best.
            last_path = save_folder + "/model.last.pth"
            tmp_fd, tmp_path = tempfile.mkstemp(dir=save_folder, suffix=".pth.tmp")
            try:
                os.close(tmp_fd)
                torch.save(self.model.state_dict(), tmp_path)
                os.replace(tmp_path, last_path)
            except Exception:
                os.unlink(tmp_path)
                raise

            # When this epoch is the new best, copy model.last.pth → model.pth
            # atomically — avoids serialising state_dict a second time.
            _es_val = _metric_values.get(_es_metric, valid_action_accuracy)
            is_best = (_es_val < _es_best) if _es_minimize else (_es_val > _es_best)
            if is_best:
                model_path = save_folder + "/model.pth"
                _es_best = _es_val
                tmp_fd2, tmp_path2 = tempfile.mkstemp(dir=save_folder, suffix=".pth.tmp")
                try:
                    os.close(tmp_fd2)
                    shutil.copyfile(last_path, tmp_path2)
                    os.replace(tmp_path2, model_path)
                except Exception:
                    os.unlink(tmp_path2)
                    raise
                epochs_no_improve = 0
                note = "*"
            else:
                epochs_no_improve += 1
                note = ""

            if valid_route_accuracy > self._best_valid_route_accuracy:
                self._best_valid_route_accuracy = valid_route_accuracy
                bestroutes_path = save_folder + "/model.bestroutes.pth"
                tmp_fd3, tmp_path3 = tempfile.mkstemp(dir=save_folder, suffix=".pth.tmp")
                try:
                    os.close(tmp_fd3)
                    shutil.copyfile(last_path, tmp_path3)
                    os.replace(tmp_path3, bestroutes_path)
                except Exception:
                    os.unlink(tmp_path3)
                    raise

            eval_routes_frequency = eval_cfg["eval_routes_frequency"]
            eval_due = (epoch % eval_routes_frequency == 0 and epoch > 0) or epoch == n_epochs - 1

            # --- Async path ---------------------------------------------------
            if use_async:
                # Collect any finished extrapolation result into the buffer so it
                # can be attached to the route-eval entry when that arrives.
                extrap_check = async_pool.collect_extrapolation_if_ready()
                if extrap_check is not None:
                    _extrap_buffer = extrap_check

                # Collect route eval; merge buffered extrapolation if epochs match.
                async_result = async_pool.collect_if_ready()
                if async_result is not None:
                    pred_routes = async_result["routes"]
                    entry: dict = {"epoch": async_result["eval_epoch"], "result": pred_routes}
                    if (
                        _extrap_buffer is not None
                        and _extrap_buffer["eval_epoch"] == async_result["eval_epoch"]
                    ):
                        entry["extrapolation"] = _extrap_buffer["metrics"]
                        _extrap_buffer = None
                    self.results_eval.append(entry)
                    fraction_targets_solved = async_result["fraction_targets_solved"]
                    fraction_valid_routes = async_result["fraction_valid_routes"]
                    print(
                        f"  [async] solved={fraction_targets_solved:.5f}"
                        f"  valid={fraction_valid_routes:.5f}"
                        f"  (lagged from epoch {async_result['eval_epoch']})",
                        flush=True,
                    )

                if eval_due:
                    async_pool.submit(self.model, self.config, self.valid_dataloader, epoch)
                    async_pool.submit_extrapolation(
                        metric_histories={
                            "train_loss":             list(training_loss),
                            "train_action_accuracy":  list(training_accuracy),
                            "train_route_accuracy":   list(training_route_accuracy),
                            "valid_action_accuracy":  list(validation_accuracy),
                            "valid_route_accuracy":   list(validation_route_accuracy),
                        },
                        n_epochs=n_epochs,
                        epoch=epoch,
                    )

            # --- Synchronous path (default) -----------------------------------
            else:
                if eval_due:
                    from .async_eval import _extrapolation_worker
                    eval_start_time = time.time()
                    print(f"Epoch {epoch}: running route evaluation …", flush=True)
                    route_predictor.set_model(self.model)
                    pred_routes = route_predictor.eval_predicted_routes(self.valid_dataloader)
                    extrap = _extrapolation_worker(
                        metric_histories={
                            "train_loss":             list(training_loss),
                            "train_action_accuracy":  list(training_accuracy),
                            "train_route_accuracy":   list(training_route_accuracy),
                            "valid_action_accuracy":  list(validation_accuracy),
                            "valid_route_accuracy":   list(validation_route_accuracy),
                        },
                        n_epochs=n_epochs,
                        eval_epoch=epoch,
                    )
                    self.results_eval.append({
                        "epoch": epoch,
                        "result": pred_routes,
                        "extrapolation": extrap["metrics"],
                    })
                    solved_routes = [r["route_solved"] for r in pred_routes]
                    fraction_targets_solved = sum(solved_routes) / len(solved_routes)
                    valid_routes = [r["valid_route"] for r in pred_routes]
                    fraction_valid_routes = sum(valid_routes) / len(valid_routes)
                    print(
                        f"  solved={fraction_targets_solved:.5f}  valid={fraction_valid_routes:.5f}"
                        f"  ({len(pred_routes)} routes, {(time.time()-eval_start_time)/60:.1f} min)",
                        flush=True,
                    )
                else:
                    fraction_targets_solved = None

            EpochLogger.update_many(
                train_loss=train_loss,
                train_action_accuracy=train_action_accuracy,
                train_route_accuracy=train_route_accuracy,
                valid_loss=valid_loss,
                valid_action_accuracy=valid_action_accuracy,
                valid_route_accuracy=valid_route_accuracy,
                seconds_per_epoch=seconds_this_epoch,
                learning_rate=current_lr,
                is_best=is_best,
                epochs_without_improvement=epochs_no_improve,
                n_lr_reductions=self._n_lr_reductions,
                best_valid_route_accuracy=self._best_valid_route_accuracy,
            )
            # gradient_norm was already set inside train_one_epoch via EpochLogger.update()
            # elapsed_seconds and timestamp are registered providers, evaluated at flush()
            record = EpochLogger.flush()
            self.result_df = pd.concat(
                [self.result_df, pd.DataFrame([record])], ignore_index=True
            )

            if verbose:
                if epoch != start_epoch and (epoch - start_epoch) % 10 == 0:
                    _print_header()
                _row_prefix = []
                if trial_number is not None:
                    _row_prefix.append(f"{trial_number:>5}")
                _row_suffix = [str(study_name)] if study_name is not None else []
                print("  ".join(_row_prefix + [
                    f"{epoch:>5}",
                    f"{train_loss:>7.5f}",
                    f"{train_action_accuracy:>7.5f}",
                    f"{train_route_accuracy:>7.5f}",
                    f"{valid_loss:>7.5f}",
                    f"{valid_action_accuracy:>7.5f}",
                    f"{valid_route_accuracy:>7.5f}",
                    f"{seconds_this_epoch:>6.1f}",
                    f"{note:<4}",
                ] + _row_suffix), flush=True)

            with open(eval_path, "w") as results:
                json.dump(self.results_eval, results)

            logger.debug("epoch %d es_check: epochs_no_improve=%d patience=%d _es_val=%.6f _es_best=%.6f", epoch, epochs_no_improve, patience, _es_val, _es_best)
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping: {_es_metric} has not improved for {patience} consecutive epochs.")
                break

            if _interrupted:
                print(f"[trainer] Stopped after epoch {epoch} (Ctrl-C). Running final route eval before exit.")
                eval_routes_at_end = True
                break

        # Run final route evaluation when explicitly requested (hypertune) or
        # when early stopping skipped the last scheduled eval epoch.
        if eval_routes_at_end or fraction_targets_solved is None:
            if use_async:
                # Collect any in-flight job first; if nothing is running (or
                # eval_routes_at_end forced a fresh run), submit one now and block.
                _final_fresh = not async_pool.is_running()
                if _final_fresh:
                    print("[async_eval] Submitting final blocking route eval …", flush=True)
                    async_pool.submit(self.model, self.config, self.valid_dataloader, epoch)
                    async_pool.submit_extrapolation(
                        metric_histories={
                            "train_loss":             list(training_loss),
                            "train_action_accuracy":  list(training_accuracy),
                            "train_route_accuracy":   list(training_route_accuracy),
                            "valid_action_accuracy":  list(validation_accuracy),
                            "valid_route_accuracy":   list(validation_route_accuracy),
                        },
                        n_epochs=n_epochs,
                        epoch=epoch,
                    )
                else:
                    print("[async_eval] Waiting for in-flight eval to finish …", flush=True)
                async_result = async_pool.collect_blocking(timeout=900)
                extrap_final = async_pool.collect_extrapolation_blocking(timeout=60)
                async_pool.shutdown()
                if async_result is not None:
                    pred_routes = async_result["routes"]
                    entry = {"epoch": epoch, "result": pred_routes}
                    if extrap_final is not None:
                        entry["extrapolation"] = extrap_final["metrics"]
                    elif _extrap_buffer is not None:
                        entry["extrapolation"] = _extrap_buffer["metrics"]
                    self.results_eval.append(entry)
                    fraction_targets_solved = async_result["fraction_targets_solved"]
                    print(
                        f"Final route eval: {fraction_targets_solved:.4f} fraction solved "
                        f"({async_result['n_routes']} routes, "
                        f"{async_result['eval_elapsed_seconds'] / 60:.1f} min on CPU)"
                    )
                else:
                    fraction_targets_solved = 0.0
            else:
                from .async_eval import _extrapolation_worker
                print("Running final route evaluation …")
                eval_start_time = time.time()
                route_predictor.set_model(self.model)
                pred_routes = route_predictor.eval_predicted_routes(self.valid_dataloader)
                extrap = _extrapolation_worker(
                    metric_histories={
                        "train_loss":             list(training_loss),
                        "train_action_accuracy":  list(training_accuracy),
                        "train_route_accuracy":   list(training_route_accuracy),
                        "valid_action_accuracy":  list(validation_accuracy),
                        "valid_route_accuracy":   list(validation_route_accuracy),
                    },
                    n_epochs=n_epochs,
                    eval_epoch=epoch,
                )
                solved_routes = [r["route_solved"] for r in pred_routes]
                fraction_targets_solved = sum(solved_routes) / len(solved_routes) if solved_routes else 0.0
                self.results_eval.append({
                    "epoch": epoch,
                    "result": pred_routes,
                    "extrapolation": extrap["metrics"],
                })
                print(
                    f"Final route eval: {fraction_targets_solved:.4f} fraction solved "
                    f"({sum(solved_routes)}/{len(solved_routes)}) "
                    f"in {(time.time() - eval_start_time) / 60:.1f} min"
                )
            with open(eval_path, "w") as results:
                json.dump(self.results_eval, results)

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
