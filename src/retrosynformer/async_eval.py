"""Async CPU-side route evaluation during GPU training.

Usage in the training loop::

    pool = AsyncRouteEvalPool(n_workers=config["evaluation"].get("eval_n_workers", 24))

    # Non-blocking submit at eval epoch — GPU training continues immediately.
    pool.submit(model, config, valid_dataloader, epoch)

    # Check at the top of every epoch; returns result dict or None.
    result = pool.collect_if_ready()
    if result is not None:
        fraction_targets_solved = result["fraction_targets_solved"]

    # At the very end of training, block until any pending job finishes.
    result = pool.collect_blocking(timeout=900)
    pool.shutdown()

Configuration (``evaluation:`` section of ``model.config.yaml``)::

    evaluation:
      async_route_eval: true   # opt-in; false/absent → synchronous (default)
      eval_n_workers: 24       # CPU cores for the worker pool (default 24)
"""
from __future__ import annotations

import io
import logging
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Metric extrapolation constants
# ---------------------------------------------------------------------------

# Metrics extracted from train_progress.jsonl for extrapolation.
_EXTRAP_METRICS = (
    "train_loss",
    "train_action_accuracy",
    "train_route_accuracy",
    "valid_action_accuracy",
    "valid_route_accuracy",
)

# Horizon multipliers applied to n_epochs when extrapolating.
_EXTRAP_HORIZONS: dict[str, float] = {
    "1.0x": 1.0,
    "1.1x": 1.1,
    "1.5x": 1.5,
    "2.0x": 2.0,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker — must live here (not in a class/closure) so
# multiprocessing can pickle it by qualified name.
# ---------------------------------------------------------------------------

def _eval_worker_chunk(
    state_dict_bytes: bytes,
    config: dict,
    compounds: list[dict],
) -> list[dict]:
    """Run beam-search route prediction for *compounds* on CPU.

    Parameters
    ----------
    state_dict_bytes:
        ``torch.save(model.state_dict(), buf); buf.getvalue()``
    config:
        Full training config dict.
    compounds:
        List of per-compound dicts::

            {"smiles": str, "target_tree": list,
             "actions": list, "rewards": list}

    Returns
    -------
    list[dict]
        Route-result dicts (same schema as ``RoutePredictor.eval_predicted_routes``).
    """
    import io
    import time

    import torch
    from transformers import DecisionTransformerConfig, DecisionTransformerModel

    from retrosynformer.inference import RoutePredictor

    # --- Reconstruct model on CPU -------------------------------------------
    device = "cpu"

    dt_config = DecisionTransformerConfig(bos_token_id=None, eos_token_id=None)
    dt_config.act_dim = config["dataset"]["action_dim"]
    dt_config.state_dim = int(
        config["dataset"]["fp_dim"] * config["dataset"]["n_in_state"]
    )
    dt_config.max_ep_len = config["model"]["max_ep_len"]
    dt_config.hidden_size = config["model"]["hidden_size"]
    dt_config.n_layer = config["model"]["n_layers"]
    dt_config.n_head = config["model"]["n_heads"]
    dt_config.activation_function = config["model"]["activation_function"]
    dt_config.action_tanh = config["model"]["action_tanh"]
    dt_config.attn_pdrop = config["model"]["attn_pdrop"]
    dt_config.embd_pdrop = config["model"]["embd_pdrop"]
    dt_config.resid_pdrop = config["model"]["resid_pdrop"]

    if config["model"].get("use_structured_dropout", False):
        from retrosynformer.structured_dropout import StructuredDropoutDecisionTransformer
        fp_dim = config["dataset"]["fp_dim"]
        bottleneck = config["model"].get("structured_dropout_bottleneck", 128)
        rate = config["model"].get("structured_dropout_rate", 1.0)
        model = StructuredDropoutDecisionTransformer(dt_config, fp_dim, bottleneck, rate=rate)
    else:
        model = DecisionTransformerModel(dt_config)

    model.load_state_dict(
        torch.load(io.BytesIO(state_dict_bytes), map_location=device)
    )
    model.to(device)
    model.eval()

    # Force CPU inside this subprocess before RoutePredictor reads get_device().
    import retrosynformer.utils.utils as _utils
    _utils.get_device = lambda: device

    predictor = RoutePredictor(model, config, beam_width=config["evaluation"]["beam_width"])
    beam_width = config["evaluation"]["beam_width"]

    # --- Beam search per compound -------------------------------------------
    routes: list[dict] = []
    for compound in compounds:
        smiles = compound["smiles"]
        target_tree = compound["target_tree"]
        actions_list = compound["actions"]
        rewards_list = compound["rewards"]

        t0 = time.time()
        route: dict[str, Any] = {
            "target": smiles,
            "target_tree": target_tree,
            "pred_tree": None,
        }

        try:
            best_beam = predictor.predict_route(smiles, beam_width=beam_width)
        except Exception as exc:
            logger.error("predict_route failed for %s: %s", smiles, exc)
            best_beam = None

        route["time"] = time.time() - t0

        if best_beam:
            route["route_solved"] = best_beam.route_solved
            route["n_reactions"] = len(best_beam.reaction_list)
            route["leafs"] = best_beam.env.leafs
            route["n_branchings"] = best_beam.env.number_of_branchings
            route["n_dead_ends"] = best_beam.env.dead_ends
            route["predicted_reaction_lists"] = best_beam.reaction_list
            route["predicted_action_list"] = [a.item() for a in best_beam.predicted_actions]
            route["target_action_list"] = [
                torch.argmax(torch.tensor(a), dim=0).tolist() for a in actions_list
            ]
            route["predicted_rewards"] = best_beam.env.rewards
            route["target_rewards"] = rewards_list
            route["trajectory_prob"] = best_beam.trajectory_prob

            try:
                from rxnutils.routes import base
                from retrosynformer.utils import utils as _u
                pred_tree = _u.list2route(best_beam.reaction_list).reaction_tree
                pred_tree, route_solved = _u.add_in_stock_property_to_trees(
                    pred_tree, predictor.building_blocks
                )
                route["pred_tree"] = pred_tree
                route["route_solved"] = route_solved
                route["TED to target"], most_similar_idx = _u.calculate_ted(
                    predictor.calculator,
                    base.SynthesisRoute(pred_tree),
                    [base.SynthesisRoute(r) for r in target_tree],
                )
                route["target_tree"] = target_tree[most_similar_idx]
                route["valid_route"] = True
            except Exception as exc:
                logger.error("TED/tree computation failed: %s", exc)
                route["valid_route"] = False
        else:
            route["route_solved"] = False
            route["valid_route"] = False

        routes.append(route)

    return routes


# ---------------------------------------------------------------------------
# Extrapolation worker (runs in a thread — pure numpy, no process needed)
# ---------------------------------------------------------------------------

def _extrapolation_worker(
    metric_histories: dict[str, list[float]],
    n_epochs: int,
    eval_epoch: int,
) -> dict:
    """Extrapolate each metric to multiple time horizons.

    Parameters
    ----------
    metric_histories:
        ``{metric_name: [value_epoch0, value_epoch1, …]}`` — a snapshot of
        the training lists at the moment of submission.
    n_epochs:
        Planned total epochs (1.0× horizon target = ``n_epochs - 1``).
    eval_epoch:
        The epoch at which the snapshot was taken (stored in the result for
        traceability).

    Returns
    -------
    dict
        ``{"eval_epoch": int, "metrics": {name: {"n_observed": int,
        "horizons": {"1.0x": {"target_epoch", "estimate", "se"}, …}}}}``
    """
    from retrosynformer.extrapolate import extrapolate_objective

    metrics_result: dict[str, Any] = {}
    for metric, values in metric_histories.items():
        if not values:
            metrics_result[metric] = {"n_observed": 0, "horizons": {}}
            continue
        horizons: dict[str, dict] = {}
        for label, factor in _EXTRAP_HORIZONS.items():
            target_n = max(int(round(factor * n_epochs)), len(values) + 1)
            result = extrapolate_objective(values, n_epochs=target_n)
            if result is None:
                horizons[label] = {
                    "target_epoch": target_n - 1,
                    "estimate": None,
                    "se": None,
                }
            else:
                horizons[label] = {
                    "target_epoch": result["target_epoch"],
                    "estimate": round(result["estimate"], 6),
                    "se": round(result["se"], 6),
                }
        metrics_result[metric] = {"n_observed": len(values), "horizons": horizons}

    return {"eval_epoch": eval_epoch, "metrics": metrics_result}


# ---------------------------------------------------------------------------
# Helpers (main-process side)
# ---------------------------------------------------------------------------

def _collect_compounds_from_dataloader(
    dataloader, n_batches: int | None
) -> list[dict]:
    """Drain up to *n_batches* from the dataloader; return serialisable compound dicts."""
    compounds: list[dict] = []
    for batch_no, data in enumerate(dataloader):
        if n_batches is not None and batch_no >= n_batches:
            break
        (
            (states, actions, rewards, timesteps, attention_mask),
            action_labels,
            target_routes,
        ) = data
        for j in range(len(states)):
            compounds.append({
                "smiles": target_routes[j][0]["smiles"],
                "target_tree": target_routes[j],
                "actions": actions[j].tolist(),
                "rewards": rewards[j].tolist(),
            })
    return compounds


def _summarise(routes: list[dict], eval_epoch: int, elapsed: float) -> dict:
    solved = [r.get("route_solved", False) for r in routes]
    valid = [r.get("valid_route", False) for r in routes]
    n = len(routes)
    return {
        "eval_epoch": eval_epoch,
        "eval_elapsed_seconds": elapsed,
        "n_routes": n,
        "fraction_targets_solved": sum(solved) / n if n else 0.0,
        "fraction_valid_routes": sum(valid) / n if n else 0.0,
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AsyncRouteEvalPool:
    """Non-blocking route-eval runner backed by a CPU ``ProcessPoolExecutor``.

    Compounds are sharded across *n_workers* processes; all chunks run
    concurrently.  The main training loop calls ``collect_if_ready()`` once
    per epoch — it returns ``None`` while chunks are still running and the
    merged result dict the first time all chunks are done.

    Only one eval job runs at a time.  If ``submit()`` is called while a
    previous job is still running, the call is a no-op and returns ``False``.
    """

    def __init__(self, n_workers: int = 24) -> None:
        self._n_workers = n_workers
        # Route-eval pool (process-based — heavy beam search)
        self._executor: ProcessPoolExecutor | None = None
        self._chunk_futures: list[Future] = []
        self._pending_epoch: int | None = None
        self._submit_time: float = 0.0
        # Extrapolation pool (thread-based — fast numpy, no process overhead needed)
        self._extrap_executor: ThreadPoolExecutor | None = None
        self._extrap_future: Future | None = None
        self._extrap_epoch: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_extrap_executor(self) -> ThreadPoolExecutor:
        if self._extrap_executor is None:
            self._extrap_executor = ThreadPoolExecutor(max_workers=1)
        return self._extrap_executor

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._n_workers,
                mp_context=__import__("multiprocessing").get_context("spawn"),
            )
        return self._executor

    def _serialize_weights(self, model: torch.nn.Module) -> bytes:
        buf = io.BytesIO()
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, buf)
        return buf.getvalue()

    def _all_done(self) -> bool:
        return bool(self._chunk_futures) and all(f.done() for f in self._chunk_futures)

    def _merge(self) -> dict:
        routes: list[dict] = []
        for f in self._chunk_futures:
            routes.extend(f.result())
        result = _summarise(routes, self._pending_epoch, time.time() - self._submit_time)
        self._chunk_futures = []
        self._pending_epoch = None
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(
        self,
        model: torch.nn.Module,
        config: dict,
        dataloader,
        epoch: int,
    ) -> bool:
        """Snapshot model weights and dispatch route eval to the CPU pool.

        The dataloader is drained in the *main process* (it isn't picklable),
        then the serialised compound list and weight bytes are sent to workers.

        Returns ``True`` if submitted, ``False`` if a previous job is still running.
        """
        if self._chunk_futures and not self._all_done():
            logger.info(
                "[async_eval] Epoch %d: previous eval (epoch %d) still running — skipping.",
                epoch, self._pending_epoch,
            )
            return False

        n_batches = config["evaluation"].get("eval_n_batches")
        compounds = _collect_compounds_from_dataloader(dataloader, n_batches)
        if not compounds:
            logger.warning("[async_eval] No compounds extracted from dataloader.")
            return False

        state_dict_bytes = self._serialize_weights(model)

        # Shard compounds evenly across workers.
        n = self._n_workers
        chunk_size = max(1, (len(compounds) + n - 1) // n)
        chunks = [compounds[i:i + chunk_size] for i in range(0, len(compounds), chunk_size)]

        executor = self._ensure_executor()
        self._chunk_futures = [
            executor.submit(_eval_worker_chunk, state_dict_bytes, config, chunk)
            for chunk in chunks
        ]
        self._pending_epoch = epoch
        self._submit_time = time.time()

        print(
            f"[async_eval] Epoch {epoch}: {len(compounds)} compounds dispatched "
            f"to {len(chunks)} CPU workers — GPU training continues.",
            flush=True,
        )
        return True

    def collect_if_ready(self) -> dict | None:
        """Return merged result if all worker chunks are done, else ``None``."""
        if not self._all_done():
            return None
        result = self._merge()
        fts = result["fraction_targets_solved"]
        elapsed = result["eval_elapsed_seconds"]
        print(
            f"[async_eval] Collected eval (epoch {result['eval_epoch']}): "
            f"{fts:.4f} solved ({result['n_routes']} routes, {elapsed / 60:.1f} min).",
            flush=True,
        )
        return result

    def is_running(self) -> bool:
        """True if a job has been submitted and not yet collected."""
        return bool(self._chunk_futures) and not self._all_done()

    def collect_blocking(self, timeout: float = 900.0) -> dict | None:
        """Block up to *timeout* seconds for the running job and return its result."""
        if not self._chunk_futures:
            return None
        deadline = time.time() + timeout
        while not self._all_done():
            if time.time() > deadline:
                logger.warning("[async_eval] collect_blocking: timed out after %.0fs.", timeout)
                return None
            time.sleep(2.0)
        return self.collect_if_ready()

    # ------------------------------------------------------------------
    # Extrapolation interface
    # ------------------------------------------------------------------

    def submit_extrapolation(
        self,
        metric_histories: dict[str, list[float]],
        n_epochs: int,
        epoch: int,
    ) -> bool:
        """Dispatch metric extrapolation to the thread pool (non-blocking).

        Parameters
        ----------
        metric_histories:
            Snapshot of training metric lists, keyed by metric name.
            Pass ``list(training_loss)`` etc. to avoid mutation by the main
            training loop before the thread reads the data.
        n_epochs:
            Planned total epochs (1.0× horizon = ``n_epochs - 1``).
        epoch:
            Current epoch (stored in the result for traceability).

        Returns
        -------
        bool
            ``True`` if submitted; ``False`` if the previous job is still running.
        """
        if self._extrap_future is not None and not self._extrap_future.done():
            logger.info(
                "[async_eval] Extrapolation epoch %d: previous (epoch %d) still running — skipping.",
                epoch, self._extrap_epoch,
            )
            return False
        executor = self._ensure_extrap_executor()
        self._extrap_future = executor.submit(
            _extrapolation_worker, metric_histories, n_epochs, epoch
        )
        self._extrap_epoch = epoch
        return True

    def collect_extrapolation_if_ready(self) -> dict | None:
        """Return extrapolation result if the thread has finished, else ``None``."""
        if self._extrap_future is None or not self._extrap_future.done():
            return None
        result = self._extrap_future.result()
        self._extrap_future = None
        self._extrap_epoch = None
        return result

    def collect_extrapolation_blocking(self, timeout: float = 60.0) -> dict | None:
        """Block up to *timeout* seconds for the extrapolation thread."""
        if self._extrap_future is None:
            return None
        try:
            result = self._extrap_future.result(timeout=timeout)
            self._extrap_future = None
            self._extrap_epoch = None
            return result
        except Exception as exc:
            logger.warning("[async_eval] Extrapolation collect failed: %s", exc)
            return None

    def shutdown(self) -> None:
        """Shut down the worker pool. Safe to call even if no job was submitted."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._extrap_executor is not None:
            self._extrap_executor.shutdown(wait=True)  # fast — always finishes quickly
            self._extrap_executor = None
