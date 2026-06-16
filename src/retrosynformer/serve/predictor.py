"""ModelPredictor: loads the model once at startup and exposes predict_sync for thread-pool use."""

import hashlib
import threading
import time

import torch

from retrosynformer.inference import RoutePredictor
from retrosynformer.runner import init_model, read_config


def _beam_to_route_dict(beam) -> dict:
    """Convert a single terminal Beam namedtuple to a plain dict matching Route schema.

    Zips reaction_list, predicted_actions, and env.rewards into per-step dicts.
    All three lists have the same length for any beam produced by predict_all_routes.

    >>> # Lightweight test using a mock beam-like object.
    >>> from types import SimpleNamespace
    >>> env = SimpleNamespace(leafs=["CC"], dead_ends=[], rewards=[0.8])
    >>> beam = SimpleNamespace(
    ...     route_solved=True,
    ...     trajectory_prob=0.9,
    ...     reaction_list=["C>>CC"],
    ...     predicted_actions=[42],
    ...     env=env,
    ... )
    >>> d = _beam_to_route_dict(beam)
    >>> d["route_solved"]
    True
    >>> d["n_steps"]
    1
    >>> d["reactions"][0]["template_index"]
    42
    >>> d["leaf_smiles"]
    ['CC']
    """
    steps = [
        {
            "reaction_smarts": rxn,
            "template_index": int(tidx),
            "reward": float(r),
        }
        for rxn, tidx, r in zip(
            beam.reaction_list,
            beam.predicted_actions,
            beam.env.rewards,
        )
    ]
    return {
        "route_solved": beam.route_solved,
        "trajectory_prob": float(beam.trajectory_prob),
        "n_steps": len(steps),
        "reactions": steps,
        "leaf_smiles": list(beam.env.leafs),
        "dead_ends": list(beam.env.dead_ends),
    }


def _beam_to_rsgen_route(beam) -> dict:
    """Convert a terminal Beam to a dict matching RouteResponse (synthesis-routes-generator format).

    Each entry in ``reaction_list`` is ``"A.B>>C"`` where C is the molecule being
    disconnected (the target of that step) and A, B are the reactants.

    >>> from types import SimpleNamespace
    >>> env = SimpleNamespace(leafs=["CC", "OC"], dead_ends=[])
    >>> beam = SimpleNamespace(
    ...     route_solved=True,
    ...     trajectory_prob=0.75,
    ...     reaction_list=["CC.OC>>CCO"],
    ...     predicted_actions=[5],
    ...     env=env,
    ... )
    >>> d = _beam_to_rsgen_route(beam)
    >>> d["model"]
    'model1'
    >>> d["steps"][0]["target"]
    'CCO'
    >>> d["steps"][0]["reactants"]
    ['CC', 'OC']
    >>> d["all_leaves_purchasable"]
    True
    """
    steps = []
    for i, rxn in enumerate(beam.reaction_list):
        sep = rxn.find(">>")
        if sep >= 0:
            reactants = rxn[:sep].split(".")
            target = rxn[sep + 2:]
        else:
            reactants = []
            target = rxn
        steps.append({
            "step": i + 1,
            "target": target,
            "reaction_id": hashlib.md5(rxn.encode()).hexdigest()[:16],
            "reactants": reactants,
            "reagents": [],
            "co_products": [],
            "yield_pct": None,
            "confidence": round(float(beam.trajectory_prob), 4),
            "source": "retrosynformer",
            "dataset_name": "PaRoutes",
            "doi": "10.26434/chemrxiv-2025-kd6gb",
        })

    leaf_mols = (
        [{"smiles": s, "purchasable": True} for s in beam.env.leafs]
        + [{"smiles": s, "purchasable": False} for s in beam.env.dead_ends]
    )
    return {
        "model": "model1",
        "steps": steps,
        "score": round(float(beam.trajectory_prob), 4),
        "depth": len(beam.reaction_list),
        "leaf_molecules": leaf_mols,
        "all_leaves_purchasable": beam.route_solved and len(beam.env.dead_ends) == 0,
    }


class ModelPredictor:
    """Wraps RoutePredictor; safe to call from multiple threads (serialised via lock).

    Parameters
    ----------
    config_path:
        Path to a ``config.yaml`` that was used to train the model.
    model_path:
        Path to the ``model.pth`` checkpoint.
    building_blocks_path:
        Override for ``config["context"]["building_blocks"]``.
    templates_path:
        Override for ``config["context"]["templates_path"]``.
    """

    def __init__(
        self,
        config_path: str,
        model_path: str,
        building_blocks_path: str,
        templates_path: str,
    ) -> None:
        config = read_config(config_path)
        config["context"]["building_blocks"] = building_blocks_path
        config["context"]["templates_path"] = templates_path

        model = init_model(config, model_path=model_path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        self._predictor = RoutePredictor(model, config)
        self._config = config
        self.device = device
        # Serialise inference calls — predict_all_routes mutates self.env on the
        # RoutePredictor instance, so concurrent calls from the thread pool must queue.
        self._lock = threading.Lock()

    @property
    def action_dim(self) -> int:
        return int(self._config["dataset"]["action_dim"])

    @property
    def beam_width_default(self) -> int:
        return int(self._config["evaluation"].get("beam_width", 10))

    def predict_sync(
        self,
        smiles: str,
        beam_width: int,
        target_reward: float,
        sort_on: str,
    ) -> dict:
        """Run beam search and return a plain dict matching PredictResponse (minus smiles).

        Intended to be called from ``asyncio.get_event_loop().run_in_executor``
        so the async event loop stays responsive during the synchronous inference.
        """
        t0 = time.perf_counter()
        with self._lock:
            beams = self._predictor.predict_all_routes(smiles, beam_width, target_reward)
        elapsed = time.perf_counter() - t0

        if sort_on == "total_reward":
            beams.sort(key=lambda b: b.total_reward, reverse=True)
        # default sort from predict_all_routes is trajectory_prob desc — keep as-is

        routes = [_beam_to_route_dict(b) for b in beams]
        return {
            "n_routes": len(routes),
            "n_solved": sum(r["route_solved"] for r in routes),
            "routes": routes,
            "elapsed_s": round(elapsed, 3),
        }

    def predict_retrosynthesis_sync(
        self,
        smiles: str,
        max_routes: int,
        max_steps: int,
    ) -> list[dict]:
        """Run beam search and return routes in synthesis-routes-generator RouteResponse format.

        ``max_routes`` is used as the beam width so the search explores at least that
        many parallel hypotheses.  ``max_steps`` overrides the config's ``max_depth``
        for this request only (thread-safe: passed into predict_all_routes, not mutated
        on self).

        Returns a list of route dicts (to be placed in ``ai_routes``).
        """
        with self._lock:
            beams = self._predictor.predict_all_routes(
                smiles,
                beam_width=max_routes,
                max_depth=max_steps,
            )
        return [_beam_to_rsgen_route(b) for b in beams[:max_routes]]
