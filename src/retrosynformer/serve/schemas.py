"""Pydantic v2 request/response schemas for the RetroSynFormer inference API."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Body accepted by ``POST /predict``.

    >>> r = PredictRequest(smiles="CC(=O)O")
    >>> r.beam_width
    10
    >>> r.target_reward
    0.5
    >>> r.sort_on
    'trajectory_prob'

    Beam width is capped at 50:

    >>> PredictRequest(smiles="C", beam_width=0)
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: ...
    """

    smiles: str
    beam_width: int = Field(default=10, ge=1, le=50)
    target_reward: float = Field(default=0.5, ge=0.0, le=1.0)
    sort_on: Literal["trajectory_prob", "total_reward"] = "trajectory_prob"


class ReactionStep(BaseModel):
    """One step in a predicted retrosynthesis route.

    >>> step = ReactionStep(reaction_smarts="C>>CC", template_index=7, reward=0.8)
    >>> step.template_index
    7
    """

    reaction_smarts: str
    template_index: int
    reward: float


class Route(BaseModel):
    """A single retrosynthesis route (one beam from beam search).

    >>> from retrosynformer.serve.schemas import Route, ReactionStep
    >>> r = Route(
    ...     route_solved=True,
    ...     trajectory_prob=0.42,
    ...     n_steps=2,
    ...     reactions=[ReactionStep(reaction_smarts="A>>B", template_index=0, reward=1.0)],
    ...     leaf_smiles=["CC", "OC"],
    ...     dead_ends=[],
    ... )
    >>> r.route_solved
    True
    >>> r.n_steps
    2
    """

    route_solved: bool
    trajectory_prob: float
    n_steps: int
    reactions: list[ReactionStep]
    leaf_smiles: list[str]
    dead_ends: list[str]


class PredictResponse(BaseModel):
    """Response from ``POST /predict``.

    >>> from retrosynformer.serve.schemas import PredictResponse, Route
    >>> resp = PredictResponse(
    ...     smiles="C",
    ...     n_routes=0,
    ...     n_solved=0,
    ...     routes=[],
    ...     elapsed_s=0.1,
    ... )
    >>> resp.n_solved
    0
    """

    smiles: str
    n_routes: int
    n_solved: int
    routes: list[Route]
    elapsed_s: float


class HealthResponse(BaseModel):
    """Response from ``GET /health``.

    >>> h = HealthResponse(
    ...     status="ok",
    ...     model_loaded=True,
    ...     device="cpu",
    ...     beam_width_default=10,
    ...     action_dim=589,
    ... )
    >>> h.status
    'ok'
    """

    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str
    beam_width_default: int
    action_dim: int
