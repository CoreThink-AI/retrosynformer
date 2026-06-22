"""Pydantic v2 request/response schemas for the RetroSynFormer inference API."""

from typing import Literal, Optional

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
    max_depth: Optional[int] = Field(default=None, ge=1, le=20, description="Max retrosynthesis depth (overrides model config; default is config value, typically 6)")
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


# ---------------------------------------------------------------------------
# Synthesis-routes-generator compatible schemas (POST /retrosynthesis)
# ---------------------------------------------------------------------------

class RetrosynthesisRequest(BaseModel):
    """Input for ``POST /retrosynthesis`` — matches the synthesis-routes-generator API.

    ``molecule_name`` is accepted for API compatibility but is not used by
    RetroSynFormer (the model operates on SMILES, not names).

    >>> r = RetrosynthesisRequest(smiles="CC(=O)O")
    >>> r.max_steps
    6
    >>> r.max_routes
    5
    """

    smiles: str = Field(..., description="Target molecule SMILES", examples=["CC(=O)Oc1ccccc1C(=O)O"])
    max_steps: int = Field(default=6, ge=1, le=20, description="Max retrosynthesis depth")
    max_routes: int = Field(default=5, ge=1, le=50, description="Max routes to return")
    molecule_name: Optional[str] = Field(
        default=None,
        description="Common or IUPAC name (accepted for API compatibility; not used by this model)",
    )


class RouteStepResponse(BaseModel):
    """One reaction step within a route — mirrors the synthesis-routes-generator field set."""

    step: int
    target: str
    reaction_id: str
    reactants: list[str]
    reagents: list[str]
    co_products: list[str] = []
    yield_pct: Optional[float]
    confidence: float
    source: str
    dataset_name: str
    doi: str


class LeafMoleculeResponse(BaseModel):
    smiles: str
    purchasable: bool


class RouteResponse(BaseModel):
    """A single retrosynthesis route in synthesis-routes-generator format.

    ``model`` is set to ``"model1"`` so the webapp can distinguish RetroSynFormer
    routes from the synthesis-routes-generator's own routes (``"model2"``).
    """

    model: str = "model1"
    steps: list[RouteStepResponse]
    score: float
    depth: int
    leaf_molecules: list[LeafMoleculeResponse]
    all_leaves_purchasable: bool


class RetrosynthesisResponse(BaseModel):
    """Response from ``POST /retrosynthesis`` — mirrors the synthesis-routes-generator contract.

    RetroSynFormer routes are placed in ``ai_routes`` (the model predicts
    disconnections; it does not query literature databases).
    ``literature_routes`` is always empty.
    """

    target_smiles: str
    canonical_smiles: str
    literature_routes: list[RouteResponse] = []
    ai_routes: list[RouteResponse]


class HealthResponse(BaseModel):
    """Response from ``GET /health``.

    >>> h = HealthResponse(
    ...     status="ok",
    ...     model_loaded=True,
    ...     device="cpu",
    ...     beam_width_default=10,
    ...     action_dim=589,
    ...     model_path="gs://bucket/retrosynformer/v3/model.pth.gz",
    ...     model_released_at="2026-06-21T12:00:00",
    ...     model_sha256_hash="abc123",
    ...     model_file_size_bytes=1_200_000_000,
    ... )
    >>> h.status
    'ok'
    >>> h.model_path
    'gs://bucket/retrosynformer/v3/model.pth.gz'
    >>> h.model_file_size_bytes
    1200000000
    """

    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str
    beam_width_default: int
    action_dim: int
    retrosynformer_version: Optional[str] = None
    model_path: Optional[str] = None
    model_released_at: Optional[str] = None
    model_sha256_hash: Optional[str] = None
    model_file_size_bytes: Optional[int] = None
