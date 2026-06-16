"""FastAPI application for the RetroSynFormer inference service."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from rdkit import Chem

from .predictor import ModelPredictor
from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RetrosynthesisRequest,
    RetrosynthesisResponse,
    RouteResponse,
)

# Module-level state populated during lifespan startup.
_state: dict = {}

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


def _verify_key(key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.environ.get("API_KEY", "")
    if not expected or key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["sem"] = asyncio.Semaphore(1)
    _state["predictor"] = ModelPredictor(
        config_path=os.environ["MODEL_CONFIG_PATH"],
        model_path=os.environ["MODEL_WEIGHTS_PATH"],
        building_blocks_path=os.environ["BUILDING_BLOCKS_PATH"],
        templates_path=os.environ["TEMPLATES_PATH"],
    )
    yield
    _state.clear()


app = FastAPI(
    title="RetroSynFormer Inference API",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + readiness check.  Returns 200 even while model is loading."""
    predictor: ModelPredictor | None = _state.get("predictor")
    return HealthResponse(
        status="ok" if predictor else "loading",
        model_loaded=predictor is not None,
        device=predictor.device if predictor else "unknown",
        beam_width_default=predictor.beam_width_default if predictor else 0,
        action_dim=predictor.action_dim if predictor else 0,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Security(_verify_key)])
async def predict(req: PredictRequest) -> PredictResponse:
    """Run beam search for a target molecule SMILES and return all terminal routes."""
    predictor: ModelPredictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if Chem.MolFromSmiles(req.smiles) is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")

    loop = asyncio.get_event_loop()
    sem: asyncio.Semaphore = _state["sem"]
    async with sem:
        result = await loop.run_in_executor(
            None,
            predictor.predict_sync,
            req.smiles,
            req.beam_width,
            req.target_reward,
            req.sort_on,
        )

    return PredictResponse(smiles=req.smiles, **result)


@app.post("/retrosynthesis", response_model=RetrosynthesisResponse, dependencies=[Security(_verify_key)])
async def retrosynthesis(req: RetrosynthesisRequest) -> RetrosynthesisResponse:
    """Retrosynthesis endpoint with the same request/response contract as synthesis-routes-generator.

    Routes are placed in ``ai_routes``; ``literature_routes`` is always empty
    (RetroSynFormer predicts disconnections from a trained model, not a literature DB).
    The ``model`` field in each route is ``"model1"`` so the webapp can distinguish
    these from synthesis-routes-generator routes (``"model2"``).
    """
    predictor: ModelPredictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")
    canon = Chem.MolToSmiles(mol)

    loop = asyncio.get_event_loop()
    sem: asyncio.Semaphore = _state["sem"]
    async with sem:
        route_dicts = await loop.run_in_executor(
            None,
            predictor.predict_retrosynthesis_sync,
            req.smiles,
            req.max_routes,
            req.max_steps,
        )

    ai_routes = [RouteResponse(**d) for d in route_dicts]
    return RetrosynthesisResponse(
        target_smiles=req.smiles,
        canonical_smiles=canon,
        literature_routes=[],
        ai_routes=ai_routes,
    )


def run() -> None:
    """Entry point used by the ``rs-serve`` CLI command."""
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "retrosynformer.serve.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )
