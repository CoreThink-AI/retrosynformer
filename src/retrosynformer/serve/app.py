"""FastAPI application for the RetroSynFormer inference service."""

import asyncio
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    _RETROSYNFORMER_VERSION: str | None = _pkg_version("retrosynformer")
except PackageNotFoundError:
    _RETROSYNFORMER_VERSION = None

from fastapi import FastAPI, HTTPException, Request, Security, status
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

# ---------------------------------------------------------------------------
# In-memory LRU response cache keyed on (canonical_smiles, max_routes, max_steps).
# Bypasses the inference semaphore entirely for cache hits.
# ---------------------------------------------------------------------------
_ROUTE_CACHE_MAX = 256
_route_cache: OrderedDict[tuple, list] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: tuple) -> list | None:
    with _cache_lock:
        val = _route_cache.get(key)
        if val is not None:
            _route_cache.move_to_end(key)
        return val


def _cache_put(key: tuple, value: list) -> None:
    with _cache_lock:
        if key in _route_cache:
            _route_cache.move_to_end(key)
        else:
            if len(_route_cache) >= _ROUTE_CACHE_MAX:
                _route_cache.popitem(last=False)
            _route_cache[key] = value

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
    with _cache_lock:
        cache_size = len(_route_cache)
    return HealthResponse(
        status="ok" if predictor else "loading",
        model_loaded=predictor is not None,
        device=predictor.device if predictor else "unknown",
        beam_width_default=predictor.beam_width_default if predictor else 0,
        action_dim=predictor.action_dim if predictor else 0,
        retrosynformer_version=_RETROSYNFORMER_VERSION,
        model_path=predictor.model_path if predictor else None,
        model_released_at=predictor.model_released_at if predictor else None,
        model_sha256_hash=predictor.model_sha256_hash if predictor else None,
        model_file_size_bytes=predictor.model_file_size_bytes if predictor else None,
        route_cache_size=cache_size,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Security(_verify_key)])
async def predict(request: Request, req: PredictRequest) -> PredictResponse:
    """Run beam search for a target molecule SMILES and return all terminal routes."""
    predictor: ModelPredictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if Chem.MolFromSmiles(req.smiles) is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")

    loop = asyncio.get_event_loop()
    sem: asyncio.Semaphore = _state["sem"]
    async with sem:
        future = loop.run_in_executor(
            None,
            predictor.predict_sync,
            req.smiles,
            req.beam_width,
            req.target_reward,
            req.sort_on,
            req.max_depth,
        )
        while not future.done():
            if await request.is_disconnected():
                # Release the semaphore immediately so the next request is not
                # blocked.  The executor thread continues on the GPU until it
                # finishes naturally, but it no longer holds the lock.
                raise HTTPException(status_code=499, detail="Client disconnected")
            await asyncio.sleep(10)
        result = future.result()

    return PredictResponse(smiles=req.smiles, **result)


@app.post("/retrosynthesis", response_model=RetrosynthesisResponse, dependencies=[Security(_verify_key)])
async def retrosynthesis(request: Request, req: RetrosynthesisRequest) -> RetrosynthesisResponse:
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

    cache_key = (canon, req.max_routes, req.max_steps)
    cached = _cache_get(cache_key)
    if cached is not None:
        return RetrosynthesisResponse(
            target_smiles=req.smiles,
            canonical_smiles=canon,
            literature_routes=[],
            ai_routes=[RouteResponse(**d) for d in cached],
        )

    loop = asyncio.get_event_loop()
    sem: asyncio.Semaphore = _state["sem"]
    async with sem:
        future = loop.run_in_executor(
            None,
            predictor.predict_retrosynthesis_sync,
            req.smiles,
            req.max_routes,
            req.max_steps,
        )
        while not future.done():
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")
            await asyncio.sleep(10)
        route_dicts = future.result()

    _cache_put(cache_key, route_dicts)
    return RetrosynthesisResponse(
        target_smiles=req.smiles,
        canonical_smiles=canon,
        literature_routes=[],
        ai_routes=[RouteResponse(**d) for d in route_dicts],
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
