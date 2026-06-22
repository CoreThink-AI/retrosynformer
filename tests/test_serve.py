"""Tests for the RetroSynFormer inference API (serve package).

Covers:
- Schema validation (PredictRequest, Route, PredictResponse, HealthResponse)
- Predictor helper (_beam_to_route_dict)
- FastAPI endpoints: GET /health, POST /predict (auth, SMILES validation, success)

The FastAPI tests mock ModelPredictor so no model files are needed.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from retrosynformer.serve.predictor import _beam_to_route_dict
from retrosynformer.serve.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReactionStep,
    Route,
)

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestPredictRequest:
    def test_defaults(self):
        r = PredictRequest(smiles="CC")
        assert r.beam_width == 10
        assert r.target_reward == 0.5
        assert r.sort_on == "trajectory_prob"

    def test_custom_values(self):
        r = PredictRequest(smiles="c1ccccc1", beam_width=5, target_reward=0.8, sort_on="total_reward")
        assert r.beam_width == 5
        assert r.sort_on == "total_reward"

    def test_beam_width_min_bound(self):
        with pytest.raises(ValidationError):
            PredictRequest(smiles="C", beam_width=0)

    def test_beam_width_max_bound(self):
        with pytest.raises(ValidationError):
            PredictRequest(smiles="C", beam_width=51)

    def test_target_reward_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictRequest(smiles="C", target_reward=1.1)
        with pytest.raises(ValidationError):
            PredictRequest(smiles="C", target_reward=-0.1)

    def test_invalid_sort_on(self):
        with pytest.raises(ValidationError):
            PredictRequest(smiles="C", sort_on="unknown_key")  # type: ignore[arg-type]

    def test_smiles_is_required(self):
        with pytest.raises(ValidationError):
            PredictRequest()  # type: ignore[call-arg]


class TestReactionStep:
    def test_basic(self):
        s = ReactionStep(reaction_smarts="A.B>>C", template_index=3, reward=0.5)
        assert s.template_index == 3
        assert s.reaction_smarts == "A.B>>C"

    def test_negative_reward_allowed(self):
        s = ReactionStep(reaction_smarts="X>>Y", template_index=0, reward=-2.0)
        assert s.reward == -2.0


class TestRoute:
    def _make(self, **kwargs):
        defaults = dict(
            route_solved=True,
            trajectory_prob=0.7,
            n_steps=1,
            reactions=[ReactionStep(reaction_smarts="A>>B", template_index=0, reward=1.0)],
            leaf_smiles=["CC"],
            dead_ends=[],
        )
        defaults.update(kwargs)
        return Route(**defaults)

    def test_solved_route(self):
        r = self._make(route_solved=True)
        assert r.route_solved is True
        assert r.dead_ends == []

    def test_unsolved_route(self):
        r = self._make(route_solved=False, dead_ends=["C#N"])
        assert not r.route_solved
        assert "C#N" in r.dead_ends


class TestPredictResponse:
    def test_empty_routes(self):
        resp = PredictResponse(smiles="C", n_routes=0, n_solved=0, routes=[], elapsed_s=0.01)
        assert resp.n_routes == 0
        assert resp.routes == []


class TestHealthResponse:
    def test_ok(self):
        h = HealthResponse(status="ok", model_loaded=True, device="cpu",
                           beam_width_default=10, action_dim=589)
        assert h.status == "ok"
        assert h.model_loaded is True

    def test_loading(self):
        h = HealthResponse(status="loading", model_loaded=False, device="unknown",
                           beam_width_default=0, action_dim=0)
        assert h.status == "loading"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="ready", model_loaded=True, device="cpu",  # type: ignore[arg-type]
                           beam_width_default=10, action_dim=0)


# ---------------------------------------------------------------------------
# Predictor helper tests
# ---------------------------------------------------------------------------


def _make_beam(
    route_solved=True,
    trajectory_prob=0.9,
    reaction_list=None,
    predicted_actions=None,
    rewards=None,
    leafs=None,
    dead_ends=None,
):
    """Construct a minimal fake Beam namedtuple-like object."""
    env = SimpleNamespace(
        leafs=leafs or ["CC"],
        dead_ends=dead_ends or [],
        rewards=rewards or [0.8],
    )
    return SimpleNamespace(
        route_solved=route_solved,
        trajectory_prob=trajectory_prob,
        reaction_list=reaction_list or ["C>>CC"],
        predicted_actions=predicted_actions or [42],
        env=env,
        total_reward=sum(rewards or [0.8]),
    )


class TestBeamToRouteDict:
    def test_basic_conversion(self):
        beam = _make_beam()
        d = _beam_to_route_dict(beam)
        assert d["route_solved"] is True
        assert d["trajectory_prob"] == 0.9
        assert d["n_steps"] == 1
        assert d["reactions"][0]["template_index"] == 42
        assert d["reactions"][0]["reaction_smarts"] == "C>>CC"
        assert abs(d["reactions"][0]["reward"] - 0.8) < 1e-9
        assert d["leaf_smiles"] == ["CC"]
        assert d["dead_ends"] == []

    def test_multi_step(self):
        beam = _make_beam(
            reaction_list=["A>>B", "B>>C"],
            predicted_actions=[1, 2],
            rewards=[0.5, -1.0],
        )
        d = _beam_to_route_dict(beam)
        assert d["n_steps"] == 2
        assert d["reactions"][0]["template_index"] == 1
        assert d["reactions"][1]["template_index"] == 2

    def test_unsolved_with_dead_ends(self):
        beam = _make_beam(route_solved=False, dead_ends=["C#N"])
        d = _beam_to_route_dict(beam)
        assert not d["route_solved"]
        assert "C#N" in d["dead_ends"]

    def test_template_index_coerced_to_int(self):
        import torch
        beam = _make_beam(predicted_actions=[torch.tensor(7)])
        d = _beam_to_route_dict(beam)
        assert d["reactions"][0]["template_index"] == 7
        assert isinstance(d["reactions"][0]["template_index"], int)


# ---------------------------------------------------------------------------
# FastAPI app tests (mocked predictor)
# ---------------------------------------------------------------------------

_FAKE_PREDICT_RESULT = {
    "n_routes": 2,
    "n_solved": 1,
    "routes": [
        {
            "route_solved": True,
            "trajectory_prob": 0.8,
            "n_steps": 1,
            "reactions": [{"reaction_smarts": "A>>B", "template_index": 0, "reward": 1.0}],
            "leaf_smiles": ["CC"],
            "dead_ends": [],
        },
        {
            "route_solved": False,
            "trajectory_prob": 0.2,
            "n_steps": 1,
            "reactions": [{"reaction_smarts": "X>>Y", "template_index": 1, "reward": -1.0}],
            "leaf_smiles": [],
            "dead_ends": ["C#N"],
        },
    ],
    "elapsed_s": 0.5,
}


@pytest.fixture(scope="module")
def test_client():
    """TestClient with a mocked ModelPredictor — no model files required."""
    import retrosynformer.serve.app as app_module

    mock_pred = MagicMock()
    mock_pred.device = "cpu"
    mock_pred.action_dim = 589
    mock_pred.beam_width_default = 10
    mock_pred.model_path = "gs://fake-bucket/fake/model.pth.gz"
    mock_pred.model_released_at = "2026-01-01T00:00:00"
    mock_pred.model_sha256_hash = "a" * 64
    mock_pred.model_file_size_bytes = 1_234_567_890
    mock_pred.predict_sync.return_value = _FAKE_PREDICT_RESULT

    env_patch = {
        "MODEL_CONFIG_PATH": "/fake/config.yaml",
        "MODEL_WEIGHTS_PATH": "/fake/model.pth",
        "BUILDING_BLOCKS_PATH": "/fake/bb.csv",
        "TEMPLATES_PATH": "/fake/templates.pkl",
        "API_KEY": "test-secret",
    }

    with patch.dict(os.environ, env_patch):
        with patch(
            "retrosynformer.serve.app.ModelPredictor", return_value=mock_pred
        ):
            with TestClient(app_module.app, raise_server_exceptions=True) as client:
                yield client


API_KEY = "test-secret"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, test_client):
        data = resp = test_client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"
        assert data["device"] == "cpu"
        assert data["action_dim"] == 589


class TestPredictEndpoint:
    def test_valid_request(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN, "beam_width": 5},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["smiles"] == ASPIRIN
        assert data["n_routes"] == 2
        assert data["n_solved"] == 1
        assert len(data["routes"]) == 2

    def test_missing_api_key_returns_4xx(self, test_client):
        # FastAPI ≥ 0.100 returns 401 when Security header is absent (auto_error=True),
        # and 403 when the header is present but the value is wrong.
        resp = test_client.post("/predict", json={"smiles": ASPIRIN})
        assert resp.status_code in (401, 403)

    def test_wrong_api_key_returns_403(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_invalid_smiles_returns_422(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": "not-a-smiles!!!"},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 422

    def test_beam_width_out_of_range_returns_422(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN, "beam_width": 0},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 422

    def test_response_schema(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN},
            headers={"X-API-Key": API_KEY},
        )
        data = resp.json()
        route = data["routes"][0]
        assert "route_solved" in route
        assert "trajectory_prob" in route
        assert "reactions" in route
        assert "leaf_smiles" in route
        assert "dead_ends" in route

    def test_sort_on_total_reward_accepted(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN, "sort_on": "total_reward"},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200

    def test_sort_on_invalid_rejected(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN, "sort_on": "bad_key"},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 422

    def test_elapsed_s_in_response(self, test_client):
        resp = test_client.post(
            "/predict",
            json={"smiles": ASPIRIN},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.json()["elapsed_s"] >= 0
