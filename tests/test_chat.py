"""Tests for POST /chat. Both the routing step and the model call are replaced
by fakes, so these tests need no network, no API key, no quota and no embedding model."""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, DispatchResult
from app.schemas import MAX_PROMPT_LENGTH, SemanticRouteResponse, TokenUsage
from app.semantic import SemanticModelError

client = TestClient(main.app)


def decision(tier="cheap", category="coding"):
    return SemanticRouteResponse(
        task_category=category,
        model_tier=tier,
        routing_method="semantic_plus_rules",
        similarity_score=0.8123,
        reason="fake routing reason",
    )


class FakeDispatcher:
    """Records calls and returns a canned answer (or raises a canned error)."""

    def __init__(self, result=None, error=None):
        self.result = result or DispatchResult("Fake answer.", TokenUsage(
            prompt_tokens=3, completion_tokens=4, total_tokens=7))
        self.error = error
        self.calls = []

    def __call__(self, model, prompt):
        self.calls.append((model, prompt))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def dispatcher(monkeypatch):
    fake = FakeDispatcher()
    monkeypatch.setattr(main, "dispatch_to_model", fake)
    monkeypatch.setattr(main, "route_prompt_semantic", lambda prompt: decision())
    return fake


def test_model_mapping_covers_both_tiers():
    assert set(MODEL_BY_TIER) == {"cheap", "strong"}
    assert MODEL_BY_TIER["cheap"] != MODEL_BY_TIER["strong"]


def test_cheap_tier_uses_cheap_model_and_returns_full_response(dispatcher):
    response = client.post("/chat", json={"prompt": "Explain what a Python list is."})
    assert response.status_code == 200
    data = response.json()

    assert dispatcher.calls == [(MODEL_BY_TIER["cheap"], "Explain what a Python list is.")]
    assert data["answer"] == "Fake answer."
    assert data["model"] == MODEL_BY_TIER["cheap"]
    assert data["task_category"] == "coding"
    assert data["model_tier"] == "cheap"
    assert data["routing_method"] == "semantic_plus_rules"
    assert data["similarity_score"] == pytest.approx(0.8123)
    assert data["reason"] == "fake routing reason"
    assert isinstance(data["latency_ms"], int) and data["latency_ms"] >= 0
    assert data["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert set(data) == {
        "answer", "model", "task_category", "model_tier", "routing_method",
        "similarity_score", "reason", "latency_ms", "usage",
    }


def test_strong_tier_uses_strong_model(dispatcher, monkeypatch):
    monkeypatch.setattr(main, "route_prompt_semantic", lambda prompt: decision("strong", "reasoning"))
    data = client.post("/chat", json={"prompt": "Prove something hard."}).json()
    assert dispatcher.calls[0][0] == MODEL_BY_TIER["strong"]
    assert data["model"] == MODEL_BY_TIER["strong"]
    assert data["model_tier"] == "strong"
    assert data["task_category"] == "reasoning"


def test_usage_is_null_when_provider_gives_none(dispatcher):
    dispatcher.result = DispatchResult("No usage here.", None)
    data = client.post("/chat", json={"prompt": "hello there"}).json()
    assert data["usage"] is None


@pytest.mark.parametrize(
    "status_code, message",
    [
        (503, "GEMINI_API_KEY is not configured on the server."),
        (429, "The model provider's rate limit or quota was reached. Try again later."),
        (504, "The model provider did not respond in time."),
        (502, "The model provider returned an error. Try again later."),
    ],
)
def test_dispatch_errors_become_http_errors(dispatcher, status_code, message):
    dispatcher.error = DispatchError(status_code, message)
    response = client.post("/chat", json={"prompt": "hello there"})
    assert response.status_code == status_code
    assert response.json() == {"detail": message}


def test_semantic_model_failure_returns_503_and_skips_model_call(dispatcher, monkeypatch):
    def broken(prompt):
        raise SemanticModelError("Could not load embedding model.")

    monkeypatch.setattr(main, "route_prompt_semantic", broken)
    response = client.post("/chat", json={"prompt": "hello there"})
    assert response.status_code == 503
    assert dispatcher.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "     "},
        {},
        {"prompt": 123},
        {"prompt": "a" * (MAX_PROMPT_LENGTH + 1)},
    ],
)
def test_invalid_prompts_are_rejected_before_any_model_call(dispatcher, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    assert dispatcher.calls == []