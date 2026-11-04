"""Tests for POST /chat operating modes (auto, fast, quality) and observability."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, DispatchResult
from app.observability import fetch_recent_events
from app.schemas import SemanticRouteResponse, TokenUsage


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point observability at a temporary SQLite database for every test."""
    db_file = tmp_path / "test_routewise.db"
    monkeypatch.setenv("ROUTEWISE_DB_PATH", str(db_file))
    return db_file


client = TestClient(main_module.app)


def _fake_semantic_decision(tier: str = "cheap") -> SemanticRouteResponse:
    return SemanticRouteResponse(
        task_category="coding",
        model_tier=tier,  # type: ignore[arg-type]
        routing_method="semantic_plus_rules",
        similarity_score=0.82,
        reason="Semantic match for coding.",
    )


def _fake_dispatch_result(text: str = "Mocked model response.") -> DispatchResult:
    return DispatchResult(
        answer=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def test_chat_defaults_to_auto_when_mode_omitted(monkeypatch: pytest.MonkeyPatch):
    """1 & 2: Omitting mode defaults to 'auto' and calls route_prompt_semantic."""
    semantic_calls: list[str] = []
    dispatched_models: list[str] = []

    def mock_semantic(prompt: str) -> SemanticRouteResponse:
        semantic_calls.append(prompt)
        return _fake_semantic_decision(tier="cheap")

    def mock_dispatch(model: str, prompt: str) -> DispatchResult:
        dispatched_models.append(model)
        return _fake_dispatch_result("Auto answer")

    monkeypatch.setattr(main_module, "route_prompt_semantic", mock_semantic)
    monkeypatch.setattr(main_module, "dispatch_to_model", mock_dispatch)

    response = client.post("/chat", json={"prompt": "Write a Python function."})

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    data = response.json()
    assert data["model_tier"] == "cheap"
    assert data["model"] == MODEL_BY_TIER["cheap"]
    assert data["routing_method"] == "semantic_plus_rules"
    assert data["task_category"] == "coding"
    assert data["similarity_score"] == pytest.approx(0.82)
    assert data["reason"] == "Semantic match for coding."
    assert semantic_calls == ["Write a Python function."]
    assert dispatched_models == [MODEL_BY_TIER["cheap"]]


def test_chat_explicit_auto_mode_uses_semantic_routing(monkeypatch: pytest.MonkeyPatch):
    """2 & 5: Explicit mode='auto' uses route_prompt_semantic and returns its explanation."""
    semantic_calls: list[str] = []

    def mock_semantic(prompt: str) -> SemanticRouteResponse:
        semantic_calls.append(prompt)
        return _fake_semantic_decision(tier="strong")

    monkeypatch.setattr(main_module, "route_prompt_semantic", mock_semantic)
    monkeypatch.setattr(
        main_module, "dispatch_to_model", lambda m, p: _fake_dispatch_result("Strong auto")
    )

    response = client.post("/chat", json={"prompt": "Debug a deadlock.", "mode": "auto"})

    assert response.status_code == 200
    data = response.json()
    assert data["model_tier"] == "strong"
    assert data["model"] == MODEL_BY_TIER["strong"]
    assert data["routing_method"] == "semantic_plus_rules"
    assert data["reason"] == "Semantic match for coding."
    assert len(semantic_calls) == 1


def test_chat_fast_mode_selects_cheap_and_bypasses_semantic(monkeypatch: pytest.MonkeyPatch):
    """3 & 5: Fast mode selects cheap model directly and never calls route_prompt_semantic."""
    def forbid_semantic(prompt: str):
        pytest.fail("route_prompt_semantic must not be called in fast mode")

    dispatched: list[tuple[str, str]] = []

    def mock_dispatch(model: str, prompt: str) -> DispatchResult:
        dispatched.append((model, prompt))
        return _fake_dispatch_result("Fast answer")

    monkeypatch.setattr(main_module, "route_prompt_semantic", forbid_semantic)
    monkeypatch.setattr(main_module, "dispatch_to_model", mock_dispatch)

    response = client.post(
        "/chat",
        json={"prompt": "Prove step by step that sqrt(2) is irrational.", "mode": "fast"},
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    data = response.json()
    assert data["model_tier"] == "cheap"
    assert data["model"] == MODEL_BY_TIER["cheap"]
    assert data["routing_method"] == "manual_fast"
    assert data["task_category"] is None
    assert data["similarity_score"] is None
    assert "fast" in data["reason"].lower()
    assert dispatched == [
        (MODEL_BY_TIER["cheap"], "Prove step by step that sqrt(2) is irrational.")
    ]


def test_chat_quality_mode_selects_strong_and_bypasses_semantic(
    monkeypatch: pytest.MonkeyPatch,
):
    """4 & 5: Quality mode selects strong model directly and never calls route_prompt_semantic."""
    def forbid_semantic(prompt: str):
        pytest.fail("route_prompt_semantic must not be called in quality mode")

    dispatched: list[tuple[str, str]] = []

    def mock_dispatch(model: str, prompt: str) -> DispatchResult:
        dispatched.append((model, prompt))
        return _fake_dispatch_result("Quality answer")

    monkeypatch.setattr(main_module, "route_prompt_semantic", forbid_semantic)
    monkeypatch.setattr(main_module, "dispatch_to_model", mock_dispatch)

    response = client.post(
        "/chat",
        json={"prompt": "Say hello.", "mode": "quality"},
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    data = response.json()
    assert data["model_tier"] == "strong"
    assert data["model"] == MODEL_BY_TIER["strong"]
    assert data["routing_method"] == "manual_quality"
    assert data["task_category"] is None
    assert data["similarity_score"] is None
    assert "quality" in data["reason"].lower()
    assert dispatched == [(MODEL_BY_TIER["strong"], "Say hello.")]


def test_chat_invalid_mode_returns_422_without_dispatching(
    monkeypatch: pytest.MonkeyPatch,
):
    """Invalid mode fails Pydantic validation with HTTP 422 and never calls dispatcher."""
    def forbid_dispatch(model: str, prompt: str):
        pytest.fail("dispatch_to_model must not be called when mode is invalid")

    monkeypatch.setattr(main_module, "dispatch_to_model", forbid_dispatch)

    response = client.post(
        "/chat",
        json={"prompt": "Hello world.", "mode": "invalid_mode"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("mode", "expected_method", "expected_tier", "expected_category"),
    [
        ("auto", "semantic_plus_rules", "cheap", "coding"),
        ("fast", "manual_fast", "cheap", None),
        ("quality", "manual_quality", "strong", None),
    ],
)
def test_observability_records_mode_and_model(
    mode: str,
    expected_method: str,
    expected_tier: str,
    expected_category: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    """7 & 9: SQLite observability records the routing_method, tier, model, and X-Request-ID."""
    monkeypatch.setattr(
        main_module,
        "route_prompt_semantic",
        lambda p: _fake_semantic_decision(tier="cheap"),
    )
    monkeypatch.setattr(
        main_module, "dispatch_to_model", lambda m, p: _fake_dispatch_result("OK")
    )

    response = client.post("/chat", json={"prompt": "Test prompt", "mode": mode})
    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id

    events = fetch_recent_events(limit=1)
    assert len(events) == 1
    row = events[0]
    assert row["request_id"] == request_id
    assert row["status"] == "success"
    assert row["http_status"] == 200
    assert row["routing_method"] == expected_method
    assert row["model_tier"] == expected_tier
    assert row["model"] == MODEL_BY_TIER[expected_tier]
    assert row["task_category"] == expected_category


@pytest.mark.parametrize(
    ("mode", "expected_method", "expected_tier"),
    [
        ("fast", "manual_fast", "cheap"),
        ("quality", "manual_quality", "strong"),
    ],
)
def test_dispatch_failure_handled_and_logged_in_manual_modes(
    mode: str,
    expected_method: str,
    expected_tier: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """6 & 9: DispatchError returns proper status, preserves X-Request-ID, and logs failure."""
    def failing_dispatch(model: str, prompt: str) -> DispatchResult:
        raise DispatchError(429, "Rate limit reached.", "rate_limit")

    monkeypatch.setattr(main_module, "dispatch_to_model", failing_dispatch)

    response = client.post("/chat", json={"prompt": "Hello", "mode": mode})

    assert response.status_code == 429
    request_id = response.headers.get("X-Request-ID")
    assert request_id
    assert response.json()["detail"] == "Rate limit reached."

    events = fetch_recent_events(limit=1)
    assert len(events) == 1
    row = events[0]
    assert row["request_id"] == request_id
    assert row["status"] == "failure"
    assert row["http_status"] == 429
    assert row["error_category"] == "rate_limit"
    assert row["routing_method"] == expected_method
    assert row["model_tier"] == expected_tier
    assert row["model"] == MODEL_BY_TIER[expected_tier]


def test_existing_route_and_semantic_route_contracts_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    """8: /route and /route/semantic schemas and behaviour remain unchanged."""
    monkeypatch.setattr(
        main_module,
        "route_prompt_semantic",
        lambda p: _fake_semantic_decision(tier="cheap"),
    )

    res_rule = client.post("/route", json={"prompt": "Write a python script."})
    assert res_rule.status_code == 200
    assert set(res_rule.json().keys()) == {
        "task_category",
        "model_tier",
        "routing_method",
        "reason",
    }

    res_sem = client.post("/route/semantic", json={"prompt": "Write a python script."})
    assert res_sem.status_code == 200
    assert set(res_sem.json().keys()) == {
        "task_category",
        "model_tier",
        "routing_method",
        "similarity_score",
        "reason",
    }