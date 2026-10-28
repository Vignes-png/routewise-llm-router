"""POST /chat observability. Routing and the model call are replaced by fakes and
the database is a temporary file, so no network, API key or quota is used."""

import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import MODEL_BY_TIER, get_database_path
from app.dispatcher import DispatchError, DispatchResult
from app.observability import ERROR_CATEGORIES, fetch_recent_events
from app.pricing import PRICING_CHECKED_ON
from app.schemas import SemanticRouteResponse, TokenUsage
from app.semantic import SemanticModelError

client = TestClient(main.app)
no_raise_client = TestClient(main.app, raise_server_exceptions=False)

SECRET_PROMPT = "UNIQUE-PROMPT-TEXT-12345 explain lists"
SECRET_ANSWER = "UNIQUE-ANSWER-TEXT-67890"


def decision(tier="cheap", category="coding"):
    return SemanticRouteResponse(
        task_category=category, model_tier=tier,
        routing_method="semantic_plus_rules", similarity_score=0.8123,
        reason="fake routing reason",
    )


class FakeDispatcher:
    def __init__(self):
        self.result = DispatchResult(
            SECRET_ANSWER, TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        )
        self.error = None

    def __call__(self, model, prompt):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def fake(monkeypatch):
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(main, "dispatch_to_model", dispatcher)
    monkeypatch.setattr(main, "route_prompt_semantic", lambda prompt: decision())
    return dispatcher


def rows():
    return fetch_recent_events(50)


# --- success ---------------------------------------------------------------

def test_success_is_recorded_with_all_metadata(fake):
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})
    assert response.status_code == 200

    [row] = rows()
    assert row["request_id"] == response.headers["X-Request-ID"]
    assert row["status"] == "success"
    assert row["error_category"] is None
    assert row["http_status"] == 200
    assert row["routing_method"] == "semantic_plus_rules"
    assert row["task_category"] == "coding"
    assert row["similarity_score"] == pytest.approx(0.8123)
    assert row["model_tier"] == "cheap"
    assert row["model"] == MODEL_BY_TIER["cheap"]
    assert isinstance(row["latency_ms"], int) and row["latency_ms"] >= 0
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (3, 4, 7)
    assert row["estimated_cost_usd"] == pytest.approx((3 * 0.30 + 4 * 2.50) / 1_000_000)
    assert row["pricing_checked_on"] == PRICING_CHECKED_ON
    assert row["created_at"]


def test_response_body_schema_is_unchanged(fake):
    data = client.post("/chat", json={"prompt": SECRET_PROMPT}).json()
    assert set(data) == {
        "answer", "model", "task_category", "model_tier", "routing_method",
        "similarity_score", "reason", "latency_ms", "usage",
    }


def test_strong_tier_row_uses_strong_model(fake, monkeypatch):
    monkeypatch.setattr(main, "route_prompt_semantic", lambda p: decision("strong", "reasoning"))
    client.post("/chat", json={"prompt": SECRET_PROMPT})
    [row] = rows()
    assert row["model_tier"] == "strong"
    assert row["model"] == MODEL_BY_TIER["strong"]


def test_missing_usage_gives_null_tokens_and_unknown_cost(fake):
    fake.result = DispatchResult(SECRET_ANSWER, None)
    assert client.post("/chat", json={"prompt": SECRET_PROMPT}).status_code == 200
    [row] = rows()
    assert row["status"] == "success"
    assert row["prompt_tokens"] is None
    assert row["completion_tokens"] is None
    assert row["total_tokens"] is None
    assert row["estimated_cost_usd"] is None
    assert row["pricing_checked_on"] is None


def test_partial_usage_gives_unknown_cost(fake):
    fake.result = DispatchResult(SECRET_ANSWER, TokenUsage(prompt_tokens=3))
    client.post("/chat", json={"prompt": SECRET_PROMPT})
    [row] = rows()
    assert row["prompt_tokens"] == 3
    assert row["completion_tokens"] is None
    assert row["estimated_cost_usd"] is None


def test_each_request_gets_its_own_id(fake):
    a = client.post("/chat", json={"prompt": SECRET_PROMPT}).headers["X-Request-ID"]
    b = client.post("/chat", json={"prompt": SECRET_PROMPT}).headers["X-Request-ID"]
    assert a != b
    assert {r["request_id"] for r in rows()} == {a, b}


# --- failures --------------------------------------------------------------

DISPATCH_FAILURES = [
    (503, "missing_credentials"),
    (503, "provider_auth"),
    (429, "rate_limit"),
    (504, "timeout"),
    (502, "model_not_found"),
    (502, "bad_request"),
    (502, "provider_error"),
    (502, "empty_response"),
]


@pytest.mark.parametrize("status_code, category", DISPATCH_FAILURES)
def test_model_call_failures_are_recorded(fake, status_code, category):
    fake.error = DispatchError(status_code, "safe message", category)
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})
    assert response.status_code == status_code
    assert response.json() == {"detail": "safe message"}

    [row] = rows()
    assert row["request_id"] == response.headers["X-Request-ID"]
    assert row["status"] == "failure"
    assert row["error_category"] == category
    assert category in ERROR_CATEGORIES
    assert row["http_status"] == status_code
    assert row["model"] == MODEL_BY_TIER["cheap"]          # we know which model was tried
    assert row["task_category"] == "coding"                # routing succeeded
    assert row["latency_ms"] is not None                   # time spent before failing
    assert row["prompt_tokens"] is None
    assert row["estimated_cost_usd"] is None


def test_semantic_routing_failure_is_recorded(fake, monkeypatch):
    def broken(prompt):
        raise SemanticModelError("Could not load embedding model.")

    monkeypatch.setattr(main, "route_prompt_semantic", broken)
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})
    assert response.status_code == 503

    [row] = rows()
    assert row["status"] == "failure"
    assert row["error_category"] == "routing_model_unavailable"
    assert row["http_status"] == 503
    assert row["routing_method"] is None
    assert row["model"] is None
    assert row["latency_ms"] is None  # the model was never called


def test_unexpected_routing_error_is_recorded_as_500(fake, monkeypatch):
    def buggy(prompt):
        raise ValueError("boom")

    monkeypatch.setattr(main, "route_prompt_semantic", buggy)
    response = no_raise_client.post("/chat", json={"prompt": SECRET_PROMPT})
    assert response.status_code == 500
    [row] = rows()
    assert row["status"] == "failure"
    assert row["error_category"] == "routing_error"
    assert row["http_status"] == 500


def test_unexpected_dispatch_error_is_recorded_as_500(fake):
    fake.error = RuntimeError("not a DispatchError")
    response = no_raise_client.post("/chat", json={"prompt": SECRET_PROMPT})
    assert response.status_code == 500
    [row] = rows()
    assert row["status"] == "failure"
    assert row["error_category"] == "unexpected_error"
    assert row["model"] == MODEL_BY_TIER["cheap"]


def test_invalid_requests_are_rejected_before_the_endpoint_and_not_recorded(fake):
    assert client.post("/chat", json={"prompt": "   "}).status_code == 422
    assert rows() == []


# --- privacy ---------------------------------------------------------------

def test_prompt_and_answer_never_reach_the_database_file(fake):
    fake.error = None
    client.post("/chat", json={"prompt": SECRET_PROMPT})
    fake.error = DispatchError(502, "safe message", "provider_error")
    client.post("/chat", json={"prompt": SECRET_PROMPT})

    raw = get_database_path().read_bytes()
    assert SECRET_PROMPT.encode() not in raw
    assert b"UNIQUE-PROMPT-TEXT" not in raw
    assert SECRET_ANSWER.encode() not in raw
    assert b"safe message" not in raw  # only the category is stored, not message text


# --- database failures must not break requests -----------------------------

def test_success_still_succeeds_when_database_write_fails(fake, monkeypatch, tmp_path, caplog):
    # Point the database at a directory, which SQLite cannot open.
    monkeypatch.setenv("ROUTEWISE_DB_PATH", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="routewise.observability"):
        response = client.post("/chat", json={"prompt": SECRET_PROMPT})

    assert response.status_code == 200
    assert response.json()["answer"] == SECRET_ANSWER
    assert "Could not record chat event" in caplog.text
    assert SECRET_PROMPT not in caplog.text
    assert SECRET_ANSWER not in caplog.text


def test_failure_response_is_unchanged_when_database_write_fails(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("ROUTEWISE_DB_PATH", str(tmp_path))
    fake.error = DispatchError(429, "rate limited message", "rate_limit")
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})
    assert response.status_code == 429
    assert response.json() == {"detail": "rate limited message"}