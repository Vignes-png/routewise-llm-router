"""Tests for the SQLite layer in app/observability.py and the error categories
produced by app/dispatcher.py. Every test uses a temporary database."""

import sqlite3
from datetime import datetime

import httpx
import litellm
import pytest
from types import SimpleNamespace

from app.dispatcher import DispatchError, dispatch_to_model
from app.observability import (
    ERROR_CATEGORIES,
    ChatEvent,
    fetch_recent_events,
    record_chat_event,
)


def test_new_event_defaults_to_failure_not_success():
    event = ChatEvent()
    assert event.status == "failure"
    assert event.error_category == "unexpected_error"
    assert event.http_status == 500


def test_event_ids_are_unique_and_timestamps_are_utc_iso():
    a, b = ChatEvent(), ChatEvent()
    assert a.request_id != b.request_id
    assert len(a.request_id) == 32
    parsed = datetime.fromisoformat(a.created_at)
    assert parsed.utcoffset().total_seconds() == 0


def test_record_and_read_back_round_trip(tmp_path):
    db = tmp_path / "t.db"
    event = ChatEvent(
        status="success", error_category=None, http_status=200,
        routing_method="semantic_plus_rules", task_category="coding",
        similarity_score=0.81, model_tier="cheap", model="gemini/m",
        latency_ms=123, prompt_tokens=5, completion_tokens=7, total_tokens=12,
        estimated_cost_usd=0.00001, pricing_checked_on="2026-10-28",
    )
    assert record_chat_event(event, db) is True

    rows = fetch_recent_events(10, db)
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == event.request_id
    assert row["status"] == "success"
    assert row["error_category"] is None
    assert row["latency_ms"] == 123
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (5, 7, 12)


def test_unknown_values_are_stored_as_null_not_zero(tmp_path):
    db = tmp_path / "t.db"
    record_chat_event(ChatEvent(), db)
    row = fetch_recent_events(1, db)[0]
    for column in ("prompt_tokens", "completion_tokens", "total_tokens",
                   "estimated_cost_usd", "latency_ms", "model"):
        assert row[column] is None


def test_table_has_no_column_for_prompts_answers_or_error_text(tmp_path):
    db = tmp_path / "t.db"
    record_chat_event(ChatEvent(), db)
    conn = sqlite3.connect(db)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(chat_requests)")}
    conn.close()
    forbidden = {"prompt", "answer", "response", "error", "error_message", "api_key", "key"}
    assert columns.isdisjoint(forbidden)
    assert "estimated_cost_usd" in columns  # labelled as an estimate


def test_duplicate_request_id_is_rejected_without_raising(tmp_path):
    db = tmp_path / "t.db"
    event = ChatEvent()
    assert record_chat_event(event, db) is True
    assert record_chat_event(event, db) is False  # PRIMARY KEY violation, but no exception
    assert len(fetch_recent_events(10, db)) == 1


def test_record_never_raises_when_database_cannot_be_opened(tmp_path):
    # A directory cannot be opened as a database file.
    assert record_chat_event(ChatEvent(), tmp_path) is False


def test_fetch_returns_empty_list_when_database_missing(tmp_path):
    assert fetch_recent_events(10, tmp_path / "does_not_exist.db") == []


def test_newest_rows_come_first(tmp_path):
    db = tmp_path / "t.db"
    first, second = ChatEvent(), ChatEvent()
    record_chat_event(first, db)
    record_chat_event(second, db)
    ids = [r["request_id"] for r in fetch_recent_events(10, db)]
    assert ids[0] == second.request_id


# --- dispatcher error categories -------------------------------------------

def _http_response(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))


CATEGORY_CASES = [
    (lambda: litellm.AuthenticationError("x", "gemini", "m"), "provider_auth"),
    (lambda: litellm.PermissionDeniedError("x", "gemini", "m", response=_http_response(403)), "provider_auth"),
    (lambda: litellm.RateLimitError("x", "gemini", "m"), "rate_limit"),
    (lambda: litellm.Timeout("x", "m", "gemini"), "timeout"),
    (lambda: litellm.NotFoundError("x", "m", "gemini"), "model_not_found"),
    (lambda: litellm.BadRequestError("x", "m", "gemini"), "bad_request"),
    (lambda: litellm.InternalServerError("x", "gemini", "m"), "provider_error"),
    (lambda: RuntimeError("x"), "provider_error"),
]


@pytest.mark.parametrize("make_exception, expected_category", CATEGORY_CASES)
def test_dispatcher_assigns_safe_category(monkeypatch, make_exception, expected_category):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    def raise_it(**kwargs):
        raise make_exception()

    monkeypatch.setattr(litellm, "completion", raise_it)
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.category == expected_category
    assert expected_category in ERROR_CATEGORIES


def test_missing_key_and_empty_response_categories(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.category == "missing_credentials"

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))], usage=None)
    monkeypatch.setattr(litellm, "completion", lambda **kw: empty)
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.category == "empty_response"


def test_dispatch_error_default_category_keeps_old_two_argument_form():
    assert DispatchError(502, "msg").category == "provider_error"