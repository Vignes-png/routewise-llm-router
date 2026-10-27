"""Tests for app/dispatcher.py. litellm.completion is always replaced by a fake,
so no network access, API key or quota is needed."""

from types import SimpleNamespace

import httpx
import litellm
import pytest

from app import dispatcher
from app.dispatcher import DispatchError, dispatch_to_model

FAKE_KEY = "test-key-not-a-real-secret"


def fake_response(content="Hello!", usage=None):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)


def test_success_returns_answer_and_usage(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        return fake_response("Four.", usage)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    result = dispatch_to_model("gemini/some-model", "What is 2+2?")

    assert result.answer == "Four."
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 12
    assert seen["model"] == "gemini/some-model"
    assert seen["messages"] == [{"role": "user", "content": "What is 2+2?"}]
    assert seen["api_key"] == FAKE_KEY


def test_usage_is_none_when_provider_supplies_none(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("Hi", usage=None))
    assert dispatch_to_model("gemini/m", "hi").usage is None


def test_missing_api_key_raises_503_without_calling_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def must_not_be_called(**kwargs):
        raise AssertionError("provider should not be called without a key")

    monkeypatch.setattr(litellm, "completion", must_not_be_called)
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.status_code == 503


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.status_code == 503


def _http_response(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))


# Each case is a small function that BUILDS the exception, so a LiteLLM version
# change in exception constructors fails one test instead of breaking collection.
ERROR_CASES = [
    (lambda: litellm.AuthenticationError("bad key", "gemini", "m"), 503),
    (lambda: litellm.PermissionDeniedError("denied", "gemini", "m", response=_http_response(403)), 503),
    (lambda: litellm.RateLimitError("slow down", "gemini", "m"), 429),
    (lambda: litellm.Timeout("too slow", "m", "gemini"), 504),
    (lambda: litellm.NotFoundError("no such model", "m", "gemini"), 502),
    (lambda: litellm.BadRequestError("bad request", "m", "gemini"), 502),
    (lambda: litellm.InternalServerError("boom", "gemini", "m"), 502),
    (lambda: RuntimeError("something unexpected"), 502),
]


@pytest.mark.parametrize("make_exception, expected_status", ERROR_CASES)
def test_provider_errors_map_to_http_status(monkeypatch, make_exception, expected_status):
    def raise_it(**kwargs):
        raise make_exception()

    monkeypatch.setattr(litellm, "completion", raise_it)
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.status_code == expected_status


def test_error_message_never_contains_secret_or_provider_text(monkeypatch):
    leaky = litellm.AuthenticationError(f"invalid key {FAKE_KEY} at https://x?key={FAKE_KEY}", "gemini", "m")

    def raise_it(**kwargs):
        raise leaky

    monkeypatch.setattr(litellm, "completion", raise_it)
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert FAKE_KEY not in info.value.message
    assert FAKE_KEY not in str(info.value)
    assert info.value.__cause__ is None  # original exception is not chained


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_model_response_is_a_502(monkeypatch, content):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response(content))
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.status_code == 502


def test_malformed_response_is_a_502(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kw: SimpleNamespace(choices=[]))
    with pytest.raises(DispatchError) as info:
        dispatch_to_model("gemini/m", "hi")
    assert info.value.status_code == 502


def test_dispatcher_module_exposes_no_api_key():
    assert not hasattr(dispatcher, "GEMINI_API_KEY")