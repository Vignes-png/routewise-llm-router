"""Unit tests for the routing functions (no HTTP involved)."""

import pytest

from app.routing import (
    LONG_PROMPT_CHARS,
    assess_complexity,
    build_reason,
    classify_task,
    find_matches,
    route_prompt,
    select_tier,
)


def test_find_matches_is_case_insensitive():
    assert find_matches("Fix this PYTHON bug", ("python", "bug")) == ["python", "bug"]


def test_find_matches_respects_word_start():
    # "class" must not match inside "declassified".
    assert find_matches("declassified memo", ("class",)) == []


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Write a Python function to reverse a string.", "coding"),
        ("Summarize this article in three sentences.", "summarization"),
        ("Write a poem about autumn.", "creative_writing"),
        ("Compare the pros and cons of renting versus buying.", "reasoning"),
        ("What is the capital of France?", "general"),
    ],
)
def test_classify_task(prompt, expected):
    category, _ = classify_task(prompt)
    assert category == expected


def test_assess_complexity_simple():
    is_complex, signals = assess_complexity("What is a variable?")
    assert is_complex is False
    assert signals == []


def test_assess_complexity_keyword():
    is_complex, signals = assess_complexity("Find the deadlock in this code.")
    assert is_complex is True
    assert "deadlock" in signals


def test_assess_complexity_long_prompt():
    is_complex, signals = assess_complexity("x " * LONG_PROMPT_CHARS)
    assert is_complex is True
    assert any("long prompt" in s for s in signals)


def test_select_tier():
    assert select_tier(False) == "cheap"
    assert select_tier(True) == "strong"


def test_build_reason_mentions_signals():
    reason = build_reason("coding", ["python"], "strong", ["deadlock"])
    assert "python" in reason
    assert "deadlock" in reason


def test_route_prompt_returns_baseline_method():
    decision = route_prompt("Tell me a fun fact.")
    assert decision.routing_method == "rule_based"
    assert decision.task_category == "general"
    assert decision.model_tier == "cheap"