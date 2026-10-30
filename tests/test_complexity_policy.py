"""Regression tests for the complexity (model-tier) policy in app/routing.py.

The policy looks at the prompt text only. A single cue never escalates a prompt
and neither does length: a demanding request has to meet a hard context.
All prompts here are written for these tests; none come from the benchmark file.
"""

import inspect

import pytest

import app.semantic as semantic
from app.routing import assess_complexity, route_prompt, select_tier
from app.semantic import SemanticClassification, route_prompt_semantic


def tier(prompt):
    return route_prompt(prompt).model_tier


# --- problem 1: one "prove" must not escalate ------------------------------

@pytest.mark.parametrize("prompt", [
    "Prove that the product of two even numbers is even.",
    "Prove that 17 is a prime number.",
    "Write a story about a detective who must prove a suspect is innocent.",
])
def test_a_bare_proof_request_is_cheap(prompt):
    assert tier(prompt) == "cheap"


@pytest.mark.parametrize("prompt", [
    "Prove rigorously that the sum of the first n odd numbers equals n squared.",
    "Derive the quadratic formula step by step.",
    "Formally prove that there are infinitely many primes.",
])
def test_a_proof_with_an_explicit_demand_for_rigor_is_strong(prompt):
    assert tier(prompt) == "strong"


# --- problem 2: technical words used as a creative TOPIC -------------------

@pytest.mark.parametrize("prompt", [
    "Write a limerick about a memory leak.",
    "Compose a song about a distributed system that can never agree.",
    "Write a haiku about latency in production.",
])
def test_technical_topic_words_in_creative_requests_stay_cheap(prompt):
    assert tier(prompt) == "cheap"


# --- problem 3: length alone must not escalate -----------------------------

def test_a_long_plain_summarization_prompt_is_cheap():
    prompt = "Summarize this: " + "The committee met on Tuesday and reviewed the budget. " * 40
    assert len(prompt) > 1500
    assert assess_complexity(prompt) == (False, [])
    assert tier(prompt) == "cheap"


# --- problem 4: demanding technical work is strong -------------------------

@pytest.mark.parametrize("prompt", [
    "Our checkout service times out under load and latency climbs after every deploy. Help me find the cause.",
    "Troubleshoot a race condition that only appears in production.",
    "Diagnose why our payment queue loses messages when a node restarts in production.",
    "Design a distributed job queue that never runs a job twice.",
    "Design a scalable chat service with no single point of failure.",
    "Optimize this endpoint for high throughput under concurrent load.",
    "Explain the trade-offs of microservices at scale.",
])
def test_hard_technical_requests_are_strong(prompt):
    assert tier(prompt) == "strong"


# --- easy prompts that merely contain a "hard" word ------------------------

@pytest.mark.parametrize("prompt", [
    "What does 'distributed' mean in statistics?",
    "Explain what a race condition is.",
    "What is the complex conjugate of 3 + 4i?",
    "How do I write a formal email to my landlord?",
    "Give me a step by step recipe for pancakes.",
    "Design a birthday card for my dad.",
    "Debug my sentence: 'me and him goes to school'.",
    "Describe the architecture of a Gothic cathedral.",
    "What is a deadlock?",
])
def test_easy_prompts_with_misleading_words_are_cheap(prompt):
    assert tier(prompt) == "cheap"


# --- the signals explain the decision --------------------------------------

def test_strong_prompts_report_the_rule_and_the_cues():
    is_complex, signals = assess_complexity("Please debug a race condition in my multithreaded code.")
    assert is_complex is True
    assert signals == ["technical task in a hard-systems context", "debug", "race condition"]


def test_proof_rule_reports_its_cues():
    _, signals = assess_complexity("Prove rigorously that 2 + 2 = 4.")
    assert signals == ["proof or derivation with explicit rigor", "prove", "rigorous"]


def test_cheap_prompts_have_no_signals():
    assert assess_complexity("What is the capital of France?") == (False, [])


def test_reason_text_names_the_signals():
    reason = route_prompt("Please debug a race condition in my multithreaded code.").reason
    assert "technical task" in reason and "race condition" in reason


# --- one shared policy, independent of the category ------------------------

SHARED_POLICY_PROMPTS = [
    "Prove that 17 is a prime number.",
    "Formally prove that there are infinitely many primes.",
    "Write a limerick about a memory leak.",
    "Design a distributed job queue that never runs a job twice.",
    "Explain what a race condition is.",
    "Summarize this: " + "The committee met on Tuesday. " * 80,
]


def stub_category(monkeypatch, category):
    """Make the semantic router return a fixed category without loading any model."""
    scores = {name: 0.1 for name in semantic.CATEGORY_EXAMPLES}
    scores[category] = 0.9
    fixed = SemanticClassification(category, 0.9, category, False, scores)
    monkeypatch.setattr(semantic, "classify_semantic", lambda prompt: fixed)


@pytest.mark.parametrize("category", list(semantic.CATEGORY_EXAMPLES))
def test_both_routers_choose_the_same_tier_whatever_the_category(monkeypatch, category):
    stub_category(monkeypatch, category)
    for prompt in SHARED_POLICY_PROMPTS:
        expected = select_tier(assess_complexity(prompt)[0])
        assert route_prompt(prompt).model_tier == expected
        assert route_prompt_semantic(prompt).model_tier == expected


def test_the_policy_takes_only_the_prompt():
    assert list(inspect.signature(assess_complexity).parameters) == ["prompt"]


# --- known limitations, written down on purpose ----------------------------
# strict xfail: if a later change makes one of these pass, the suite fails and
# tells you to remove the marker and celebrate (or revisit the policy).

@pytest.mark.xfail(strict=True, reason="a task word plus a technical TOPIC word cannot be told from a task")
@pytest.mark.parametrize("prompt", [
    "Write a poem about debugging a deadlock.",
    "Fix the grammar in this paragraph about network latency.",
])
def test_limitation_topic_words_next_to_a_task_word_still_escalate(prompt):
    assert tier(prompt) == "cheap"


@pytest.mark.xfail(strict=True, reason="difficulty without technical or rigor words is not detected by keywords")
def test_limitation_open_ended_decision_analysis_is_not_detected():
    prompt = "Should we rewrite our monolith as microservices? Argue both sides and recommend one."
    assert tier(prompt) == "strong"
