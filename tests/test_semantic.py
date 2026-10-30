"""Tests for semantic routing.

The real embedding model is NEVER loaded here. A deterministic FakeModel maps
text to a 5-dimensional vector by counting cue words, so the tests need no
network, no model download and no sentence-transformers import.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.semantic as semantic
from app.main import app
from app.schemas import MAX_PROMPT_LENGTH
from app.semantic import (
    CATEGORY_EXAMPLES,
    SemanticModelError,
    aggregate_similarity,
    classify_semantic,
    get_category_embeddings,
    route_prompt_semantic,
)

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One axis per category, in this order.
CUES = {
    "coding": ("python", "code", "function", "bug", "sql", "javascript", "algorithm",
               "stack trace", "refactor", "class", "multithreaded", "typeerror", "debug"),
    "reasoning": ("prove", "logic", "trade-offs", "step by step", "argument",
                  "probability", "weigh", "follows", "justify"),
    "summarization": ("summar", "key points", "condense", "tl;dr", "recap",
                      "shorten", "takeaways"),
    "creative_writing": ("poem", "story", "lyrics", "haiku", "fictional", "novel",
                         "limerick", "screenplay"),
    "general": ("capital", "fun fact", "how tall", "motivated", "breakfast",
                "who wrote", "boil", "weather"),
}


class FakeModel:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self):
        self.calls = []  # every list of texts that was encoded

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            lowered = text.lower()
            vec = np.array(
                [sum(lowered.count(cue) for cue in cues) for cues in CUES.values()],
                dtype=float,
            )
            vec += 1e-3  # avoid zero vectors
            rows.append(vec / np.linalg.norm(vec))
        return np.array(rows)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    get_category_embeddings.cache_clear()
    monkeypatch.setattr(semantic, "get_model", lambda: model)
    yield model
    get_category_embeddings.cache_clear()  # don't leak fake embeddings to other tests


# --- import safety ---------------------------------------------------------

def test_importing_app_does_not_load_embedding_library():
    code = (
        "import sys; import app.main; "
        "assert 'sentence_transformers' not in sys.modules, 'model library imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- aggregation -----------------------------------------------------------

def test_aggregate_similarity_is_mean_of_top_k():
    assert aggregate_similarity(np.array([0.9, 0.5, 0.1]), top_k=2) == pytest.approx(0.7)


def test_aggregate_similarity_handles_fewer_examples_than_k():
    assert aggregate_similarity(np.array([0.4]), top_k=2) == pytest.approx(0.4)


# --- fake-model sanity (guards the test fixture itself) --------------------

def test_fake_model_maps_every_example_to_its_own_category(fake_model):
    for category, examples in CATEGORY_EXAMPLES.items():
        for text in examples:
            assert classify_semantic(text).best_match == category, text


# --- category selection ----------------------------------------------------

@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Write a Python function to sort a list.", "coding"),
        ("Summarize this article for me.", "summarization"),
        ("Write a poem about the sea.", "creative_writing"),
        ("Prove that there are infinitely many primes.", "reasoning"),
        ("What is the capital of Japan?", "general"),
    ],
)
def test_classify_semantic_picks_expected_category(fake_model, prompt, expected):
    result = classify_semantic(prompt)
    assert result.category == expected
    assert result.low_confidence is False
    assert result.similarity_score > 0.9


def test_low_similarity_falls_back_to_general(fake_model, monkeypatch):
    monkeypatch.setattr(semantic, "MIN_SIMILARITY", 0.99)
    result = classify_semantic("Zzz qqq xxx.")  # no cue words -> weak match
    assert result.low_confidence is True
    assert result.category == "general"
    assert "falls back to general" in route_prompt_semantic("Zzz qqq xxx.").reason


# --- tier selection (rule-based policy reused) -----------------------------

def test_simple_prompt_gets_cheap_tier(fake_model):
    decision = route_prompt_semantic("Explain what a Python list is.")
    assert decision.task_category == "coding"
    assert decision.model_tier == "cheap"
    assert decision.routing_method == "semantic_plus_rules"


def test_complex_prompt_gets_strong_tier(fake_model):
    decision = route_prompt_semantic("Please debug a race condition in my multithreaded code.")
    assert decision.task_category == "coding"
    assert decision.model_tier == "strong"
    assert "race condition" in decision.reason


def test_complex_reasoning_prompt_gets_strong_tier(fake_model):
    decision = route_prompt_semantic("Prove step by step that there are infinitely many primes.")
    assert decision.task_category == "reasoning"
    assert decision.model_tier == "strong"


# --- caching ---------------------------------------------------------------

def test_category_embeddings_are_computed_once(fake_model):
    route_prompt_semantic("Write a poem about the sea.")
    route_prompt_semantic("What is the capital of Japan?")
    example_batches = [c for c in fake_model.calls if len(c) > 1]
    assert len(example_batches) == len(CATEGORY_EXAMPLES)  # one per category, once


# --- API integration -------------------------------------------------------

def test_semantic_endpoint_returns_expected_shape(fake_model):
    response = client.post("/route/semantic", json={"prompt": "Summarize this article for me."})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "task_category", "model_tier", "routing_method", "similarity_score", "reason",
    }
    assert data["task_category"] == "summarization"
    assert data["model_tier"] == "cheap"
    assert data["routing_method"] == "semantic_plus_rules"
    assert isinstance(data["similarity_score"], float)
    assert data["reason"]


def test_semantic_and_rule_endpoints_can_disagree_on_category(fake_model):
    prompt = "Condense the list of notes."
    rule = client.post("/route", json={"prompt": prompt}).json()
    sem = client.post("/route/semantic", json={"prompt": prompt}).json()
    assert rule["routing_method"] == "rule_based"
    assert rule["task_category"] == "coding"            # keyword "list"
    assert sem["task_category"] == "summarization"      # semantic match
    assert rule["model_tier"] == sem["model_tier"]      # same complexity policy


def test_rule_endpoint_response_is_unchanged(fake_model):
    data = client.post("/route", json={"prompt": "Explain what a Python list is."}).json()
    assert set(data) == {"task_category", "model_tier", "routing_method", "reason"}


def test_semantic_endpoint_returns_503_when_model_unavailable(monkeypatch):
    def broken():
        raise SemanticModelError("Could not load embedding model.")

    get_category_embeddings.cache_clear()
    monkeypatch.setattr(semantic, "get_model", broken)
    response = client.post("/route/semantic", json={"prompt": "Write a poem."})
    assert response.status_code == 503
    assert "Could not load embedding model" in response.json()["detail"]


# --- invalid input (model must never be touched) ---------------------------

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
def test_semantic_endpoint_rejects_invalid_prompts(fake_model, payload):
    response = client.post("/route/semantic", json=payload)
    assert response.status_code == 422
    assert fake_model.calls == []