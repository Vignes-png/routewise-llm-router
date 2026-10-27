"""Semantic task classification for RouteWise.

The category comes from embedding similarity (all-MiniLM-L6-v2, run locally).
The model tier still comes from the rule-based complexity policy in app.routing.

The embedding model is loaded lazily on first use and then cached, so importing
this module (or the FastAPI app) never downloads or loads any model.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.routing import assess_complexity, select_tier
from app.schemas import ModelTier, SemanticRouteResponse, TaskCategory

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# How many of a category's best-matching examples are averaged into its score.
TOP_K = 2

# If even the best category scores below this, we do not trust the match and
# fall back to "general". This is an untuned starting value: calibrate it on
# your benchmark prompts.
MIN_SIMILARITY = 0.25

CATEGORY_EXAMPLES: dict[str, tuple[str, ...]] = {
    "coding": (
        "Write a Python function that reverses a string.",
        "Explain what a Python list is.",
        "Why does my JavaScript code throw a TypeError?",
        "Debug this stack trace for me.",
        "How do I write a SQL query that joins two tables?",
        "Refactor this class to make it more readable.",
        "What is the time complexity of this algorithm?",
        "Fix the race condition in my multithreaded program.",
    ),
    "reasoning": (
        "Prove that the square root of 2 is irrational.",
        "If all A are B and some B are C, what follows?",
        "Analyze the trade-offs between renting and buying a home.",
        "Solve this logic puzzle step by step.",
        "Why did the Roman Empire collapse? Weigh the competing explanations.",
        "Compare the pros and cons of two strategies and justify a recommendation.",
        "Work through this probability problem carefully.",
        "Evaluate whether this argument is valid.",
    ),
    "summarization": (
        "Summarize this article in three sentences.",
        "Give me the key points of this document.",
        "Condense this paragraph into a short summary.",
        "TL;DR of this email thread, please.",
        "Write a brief recap of these meeting notes.",
        "Shorten this text while keeping the main ideas.",
        "Summarize the findings of this research paper.",
        "Extract the main takeaways from this report.",
    ),
    "creative_writing": (
        "Write a poem about autumn.",
        "Tell me a short story about a dragon who is afraid of fire.",
        "Compose song lyrics about summer road trips.",
        "Write a haiku about the ocean.",
        "Create a fictional character with a tragic backstory.",
        "Write the opening scene of a mystery novel.",
        "Draft a funny limerick about a cat.",
        "Write a short screenplay dialogue between two old friends.",
    ),
    "general": (
        "What is the capital of France?",
        "Tell me a fun fact about octopuses.",
        "How tall is Mount Everest?",
        "What's a good way to stay motivated?",
        "Recommend a healthy breakfast.",
        "Who wrote Pride and Prejudice?",
        "How do I boil an egg?",
        "What is the difference between weather and climate?",
    ),
}


class SemanticModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded (not installed, offline, ...)."""


@dataclass(frozen=True)
class SemanticClassification:
    category: TaskCategory       # final category (may be the "general" fallback)
    similarity_score: float      # aggregated score of the best-matching category
    best_match: str              # best-matching category before any fallback
    low_confidence: bool         # True if the score was below MIN_SIMILARITY
    all_scores: dict[str, float]


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once, on first use, then reuse it."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # not installed, offline on first run, corrupt cache...
        raise SemanticModelError(
            f"Could not load embedding model '{MODEL_NAME}'. Check that "
            "sentence-transformers is installed and that the model can be "
            "downloaded on first use."
        ) from exc


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts into L2-normalised vectors (so dot product == cosine similarity)."""
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype=float)


@lru_cache(maxsize=1)
def get_category_embeddings() -> dict[str, np.ndarray]:
    """Embed every category's examples once and cache the result."""
    return {
        category: encode(list(examples))
        for category, examples in CATEGORY_EXAMPLES.items()
    }


def aggregate_similarity(similarities: np.ndarray, top_k: int = TOP_K) -> float:
    """Collapse one category's per-example similarities into a single score.

    Mean of the top_k best examples: one lucky example cannot decide alone (max),
    and unrelated examples do not drag the score down (mean over all).
    """
    k = min(top_k, len(similarities))
    return float(np.sort(similarities)[-k:].mean())


def score_categories(prompt: str) -> dict[str, float]:
    """Return the aggregated cosine similarity between the prompt and each category."""
    prompt_vector = encode([prompt])[0]
    return {
        category: aggregate_similarity(example_vectors @ prompt_vector)
        for category, example_vectors in get_category_embeddings().items()
    }


def classify_semantic(prompt: str) -> SemanticClassification:
    """Pick the best-matching category, falling back to 'general' if the match is weak."""
    scores = score_categories(prompt)
    best = max(scores, key=scores.get)  # ties resolve to the first category listed
    low_confidence = scores[best] < MIN_SIMILARITY
    return SemanticClassification(
        category="general" if low_confidence else best,  # type: ignore[arg-type]
        similarity_score=scores[best],
        best_match=best,
        low_confidence=low_confidence,
        all_scores=scores,
    )


def build_semantic_reason(
    classification: SemanticClassification,
    tier: ModelTier,
    complexity_signals: list[str],
) -> str:
    """Explain the decision using only values that were actually computed."""
    ranked = sorted(classification.all_scores.items(), key=lambda kv: kv[1], reverse=True)
    if classification.low_confidence:
        category_part = (
            f"Best semantic match was '{classification.best_match}' "
            f"(similarity {classification.similarity_score:.2f}), below the "
            f"{MIN_SIMILARITY:.2f} threshold, so the category falls back to general."
        )
    else:
        runner_up_name, runner_up_score = ranked[1]
        category_part = (
            f"Semantically closest to '{classification.category}' examples "
            f"(similarity {classification.similarity_score:.2f}; "
            f"runner-up: {runner_up_name} {runner_up_score:.2f})."
        )
    if tier == "strong":
        tier_part = f"Complexity signals: {', '.join(complexity_signals[:3])}."
    else:
        tier_part = "No complexity signals found."
    return f"{category_part} {tier_part}"


def route_prompt_semantic(prompt: str) -> SemanticRouteResponse:
    """Semantic category + existing rule-based complexity policy for the tier."""
    classification = classify_semantic(prompt)
    is_complex, signals = assess_complexity(prompt)
    tier = select_tier(is_complex)
    return SemanticRouteResponse(
        task_category=classification.category,
        model_tier=tier,
        routing_method="semantic_plus_rules",
        similarity_score=round(classification.similarity_score, 4),
        reason=build_semantic_reason(classification, tier, signals),
    )