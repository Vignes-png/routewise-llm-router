"""Rule-based routing baseline for RouteWise.

Transparent keyword/heuristic rules only: no external services, no ML.
Each function is small and pure so it can be unit-tested on its own.
"""

import re

from app.schemas import ModelTier, RouteResponse, TaskCategory

# Category rules are checked in this order; the first category with a match wins.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "coding": (
        "python", "javascript", "typescript", "java", "c++", "sql", "html", "css",
        "code", "function", "class", "script", "api", "bug", "debug", "compile",
        "stack trace", "exception", "regex", "algorithm", "race condition",
        "deadlock", "refactor", "list", "dictionary", "array",
    ),
    "summarization": (
        "summarize", "summarise", "summary", "tl;dr", "tldr", "condense",
        "shorten", "key points", "main points", "recap",
    ),
    "creative_writing": (
        "poem", "story", "song", "lyrics", "haiku", "novel", "screenplay",
        "creative", "fictional", "write a limerick",
    ),
    "reasoning": (
        "why", "prove", "proof", "analyze", "analyse", "compare", "trade-off",
        "tradeoff", "trade off", "step by step", "logic", "puzzle", "deduce",
        "evaluate", "implications", "pros and cons",
    ),
}

# Phrases that signal a hard task and push the request to the strong tier.
COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "race condition", "deadlock", "concurrency", "concurrent", "multithreaded",
    "memory leak", "distributed", "architecture", "system design", "scalab",
    "optimize", "optimise", "performance bottleneck", "prove", "proof",
    "step by step", "trade-off", "tradeoff", "trade off", "complex",
    "multi-step", "formal", "rigorous", "root cause", "security vulnerability",
)

# Prompts at least this long are treated as likely complex.
LONG_PROMPT_CHARS = 1500


def find_matches(text: str, keywords: tuple[str, ...]) -> list[str]:
    """Return keywords found in text, matched at word starts (case-insensitive)."""
    lowered = text.lower()
    return [
        kw
        for kw in keywords
        if re.search(r"(?<![a-z0-9])" + re.escape(kw), lowered)
    ]


def classify_task(prompt: str) -> tuple[TaskCategory, list[str]]:
    """Pick a task category and return it with the keywords that triggered it."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = find_matches(prompt, keywords)
        if matches:
            return category, matches  # type: ignore[return-value]
    return "general", []


def assess_complexity(prompt: str) -> tuple[bool, list[str]]:
    """Return (is_complex, signals) using keyword and length heuristics."""
    signals = find_matches(prompt, COMPLEXITY_KEYWORDS)
    if len(prompt) >= LONG_PROMPT_CHARS:
        signals.append(f"long prompt (>= {LONG_PROMPT_CHARS} characters)")
    return bool(signals), signals


def select_tier(is_complex: bool) -> ModelTier:
    """Map the complexity decision to a model tier."""
    return "strong" if is_complex else "cheap"


def build_reason(
    category: TaskCategory,
    category_matches: list[str],
    tier: ModelTier,
    complexity_signals: list[str],
) -> str:
    """Build a short human-readable explanation of the decision."""
    if category_matches:
        category_part = f"Matched {category} keywords: {', '.join(category_matches[:3])}."
    else:
        category_part = "No specific task keywords matched, so category is general."
    if tier == "strong":
        tier_part = f"Complexity signals: {', '.join(complexity_signals[:3])}."
    else:
        tier_part = "No complexity signals found."
    return f"{category_part} {tier_part}"


def route_prompt(prompt: str) -> RouteResponse:
    """Run the full rule-based routing pipeline for one prompt."""
    category, category_matches = classify_task(prompt)
    is_complex, signals = assess_complexity(prompt)
    tier = select_tier(is_complex)
    return RouteResponse(
        task_category=category,
        model_tier=tier,
        routing_method="rule_based",
        reason=build_reason(category, category_matches, tier, signals),
    )