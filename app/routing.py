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

# --- Complexity policy -----------------------------------------------------
# Chooses the model tier from the prompt text ONLY. It never looks at the task
# category, so the rule-based and the semantic router share it unchanged.
#
# A single word never makes a prompt "strong", and neither does length. A prompt
# is strong only when a demanding REQUEST meets a context that makes it hard:
#   1. a technical task (debug / design / optimise ...) in a hard-systems context
#   2. a proof or derivation with an explicit demand for rigor or steps

# What the user asks us to DO (diagnose, repair, design, optimise, weigh options).
TECHNICAL_TASK_CUES: tuple[str, ...] = (
    "debug", "diagnos", "troubleshoot", "root cause", "causing", "cause of",
    "find the", "fix", "track down", "optimi", "design",
    "trade-off", "tradeoff", "trade off",
)

# Terms for hard-to-reason-about system behaviour (the topic, not the task).
HARD_SYSTEMS_CUES: tuple[str, ...] = (
    "race condition", "deadlock", "memory leak", "concurren", "multithread",
    "distributed", "scalab", "at scale", "latency", "throughput", "in production",
    "single point of failure", "high availability", "fault toleran",
    "performance bottleneck",
)

PROOF_CUES: tuple[str, ...] = ("prove", "derive", "proof")
RIGOR_CUES: tuple[str, ...] = ("step by step", "rigorous", "formally")


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
    """Return (is_complex, signals). Each rule needs TWO ingredients; one cue is never enough."""
    signals: list[str] = []

    tasks = find_matches(prompt, TECHNICAL_TASK_CUES)
    systems = find_matches(prompt, HARD_SYSTEMS_CUES)
    if tasks and systems:
        signals += ["technical task in a hard-systems context", tasks[0], systems[0]]

    proofs = find_matches(prompt, PROOF_CUES)
    rigor = find_matches(prompt, RIGOR_CUES)
    if proofs and rigor:
        signals += ["proof or derivation with explicit rigor", proofs[0], rigor[0]]

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