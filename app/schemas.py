
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MAX_PROMPT_LENGTH = 4000

TaskCategory = Literal[
    "coding",
    "reasoning",
    "summarization",
    "creative_writing",
    "general",
]

ModelTier = Literal["cheap", "strong"]

ChatMode = Literal["auto", "fast", "quality"]

ChatRoutingMethod = Literal[
    "semantic_plus_rules",
    "manual_fast",
    "manual_quality",
]


class RouteRequest(BaseModel):
    """Incoming request body for POST /route and POST /route/semantic."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="The user prompt to classify and route.",
        examples=["Explain what a Python list is."],
    )


class ChatRequest(BaseModel):
    """Incoming request body for POST /chat."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="The user prompt to route and answer.",
        examples=["Explain what a Python list is."],
    )

    mode: ChatMode = Field(
        default="auto",
        description=(
            "Operating mode: auto (semantic + rules), "
            "fast (cheap tier), or quality (strong tier)."
        ),
    )


class RouteResponse(BaseModel):
    """Structured routing decision returned by POST /route."""

    task_category: TaskCategory
    model_tier: ModelTier
    routing_method: Literal["rule_based"] = "rule_based"
    reason: str


class SemanticRouteResponse(BaseModel):
    """Routing decision returned by POST /route/semantic."""

    task_category: TaskCategory
    model_tier: ModelTier
    routing_method: Literal["semantic_plus_rules"] = "semantic_plus_rules"

    similarity_score: float = Field(
        ...,
        description=(
            "Aggregated cosine similarity of the best category "
            "(not a probability)."
        ),
    )

    reason: str


class TokenUsage(BaseModel):
    """Token counts reported by the provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    """
    Response of POST /chat.

    The response body deliberately preserves the original contract.
    The selected mode can be identified through routing_method.
    """

    answer: str

    model: str = Field(
        ...,
        description="LiteLLM ID of the model that produced the answer.",
    )

    task_category: TaskCategory | None

    model_tier: ModelTier

    routing_method: ChatRoutingMethod = "semantic_plus_rules"

    similarity_score: float | None = None

    reason: str

    latency_ms: int = Field(
        ...,
        description="Wall-clock time of the model call only.",
    )

    usage: TokenUsage | None = Field(
        None,
        description="Token usage, or null when the provider did not supply it.",
    )
