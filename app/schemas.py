from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_PROMPT_LENGTH = 4000

TaskCategory = Literal[
    "coding", "reasoning", "summarization", "creative_writing", "general"
]
ModelTier = Literal["cheap", "strong"]


class RouteRequest(BaseModel):
    """Incoming request body for POST /route and POST /route/semantic."""

    # Strip leading/trailing whitespace BEFORE length checks, so "   " is rejected.
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="The user prompt to classify and route.",
        examples=["Explain what a Python list is."],
    )


class RouteResponse(BaseModel):
    """Structured routing decision returned by POST /route (unchanged)."""

    task_category: TaskCategory
    model_tier: ModelTier
    routing_method: Literal["rule_based"] = "rule_based"
    reason: str


class SemanticRouteResponse(BaseModel):
    """Routing decision returned by POST /route/semantic.

    A separate model (not a subclass of RouteResponse) so the /route contract
    stays exactly as it was.
    """

    task_category: TaskCategory
    model_tier: ModelTier
    routing_method: Literal["semantic_plus_rules"] = "semantic_plus_rules"
    similarity_score: float = Field(
        ...,
        description="Aggregated cosine similarity of the best category (not a probability).",
    )
    reason: str