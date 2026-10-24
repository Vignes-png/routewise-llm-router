from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_PROMPT_LENGTH = 4000

TaskCategory = Literal[
    "coding", "reasoning", "summarization", "creative_writing", "general"
]
ModelTier = Literal["cheap", "strong"]


class RouteRequest(BaseModel):
    """Incoming request body for POST /route."""

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
    """Structured routing decision returned by POST /route."""

    task_category: TaskCategory
    model_tier: ModelTier
    routing_method: Literal["rule_based"] = "rule_based"
    reason: str