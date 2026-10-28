"""Model dispatch: send one prompt to one model through LiteLLM.

This file receives a model name and a prompt, calls the model,
and returns its answer or raises a safe DispatchError.
"""

import logging
import os
from dataclasses import dataclass

# Prevent LiteLLM from downloading its price/context map during import.
# This must be set before importing LiteLLM.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

from app.config import (
    REQUEST_TIMEOUT_SECONDS,
    get_gemini_api_key,
)
from app.schemas import TokenUsage

litellm.suppress_debug_info = True

logger = logging.getLogger("routewise.dispatcher")


class DispatchError(Exception):
    """A model-call error with a safe message for API callers.

    `category` is a short fixed label (for example "rate_limit") that is
    recorded in the observability database instead of any provider text.
    """

    def __init__(self, status_code: int, message: str, category: str = "provider_error"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.category = category


@dataclass(frozen=True)
class DispatchResult:
    """The answer returned by a model and its reported token usage."""

    answer: str
    usage: TokenUsage | None


def to_dispatch_error(exc: Exception) -> DispatchError:
    """Convert provider exceptions into safe HTTP errors."""

    if isinstance(
        exc,
        (litellm.AuthenticationError, litellm.PermissionDeniedError),
    ):
        return DispatchError(
            503,
            "The model provider rejected the server's API credentials.",
            "provider_auth",
        )

    if isinstance(exc, litellm.RateLimitError):
        return DispatchError(
            429,
            "The model provider's rate limit or quota was reached. Try again later.",
            "rate_limit",
        )

    if isinstance(exc, litellm.Timeout):
        return DispatchError(
            504,
            "The model provider did not respond in time.",
            "timeout",
        )

    if isinstance(exc, litellm.NotFoundError):
        return DispatchError(
            502,
            "The configured model was not found by the provider.",
            "model_not_found",
        )

    if isinstance(exc, litellm.BadRequestError):
        return DispatchError(
            502,
            "The model provider rejected the request.",
            "bad_request",
        )

    return DispatchError(
        502,
        "The model provider returned an error. Try again later.",
        "provider_error",
    )


def extract_usage(response) -> TokenUsage | None:
    """Extract token counts when the provider supplies them."""

    usage = getattr(response, "usage", None)

    if usage is None:
        return None

    counts = TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )

    if all(value is None for value in counts.model_dump().values()):
        return None

    return counts


def dispatch_to_model(model: str, prompt: str) -> DispatchResult:
    """Send a prompt to the selected model and return its answer."""

    api_key = get_gemini_api_key()

    if api_key is None:
        raise DispatchError(
            503,
            "GEMINI_API_KEY is not configured on the server.",
            "missing_credentials",
        )

    try:
        completion_kwargs = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "api_key": api_key,
            "timeout": REQUEST_TIMEOUT_SECONDS,
            "num_retries": 0,
        }

        # Reduce reasoning effort for the strong model to try
        # to reduce response time.
        if model == "gemini/gemini-3.6-flash":
            completion_kwargs["reasoning_effort"] = "low"

        response = litellm.completion(**completion_kwargs)

    except Exception as exc:
        # Log only the exception type. Provider messages may contain
        # sensitive request details, so do not expose the original error.
        logger.warning(
            "Model call failed (model=%s): %s",
            model,
            type(exc).__name__,
        )

        raise to_dispatch_error(exc) from None

    try:
        answer = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        answer = None

    if not isinstance(answer, str) or not answer.strip():
        raise DispatchError(
            502,
            "The model returned an empty response.",
            "empty_response",
        )

    return DispatchResult(
        answer=answer,
        usage=extract_usage(response),
    )