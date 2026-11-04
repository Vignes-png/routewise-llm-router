
import time

from fastapi import FastAPI, HTTPException, Response

from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, dispatch_to_model
from app.observability import ChatEvent, record_chat_event
from app.pricing import PRICING_CHECKED_ON, estimate_cost_usd
from app.routing import route_prompt
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatRoutingMethod,
    ModelTier,
    RouteRequest,
    RouteResponse,
    SemanticRouteResponse,
    TaskCategory,
)
from app.semantic import SemanticModelError, route_prompt_semantic


app = FastAPI(
    title="RouteWise API",
    description="An intelligent LLM routing application.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "RouteWise",
        "status": "running",
        "message": "The RouteWise API is working!",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    """Rule-based routing baseline."""
    return route_prompt(request.prompt)


@app.post("/route/semantic", response_model=SemanticRouteResponse)
def route_semantic(request: RouteRequest) -> SemanticRouteResponse:
    """Semantic category classification with rule-based tier selection."""

    try:
        return route_prompt_semantic(request.prompt)

    except SemanticModelError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, response: Response) -> ChatResponse:
    """
    Supports three operating modes:

    auto:
        Uses the existing semantic + rule-based routing system.

    fast:
        Directly selects the cheap model tier.

    quality:
        Directly selects the strong model tier.

    Every request is recorded in SQLite, including failures.
    """

    # Create an observability event for this request.
    event = ChatEvent()

    id_header = {"X-Request-ID": event.request_id}
    response.headers["X-Request-ID"] = event.request_id

    try:
        mode = request.mode

        # --------------------------------------------------
        # STEP 1: Select the operating mode
        # --------------------------------------------------

        if mode == "fast":

            # Manual fast mode bypasses semantic classification.
            task_category: TaskCategory | None = None
            similarity_score: float | None = None

            model_tier: ModelTier = "cheap"

            routing_method: ChatRoutingMethod = "manual_fast"

            reason = (
                "Manual fast mode selected; "
                "routed directly to the cheap model tier."
            )

        elif mode == "quality":

            # Manual quality mode bypasses semantic classification.
            task_category = None
            similarity_score = None

            model_tier = "strong"

            routing_method = "manual_quality"

            reason = (
                "Manual quality mode selected; "
                "routed directly to the strong model tier."
            )

        else:

            # Auto mode preserves the existing routing behaviour.
            try:
                decision = route_prompt_semantic(request.prompt)

            except SemanticModelError as exc:
                event.error_category = "routing_model_unavailable"
                event.http_status = 503

                raise HTTPException(
                    status_code=503,
                    detail=str(exc),
                    headers=id_header,
                ) from exc

            except Exception:
                event.error_category = "routing_error"
                raise

            task_category = decision.task_category
            similarity_score = decision.similarity_score
            model_tier = decision.model_tier
            routing_method = decision.routing_method
            reason = decision.reason

        # --------------------------------------------------
        # STEP 2: Record routing decision
        # --------------------------------------------------

        event.routing_method = routing_method
        event.task_category = task_category
        event.similarity_score = similarity_score
        event.model_tier = model_tier

        # Map selected tier to actual model ID.
        model = MODEL_BY_TIER[model_tier]

        event.model = model

        # --------------------------------------------------
        # STEP 3: Dispatch prompt to selected model
        # --------------------------------------------------

        started = time.perf_counter()

        try:
            result = dispatch_to_model(
                model,
                request.prompt,
            )

        except DispatchError as exc:

            event.latency_ms = round(
                (time.perf_counter() - started) * 1000
            )

            event.error_category = exc.category
            event.http_status = exc.status_code

            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.message,
                headers=id_header,
            ) from exc

        except Exception:
            event.latency_ms = round(
                (time.perf_counter() - started) * 1000
            )
            raise

        latency_ms = round(
            (time.perf_counter() - started) * 1000
        )

        # --------------------------------------------------
        # STEP 4: Record successful request metadata
        # --------------------------------------------------

        event.status = "success"
        event.error_category = None
        event.http_status = 200
        event.latency_ms = latency_ms

        if result.usage is not None:

            event.prompt_tokens = result.usage.prompt_tokens
            event.completion_tokens = result.usage.completion_tokens
            event.total_tokens = result.usage.total_tokens

            event.estimated_cost_usd = estimate_cost_usd(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
            )

            if event.estimated_cost_usd is not None:
                event.pricing_checked_on = PRICING_CHECKED_ON

        # --------------------------------------------------
        # STEP 5: Return the original response structure
        # --------------------------------------------------

        return ChatResponse(
            answer=result.answer,
            model=model,
            task_category=task_category,
            model_tier=model_tier,
            routing_method=routing_method,
            similarity_score=similarity_score,
            reason=reason,
            latency_ms=latency_ms,
            usage=result.usage,
        )

    finally:
        # Record every request, whether successful or unsuccessful.
        # Observability errors must never break the API response.
        record_chat_event(event)
