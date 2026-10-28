import time

from fastapi import FastAPI, HTTPException, Response

from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, dispatch_to_model
from app.observability import ChatEvent, record_chat_event
from app.pricing import PRICING_CHECKED_ON, estimate_cost_usd
from app.routing import route_prompt
from app.schemas import (
    ChatResponse,
    RouteRequest,
    RouteResponse,
    SemanticRouteResponse,
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
    """Rule-based routing baseline: classify the prompt and pick a model tier."""
    return route_prompt(request.prompt)


@app.post("/route/semantic", response_model=SemanticRouteResponse)
def route_semantic(request: RouteRequest) -> SemanticRouteResponse:
    """Semantic category + rule-based tier. The embedding model loads on first use."""
    try:
        return route_prompt_semantic(request.prompt)
    except SemanticModelError as exc:
        # Controlled error: no traceback is sent to the client.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: RouteRequest, response: Response) -> ChatResponse:
    """Route the prompt (semantic + rules), call the chosen model, return its answer.

    Every request that reaches this function is recorded in SQLite (metadata
    only) in the `finally` block, whether it succeeded or failed. The request
    id is also returned in the X-Request-ID header so you can find the row.
    """
    # The event starts out as "unexpected failure" and is updated as we go.
    event = ChatEvent()
    id_header = {"X-Request-ID": event.request_id}
    response.headers["X-Request-ID"] = event.request_id

    try:
        # Step 1: routing decision (same pipeline as /route/semantic).
        try:
            decision = route_prompt_semantic(request.prompt)
        except SemanticModelError as exc:
            event.error_category = "routing_model_unavailable"
            event.http_status = 503
            raise HTTPException(status_code=503, detail=str(exc), headers=id_header) from exc
        except Exception:
            event.error_category = "routing_error"
            raise  # FastAPI turns this into a plain 500; the row is written below

        event.routing_method = decision.routing_method
        event.task_category = decision.task_category
        event.similarity_score = decision.similarity_score
        event.model_tier = decision.model_tier

        # Step 2: tier -> model id (single mapping lives in app/config.py).
        model = MODEL_BY_TIER[decision.model_tier]
        event.model = model

        # Step 3: call the model and time the call.
        started = time.perf_counter()
        try:
            result = dispatch_to_model(model, request.prompt)
        except DispatchError as exc:
            event.latency_ms = round((time.perf_counter() - started) * 1000)
            event.error_category = exc.category
            event.http_status = exc.status_code
            raise HTTPException(
                status_code=exc.status_code, detail=exc.message, headers=id_header
            ) from exc
        except Exception:
            event.latency_ms = round((time.perf_counter() - started) * 1000)
            raise  # error_category stays "unexpected_error"
        latency_ms = round((time.perf_counter() - started) * 1000)

        # Step 4: success. Record usage only if the provider supplied it.
        event.status = "success"
        event.error_category = None
        event.http_status = 200
        event.latency_ms = latency_ms
        if result.usage is not None:
            event.prompt_tokens = result.usage.prompt_tokens
            event.completion_tokens = result.usage.completion_tokens
            event.total_tokens = result.usage.total_tokens
            event.estimated_cost_usd = estimate_cost_usd(
                model, result.usage.prompt_tokens, result.usage.completion_tokens
            )
            if event.estimated_cost_usd is not None:
                event.pricing_checked_on = PRICING_CHECKED_ON

        return ChatResponse(
            answer=result.answer,
            model=model,
            task_category=decision.task_category,
            model_tier=decision.model_tier,
            routing_method=decision.routing_method,
            similarity_score=decision.similarity_score,
            reason=decision.reason,
            latency_ms=latency_ms,
            usage=result.usage,
        )
    finally:
        # Runs on success AND on every failure. record_chat_event never raises,
        # so a database problem cannot change the response the caller gets.
        record_chat_event(event)