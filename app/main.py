import time

from fastapi import FastAPI, HTTPException

from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, dispatch_to_model
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
def chat(request: RouteRequest) -> ChatResponse:
    """Route the prompt (semantic + rules), call the chosen model, return its answer."""
    # Step 1: routing decision (same pipeline as /route/semantic).
    try:
        decision = route_prompt_semantic(request.prompt)
    except SemanticModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Step 2: tier -> model id (single mapping lives in app/config.py).
    model = MODEL_BY_TIER[decision.model_tier]

    # Step 3: call the model and time the call.
    started = time.perf_counter()
    try:
        result = dispatch_to_model(model, request.prompt)
    except DispatchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

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