from fastapi import FastAPI, HTTPException

from app.routing import route_prompt
from app.schemas import (
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