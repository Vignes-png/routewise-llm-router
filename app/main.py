from fastapi import FastAPI

from app.routing import route_prompt
from app.schemas import RouteRequest, RouteResponse

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