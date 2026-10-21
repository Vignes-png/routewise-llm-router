
from fastapi import FastAPI

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
