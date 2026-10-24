from fastapi.testclient import TestClient

from app.main import app
from app.schemas import MAX_PROMPT_LENGTH

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["project"] == "RouteWise"
    assert data["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_route_simple_prompt_is_cheap():
    response = client.post("/route", json={"prompt": "Explain what a Python list is."})

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"task_category", "model_tier", "routing_method", "reason"}
    assert data["task_category"] == "coding"
    assert data["model_tier"] == "cheap"
    assert data["routing_method"] == "rule_based"
    assert data["reason"]


def test_route_complex_prompt_is_strong():
    response = client.post(
        "/route",
        json={"prompt": "Help me debug a race condition in my multithreaded Python service."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task_category"] == "coding"
    assert data["model_tier"] == "strong"
    assert data["routing_method"] == "rule_based"


def test_route_complex_reasoning_prompt_is_strong():
    response = client.post(
        "/route",
        json={"prompt": "Prove step by step that the square root of 2 is irrational."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task_category"] == "reasoning"
    assert data["model_tier"] == "strong"


def test_route_empty_prompt_rejected():
    response = client.post("/route", json={"prompt": ""})
    assert response.status_code == 422


def test_route_whitespace_only_prompt_rejected():
    response = client.post("/route", json={"prompt": "    "})
    assert response.status_code == 422


def test_route_missing_prompt_rejected():
    response = client.post("/route", json={})
    assert response.status_code == 422


def test_route_non_string_prompt_rejected():
    response = client.post("/route", json={"prompt": 123})
    assert response.status_code == 422


def test_route_too_long_prompt_rejected():
    response = client.post("/route", json={"prompt": "a" * (MAX_PROMPT_LENGTH + 1)})
    assert response.status_code == 422


def test_route_max_length_prompt_accepted():
    response = client.post("/route", json={"prompt": "a" * MAX_PROMPT_LENGTH})
    assert response.status_code == 200