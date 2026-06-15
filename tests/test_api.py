"""Tests for API schemas and route wiring."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient

from api.dependencies import get_retrieval_server
from api.main import app
from api.schemas import SearchRequest


class _DummyResponse:
    def __init__(self, hits):
        self.hits = hits


class _DummyServer:
    def search_text_to_image(self, *, query: str, top_k: int | None = None):
        if not query:
            raise ValueError("query must be non-empty")
        return _DummyResponse(
            hits=[
                {
                    "rank": 1,
                    "item_id": "img_1.jpg",
                    "score": 0.99,
                    "metadata": {"source": "dummy"},
                }
            ]
        )


def test_search_request_requires_non_empty_query() -> None:
    with pytest.raises(Exception):
        SearchRequest(query="", top_k=5)


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_text_to_image_route_with_dependency_override() -> None:
    app.dependency_overrides[get_retrieval_server] = lambda: _DummyServer()
    try:
        client = TestClient(app)
        response = client.post("/search/text-to-image", json={"query": "a red car", "top_k": 3})
        assert response.status_code == 200
        payload = response.json()
        assert "hits" in payload
        assert payload["hits"][0]["item_id"] == "img_1.jpg"
    finally:
        app.dependency_overrides.clear()
