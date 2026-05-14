"""Tests for GET /health endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    """Create a test client with mocked vector store and LLM client."""
    with (
        patch("retrieval.vector_store.load"),
        patch("retrieval.vector_store.get_all_urls", return_value=set()),
        patch("openai.OpenAI"),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.json() == {"status": "ok"}


def test_health_content_type(client):
    resp = client.get("/health")
    assert "application/json" in resp.headers["content-type"]
