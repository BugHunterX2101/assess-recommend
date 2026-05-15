"""Tests for POST /chat schema validation."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


VALID_CATALOG_URLS = {
    "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
    "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-reasoning/",
}

MOCK_AGENT_RESPONSE = {
    "reply": "Here are my recommendations for a Java developer:",
    "recommendations": [
        {
            "name": "Java 8 (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
            "test_type": ["Knowledge & Skills"],
        }
    ],
    "end_of_conversation": False,
}


@pytest.fixture
def client():
    # Ensure agent.agent submodule is loaded before patch() tries to resolve it
    import agent.agent  # noqa: F401

    with (
        patch("retrieval.vector_store.load"),
        patch("retrieval.vector_store.get_all_urls", return_value=VALID_CATALOG_URLS),
        patch("retrieval.retriever.retrieve", return_value=[]),
        patch("agent.agent.run", return_value=MOCK_AGENT_RESPONSE),
        patch("openai.OpenAI"),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestChatSchema:
    def test_valid_request_returns_200(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "I need a Java developer assessment."}]},
        )
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "I need an assessment."}]},
        )
        data = resp.json()
        assert "reply" in data
        assert "recommendations" in data
        assert "end_of_conversation" in data

    def test_reply_is_non_empty_string(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "I need an assessment."}]},
        )
        data = resp.json()
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

    def test_recommendations_is_list(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Java developer."}]},
        )
        data = resp.json()
        assert isinstance(data["recommendations"], list)

    def test_recommendation_item_schema(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Java developer, 3 years."}]},
        )
        data = resp.json()
        for rec in data["recommendations"]:
            assert "name" in rec
            assert "url" in rec
            assert "test_type" in rec
            assert isinstance(rec["test_type"], list)

    def test_end_of_conversation_is_bool(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Java developer."}]},
        )
        data = resp.json()
        assert isinstance(data["end_of_conversation"], bool)

    def test_empty_messages_returns_422(self, client):
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code == 422

    def test_missing_messages_key_returns_422(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_invalid_role_returns_422(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "system", "content": "Hi"}]},
        )
        assert resp.status_code == 422

    def test_first_message_must_be_user(self, client):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "assistant", "content": "Hello"}]},
        )
        assert resp.status_code == 422

    def test_too_many_turns_returns_400(self, client):
        messages = []
        for i in range(9):
            messages.append({"role": "user", "content": f"Question {i}"})
            if i < 8:
                messages.append({"role": "assistant", "content": f"Answer {i}"})
        resp = client.post("/chat", json={"messages": messages})
        assert resp.status_code in (400, 422)
