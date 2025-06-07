import os

from fastapi.testclient import TestClient

from tunedd_api.main import app

client = TestClient(app)

conversation_id = None

RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL")


def test_new_conversation():
    """
    Test new conversations endpoint
    """
    response = client.post("/conversations/")
    assert response.status_code == 200
    assert "conversation_id" in response.json()

    global conversation_id
    conversation_id = response.json()["conversation_id"]


def test_conversation():
    """
    Test conversations endpoint
    """
    message = "Say something smart"

    response = client.post(
        f"/conversations/{conversation_id}/message", params={"message": message}
    )
    assert response.status_code == 200

    data = response.json()

    # Check the main "response" box
    assert "response" in data

    # Check the "embedder" box
    assert "embedder" in data["response"]
    assert "meta" in data["response"]["embedder"]
    assert "model" in data["response"]["embedder"]["meta"]
    assert data["response"]["embedder"]["meta"]["model"] == RAG_EMBEDDING_MODEL

    # Check the "generator" box
    assert "generator" in data["response"]
    assert "replies" in data["response"]["generator"]
    # Check if replies is a list
    assert isinstance(data["response"]["generator"]["replies"], list)
    assert "meta" in data["response"]["generator"]
    # Check if meta is a list
    assert isinstance(data["response"]["generator"]["meta"], list)

    # Check the "generator.meta" box
    # Check if meta has at least one item
    assert len(data["response"]["generator"]["meta"]) > 0
    assert "model" in data["response"]["generator"]["meta"][0]
    assert "finish_reason" in data["response"]["generator"]["meta"][0]
    assert "usage" in data["response"]["generator"]["meta"][0]
    assert "completion_tokens" in data["response"]["generator"]["meta"][0]["usage"]

    # Check that the replies contains the question and answer
    assert "Answer:" in data["response"]["generator"]["replies"][0]
    assert "Question:" in data["response"]["generator"]["replies"][0]
