import json
import os
from types import SimpleNamespace

# Minimal env for importing app.py in local smoke mode
os.environ.setdefault("OPENAI_API_KEY", "dummy-openai")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-anthropic")
os.environ.setdefault("PINECONE_API_KEY", "dummy-pinecone")
os.environ.setdefault("PINECONE_INDEX_NAME", "dummy-index")
os.environ.setdefault("PINECONE_HOST", "https://example.com")

from fastapi.testclient import TestClient
import app


def fake_get_matches(query, top_k, request_id=None):
    # Mimic successful retrieval with Diadem-specific content
    return [
        {
            "id": "master-negotiator-slides:p1:c0",
            "score": 0.91,
            "metadata": {
                "type": "text",
                "file_name": "Master Negotiator Slides.pdf",
                "doc_id": "master_negotiator_slides",
                "priority": 2,
                "text": "MASTER framework: ambition, variables, trading strategy, and steering phrases.",
            },
        }
    ]


def fake_create(*, model, messages, max_tokens, system, temperature):
    # Validate that /chat now sends conversation context and information
    user_payload = messages[0]["content"] if messages else ""
    has_info = "INFORMATION:" in user_payload and "MASTER framework" in user_payload
    has_conv = "CONVERSATION_CONTEXT:" in user_payload

    text = (
        "Use the MASTER framework with ambition, variable mapping, and a steer-back line."
        if has_info
        else "I don't have that material available right now."
    )

    if not has_conv:
        text = "Conversation context missing."

    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def fake_stream_text(messages, model, temperature=0.2):
    # Deterministic SSE text without external API calls
    yield "Use the MASTER framework with ambition, variable mapping, and a steer-back line."


def main():
    # Monkeypatch external dependencies for deterministic smoke test
    app.get_matches = fake_get_matches
    app.anthropod.messages.create = fake_create
    app._openai_stream_text = fake_stream_text

    client = TestClient(app.app)

    payload = {
        "query": "What Diadem framework should I use first?",
        "top_k": 10,
        "user_name": "Igor",
        "history": [
            {"role": "user", "content": "Need framework for buyer pressure"},
            {"role": "assistant", "content": "Let's use MASTER structure"},
        ],
    }

    r1 = client.post("/chat", json=payload)
    r1.raise_for_status()
    body1 = r1.json()

    r2 = client.post("/chat/sse", json=payload)
    r2.raise_for_status()
    sse_text = r2.text

    print("CHAT_STATUS", r1.status_code)
    print("CHAT_ANSWER", body1.get("answer", ""))
    print("SSE_STATUS", r2.status_code)
    print("SSE_HAS_CHUNK", "event: chunk" in sse_text)
    print("SSE_HAS_DONE", "event: done" in sse_text)
    print("SSE_HAS_MASTER", "MASTER framework" in sse_text)

    ok = True
    ok = ok and ("MASTER framework" in body1.get("answer", ""))
    ok = ok and ("event: chunk" in sse_text)
    ok = ok and ("event: done" in sse_text)
    ok = ok and ("MASTER framework" in sse_text)

    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
