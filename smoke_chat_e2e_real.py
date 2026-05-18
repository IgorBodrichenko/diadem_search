import os
import json
from fastapi.testclient import TestClient

REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_HOST",
]


def _masked(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        return "<missing>"
    if len(v) <= 8:
        return "*" * len(v)
    return v[:4] + "..." + v[-4:]


def main() -> int:
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    print("ENV_CHECK")
    for k in REQUIRED_ENV:
        print(f"  {k}: {_masked(k)}")

    if missing:
        print("RESULT FAIL")
        print("REASON Missing required env vars:", ", ".join(missing))
        return 2

    # Import only after env check because app.py validates env at import-time
    import app

    client = TestClient(app.app)

    payload = {
        "query": "What Diadem framework should I use first for a tough buyer pushback?",
        "top_k": 10,
        "user_name": "Igor",
        "history": [
            {"role": "user", "content": "I need Diadem materials for frameworks and slides"},
            {"role": "assistant", "content": "Let's use the MASTER approach"},
        ],
    }

    # /chat
    r_chat = client.post("/chat", json=payload)
    chat_ok = r_chat.status_code == 200
    chat_answer = ""
    if chat_ok:
        body = r_chat.json()
        chat_answer = (body.get("answer") or "").strip()

    # /chat/sse
    r_sse = client.post("/chat/sse", json=payload)
    sse_ok = r_sse.status_code == 200
    sse_text = r_sse.text if sse_ok else ""

    # Heuristics for regression reported by clients
    bad_phrase = "don't have that material available right now"
    chat_has_bad = bad_phrase.lower() in chat_answer.lower()
    sse_has_chunk = "event: chunk" in sse_text
    sse_has_done = "event: done" in sse_text

    print("CHAT_STATUS", r_chat.status_code)
    print("CHAT_LEN", len(chat_answer))
    print("CHAT_PREVIEW", chat_answer[:240].replace("\n", " "))

    print("SSE_STATUS", r_sse.status_code)
    print("SSE_HAS_CHUNK", sse_has_chunk)
    print("SSE_HAS_DONE", sse_has_done)

    # Soft pass criteria: endpoints work and no known regression phrase in /chat
    ok = chat_ok and sse_ok and sse_has_chunk and sse_has_done and not chat_has_bad
    print("RESULT", "PASS" if ok else "FAIL")
    if chat_has_bad:
        print("REASON Regression phrase detected in /chat answer")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
