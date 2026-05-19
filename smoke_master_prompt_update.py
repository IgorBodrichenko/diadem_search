import os
from types import SimpleNamespace

# Minimal env for importing app.py locally without real keys
os.environ.setdefault("OPENAI_API_KEY", "dummy-openai")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-anthropic")
os.environ.setdefault("PINECONE_API_KEY", "dummy-pinecone")
os.environ.setdefault("PINECONE_INDEX_NAME", "dummy-index")
os.environ.setdefault("PINECONE_HOST", "https://example.com")

from fastapi.testclient import TestClient
import app


def fake_get_matches(query, top_k, request_id=None):
    return [
        {
            "id": "master_negotiator_slides:p1:c0",
            "score": 0.9,
            "metadata": {
                "type": "text",
                "file": "Master Negotiator Slides.pdf",
                "source": "Master Negotiator Slides.pdf",
                "text": "Use ABC model: Awareness, Balanced playing field, Confidence.",
            },
        }
    ]


state = {
    "create_system_ok": False,
    "stream_system_ok": False,
}


def fake_create(*, model, messages, max_tokens, system, temperature):
    has_v12 = "Version 1.2" in system
    has_part2 = "PART 2" in system and "OFFER SLIDES AND MATERIALS" in system
    state["create_system_ok"] = has_v12 and has_part2
    text = "Use the ABC model: Awareness, Balanced playing field, then project Confidence."
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def fake_stream_text(messages, model, temperature=0.2):
    system_text = ""
    if messages and isinstance(messages[0], dict):
        system_text = str(messages[0].get("content") or "")
    has_v12 = "Version 1.2" in system_text
    has_part2 = "PART 2" in system_text and "OFFER SLIDES AND MATERIALS" in system_text
    state["stream_system_ok"] = has_v12 and has_part2
    yield "Use the ABC model and anchor commercial control in your opening position."


def main():
    loaded = app._load_master_system_prompt_text()
    file_has_v12 = "Version 1.2" in loaded
    file_has_part2 = "PART 2" in loaded and "OFFER SLIDES AND MATERIALS" in loaded

    app.get_matches = fake_get_matches
    app.anthropod.messages.create = fake_create
    app._openai_stream_text = fake_stream_text

    client = TestClient(app.app)

    payload = {
        "user_message": "How do I handle buyer pressure in this negotiation?",
        "active_section_id": "my_list",
        "focus_field": "variable_name",
        "user_name": "Igor",
        "help_accepted": True,
    }

    r1 = client.post("/master/template", json=payload)
    r2 = client.post("/master/template/sse", json=payload)

    txt1 = r1.json().get("text", "") if r1.status_code == 200 else ""
    sse_text = r2.text if r2.status_code == 200 else ""

    ok = True
    ok = ok and file_has_v12 and file_has_part2
    ok = ok and (r1.status_code == 200)
    ok = ok and ("ABC" in txt1)
    ok = ok and state["create_system_ok"]
    ok = ok and (r2.status_code == 200)
    ok = ok and ("event: chunk" in sse_text) and ("event: done" in sse_text)
    ok = ok and state["stream_system_ok"]

    print("FILE_HAS_V12", file_has_v12)
    print("FILE_HAS_PART2", file_has_part2)
    print("MASTER_STATUS", r1.status_code)
    print("MASTER_TEXT_PREVIEW", txt1[:180].replace("\n", " "))
    print("MASTER_SYSTEM_USED", state["create_system_ok"])
    print("MASTER_SSE_STATUS", r2.status_code)
    print("MASTER_SSE_HAS_CHUNK", "event: chunk" in sse_text)
    print("MASTER_SSE_HAS_DONE", "event: done" in sse_text)
    print("MASTER_SSE_SYSTEM_USED", state["stream_system_ok"])
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
