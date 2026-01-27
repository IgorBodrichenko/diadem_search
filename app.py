import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from openai import OpenAI
from pinecone import Pinecone

# =========================
# CONFIG
# =========================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))

# in-memory session store (Variant B)
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h default

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_HOST = os.getenv("PINECONE_HOST")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME missing")
if not PINECONE_HOST:
    raise RuntimeError("PINECONE_HOST missing")

openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом ограничишь доменом Bubble
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SIMPLE SESSION STORE (in-memory)
# =========================
# session_id -> {"state": {...}, "updated_at": epoch}
SESSIONS: Dict[str, Dict[str, Any]] = {}


def _now() -> int:
    return int(time.time())


def _cleanup_sessions():
    """Lazy cleanup on each request to avoid growing forever."""
    cutoff = _now() - SESSION_TTL_SECONDS
    to_delete = []
    for sid, entry in list(SESSIONS.items()):
        if int(entry.get("updated_at", 0)) < cutoff:
            to_delete.append(sid)
    for sid in to_delete:
        SESSIONS.pop(sid, None)


def _get_or_create_session_id(payload: Dict[str, Any]) -> str:
    sid = str(payload.get("session_id") or "").strip()
    if not sid:
        sid = uuid.uuid4().hex
    return sid


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    return max(lo, min(hi, n))


# =========================
# RAG HELPERS
# =========================
def embed_query(text: str) -> List[float]:
    resp = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        dimensions=EMBED_DIM,
    )
    return resp.data[0].embedding


def build_context(matches: List[Dict]) -> str:
    """
    Берём только metadata.text, без file/page. Нам в промпте это ок,
    но модель НЕ должна это упоминать пользователю.
    """
    parts: List[str] = []
    total = 0

    for m in matches:
        md = (m.get("metadata") or {})
        text = (md.get("text") or "").strip()
        if not text:
            continue

        snippet = text
        if len(snippet) > 2500:
            snippet = snippet[:2500] + "…"

        block = snippet + "\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)


def get_matches(query: str, top_k: int) -> List[Dict]:
    qvec = embed_query(query)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    return res.get("matches") or []


# =========================
# BASE (RAG Q&A) PROMPT
# =========================
SYSTEM_PROMPT_QA = (
    "You are a helpful assistant answering ONLY using the provided information.\n"
    "Rules:\n"
    "- Do NOT mention document names, page numbers, sources, citations, or the word 'context'.\n"
    "- Write a natural chatbot answer as plain text.\n"
    "- If the answer is not present, say: \"I can't find this in the provided documents.\".\n"
    "- Be concise and practical.\n"
)

# =========================
# COACH / TEMPLATE ENGINE
# =========================
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "build_confidence": {
        "title": "Build my confidence",
        "steps": [
            {"key": "company", "question": "Let’s start with your company. What about your company gives you a strong position here?"},
            {"key": "situation", "question": "Now the situation itself. What’s the most important thing you want to achieve in this conversation?"},
            {"key": "relationship", "question": "About the relationship: what do you know about the other person’s priorities or pressures?"},
            {"key": "myself", "question": "About you: what strengths or skills do you bring that will help you handle this well?"},
            {"key": "why_confident", "question": "Great. Now list 3–5 reasons you *should* feel confident going into this."},
            {"key": "summary", "question": "Want me to summarise your confidence plan in 5–7 bullet points you can read right before the call?"},
        ],
    },

    "prepare_difficult_behaviours": {
        "title": "Prepare for difficult behaviours",
        "steps": [
            {"key": "scenario", "question": "What’s the situation—who are you speaking to, and what decision are you trying to influence? (1–2 sentences)"},
            {"key": "anticipate_tactics", "question": "What is the *first* difficult thing they are likely to say or do? Write it as a direct quote if you can."},
            {"key": "purpose", "question": "What do you think their purpose is with that move—pressure, delay, anchoring, saving face, something else?"},
            {"key": "response_bullet", "question": "Let’s craft your response. What’s the one key point you must hold your ground on? (One sentence)"},
            {"key": "move_on_air", "question": "Now write a short linking phrase to steer back on track (e.g., “That’s helpful—so to move this forward…”). What’s your version?"},
            {"key": "rehearse", "question": "Do you want a 2-turn rehearsal? I’ll play them once, you reply, then I’ll improve your wording."},
            {"key": "summary", "question": "Want the final ‘cheat sheet’ (their likely line → your bullet → your steer-back phrase) in a clean format?"},
        ],
    },
}

SYSTEM_PROMPT_COACH = (
    "You are a professional negotiation coach running a structured guided dialogue.\n"
    "You must follow the selected TEMPLATE and step-by-step flow.\n"
    "Rules:\n"
    "- Keep it interactive: ask ONE clear question or give ONE short instruction at a time.\n"
    "- If the user answers, briefly reflect it in 1–2 lines, then move to the next step.\n"
    "- Do NOT lecture. Do NOT dump long explanations.\n"
    "- NEVER repeat a previous question unless the user explicitly asks you to repeat.\n"
    "- IMPORTANT: The next question must be EXACTLY the provided 'NEXT QUESTION' line. Ask it verbatim and stop.\n"
    "- Use the provided INFORMATION only as background support for phrasing and best-practice, but NEVER mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Output plain text only.\n"
)


def _extract_mode(payload: Dict[str, Any]) -> str:
    mode = _safe_str(payload.get("mode")) or "build_confidence"
    if mode not in TEMPLATES:
        mode = "build_confidence"
    return mode


def _default_state(mode: str) -> Dict[str, Any]:
    return {"mode": mode, "step_index": 0, "answers": {}}


def _load_state(session_id: str, mode: str) -> Dict[str, Any]:
    """Load from server memory; if missing, create new."""
    entry = SESSIONS.get(session_id)
    if not entry or not isinstance(entry.get("state"), dict):
        st = _default_state(mode)
        SESSIONS[session_id] = {"state": st, "updated_at": _now()}
        return st

    st = entry["state"]
    # if user changed shortcut/mode, reset state
    if st.get("mode") != mode:
        st = _default_state(mode)
        SESSIONS[session_id] = {"state": st, "updated_at": _now()}
        return st

    # ensure fields exist
    if "answers" not in st or not isinstance(st["answers"], dict):
        st["answers"] = {}
    if "step_index" not in st:
        st["step_index"] = 0
    st["mode"] = mode
    return st


def _save_state(session_id: str, state: Dict[str, Any]) -> None:
    SESSIONS[session_id] = {"state": state, "updated_at": _now()}


def _steps(mode: str) -> List[Dict[str, Any]]:
    return TEMPLATES[mode]["steps"]


def _current_step(mode: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps = _steps(mode)
    idx = _clamp_int(state.get("step_index"), 0, 0, len(steps))
    if idx >= len(steps):
        return None
    return steps[idx]


def _advance_with_answer(mode: str, state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    steps = _steps(mode)
    idx = _clamp_int(state.get("step_index"), 0, 0, max(len(steps) - 1, 0))

    if user_input.strip() and steps:
        key = steps[idx]["key"]
        state["answers"][key] = user_input.strip()

    state["step_index"] = idx + 1
    return state


def _retrieve_info_for_coach(mode: str, query: str, top_k: int) -> str:
    rag_query = f"{mode}: {query}"
    matches = get_matches(rag_query, top_k)
    return build_context(matches)


def _make_coach_user_message(mode: str, state: Dict[str, Any], user_input: str, info: str) -> str:
    tpl = TEMPLATES[mode]
    steps = tpl["steps"]
    idx = _clamp_int(state.get("step_index"), 0, 0, len(steps))

    next_step = steps[idx] if idx < len(steps) else None
    next_q = next_step["question"] if next_step else "Summarise the plan."

    answers_lines = []
    for i in range(min(idx, len(steps))):
        k = steps[i]["key"]
        if k in state["answers"]:
            answers_lines.append(f"- {k}: {state['answers'][k]}")

    answers_block = "\n".join(answers_lines) if answers_lines else "(none yet)"

    return (
        f"TEMPLATE: {tpl['title']} ({mode})\n"
        f"STEP_INDEX (0-based): {idx}\n\n"
        f"LAST USER MESSAGE:\n{user_input.strip()}\n\n"
        f"ANSWERS SO FAR:\n{answers_block}\n\n"
        f"NEXT QUESTION (must ask exactly this, then stop):\n{next_q}\n\n"
        f"INFORMATION (background support):\n{info}\n"
    )


def coach_turn_server_state(payload: Dict[str, Any], session_id: str, stream: bool = False):
    mode = _extract_mode(payload)
    query = _safe_str(payload.get("query")) or _safe_str(payload.get("user_input"))
    top_k = _clamp_int(payload.get("top_k"), TOP_K, 1, 30)
    reset = bool(payload.get("reset"))

    _cleanup_sessions()

    if reset:
        SESSIONS.pop(session_id, None)

    state = _load_state(session_id, mode)

    # If first call and no query: ask first question without advancing
    if not query and int(state.get("step_index", 0)) == 0:
        info = _retrieve_info_for_coach(mode, f"template {mode}", top_k)
        user_msg = _make_coach_user_message(mode, state, "", info)
    else:
        state = _advance_with_answer(mode, state, query)
        info = _retrieve_info_for_coach(mode, query, top_k)
        user_msg = _make_coach_user_message(mode, state, query, info)

    done = _current_step(mode, state) is None

    if done:
        user_msg += "\nFINAL INSTRUCTION:\nSummarise the user’s plan in a clean, practical format and offer the next action."

    _save_state(session_id, state)

    if not stream:
        resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COACH},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"text": text, "session_id": session_id, "done": done}

    def gen_text_chunks():
        stream_resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COACH},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            stream=True,
        )
        for event in stream_resp:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    return gen_text_chunks(), {"session_id": session_id, "done": done}


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"ok": True}


# ---- JSON chat (обычный RAG Q&A) ----
@app.post("/chat")
def chat(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)

    if not query:
        return JSONResponse({"answer": ""})

    matches = get_matches(query, top_k)
    context = build_context(matches)

    user = f"QUESTION:\n{query}\n\nINFORMATION:\n{context}"

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_QA},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer = (resp.choices[0].message.content or "").strip()
    return {"answer": answer}


# ---- SSE chat (обычный RAG Q&A) ----
@app.post("/chat/sse")
def chat_sse(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)

    def headers():
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

    if not query:
        def empty_gen():
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(
            empty_gen(),
            media_type="text/event-stream",
            headers=headers(),
        )

    matches = get_matches(query, top_k)
    context = build_context(matches)

    user = f"QUESTION:\n{query}\n\nINFORMATION:\n{context}"

    def gen():
        yield "event: start\ndata: {}\n\n"

        stream = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QA},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            stream=True,
        )

        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                data = json.dumps({"text": delta}, ensure_ascii=False)
                yield f"event: chunk\ndata: {data}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=headers(),
    )


# =========================
# COACH ENDPOINTS (server-stored sessions)
# =========================

# ---- Coach JSON ----
@app.post("/coach/chat")
def coach_chat(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    out = coach_turn_server_state(payload, session_id=session_id, stream=False)
    return JSONResponse(out)


# ---- Coach SSE stream: session_id в event:start, text в event:chunk, done в event:done ----
@app.post("/coach/sse")
def coach_sse(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    chunks, meta = coach_turn_server_state(payload, session_id=session_id, stream=True)

    def headers():
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

    def gen():
        start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: start\ndata: {start_payload}\n\n"

        for delta in chunks:
            data = json.dumps({"text": delta}, ensure_ascii=False)
            yield f"event: chunk\ndata: {data}\n\n"

        done_payload = json.dumps({"done": bool(meta.get("done"))}, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=headers(),
    )


# ---- Optional: reset session explicitly ----
@app.post("/coach/reset")
def coach_reset(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    SESSIONS.pop(session_id, None)
    return JSONResponse({"ok": True, "session_id": session_id})
