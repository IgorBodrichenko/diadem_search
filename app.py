import os
import json
import re
import uuid
from typing import List, Dict, Any, Tuple, Optional

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
    Берём только metadata.text, без file/page. В промпте ок,
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
    # Slide 14: Build my confidence
    "build_confidence": {
        "title": "Build my confidence",
        "steps": [
            {"key": "company", "question": "Let’s start with your company. What about your company gives you a strong position here?"},
            {"key": "situation", "question": "Thanks. In one sentence, what is your primary goal for this conversation?"},
            {"key": "outcome", "question": "Good. If this goes well, what has *actually happened* by the end (e.g. approval, clear next steps, signed proposal)?"},
            {"key": "relationship", "question": "About the relationship: what do you know about the other person’s priorities or pressures?"},
            {"key": "myself", "question": "About you: what strengths or skills do you bring that will help you handle this well?"},
            {"key": "why_confident", "question": "Great. Now list 3–5 reasons you *should* feel confident going into this."},
            {"key": "summary", "question": "Do you want me to summarise your confidence plan in 5–7 bullet points you can read right before the call? (yes/no)"},
        ],
    },

    # Slide 34: Prepare for difficult behaviours
    "prepare_difficult_behaviours": {
        "title": "Prepare for difficult behaviours",
        "steps": [
            {"key": "scenario", "question": "What’s the situation—who are you speaking to, and what decision are you trying to influence? (1–2 sentences)"},
            {"key": "anticipate_tactics", "question": "What is the *first* difficult thing they are likely to say or do? Write it as a direct quote if you can."},
            {"key": "purpose", "question": "What do you think their purpose is with that move—pressure, delay, anchoring, saving face, something else?"},
            {"key": "response_bullet", "question": "Let’s craft your response. What’s the one key point you must hold your ground on? (One sentence)"},
            {"key": "move_on_air", "question": "Now write a short linking phrase to steer back on track (e.g., “That’s helpful—so to move this forward…”). What’s your version?"},
            {"key": "rehearse", "question": "Do you want a 2-turn rehearsal? (yes/no)"},
            {"key": "summary", "question": "Want the final ‘cheat sheet’ (their likely line → your bullet → your steer-back phrase) in a clean format? (yes/no)"},
        ],
    },
}

# ВАЖНО: модель НЕ задаёт вопросы — только коротко отражает ответ
SYSTEM_PROMPT_COACH = (
    "You are a professional negotiation coach.\n"
    "You are running a structured guided dialogue, but you must NOT choose the next step.\n"
    "\n"
    "Rules:\n"
    "- Output MUST be plain text.\n"
    "- Output MUST be 1–2 short sentences maximum.\n"
    "- Do NOT ask questions.\n"
    "- Do NOT include numbered lists or long explanations.\n"
    "- Do NOT mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Your job is ONLY to acknowledge/refine what the user just said.\n"
)

def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


def _default_state(mode: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "step_index": 0,
        "answers": {},  # key -> user answer
    }


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    return max(lo, min(hi, n))


def _extract_mode(payload: Dict[str, Any]) -> str:
    mode = _safe_str(payload.get("mode")) or "build_confidence"
    if mode not in TEMPLATES:
        mode = "build_confidence"
    return mode


def _get_step(mode: str, step_index: int) -> Optional[Dict[str, Any]]:
    steps = TEMPLATES[mode]["steps"]
    if step_index < 0 or step_index >= len(steps):
        return None
    return steps[step_index]


def _is_done(mode: str, step_index: int) -> bool:
    return _get_step(mode, step_index) is None


def _update_state_with_user_answer(mode: str, state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    steps = TEMPLATES[mode]["steps"]
    idx = _clamp_int(state.get("step_index"), 0, 0, max(len(steps) - 1, 0))

    step_key = steps[idx]["key"] if steps else "step"
    if user_input.strip():
        answers = state.get("answers") or {}
        answers[step_key] = user_input.strip()
        state["answers"] = answers

    state["step_index"] = idx + 1
    return state


def _retrieve_info_for_coach(mode: str, query: str, top_k: int) -> str:
    # background retrieval
    rag_query = f"{mode}: {query}"
    matches = get_matches(rag_query, top_k)
    return build_context(matches)


def _coach_llm_ack(mode: str, state: Dict[str, Any], user_input: str, information: str) -> str:
    """
    LLM делает только ACK/рефрейм последнего ответа (без вопросов).
    """
    tpl = TEMPLATES[mode]
    last_key = ""
    # последний заполненный ключ = тот, что только что сохранили
    # (он на шаг позади текущего step_index)
    last_idx = max(0, int(state.get("step_index", 0)) - 1)
    last_step = _get_step(mode, last_idx)
    if last_step:
        last_key = last_step.get("key", "")

    answers = state.get("answers") or {}
    last_value = answers.get(last_key, "")

    return (
        f"MODE: {mode}\n"
        f"TEMPLATE: {tpl['title']}\n"
        f"LAST_STEP_KEY: {last_key}\n"
        f"USER_INPUT: {user_input}\n"
        f"LAST_SAVED_VALUE: {last_value}\n\n"
        f"INFORMATION:\n{information}\n"
        "\n"
        "Write a short acknowledgement/refinement of the user's last answer (1–2 sentences)."
    )


def coach_turn(payload: Dict[str, Any], stream: bool = False):
    mode = _extract_mode(payload)

    query = _safe_str(payload.get("query")) or _safe_str(payload.get("user_input"))
    top_k = _clamp_int(payload.get("top_k"), TOP_K, 1, 30)

    state = payload.get("state")
    if not isinstance(state, dict):
        state = _default_state(mode)
    else:
        state["mode"] = mode
        if "answers" not in state or not isinstance(state["answers"], dict):
            state["answers"] = {}
        if "step_index" not in state:
            state["step_index"] = 0

    # 1) Если это старт и текста нет — просто отдаём первый вопрос (без LLM)
    if not query and int(state.get("step_index", 0)) == 0:
        first = _get_step(mode, 0)
        text = first["question"] if first else ""
        done = _is_done(mode, 0)
        return {"text": text, "state": state, "done": done} if not stream else (iter([text]), {"state": state, "done": done})

    # 2) Сохраняем ответ пользователя в текущий шаг и двигаем step_index вперёд
    state = _update_state_with_user_answer(mode, state, query)

    done = _is_done(mode, int(state.get("step_index", 0)))

    # 3) Генерим короткий ACK через LLM (без вопросов)
    information = _retrieve_info_for_coach(mode, query, top_k)
    llm_user_msg = _coach_llm_ack(mode, state, query, information)

    # 4) Следующий вопрос берём ЖЁСТКО из шаблона (а не из LLM)
    next_step = _get_step(mode, int(state.get("step_index", 0)))
    next_q = next_step["question"] if next_step else ""

    if not stream:
        resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COACH},
                {"role": "user", "content": llm_user_msg},
            ],
            temperature=0.2,
        )
        ack = (resp.choices[0].message.content or "").strip()
        # склеиваем: ACK + следующий вопрос
        out_text = (ack + "\n\n" + next_q).strip() if next_q else ack
        return {"text": out_text, "state": state, "done": done}

    def gen_text_chunks():
        stream_resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COACH},
                {"role": "user", "content": llm_user_msg},
            ],
            temperature=0.2,
            stream=True,
        )
        # сначала стримим ack
        for event in stream_resp:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
        # затем ДОКИДЫВАЕМ следующий вопрос (жёстко)
        if next_q:
            yield "\n\n" + next_q

    return gen_text_chunks(), {"state": state, "done": done}


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


# ---- простой text/plain stream (обычный RAG Q&A) ----
@app.post("/chat/stream")
def chat_stream(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)

    if not query:
        return StreamingResponse(iter([""]), media_type="text/plain; charset=utf-8")

    matches = get_matches(query, top_k)
    context = build_context(matches)
    user = f"QUESTION:\n{query}\n\nINFORMATION:\n{context}"

    def gen():
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
                yield delta

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


# ---- SSE stream (обычный RAG Q&A) ----
@app.post("/chat/sse")
def chat_sse(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)

    if not query:
        def empty_gen():
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(
            empty_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# =========================
# COACH ENDPOINTS
# =========================

@app.post("/coach/chat")
def coach_chat(payload: Dict = Body(...)):
    out = coach_turn(payload, stream=False)
    return JSONResponse(out)


@app.post("/coach/stream")
def coach_stream(payload: Dict = Body(...)):
    chunks, _meta = coach_turn(payload, stream=True)
    return StreamingResponse(chunks, media_type="text/plain; charset=utf-8")


# ---- Coach SSE stream: session_id в event:start, text в event:chunk, done пустой ----
@app.post("/coach/sse")
def coach_sse(payload: Dict = Body(...)):
    session_id = (payload.get("session_id") or "").strip()
    if not session_id:
        session_id = uuid.uuid4().hex

    chunks, _meta = coach_turn(payload, stream=True)

    def gen():
        start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: start\ndata: {start_payload}\n\n"

        for delta in chunks:
            if not delta:
                continue
            data = json.dumps({"text": delta}, ensure_ascii=False)
            yield f"event: chunk\ndata: {data}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
