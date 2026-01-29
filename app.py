import os
import json
import uuid
import time
import random
import re
from typing import List, Dict, Any, Optional, Iterator, Tuple

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

TOP_K = int(os.getenv("TOP_K", "10"))  # final chunks passed to LLM
PINECONE_TOPK_RAW = int(os.getenv("PINECONE_TOPK_RAW", "30"))  # per-query raw candidates
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h default

# Retrieval quality knobs
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.35"))
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "700"))
MIN_OVERLAP_SCORE = float(os.getenv("MIN_OVERLAP_SCORE", "1.3"))  # if below -> treat as not found

# extra: retrieval diversity
MULTI_QUERY_K = int(os.getenv("MULTI_QUERY_K", "3"))  # number of pinecone queries to run (max 3 in this impl)
DIVERSITY_SAME_SOURCE_CAP = int(os.getenv("DIVERSITY_SAME_SOURCE_CAP", "3"))  # max chunks from same source

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SIMPLE SESSION STORE (in-memory)
# =========================
SESSIONS: Dict[str, Dict[str, Any]] = {}


def _now() -> int:
    return int(time.time())


def _cleanup_sessions():
    cutoff = _now() - SESSION_TTL_SECONDS
    for sid, entry in list(SESSIONS.items()):
        if int(entry.get("updated_at", 0)) < cutoff:
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
# USER NAME (optional)
# =========================
def _extract_user_name(payload: Dict[str, Any]) -> str:
    raw = _safe_str(payload.get("user_name")) or _safe_str(payload.get("name"))
    return raw[:40].strip()


# =========================
# TEXT CLEANUP (NO MARKDOWN)
# =========================
def strip_markdown_chars(text: str) -> str:
    if not text:
        return ""
    return text.replace("*", "").replace("`", "").replace("_", "")


# =========================
# SMALL TALK / CASUAL CHAT
# =========================
_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|good\s*(morning|afternoon|evening)|"
    r"привет|привіт|здарова|здраст[вуйте]*|добрий\s*(день|ранок|вечір)|"
    r"как\s*дела|як\s*справи|що\s*нового|thanks|thank\s*you)\s*[!.?]*\s*$",
    flags=re.IGNORECASE,
)


def _is_smalltalk(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    if len(q) <= 30 and _SMALLTALK_RE.match(q):
        return True
    return False


def _smalltalk_reply(user_name: str) -> str:
    name = (user_name or "").strip()
    if name:
        return f"Hi {name}. How can I help?"
    return "Hi. How can I help?"


# =========================
# VARIATION (avoid same opener every time)
# =========================
SOFT_OPENERS = [
    "Glad you’re thinking about this ahead of time.",
    "That makes sense — getting prepared early helps a lot.",
    "Good call to tackle this before the meeting.",
    "Nice — planning this now will make the conversation easier.",
    "Totally doable. Let’s get you set up for it.",
    "Okay, let’s make this straightforward and calm.",
    "Makes sense. Let’s work through it step by step.",
    "Alright — we can make this feel a lot more manageable.",
]

_BAD_START_RE = re.compile(r"^\s*(it['’]s\s+(great|wonderful)|great)\b", flags=re.IGNORECASE)


def _session_entry(session_id: str) -> Dict[str, Any]:
    entry = SESSIONS.get(session_id)
    if not isinstance(entry, dict):
        entry = {}
        SESSIONS[session_id] = entry
    entry["updated_at"] = _now()
    return entry


def _pick_opener(session_id: str, user_name: str, field: str) -> str:
    entry = _session_entry(session_id)
    last = _safe_str(entry.get(field))
    options = [o for o in SOFT_OPENERS if o != last] or SOFT_OPENERS[:]
    opener = random.choice(options)
    if user_name and random.random() < 0.55:
        opener = f"{opener} {user_name}."
    entry[field] = opener
    return opener


def _rewrite_bad_opening(full_text: str, opener: str) -> str:
    t = (full_text or "").strip()
    if not t:
        return ""
    if not _BAD_START_RE.match(t):
        return t
    m = re.search(r"[.!?]\s+", t)
    if m:
        rest = t[m.end():].strip()
        return f"{opener} {rest}".strip() if rest else opener
    return opener


def _stream_opening_variation(
    deltas: Iterator[str],
    session_id: str,
    user_name: str,
    field: str,
) -> Iterator[str]:
    opener = _pick_opener(session_id, user_name, field)
    buf = ""
    decided = False
    for d in deltas:
        if not decided:
            buf += d
            if len(buf) >= 140 or re.search(r"[.!?]\s+", buf):
                yield _rewrite_bad_opening(buf, opener)
                decided = True
                buf = ""
        else:
            yield d
    if not decided and buf:
        yield _rewrite_bad_opening(buf, opener)


# =========================
# SEARCH HINTS (Most common questions)
# =========================
def _norm_q(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


SEARCH_HINTS = [
    {"match_any": ["balance the power in a negotiation", "balance power in a negotiation", "balance the power"],
     "hint": "confident mindset pages 9-11"},
    {"match_any": ["push back without upsetting the relationship", "push back without upsetting", "push back"],
     "hint": "tactics prepared to respond pages 65-74 slides 7 8 15"},
    {"match_any": ["difference between selling and negotiation", "selling vs negotiation", "selling and negotiation"],
     "hint": "introduction to negotiation book page xii-vii slide 11"},
    {"match_any": ["deal with difficult questions", "handle difficult questions", "difficult questions", "deal with difficult question"],
     "hint": "slides 28-34 difficult questions"},
]


def _hint_for_question(question: str) -> str:
    qn = _norm_q(question)
    for item in SEARCH_HINTS:
        for key in item["match_any"]:
            if key in qn:
                return item["hint"]
    return ""


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


def _filter_matches_by_score(matches: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for m in matches or []:
        try:
            score = float(m.get("score") or 0.0)
        except Exception:
            score = 0.0
        if score >= MIN_MATCH_SCORE:
            out.append(m)
    return out


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "without", "is", "are", "was", "were", "be",
    "do", "does", "did", "how", "what", "why", "when", "where", "between", "into", "from", "as", "at", "by", "it", "this", "that",
}

_GENERIC_PHRASES = [
    "key techniques to consider",
    "negotiation techniques",
    "emotional intelligence",
    "win-win outcomes",
    "face-to-face negotiations",
    "self-awareness is crucial",
    "prepare thoroughly",
]


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS and len(t) > 2]
    return toks


def _keyword_query(query: str) -> str:
    toks = _tokenize(query)
    toks = toks[:18]
    return " ".join(toks)


def _source_id(md: Dict[str, Any]) -> str:
    for k in ["source", "doc_id", "document_id", "file", "filename", "title"]:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ["chunk_id", "id"]:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _merge_dedup_matches(list_of_lists: List[List[Dict]]) -> List[Dict]:
    best: Dict[str, Dict] = {}
    for matches in list_of_lists:
        for m in matches or []:
            mid = str(m.get("id") or "").strip()
            md = m.get("metadata") or {}
            txt = (md.get("text") or "").strip()
            if not mid:
                mid = f"txth:{hash(txt)}"
            if mid not in best:
                best[mid] = m
            else:
                try:
                    s_new = float(m.get("score") or 0.0)
                except Exception:
                    s_new = 0.0
                try:
                    s_old = float(best[mid].get("score") or 0.0)
                except Exception:
                    s_old = 0.0
                if s_new > s_old:
                    best[mid] = m
    return list(best.values())


def _rerank(query: str, matches: List[Dict], final_k: int) -> List[Dict]:
    qt = set(_tokenize(query))
    hint = _hint_for_question(query)
    hint_toks = set(_tokenize(hint)) if hint else set()

    scored: List[Tuple[float, Dict]] = []
    qlow = (query or "").lower()

    for m in matches or []:
        md = m.get("metadata") or {}
        text = (md.get("text") or "").strip()
        if not text:
            continue

        tlow = text.lower()
        ttoks = _tokenize(text)

        overlap_q = sum(1.0 for t in ttoks if t in qt)
        overlap_hint = sum(1.2 for t in ttoks if t in hint_toks) if hint_toks else 0.0

        bonus = 0.0
        for phrase in [
            "balance", "power", "selling", "negotiation",
            "difficult questions", "push back", "relationship",
            "tactics", "mindset",
        ]:
            if phrase in qlow and phrase in tlow:
                bonus += 0.7

        penalty = 0.0
        for gp in _GENERIC_PHRASES:
            if gp in tlow:
                penalty += 1.1

        try:
            pscore = float(m.get("score") or 0.0)
        except Exception:
            pscore = 0.0

        final = (overlap_q * 1.25) + (overlap_hint * 1.6) + bonus + (pscore * 0.25) - penalty
        scored.append((final, m))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict] = []
    per_source: Dict[str, int] = {}

    for _, m in scored:
        md = m.get("metadata") or {}
        sid = _source_id(md) or "_"
        per_source[sid] = per_source.get(sid, 0)
        if per_source[sid] >= DIVERSITY_SAME_SOURCE_CAP:
            continue
        out.append(m)
        per_source[sid] += 1
        if len(out) >= final_k:
            break

    return out


def build_context(matches: List[Dict]) -> str:
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


def is_context_relevant(query: str, matches: List[Dict]) -> bool:
    if not matches:
        return False

    top = matches[0]
    text = ((top.get("metadata") or {}).get("text") or "").strip()
    if not text:
        return False

    qt = set(_tokenize(query))
    tt = _tokenize(text)
    overlap = sum(1.0 for t in tt if t in qt)

    ctx = build_context(matches)
    if len(ctx.strip()) < MIN_CONTEXT_CHARS and overlap < MIN_OVERLAP_SCORE:
        return False

    return overlap >= MIN_OVERLAP_SCORE or len(ctx.strip()) >= MIN_CONTEXT_CHARS


def get_matches(query: str, top_k_final: int) -> List[Dict]:
    q_clean = (query or "").strip()
    if not q_clean:
        return []

    hint = _hint_for_question(q_clean)
    q_hint = f"{q_clean}\n{hint}".strip() if hint else ""
    q_kw = _keyword_query(q_clean)

    queries: List[str] = [q_clean]
    if hint and len(queries) < MULTI_QUERY_K:
        queries.append(q_hint)
    if len(queries) < MULTI_QUERY_K and q_kw:
        queries.append(q_kw)

    all_results: List[List[Dict]] = []

    for q in queries:
        try:
            vec = embed_query(q)
            res = index.query(vector=vec, top_k=PINECONE_TOPK_RAW, include_metadata=True)
            all_results.append(res.get("matches") or [])
        except Exception:
            continue

    merged = _merge_dedup_matches(all_results)
    merged = _filter_matches_by_score(merged)

    reranked = _rerank(q_clean, merged, top_k_final)

    if not is_context_relevant(q_clean, reranked):
        return []

    return reranked


# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT_QA = (
    "You are a friendly, helpful assistant.\n"
    "You must answer ONLY using the provided INFORMATION.\n\n"
    "Hard rules:\n"
    "- Do NOT mention document names, page numbers, sources, citations, or the word 'context'.\n"
    "- Output plain text only. NO markdown. Do not use *, **, _, `, #, or markdown lists.\n"
    "- Do NOT add general negotiation advice that is not explicitly supported by INFORMATION.\n"
    "- Your answer MUST include 2–4 short direct quotes from INFORMATION in double quotes (each quote 3–12 words).\n"
    "- If you cannot include those quotes because INFORMATION is missing or generic/not about the question, say exactly:\n"
    "  \"I can't find this in the provided documents.\".\n"
    "- Keep it concise and practical.\n\n"
    "Tone rules:\n"
    "- Start softly (one short supportive sentence) when appropriate.\n"
    "- Avoid starting with the same phrase every time.\n"
    "- If USER_NAME is provided, you MAY naturally mention it 0–2 times.\n"
    "- If you answered (not the 'can't find' case), end your message with one short question.\n"
)

SYSTEM_PROMPT_CHAT = (
    "You are a friendly assistant.\n"
    "If INFORMATION is provided and relevant, you should use it.\n"
    "If INFORMATION is empty or not relevant, you can still respond normally (do not say you can't find it).\n"
    "Rules:\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Keep it short and natural.\n"
    "- If USER_NAME is provided, greet/address them naturally.\n"
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
            {"key": "why_confident", "question": "Great. Now list 3–5 reasons you should feel confident going into this."},
            {"key": "summary", "question": "Want me to summarise your confidence plan in 5–7 bullet points you can read right before the call?"},
        ],
    },
    "prepare_difficult_behaviours": {
        "title": "Prepare for difficult behaviours",
        "steps": [
            {"key": "scenario", "question": "What’s the situation—who are you speaking to, and what decision are you trying to influence? (1–2 sentences)"},
            {"key": "anticipate_tactics", "question": "What is the first difficult thing they are likely to say or do? Write it as a direct quote if you can."},
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
    "- Use the provided INFORMATION only as background support, but NEVER mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Output plain text only. NO markdown.\n"
    "- Avoid starting with the same phrase every time.\n"
    "- If USER_NAME is provided, you MAY naturally mention it once.\n"
)

# ---- COACH FIXES START HERE ----

_YES_RE = re.compile(r"^\s*(yes|y|yeah|yep|sure|ok|okay|да|ага|ок|добре|звісно)\s*[!.?]*\s*$", re.IGNORECASE)


def _is_yes(s: str) -> bool:
    return bool(_YES_RE.match((s or "").strip()))


def _resolve_template_from_text(text: str) -> Optional[str]:
    q = (text or "").strip().lower()
    if not q:
        return None
    for mode_key, tpl in TEMPLATES.items():
        title = (tpl.get("title") or "").strip().lower()
        if q == title or q == mode_key:
            return mode_key
    return None


SYSTEM_PROMPT_COACH_FINAL = (
    "You are a professional negotiation coach.\n"
    "The guided dialogue is complete.\n"
    "Now produce the final output directly (do NOT ask the next question).\n"
    "Rules:\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Use the user's answers to produce a clean practical summary.\n"
    "- If a TEMPLATE expects a 'cheat sheet', format it as: likely line -> your response -> steer-back phrase.\n"
    "- Keep it concise but usable.\n"
    "- If USER_NAME is provided, you may mention it once.\n"
    "- End with one short optional next step question (e.g. rehearsal?).\n"
)


def _make_final_user_message(mode: str, state: Dict[str, Any], info: str, user_name: str) -> str:
    tpl = TEMPLATES[mode]
    steps = tpl["steps"]

    answers_lines = []
    for st in steps:
        k = st["key"]
        if k in state.get("answers", {}):
            answers_lines.append(f"- {k}: {state['answers'][k]}")
    answers_block = "\n".join(answers_lines) if answers_lines else "(none)"

    name_line = user_name if user_name else ""

    return (
        f"USER_NAME:\n{name_line}\n\n"
        f"TEMPLATE: {tpl['title']} ({mode})\n\n"
        f"ANSWERS:\n{answers_block}\n\n"
        f"INFORMATION (background support):\n{info}\n\n"
        f"FINAL TASK:\n"
        f"- If template is 'Build my confidence': summarise confidence plan in 5–7 short bullet points.\n"
        f"- If template is 'Prepare for difficult behaviours': create a final cheat sheet.\n"
    )


def _extract_mode(payload: Dict[str, Any]) -> str:
    mode = _safe_str(payload.get("mode")) or "build_confidence"
    if mode not in TEMPLATES:
        mode = "build_confidence"
    return mode


def _default_state(mode: str) -> Dict[str, Any]:
    return {"mode": mode, "step_index": 0, "answers": {}}


def _load_state(session_id: str, mode: str) -> Dict[str, Any]:
    entry = SESSIONS.get(session_id)
    if not entry or not isinstance(entry.get("state"), dict):
        st = _default_state(mode)
        SESSIONS[session_id] = {"state": st, "updated_at": _now()}
        return st

    st = entry["state"]
    if st.get("mode") != mode:
        st = _default_state(mode)
        entry["state"] = st
        entry["updated_at"] = _now()
        return st

    if "answers" not in st or not isinstance(st["answers"], dict):
        st["answers"] = {}
    if "step_index" not in st:
        st["step_index"] = 0

    st["mode"] = mode
    entry["updated_at"] = _now()
    return st


def _save_state(session_id: str, state: Dict[str, Any]) -> None:
    entry = _session_entry(session_id)
    entry["state"] = state
    entry["updated_at"] = _now()


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


def _make_coach_user_message(mode: str, state: Dict[str, Any], user_input: str, info: str, user_name: str) -> str:
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
    name_line = user_name if user_name else ""

    return (
        f"USER_NAME:\n{name_line}\n\n"
        f"TEMPLATE: {tpl['title']} ({mode})\n"
        f"STEP_INDEX (0-based): {idx}\n\n"
        f"LAST USER MESSAGE:\n{user_input.strip()}\n\n"
        f"ANSWERS SO FAR:\n{answers_block}\n\n"
        f"NEXT QUESTION (must ask exactly this, then stop):\n{next_q}\n\n"
        f"INFORMATION (background support):\n{info}\n"
    )


def _is_template_invocation_text(mode: str, query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    title = (TEMPLATES[mode]["title"] or "").strip().lower()
    mode_key = (mode or "").strip().lower()
    return q == title or q == mode_key


def coach_turn_server_state(payload: Dict[str, Any], session_id: str, stream: bool = False):
    raw_query = _safe_str(payload.get("query")) or _safe_str(payload.get("user_input"))
    top_k = _clamp_int(payload.get("top_k"), TOP_K, 1, 30)
    reset = bool(payload.get("reset"))
    user_name = _extract_user_name(payload)

    _cleanup_sessions()

    if _is_smalltalk(raw_query):
        text = _smalltalk_reply(user_name) + " Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        if not stream:
            return {"text": text, "session_id": session_id, "done": False}
        return iter([text]), {"session_id": session_id, "done": False}

    if reset:
        SESSIONS.pop(session_id, None)

    chosen_mode = _resolve_template_from_text(raw_query)
    if chosen_mode:
        SESSIONS.pop(session_id, None)
        mode = chosen_mode
        query = ""
    else:
        mode = _extract_mode(payload)
        query = raw_query

    state = _load_state(session_id, mode)

    if query and _is_template_invocation_text(mode, query):
        SESSIONS.pop(session_id, None)
        state = _load_state(session_id, mode)
        query = ""

    # start template (ask first question)
    if not query:
        info = _retrieve_info_for_coach(mode, f"template {mode}", top_k)
        user_msg = _make_coach_user_message(mode, state, "", info, user_name=user_name)
        done = False
    else:
        cur = _current_step(mode, state)
        cur_key = (cur or {}).get("key")

        state = _advance_with_answer(mode, state, query)

        # user said YES on summary step -> produce final answer NOW
        if cur_key == "summary" and _is_yes(query):
            state["step_index"] = len(_steps(mode))
            _save_state(session_id, state)

            info = _retrieve_info_for_coach(mode, "final summary", top_k)
            final_user = _make_final_user_message(mode, state, info, user_name=user_name)

            if not stream:
                resp = openai.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                        {"role": "user", "content": final_user},
                    ],
                    temperature=0.2,
                )
                text = strip_markdown_chars((resp.choices[0].message.content or "").strip())
                opener = _pick_opener(session_id, user_name, "coach_last_opener")
                text = _rewrite_bad_opening(text, opener)
                return {"text": text, "session_id": session_id, "done": True}

            def gen_final_chunks():
                stream_resp = openai.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                        {"role": "user", "content": final_user},
                    ],
                    temperature=0.2,
                    stream=True,
                )

                def raw_deltas():
                    for event in stream_resp:
                        delta = event.choices[0].delta.content
                        if delta:
                            yield strip_markdown_chars(delta)

                yield from _stream_opening_variation(
                    raw_deltas(),
                    session_id=session_id,
                    user_name=user_name,
                    field="coach_last_opener",
                )

            return gen_final_chunks(), {"session_id": session_id, "done": True}

        info = _retrieve_info_for_coach(mode, query, top_k)
        user_msg = _make_coach_user_message(mode, state, query, info, user_name=user_name)
        done = _current_step(mode, state) is None

    _save_state(session_id, state)

    # Edge case: steps already done -> final
    if done:
        info = _retrieve_info_for_coach(mode, "final summary", top_k)
        final_user = _make_final_user_message(mode, state, info, user_name=user_name)

        if not stream:
            resp = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                    {"role": "user", "content": final_user},
                ],
                temperature=0.2,
            )
            text = strip_markdown_chars((resp.choices[0].message.content or "").strip())
            opener = _pick_opener(session_id, user_name, "coach_last_opener")
            text = _rewrite_bad_opening(text, opener)
            return {"text": text, "session_id": session_id, "done": True}

        def gen_done_chunks():
            stream_resp = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                    {"role": "user", "content": final_user},
                ],
                temperature=0.2,
                stream=True,
            )

            def raw_deltas():
                for event in stream_resp:
                    delta = event.choices[0].delta.content
                    if delta:
                        yield strip_markdown_chars(delta)

            yield from _stream_opening_variation(
                raw_deltas(),
                session_id=session_id,
                user_name=user_name,
                field="coach_last_opener",
            )

        return gen_done_chunks(), {"session_id": session_id, "done": True}

    # normal coach turn: ask next question exactly
    if not stream:
        resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COACH},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        text = strip_markdown_chars((resp.choices[0].message.content or "").strip())
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = _rewrite_bad_opening(text, opener)
        return {"text": text, "session_id": session_id, "done": False}

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

        def raw_deltas():
            for event in stream_resp:
                delta = event.choices[0].delta.content
                if delta:
                    yield strip_markdown_chars(delta)

        yield from _stream_opening_variation(
            raw_deltas(),
            session_id=session_id,
            user_name=user_name,
            field="coach_last_opener",
        )

    return gen_text_chunks(), {"session_id": session_id, "done": False}

# ---- COACH FIXES END HERE ----

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"ok": True}


# =========================
# CHAT (UNCHANGED)
# =========================
@app.post("/chat")
def chat(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)
    user_name = _extract_user_name(payload)

    session_id = _get_or_create_session_id(payload)
    _cleanup_sessions()

    if not query:
        return JSONResponse({"answer": ""})

    if _is_smalltalk(query):
        return {"answer": _smalltalk_reply(user_name), "session_id": session_id}

    matches = get_matches(query, top_k)
    context = build_context(matches) if matches else ""

    if not matches:
        user = (
            f"USER_NAME:\n{user_name}\n\n"
            f"USER_MESSAGE:\n{query}\n\n"
            f"INFORMATION:\n"
        )
        resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CHAT},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        answer = strip_markdown_chars((resp.choices[0].message.content or "").strip())
        if answer:
            opener = _pick_opener(session_id, user_name, "qa_last_opener")
            answer = _rewrite_bad_opening(answer, opener)
        return {"answer": answer, "session_id": session_id}

    user = (
        f"USER_NAME:\n{user_name}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"INFORMATION:\n{context}"
    )

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_QA},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer = strip_markdown_chars((resp.choices[0].message.content or "").strip())

    if answer.strip() != "I can't find this in the provided documents.":
        opener = _pick_opener(session_id, user_name, "qa_last_opener")
        answer = _rewrite_bad_opening(answer, opener)

    return {"answer": answer, "session_id": session_id}


@app.post("/chat/sse")
def chat_sse(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)
    user_name = _extract_user_name(payload)

    session_id = _get_or_create_session_id(payload)
    _cleanup_sessions()

    def headers():
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

    if not query:
        def empty_gen():
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream", headers=headers())

    if _is_smalltalk(query):
        def gen_st():
            start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
            yield f"event: start\ndata: {start_payload}\n\n"
            data = json.dumps({"text": _smalltalk_reply(user_name)}, ensure_ascii=False)
            yield f"event: chunk\ndata: {data}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen_st(), media_type="text/event-stream", headers=headers())

    matches = get_matches(query, top_k)
    context = build_context(matches) if matches else ""

    if not matches:
        def gen_chat():
            start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
            yield f"event: start\ndata: {start_payload}\n\n"

            user = (
                f"USER_NAME:\n{user_name}\n\n"
                f"USER_MESSAGE:\n{query}\n\n"
                f"INFORMATION:\n"
            )

            stream = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CHAT},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                stream=True,
            )

            def raw_deltas():
                for event in stream:
                    delta = event.choices[0].delta.content
                    if delta:
                        yield strip_markdown_chars(delta)

            for delta in _stream_opening_variation(
                raw_deltas(),
                session_id=session_id,
                user_name=user_name,
                field="qa_last_opener",
            ):
                data = json.dumps({"text": delta}, ensure_ascii=False)
                yield f"event: chunk\ndata: {data}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(gen_chat(), media_type="text/event-stream", headers=headers())

    def gen():
        start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: start\ndata: {start_payload}\n\n"

        user = (
            f"USER_NAME:\n{user_name}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"INFORMATION:\n{context}"
        )

        stream = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QA},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            stream=True,
        )

        def raw_deltas():
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    yield strip_markdown_chars(delta)

        for delta in _stream_opening_variation(
            raw_deltas(),
            session_id=session_id,
            user_name=user_name,
            field="qa_last_opener",
        ):
            data = json.dumps({"text": delta}, ensure_ascii=False)
            yield f"event: chunk\ndata: {data}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers())


# =========================
# COACH ENDPOINTS
# =========================
@app.post("/coach/chat")
def coach_chat(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    out = coach_turn_server_state(payload, session_id=session_id, stream=False)
    return JSONResponse(out)


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

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers())


@app.post("/coach/reset")
def coach_reset(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    SESSIONS.pop(session_id, None)
    return JSONResponse({"ok": True, "session_id": session_id})
