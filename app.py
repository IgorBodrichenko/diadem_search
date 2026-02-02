import os
import json
import uuid
import time
import random
import re
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from openai import OpenAI
from pinecone import Pinecone

# =========================
# DEBUG / LOGGING
# =========================
DEBUG = os.getenv("DEBUG", "0").strip().lower() in ("1", "true", "yes", "y", "on")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("coach")


def _jlog(event: str, **fields):
    payload = {"event": event, **fields}
    try:
        log.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        log.info(f"{event} | {fields}")


def _with_debug(resp: Dict[str, Any], **dbg):
    if DEBUG:
        resp["debug"] = dbg
    return resp


# =========================
# CONFIG
# =========================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("TOP_K", "10"))
PINECONE_TOPK_RAW = int(os.getenv("PINECONE_TOPK_RAW", "30"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))

MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.35"))
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "700"))
MIN_OVERLAP_SCORE = float(os.getenv("MIN_OVERLAP_SCORE", "1.3"))

MULTI_QUERY_K = int(os.getenv("MULTI_QUERY_K", "3"))
DIVERSITY_SAME_SOURCE_CAP = int(os.getenv("DIVERSITY_SAME_SOURCE_CAP", "3"))

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
# REQUEST LOG (optional)
# =========================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if DEBUG:
        try:
            body = await request.body()
            preview = body.decode("utf-8", errors="ignore")[:800]
        except Exception:
            preview = ""
        _jlog("http_in", method=request.method, path=str(request.url.path), body_preview=preview)
    resp = await call_next(request)
    return resp


# =========================
# SESSION STORE (SQLite)
# =========================
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "sessions.sqlite3")
_DB: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        _DB = sqlite3.connect(SESSION_DB_PATH, check_same_thread=False)
        _DB.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              updated_at INTEGER NOT NULL,
              entry_json TEXT NOT NULL
            )
            """
        )
        _DB.commit()
    return _DB


def _now() -> int:
    return int(time.time())


def _db_get(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    row = _db().execute(
        "SELECT entry_json FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _db_set(session_id: str, entry: Dict[str, Any]) -> None:
    entry = entry or {}
    entry["updated_at"] = _now()
    payload = json.dumps(entry, ensure_ascii=False)
    _db().execute(
        """
        INSERT INTO sessions(session_id, updated_at, entry_json)
        VALUES(?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          updated_at=excluded.updated_at,
          entry_json=excluded.entry_json
        """,
        (session_id, int(entry["updated_at"]), payload),
    )
    _db().commit()


def _db_delete(session_id: str) -> None:
    _db().execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    _db().commit()


def _cleanup_sessions() -> None:
    cutoff = _now() - SESSION_TTL_SECONDS
    _db().execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
    _db().commit()


def _get_or_create_session_id(payload: Dict[str, Any]) -> str:
    sid = str(payload.get("session_id") or "").strip()
    if not sid:
        sid = uuid.uuid4().hex
    return sid


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off", "", "null", "none", "undefined"):
            return False
    return False


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    return max(lo, min(hi, n))


# =========================
# USER NAME / USER TEXT
# =========================
def _extract_user_name(payload: Dict[str, Any]) -> str:
    raw = _safe_str(payload.get("user_name")) or _safe_str(payload.get("name"))
    raw = raw[:40].strip()
    if raw.lower() in {"null", "none", "undefined"}:
        return ""
    return raw


_FALSEY_STRS = {"false", "true", "null", "none", "undefined", ""}


def _extract_user_text(payload: Dict[str, Any]) -> str:
    for key in ("query", "user_input", "text", "message", "input", "answer", "content"):
        v = payload.get(key)
        s = _safe_str(v)
        if s and s.lower() not in _FALSEY_STRS:
            return s
    return ""


# =========================
# TEXT CLEANUP (NO MARKDOWN)
# =========================
def strip_markdown_chars(text: str) -> str:
    if not text:
        return ""
    return text.replace("*", "").replace("`", "").replace("_", "")


# =========================
# SMALL TALK
# =========================
_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|good\s*(morning|afternoon|evening)|"
    r"привет|привіт|здарова|здраст[вуйте]*|добрий\s*(день|ранок|вечір)|"
    r"как\s*дела|як\s*справи|що\s*нового|thanks|thank\s*you)\s*[!.?]*\s*$",
    flags=re.IGNORECASE,
)


def _is_smalltalk(q: str) -> bool:
    q = (q or "").strip()
    return bool(q and len(q) <= 30 and _SMALLTALK_RE.match(q))


def _smalltalk_reply(user_name: str) -> str:
    name = (user_name or "").strip()
    if name:
        return f"Hi {name}. How can I help?"
    return "Hi. How can I help?"


# =========================
# VARIATION
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
    entry = _db_get(session_id)
    if not isinstance(entry, dict):
        entry = {}
    entry["updated_at"] = _now()
    _db_set(session_id, entry)
    return entry


def _pick_opener(session_id: str, user_name: str, field: str) -> str:
    entry = _session_entry(session_id)
    last = _safe_str(entry.get(field))
    options = [o for o in SOFT_OPENERS if o != last] or SOFT_OPENERS[:]
    opener = random.choice(options)
    if user_name and random.random() < 0.55:
        opener = f"{opener} {user_name}."
    entry[field] = opener
    _db_set(session_id, entry)
    return opener


def _rewrite_bad_opening(full_text: str, opener: str) -> str:
    t = (full_text or "").strip()
    if not t or not _BAD_START_RE.match(t):
        return t
    m = re.search(r"[.!?]\s+", t)
    if m:
        rest = t[m.end():].strip()
        return f"{opener} {rest}".strip() if rest else opener
    return opener


# =========================
# SEARCH HINTS
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
    toks = _tokenize(query)[:18]
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

        snippet = text[:2500] + "…" if len(text) > 2500 else text
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
# PROMPTS (updated to match PDF guardrails)
# =========================

VARIABLES_POLICY = (
    "\nTemplate Variables rules:\n"
    "- A Variable is a user-defined negotiation lever (e.g., price, term, volume, timing, scope, risk, concessions).\n"
    "- Variables exist only inside a specific template section; they are contextual, not global.\n"
    "- The user creates and edits Variables at all times.\n"
    "- You may suggest example Variables to consider.\n"
    "- You must never insert Variables, infer values, or overwrite existing Variables without explicit user confirmation.\n"
    "- Variables are part of the template structure, not your memory or internal state.\n"
)

TEMPLATE_FIRST_POLICY = (
    "\nTemplate-First Writing Contract:\n"
    "- Your output must map 1:1 to the template field(s) you are writing.\n"
    "- Prefer bullets / short lines. No narrative paragraphs unless the field explicitly requires them.\n"
    "- Output must be paste-ready, not advisory.\n"
)

ACTIVE_SECTION_POLICY = (
    "\nActive Section rules:\n"
    "- By default, write only for the currently active section/field.\n"
    "- If you are unsure which section is active, ask one short question to confirm.\n"
)

LIMITS_POLICY = (
    "\nInteraction limits:\n"
    "- Keep responses <= 150 words unless the user asks for more.\n"
    "- Ask at most 2 clarifying questions before producing a best-effort paste-ready output.\n"
)

SYSTEM_PROMPT_QA = (
    "You are a friendly, helpful assistant.\n"
    "You must answer ONLY using the provided INFORMATION.\n\n"
    "Hard rules:\n"
    "- Do NOT mention document names, page numbers, sources, citations, or the word 'context'.\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT add general negotiation advice that is not explicitly supported by INFORMATION.\n"
    "- Your answer MUST include 2–4 short direct quotes from INFORMATION in double quotes (each quote 3–12 words).\n"
    "- If you cannot include those quotes because INFORMATION is missing or generic/not about the question, say exactly:\n"
    "  \"I can't find this in the provided documents.\".\n"
    "- Keep it concise and practical.\n"
    "- Do not create or assume template variables unless they appear in INFORMATION.\n\n"
    "Tone rules:\n"
    "- Start softly (one short supportive sentence) when appropriate.\n"
    "- Avoid starting with the same phrase every time.\n"
    "- If USER_NAME is provided, you MAY naturally mention it 0–2 times.\n"
    "- If you answered (not the 'can't find' case), end your message with one short question.\n"
)

SYSTEM_PROMPT_CHAT = (
    "You are a friendly assistant.\n"
    "If INFORMATION is provided and relevant, you should use it.\n"
    "If INFORMATION is empty or not relevant, you can still respond normally.\n"
    "Rules:\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Keep it short and natural.\n"
    "- If USER_NAME is provided, greet/address them naturally.\n"
    + VARIABLES_POLICY
    + ACTIVE_SECTION_POLICY
    + LIMITS_POLICY
)

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
    "- End with one short optional next step question.\n"
    + VARIABLES_POLICY
    + TEMPLATE_FIRST_POLICY
    + ACTIVE_SECTION_POLICY
    + LIMITS_POLICY
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
            {"key": "summary", "question": "Want the final ‘cheat sheet’ (their likely line → your bullet → your steer-back phrase) in a clean format?"},
        ],
    },
}

# --- New: start again / confirm parsing ---
_START_AGAIN_RE = re.compile(
    r"^\s*(start again|restart|start over|reset|начать заново|почати заново|заново)\s*[!.?]*\s*$",
    flags=re.IGNORECASE,
)
_YES_RE = re.compile(r"^\s*(yes|y|ok|okay|confirm|confirmed|sure|да|ок|угу)\s*[!.?]*\s*$", flags=re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(no|n|nope|cancel|stop|нет|не)\s*[!.?]*\s*$", flags=re.IGNORECASE)


def _extract_mode(payload: Dict[str, Any], fallback_text: str = "") -> str:
    m = _safe_str(payload.get("mode"))
    if m in TEMPLATES:
        return m
    q = _norm_q(fallback_text)
    if "difficult" in q or "behav" in q:
        return "prepare_difficult_behaviours"
    if "confidence" in q:
        return "build_confidence"
    return "build_confidence"


def _default_state(mode: str) -> Dict[str, Any]:
    # variables are stored per active_section key (section-scoped)
    return {
        "mode": mode,
        "step_index": 0,
        "answers": {},
        "active_section": "",
        "variables": {},  # { section_key: [ {name, low, high, ideal, notes}, ... ] }
        "pending_write": None,  # {"key": ..., "value": ...}
        "awaiting_confirm": False,
        "clarify_count": 0,
    }


def _load_state(session_id: str, mode: str) -> Tuple[Dict[str, Any], bool]:
    entry = _db_get(session_id) or {}
    st = entry.get("state")

    if not isinstance(st, dict):
        st = _default_state(mode)
        entry["state"] = st
        _db_set(session_id, entry)
        return st, False

    if st.get("mode") != mode:
        st = _default_state(mode)
        entry["state"] = st
        _db_set(session_id, entry)
        return st, True

    if "answers" not in st or not isinstance(st["answers"], dict):
        st["answers"] = {}
    if "step_index" not in st:
        st["step_index"] = 0
    if "variables" not in st or not isinstance(st["variables"], dict):
        st["variables"] = {}
    if "active_section" not in st:
        st["active_section"] = ""
    if "pending_write" not in st:
        st["pending_write"] = None
    if "awaiting_confirm" not in st:
        st["awaiting_confirm"] = False
    if "clarify_count" not in st:
        st["clarify_count"] = 0

    st["mode"] = mode
    entry["state"] = st
    _db_set(session_id, entry)
    return st, True


def _save_state(session_id: str, state: Dict[str, Any]) -> None:
    entry = _db_get(session_id) or {}
    entry["state"] = state
    _db_set(session_id, entry)


def _steps(mode: str) -> List[Dict[str, Any]]:
    return TEMPLATES[mode]["steps"]


def _key_to_index(mode: str) -> Dict[str, int]:
    out = {}
    for i, st in enumerate(_steps(mode)):
        out[st["key"]] = i
    return out


def _current_step(mode: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps = _steps(mode)
    idx = int(state.get("step_index") or 0)
    if idx < 0:
        idx = 0
    if idx >= len(steps):
        return None
    return steps[idx]


def _retrieve_info_for_coach(mode: str, query: str, top_k: int) -> str:
    rag_query = f"{mode}: {query}"
    matches = get_matches(rag_query, top_k)
    return build_context(matches)


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

    # Include variables for current active section only (section-scoped)
    active_section = _safe_str(state.get("active_section"))
    vars_for_section = state.get("variables", {}).get(active_section, [])
    vars_lines = []
    for v in vars_for_section[:20]:
        try:
            vars_lines.append(
                f"- {v.get('name','')}: low={v.get('low','')}, high={v.get('high','')}, ideal={v.get('ideal','')}, notes={v.get('notes','')}"
            )
        except Exception:
            continue
    vars_block = "\n".join(vars_lines) if vars_lines else "(none)"

    return (
        f"USER_NAME:\n{name_line}\n\n"
        f"TEMPLATE: {tpl['title']} ({mode})\n\n"
        f"ACTIVE_SECTION:\n{active_section}\n\n"
        f"SECTION_VARIABLES:\n{vars_block}\n\n"
        f"ANSWERS:\n{answers_block}\n\n"
        f"INFORMATION (background support):\n{info}\n\n"
        f"FINAL TASK:\n"
        f"- If template is 'Build my confidence': summarise confidence plan in 5–7 short bullet points.\n"
        f"- If template is 'Prepare for difficult behaviours': create a final cheat sheet.\n"
        f"- Keep it paste-ready and mapped to the template.\n"
    )


def _reflect_line() -> str:
    return random.choice(["Got it.", "Okay.", "Thanks — noted.", "Understood.", "That helps."])


def _parse_yes_no(text: str) -> Optional[bool]:
    t = (text or "").strip()
    if not t:
        return None
    if _YES_RE.match(t):
        return True
    if _NO_RE.match(t):
        return False
    return None


def _active_section_from_payload(payload: Dict[str, Any]) -> str:
    # Bubble can send any of these
    for k in ("active_section", "active_field", "active_step", "field_key", "section_key"):
        s = _safe_str(payload.get(k))
        if s:
            return s
    return ""


def _set_active_section(mode: str, state: Dict[str, Any], active_section: str) -> bool:
    """
    Returns True if active section changed and we reset local state per PDF.
    """
    active_section = (active_section or "").strip()
    prev = _safe_str(state.get("active_section"))
    if not active_section:
        return False
    if active_section == prev:
        return False

    # If this section corresponds to a known step key, move step_index to it (field focus change)
    k2i = _key_to_index(mode)
    if active_section in k2i:
        state["step_index"] = k2i[active_section]

    state["active_section"] = active_section

    # PDF: changing field focus resets AI context -> clear pending confirm + clarify counter
    state["pending_write"] = None
    state["awaiting_confirm"] = False
    state["clarify_count"] = 0
    return True


# =========================
# COACH CORE (with confirm write-back + active section)
# =========================
def coach_turn_server_state(payload: Dict[str, Any], session_id: str, stream: bool = False):
    raw_query = _extract_user_text(payload)

    top_k = _clamp_int(payload.get("top_k"), TOP_K, 1, 30)
    reset = _as_bool(payload.get("reset"))
    start_template = _as_bool(payload.get("start_template"))
    user_name = _extract_user_name(payload)

    # new flags
    confirm_write = _as_bool(payload.get("confirm_write")) or _as_bool(payload.get("write_confirmed"))
    # if UI wants to keep old behavior (not recommended), can set auto_confirm=true
    auto_confirm = _as_bool(payload.get("auto_confirm"))
    active_section = _active_section_from_payload(payload)

    _jlog(
        "coach_in",
        session_id=session_id,
        stream=bool(stream),
        mode=_safe_str(payload.get("mode")),
        start_template=start_template,
        reset=reset,
        confirm_write=confirm_write,
        auto_confirm=auto_confirm,
        active_section=active_section,
        raw_query_len=len(raw_query or ""),
        raw_query_preview=(raw_query or "")[:160],
        payload_keys=sorted(list(payload.keys()))[:60],
    )

    _cleanup_sessions()

    if reset:
        _jlog("coach_reset", session_id=session_id)
        _db_delete(session_id)

    if _is_smalltalk(raw_query):
        text = _smalltalk_reply(user_name) + " Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        resp = {"text": text, "session_id": session_id, "done": False}
        return _with_debug(resp, case="smalltalk")

    # Start again command (PDF)
    if raw_query and _START_AGAIN_RE.match(raw_query):
        _jlog("coach_start_again", session_id=session_id)
        # keep session id but reset state
        mode = _extract_mode(payload, fallback_text=raw_query)
        st = _default_state(mode)
        _save_state(session_id, st)
        first = _current_step(mode, st)
        q_text = first["question"] if first else "Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = f"{opener} {q_text}".strip()
        resp = {"text": text, "session_id": session_id, "done": False}
        return _with_debug(resp, mode=mode, step_index=0, started=True, start_again=True)

    mode = _extract_mode(payload, fallback_text=raw_query)
    state, existed = _load_state(session_id, mode)

    # Apply active section change (field focus change reset)
    changed_section = _set_active_section(mode, state, active_section)
    if changed_section:
        _jlog("active_section_changed", session_id=session_id, mode=mode, active_section=state.get("active_section"))
        _save_state(session_id, state)

    _jlog(
        "state_loaded",
        session_id=session_id,
        mode=mode,
        existed=existed,
        step_index=int(state.get("step_index") or 0),
        active_section=_safe_str(state.get("active_section")),
        awaiting_confirm=bool(state.get("awaiting_confirm")),
        pending_key=(state.get("pending_write") or {}).get("key") if isinstance(state.get("pending_write"), dict) else None,
        answers_keys=sorted(list((state.get("answers") or {}).keys()))[:30],
    )

    # IMPORTANT: if user already sent an answer, ignore start_template (Bubble often sends it every time)
    if raw_query:
        if start_template:
            _jlog("start_template_ignored_due_to_answer", session_id=session_id, mode=mode)
        start_template = False

    # START only if explicitly requested OR session newly created
    if start_template or (not existed):
        _jlog(
            "template_started",
            session_id=session_id,
            mode=mode,
            reason=("start_template" if start_template else "new_session"),
        )
        state = _default_state(mode)
        # set active section if provided
        _set_active_section(mode, state, active_section)
        _save_state(session_id, state)
        first = _current_step(mode, state)
        q_text = first["question"] if first else "Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = f"{opener} {q_text}".strip()
        resp = {"text": text, "session_id": session_id, "done": False}
        return _with_debug(resp, mode=mode, step_index=int(state.get("step_index") or 0), existed=existed, started=True)

    # If user didn't send an answer, just repeat current question without advancing
    if not raw_query:
        cur = _current_step(mode, state)
        q_text = cur["question"] if cur else "Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        resp = {"text": q_text, "session_id": session_id, "done": False}
        return _with_debug(resp, mode=mode, step_index=int(state.get("step_index") or 0), empty_answer=True)

    # If we are awaiting confirmation, interpret yes/no
    if state.get("awaiting_confirm"):
        yn = _parse_yes_no(raw_query)
        pending = state.get("pending_write") if isinstance(state.get("pending_write"), dict) else None

        if yn is None and not confirm_write:
            # not a yes/no, prompt again
            key = pending.get("key") if pending else ""
            prompt = f"Please confirm: should I write that into '{key}'? Reply Yes or No."
            resp = {"text": prompt, "session_id": session_id, "done": False}
            return _with_debug(resp, mode=mode, awaiting_confirm=True, need_yes_no=True)

        if yn is False:
            # discard pending, ask same question again
            state["pending_write"] = None
            state["awaiting_confirm"] = False
            _save_state(session_id, state)
            cur = _current_step(mode, state)
            q_text = cur["question"] if cur else "Okay — what should I write instead?"
            resp = {"text": f"No problem. {q_text}", "session_id": session_id, "done": False}
            return _with_debug(resp, mode=mode, discarded=True)

        # yn True OR confirm_write flag
        if pending and pending.get("key"):
            cur_key = pending["key"]
            state["answers"][cur_key] = _safe_str(pending.get("value"))
            state["step_index"] = int(state.get("step_index") or 0) + 1

        state["pending_write"] = None
        state["awaiting_confirm"] = False
        _save_state(session_id, state)

        nxt = _current_step(mode, state)
        if nxt is None:
            _jlog("final_generate_after_confirm", session_id=session_id, mode=mode)
            info = _retrieve_info_for_coach(mode, "final summary", top_k)
            final_user = _make_final_user_message(mode, state, info, user_name=user_name)
            resp_llm = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                    {"role": "user", "content": final_user},
                ],
                temperature=0.2,
            )
            text = strip_markdown_chars((resp_llm.choices[0].message.content or "").strip())
            opener = _pick_opener(session_id, user_name, "coach_last_opener")
            text = _rewrite_bad_opening(text, opener)
            resp = {"text": text, "session_id": session_id, "done": True}
            return _with_debug(resp, mode=mode, done=True)

        text = f"{_reflect_line()} {nxt['question']}".strip()
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = _rewrite_bad_opening(text, opener)
        resp = {"text": text, "session_id": session_id, "done": False}
        return _with_debug(resp, mode=mode, step_index=int(state.get("step_index") or 0), next_key=nxt.get("key"))

    # Normal path: user is answering current step
    cur = _current_step(mode, state)

    # If already finished, return final
    if cur is None:
        _jlog("final_generate", session_id=session_id, mode=mode)
        info = _retrieve_info_for_coach(mode, "final summary", top_k)
        final_user = _make_final_user_message(mode, state, info, user_name=user_name)
        resp_llm = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                {"role": "user", "content": final_user},
            ],
            temperature=0.2,
        )
        text = strip_markdown_chars((resp_llm.choices[0].message.content or "").strip())
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = _rewrite_bad_opening(text, opener)
        resp = {"text": text, "session_id": session_id, "done": True}
        return _with_debug(resp, mode=mode, done=True)

    cur_key = cur["key"]
    _jlog(
        "answer_received",
        session_id=session_id,
        mode=mode,
        step_index_before=int(state.get("step_index") or 0),
        cur_key=cur_key,
        answer_preview=(raw_query or "")[:160],
    )

    # If UI already confirmed, write immediately; else require confirm step (PDF guardrail)
    if confirm_write or auto_confirm:
        state["answers"][cur_key] = raw_query.strip()
        state["step_index"] = int(state.get("step_index") or 0) + 1
        _save_state(session_id, state)

        nxt = _current_step(mode, state)
        if nxt is None:
            _jlog("final_generate_after_last_answer", session_id=session_id, mode=mode)
            info = _retrieve_info_for_coach(mode, "final summary", top_k)
            final_user = _make_final_user_message(mode, state, info, user_name=user_name)
            resp_llm = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COACH_FINAL},
                    {"role": "user", "content": final_user},
                ],
                temperature=0.2,
            )
            text = strip_markdown_chars((resp_llm.choices[0].message.content or "").strip())
            opener = _pick_opener(session_id, user_name, "coach_last_opener")
            text = _rewrite_bad_opening(text, opener)
            resp = {"text": text, "session_id": session_id, "done": True}
            return _with_debug(resp, mode=mode, done=True)

        text = f"{_reflect_line()} {nxt['question']}".strip()
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = _rewrite_bad_opening(text, opener)
        resp = {"text": text, "session_id": session_id, "done": False}
        return _with_debug(resp, mode=mode, step_index=int(state.get("step_index") or 0), next_key=nxt.get("key"))

    # Require explicit confirmation (two-step)
    state["pending_write"] = {"key": cur_key, "value": raw_query.strip()}
    state["awaiting_confirm"] = True
    _save_state(session_id, state)

    prompt = f"Got it. Should I write that into '{cur_key}'? Reply Yes or No."
    opener = _pick_opener(session_id, user_name, "coach_last_opener")
    prompt = _rewrite_bad_opening(prompt, opener)
    resp = {"text": prompt, "session_id": session_id, "done": False}
    return _with_debug(resp, mode=mode, awaiting_confirm=True, pending_key=cur_key)


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"ok": True, "debug": DEBUG}


# =========================
# CHAT (RAG)
# =========================
@app.post("/chat")
def chat(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)
    user_name = _extract_user_name(payload)

    session_id = _get_or_create_session_id(payload)
    _cleanup_sessions()

    if not query:
        return JSONResponse({"answer": "", "session_id": session_id})

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

    if DEBUG:
        _jlog(
            "coach_sse_in",
            session_id=session_id,
            mode=_safe_str(payload.get("mode")),
            start_template=_as_bool(payload.get("start_template")),
            reset=_as_bool(payload.get("reset")),
            confirm_write=_as_bool(payload.get("confirm_write")) or _as_bool(payload.get("write_confirmed")),
            active_section=_active_section_from_payload(payload),
            raw_query_preview=_extract_user_text(payload)[:120],
        )

    result = coach_turn_server_state(payload, session_id=session_id, stream=False)

    def headers():
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

    def gen():
        start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: start\ndata: {start_payload}\n\n"

        data = json.dumps({"text": result.get("text", "")}, ensure_ascii=False)
        yield f"event: chunk\ndata: {data}\n\n"

        done_payload = json.dumps({"done": bool(result.get("done"))}, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers())


@app.post("/coach/reset")
def coach_reset(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    _jlog("coach_reset_endpoint", session_id=session_id)
    _db_delete(session_id)
    return JSONResponse({"ok": True, "session_id": session_id})
