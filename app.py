import os
import json
import uuid
import time
import random
import re
import sqlite3
import urllib.request
import urllib.parse
import urllib.error
import logging
from typing import List, Dict, Any, Optional, Tuple

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

# Priority boost for specific documents (e.g., Negotiation.pdf vectors tagged with metadata priority=2)
# Applied during reranking; base retrieval is still semantic similarity.
PRIORITY_BOOST = float(os.getenv("PRIORITY_BOOST", "0.6"))  # bonus added per priority level above 1
PRIORITY_MAX = int(os.getenv("PRIORITY_MAX", "3"))          # safety clamp (e.g., 1..3)
# Search trace logging (human-readable, safe to share with clients)
SEARCH_DEBUG_LOGS = os.getenv("SEARCH_DEBUG_LOGS", "0").strip().lower() in ("1","true","yes","y","on")
SEARCH_LOG_MAX_MATCHES = int(os.getenv("SEARCH_LOG_MAX_MATCHES", "8"))   # how many matches to print
SEARCH_LOG_TEXT_PREVIEW = int(os.getenv("SEARCH_LOG_TEXT_PREVIEW", "140"))  # chars of text preview in logs

def _slog(event: str, **fields):
    """Search logs (opt-in) that you can copy from server logs."""
    if SEARCH_DEBUG_LOGS:
        _jlog(event, **fields)

# IMPORTANT: confirmation cadence (PDF intent: not after every answer)
CONFIRM_EVERY_N = int(os.getenv("COACH_CONFIRM_EVERY_N", "3"))  # checkpoint confirm after N answers (default 3)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# =========================
# BUBBLE DATA API (READ-ONLY TEMPLATE STATE)
# =========================
BUBBLE_API_BASE = os.getenv("BUBBLE_API_BASE", "").strip().rstrip("/")
BUBBLE_API_KEY = os.getenv("BUBBLE_API_KEY", "").strip()

# Expect base like: https://<app>.bubbleapps.io/<version>/api/1.1/obj
def _bubble_enabled() -> bool:
    return bool(BUBBLE_API_BASE and BUBBLE_API_KEY)

def _bubble_headers() -> Dict[str, str]:
    # Bubble Data API supports Bearer token auth. We also send api_token as query param for compatibility.
    return {"Authorization": f"Bearer {BUBBLE_API_KEY}"}

def _bubble_url(obj_type: str, obj_id: Optional[str] = None) -> str:
    if obj_id:
        return f"{BUBBLE_API_BASE}/{obj_type}/{obj_id}"
    return f"{BUBBLE_API_BASE}/{obj_type}"


def _http_get_json(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None, timeout: int = 12) -> Dict[str, Any]:
    qs = urllib.parse.urlencode(params or {}, doseq=True)
    full_url = url if not qs else f"{url}?{qs}"
    req = urllib.request.Request(full_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw) if raw else {}
            except Exception:
                return {}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        _jlog("bubble_get_error", status=int(getattr(e, "code", 0) or 0), url=full_url, body=(body or "")[:500])
        return {}
    except Exception as e:
        _jlog("bubble_get_exception", url=full_url, err=str(e)[:300])
        return {}


def _bubble_get(obj_type: str, obj_id: str, timeout: int = 12) -> Dict[str, Any]:
    if not _bubble_enabled():
        return {}
    url = _bubble_url(obj_type, obj_id)
    return _http_get_json(url, headers=_bubble_headers(), params={"api_token": BUBBLE_API_KEY}, timeout=timeout)


def _bubble_search(obj_type: str, constraints: List[Dict[str, Any]], timeout: int = 12) -> Dict[str, Any]:
    if not _bubble_enabled():
        return {}
    url = _bubble_url(obj_type)
    params = {"api_token": BUBBLE_API_KEY, "constraints": json.dumps(constraints)}
    return _http_get_json(url, headers=_bubble_headers(), params=params, timeout=timeout)

def _fetch_template_state_text(template_id: str, request_id: str = "") -> str:
    """Read-only: fetch master_negotiation_template + deals + variable items and render compact state text."""
    if not template_id:
        return ""
    if not _bubble_enabled():
        return ""
    tpl = _bubble_get("master_negotiation_template", template_id)
    resp = tpl.get("response") if isinstance(tpl.get("response"), dict) else {}
    if not resp:
        return ""

    my_deal_id = _bubble_extract_id(resp.get("my_deal"))
    their_deal_id = _bubble_extract_id(resp.get("their_deal"))
    my_items_ids_raw = resp.get("my_items") or []
    their_items_ids_raw = resp.get("their_items") or []

    my_item_ids = [x for x in (_bubble_extract_id(v) for v in my_items_ids_raw) if x]
    their_item_ids = [x for x in (_bubble_extract_id(v) for v in their_items_ids_raw) if x]

    my_deal_txt = _format_deal(_bubble_get("Deal", my_deal_id)) if my_deal_id else ""
    their_deal_txt = _format_deal(_bubble_get("Deal", their_deal_id)) if their_deal_id else ""

    # Bulk fetch all Variable_items for this template, then map by id.
    items_map: Dict[str, Dict[str, Any]] = {}
    search = _bubble_search("Variable_items", [{"key": "template", "constraint_type": "equals", "value": template_id}])
    results = search.get("response", {}).get("results") if isinstance(search.get("response"), dict) else []
    if isinstance(results, list):
        for it in results:
            _id = _bubble_extract_id(it) or it.get("_id")
            if _id:
                items_map[_id] = it

    my_items = [items_map.get(i) for i in my_item_ids if i in items_map]
    their_items = [items_map.get(i) for i in their_item_ids if i in items_map]

    my_items_txt = _format_items([x for x in my_items if x])
    their_items_txt = _format_items([x for x in their_items if x])

    parts = []
    parts.append("CURRENT MASTER NEGOTIATOR TEMPLATE STATE (read-only):")
    if my_deal_txt:
        parts.append(f"My Deal: {my_deal_txt}")
    if their_deal_txt:
        parts.append(f"Their Deal: {their_deal_txt}")
    if my_items_txt:
        parts.append("My Win Zone Variables:")
        parts.append(my_items_txt)
    if their_items_txt:
        parts.append("Their Win Zone Variables:")
        parts.append(their_items_txt)

    out = "\n".join(parts).strip()
    if request_id:
        _slog("bubble_template_state", request_id=request_id, template_id=template_id, has_state=bool(out), chars=len(out))
    return out

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

# =========================
# OPENAI STREAM HELPERS (SSE)
# =========================
def _openai_stream_text(messages: List[Dict[str, str]], model: str, temperature: float = 0.2):
    """Yield incremental text deltas from OpenAI streaming API."""
    try:
        stream = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for ev in stream:
            try:
                delta = ev.choices[0].delta
                txt = getattr(delta, "content", None)
                if txt:
                    yield txt
            except Exception:
                continue
    except Exception as e:
        # If streaming fails, fallback to a single non-streamed response
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            full = (resp.choices[0].message.content or "")
            if full:
                yield full
        except Exception:
            yield "Server error."

def _sse_headers():
    # Best-practice SSE headers (Render/Nginx friendly)
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

def _iter_text_as_sse_chunks(text_iter, *, min_chars: int = 24):
    """Coalesce tiny deltas into readable chunks for Bubble Stream."""
    buf = ""
    for piece in text_iter:
        buf += piece
        if len(buf) >= min_chars or buf.endswith("\n"):
            out = buf
            buf = ""
            yield out
    if buf:
        yield buf


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
    # MASTER template: use Bubble template_id as stable session_id so memory persists per template.
    tid = str(payload.get("template_id") or payload.get("templateId") or payload.get("template") or "").strip()
    if tid:
        return tid
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
    {
        "match_any": ["balance the power in a negotiation", "balance power in a negotiation", "balance the power"],
        "hint": "confident mindset pages 9-11",
    },
    {
        "match_any": ["push back without upsetting the relationship", "push back without upsetting", "push back"],
        "hint": "tactics prepared to respond pages 65-74 slides 7 8 15",
    },
    {
        "match_any": ["difference between selling and negotiation", "selling vs negotiation", "selling and negotiation"],
        "hint": "introduction to negotiation book page xii-vii slide 11",
    },
    {
        "match_any": [
            "deal with difficult questions",
            "handle difficult questions",
            "difficult questions",
            "deal with difficult question",
        ],
        "hint": "slides 28-34 difficult questions",
    },
{
    "match_any": [
        "earth element", "element earth", "earth rules", "7 rules earth", "seven rules earth",
        "water element", "element water", "water rules", "fire element", "element fire", "fire rules",
        "silence element", "element silence", "silence rules",
        "diadem earth", "diadem water", "diadem fire", "diadem silence"
    ],
    "hint": "Master Negotiator Slides elements water earth fire silence rules",
},

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
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "without",
    "is",
    "are",
    "was",
    "were",
    "be",
    "do",
    "does",
    "did",
    "how",
    "what",
    "why",
    "when",
    "where",
    "between",
    "into",
    "from",
    "as",
    "at",
    "by",
    "it",
    "this",
    "that",
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
        # priority is optional metadata (default 1). Higher = slightly boosted during rerank.
        try:
            pr = int(md.get("priority") or 1)
        except Exception:
            pr = 1
        if pr < 1:
            pr = 1
        if pr > PRIORITY_MAX:
            pr = PRIORITY_MAX
        text = (md.get("text") or "").strip()
        if not text:
            continue

        tlow = text.lower()
        ttoks = _tokenize(text)

        overlap_q = sum(1.0 for t in ttoks if t in qt)
        overlap_hint = sum(1.2 for t in ttoks if t in hint_toks) if hint_toks else 0.0

        bonus = 0.0
        for phrase in [
            "earth",
            "water",
            "fire",
            "silence",
            "balance",
            "power",
            "selling",
            "negotiation",
            "difficult questions",
            "push back",
            "relationship",
            "tactics",
            "mindset",
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

        final = (overlap_q * 1.25) + (overlap_hint * 1.6) + bonus + (pscore * 0.25) - penalty + (max(0, pr - 1) * PRIORITY_BOOST)
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


def build_context(matches: List[Dict], request_id: Optional[str] = None) -> str:
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

    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    _slog("context_built",
          request_id=request_id,
          chunks_used=len(parts),
          context_chars=len("\n---\n".join(parts)))
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


def _brief_match(m: Dict, label: str = "") -> Dict[str, Any]:
    md = (m.get("metadata") or {})
    try:
        score = float(m.get("score") or 0.0)
    except Exception:
        score = 0.0
    try:
        pr = int(md.get("priority") or 1)
    except Exception:
        pr = 1
    txt = (md.get("text") or "").strip().replace("\n", " ")
    if SEARCH_LOG_TEXT_PREVIEW and len(txt) > SEARCH_LOG_TEXT_PREVIEW:
        txt = txt[:SEARCH_LOG_TEXT_PREVIEW] + "…"
    return {
        "label": label,
        "score": round(score, 4),
        "priority": pr,
        "file": md.get("file_name") or "",
        "page": md.get("page"),
        "chunk_index": md.get("chunk_index"),
        "id": m.get("id") or "",
        "preview": txt,
    }

def get_matches(query: str, top_k_final: int, request_id: Optional[str] = None) -> List[Dict]:
    q_clean = (query or "").strip()
    if not q_clean:
        return []

    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    _slog("search_start",
          request_id=request_id,
          query=q_clean,
          top_k_final=top_k_final,
          pinecone_topk_raw=PINECONE_TOPK_RAW,
          multi_query_k=MULTI_QUERY_K,
          min_match_score=MIN_MATCH_SCORE,
          priority_boost=PRIORITY_BOOST)

    hint = _hint_for_question(q_clean)
    q_hint = f"{q_clean}\n{hint}".strip() if hint else ""
    q_kw = _keyword_query(q_clean)

    queries: List[str] = [q_clean]
    if hint and len(queries) < MULTI_QUERY_K:
        queries.append(q_hint)
    if len(queries) < MULTI_QUERY_K and q_kw:
        queries.append(q_kw)

    _slog("search_queries",
          request_id=request_id,
          queries=queries)
    all_results: List[List[Dict]] = []
    for q in queries:
        try:
            t0 = time.time()
            vec = embed_query(q)
            res = index.query(vector=vec, top_k=PINECONE_TOPK_RAW, include_metadata=True)
            ms = int((time.time() - t0) * 1000)
            matches = res.get("matches") or []
            all_results.append(matches)
            _slog("pinecone_query",
                  request_id=request_id,
                  q=q,
                  ms=ms,
                  matches_count=len(matches),
                  top_matches=[_brief_match(x) for x in matches[:SEARCH_LOG_MAX_MATCHES]])
        except Exception:
            continue

    merged = _merge_dedup_matches(all_results)
    _slog("search_merged",
          request_id=request_id,
          merged_count=len(merged),
          sample=[_brief_match(x) for x in merged[:SEARCH_LOG_MAX_MATCHES]])

    merged = _filter_matches_by_score(merged)
    _slog("search_filtered",
          request_id=request_id,
          filtered_count=len(merged),
          sample=[_brief_match(x) for x in merged[:SEARCH_LOG_MAX_MATCHES]])

    reranked = _rerank(q_clean, merged, top_k_final)
    _slog("search_reranked",
          request_id=request_id,
          final_count=len(reranked),
          final=[_brief_match(x) for x in reranked[:SEARCH_LOG_MAX_MATCHES]])

    # If we have a curated hint for this query, do NOT block on the relevance gate.
    # These hints exist specifically for short "definition/rules" questions (e.g. Earth/Water/Fire elements)
    # where the top chunk can be short and fail overlap/length heuristics.
    hint_gate = _hint_for_question(q_clean)
    if hint_gate and reranked:
        _slog("search_context_relevance",
              request_id=request_id,
              relevant=True,
              kept=len(reranked),
              reason="hint_override")
        return reranked

    relevant = is_context_relevant(q_clean, reranked)
    _slog("search_context_relevance",
          request_id=request_id,
          relevant=relevant,
          kept=len(reranked))
    if not relevant:
        return []

    return reranked

def _reflect_line() -> str:
    return random.choice(["Got it.", "Okay.", "Thanks — noted.", "Understood.", "That helps."])

def _retrieve_info_for_coach(mode: str, query: str, top_k: int) -> str:
    rag_query = f"{mode}: {query}"
    request_id = str(uuid.uuid4())[:8]
    matches = get_matches(rag_query, top_k, request_id=request_id)
    return build_context(matches, request_id=request_id)

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

# =========================
# PROMPTS
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
    "You are a negotiation assistant that can ONLY use the provided INFORMATION.\n"
    "Your job is to produce specific, usable negotiation output (scripts + options), grounded in INFORMATION.\n\n"

    "Hard rules:\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT mention document names, pages, sources, citations, or the word 'context'.\n"
    "- Do NOT copy sentences from INFORMATION. Paraphrase everything.\n"
    "- You may include at most ONE very short quoted phrase (max 6 words) only if it is essential.\n"
    "- If INFORMATION is missing / too generic / not clearly relevant, respond exactly:\n"
    "  \"I can't find this in the provided documents.\".\n\n"

    "Make it actionable:\n"
    "- Always include:\n"
    "  1) Recommended move (1 line)\n"
    "  2) 2–3 concrete options you can propose (numbers/terms where possible)\n"
    "  3) A word-for-word script (2–4 lines) the user can say\n"
    "  4) One pushback handling line (1–2 lines)\n"
    "- If the user didn’t give key numbers, provide placeholders like [X], [date], [volume], [payment terms].\n"
    "- Keep it concise (<=180 words).\n\n"

    "Tone:\n"
    "- Direct, practical, confident.\n"
    "- End with ONE short question only if absolutely needed to choose between options.\n"
)


# Strict doc-only chat (no apple pie)
SYSTEM_PROMPT_CHAT = (
    "You are a focused assistant specialised only in the provided materials.\n"
    "You may answer ONLY if INFORMATION is provided and clearly relevant to the user's question.\n"
    "If INFORMATION is empty, missing, or not relevant to the question, you must respond exactly with:\n"
    "\"I can only help with questions related to the provided materials.\"\n"
    "\nRules:\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Do NOT provide general knowledge, explanations, recipes, or advice outside INFORMATION.\n"
    "- Keep it short and neutral.\n"
    "- Do NOT ask follow-up questions when refusing.\n"
    "- If USER_NAME is provided, you may greet them only when answering (not when refusing).\n"
    + VARIABLES_POLICY
    + ACTIVE_SECTION_POLICY
    + LIMITS_POLICY
)

# Final output must be AI-written (not pasted) and no technical keys
SYSTEM_PROMPT_COACH_FINAL = (
    "You are a professional negotiation coach.\n"
    "You may ONLY use the provided INFORMATION to shape your guidance.\n"
    "Do not rely on general negotiation knowledge outside INFORMATION.\n"
    "The guided dialogue is complete. Produce the final output now.\n\n"

    "Hard rules:\n"
    "- Output plain text only. NO markdown.\n"
    "- Do NOT mention documents, pages, sources, citations, or the word 'context'.\n"
    "- Do NOT introduce tactics, terms, or levers that are not supported by INFORMATION.\n"
    "- Do NOT copy sentences verbatim from INFORMATION.\n"
    "- Rephrase, synthesise, and apply INFORMATION to the user's situation.\n\n"

    "Specificity rules:\n"
    "- Convert INFORMATION into concrete, usable negotiation language.\n"
    "- Where INFORMATION implies options (e.g. timing, scope, commitment), make them explicit.\n"
    "- If INFORMATION does not support a specific term (price, payment, %), use placeholders like [X], [date].\n"
    "- Avoid generic advice (e.g. 'focus on value') unless INFORMATION explains how to do so.\n\n"

    "Output format for 'Prepare for difficult behaviours':\n"
    "1) Scenario (1–2 sentences, rewritten).\n"
    "2) Cheat sheet:\n"
    "   likely line -> your response -> steer-back phrase.\n"
    "   (All lines must be something the user could say out loud.)\n"
    "3) Optional: one short line explaining the other party’s tactic, grounded in INFORMATION.\n"
    "4) Optional: 1–2 concrete trade-offs supported by INFORMATION.\n\n"

    "Style:\n"
    "- Calm, confident, practical.\n"
    "- Sound like a real coach preparing someone for a live conversation.\n"
    "- End with one short optional next-step question.\n"
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
            # summary = единственное финальное подтверждение
            {"key": "summary", "question": "Want me to pull this together into a short confidence plan you can read right before the call?"},
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
            # summary = единственное финальное подтверждение
            {"key": "summary", "question": "Want me to pull this together into a clear final cheat sheet?"},
        ],
    },
}

_START_AGAIN_RE = re.compile(
    r"^\s*(start again|restart|start over|reset|начать заново|почати заново|заново)\s*[!.?]*\s*$",
    flags=re.IGNORECASE,
)
_YES_RE = re.compile(r"^\s*(yes|y|ok|okay|confirm|confirmed|sure|да|ок|угу)\s*[!.?]*\s*$", flags=re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(no|n|nope|cancel|stop|нет|не)\s*[!.?]*\s*$", flags=re.IGNORECASE)


def _default_state(mode: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "step_index": 0,
        "answers": {},
        "active_section": "",
        "variables": {},
        # checkpoint confirmations only (no final confirmation state anymore)
        "awaiting_confirm": False,
        "pending_action": None,        # "checkpoint" only
        "pending_checkpoint_upto": 0,
        "template_done": False,
        "clarify_count": 0,
    }

def _load_state(session_id: str, mode: str) -> Tuple[Dict[str, Any], bool]:
    entry = _db_get(session_id) or {}
    st = entry.get("state")

    # new session / corrupted
    if not isinstance(st, dict):
        st = _default_state(mode)
        entry["state"] = st
        _db_set(session_id, entry)
        return st, False

    # mode switch -> reset
    if st.get("mode") != mode:
        st = _default_state(mode)
        entry["state"] = st
        _db_set(session_id, entry)
        return st, True

    # defensive defaults
    if "answers" not in st or not isinstance(st["answers"], dict):
        st["answers"] = {}
    if "step_index" not in st:
        st["step_index"] = 0
    if "variables" not in st or not isinstance(st["variables"], dict):
        st["variables"] = {}
    if "active_section" not in st:
        st["active_section"] = ""
    if "awaiting_confirm" not in st:
        st["awaiting_confirm"] = False
    if "pending_action" not in st:
        st["pending_action"] = None
    if "pending_checkpoint_upto" not in st:
        st["pending_checkpoint_upto"] = 0
    if "template_done" not in st:
        st["template_done"] = False
    if "clarify_count" not in st:
        st["clarify_count"] = 0

    st["mode"] = mode
    entry["state"] = st
    _db_set(session_id, entry)
    return st, True


def _save_state(session_id: str, state: Dict[str, Any]) -> None:
    entry = _db_get(session_id) or {}
    entry["state"] = state or {}
    _db_set(session_id, entry)

def _steps(mode: str) -> List[Dict[str, Any]]:
    return TEMPLATES[mode]["steps"]

def _current_step(mode: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps = _steps(mode)
    idx = int(state.get("step_index") or 0)
    if idx < 0:
        idx = 0
    if idx >= len(steps):
        return None
    return steps[idx]


def _should_checkpoint_confirm(mode: str, state: Dict[str, Any]) -> bool:
    if state.get("awaiting_confirm"):
        return False
    n = max(2, int(CONFIRM_EVERY_N or 3))
    step_index = int(state.get("step_index") or 0)
    steps_len = len(_steps(mode))
    if step_index <= 0:
        return False
    if step_index >= steps_len:
        return False
    # НЕ делаем checkpoint прямо перед финалом, чтобы не мешать summary
    next_step = _current_step(mode, state)
    if next_step and next_step.get("key") == "summary":
        return False
    return (step_index % n) == 0


def _checkpoint_prompt() -> str:
    # Без технички
    return "Does this feel right to lock in before we move on?"


def _generate_final(mode: str, state: Dict[str, Any], top_k: int, user_name: str, session_id: str) -> Dict[str, Any]:
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

    state["template_done"] = True
    _save_state(session_id, state)
    return {"text": text, "session_id": session_id, "done": True}

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
    for k in ("active_section", "active_field", "active_step", "field_key", "section_key"):
        s = _safe_str(payload.get(k))
        if s:
            return s
    return ""


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
def _key_to_index(mode: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, st in enumerate(_steps(mode)):
        out[st["key"]] = i
    return out


def _active_section_from_payload(payload: Dict[str, Any]) -> str:
    for k in ("active_section", "active_field", "active_step", "field_key", "section_key"):
        s = _safe_str(payload.get(k))
        if s:
            return s
    return ""


def _set_active_section(mode: str, state: Dict[str, Any], active_section: str) -> bool:
    active_section = (active_section or "").strip()
    prev = _safe_str(state.get("active_section"))

    if not active_section or active_section == prev:
        return False

    k2i = _key_to_index(mode)
    if active_section in k2i:
        state["step_index"] = k2i[active_section]

    state["active_section"] = active_section

    # when user jumps fields, clear confirmations + counters
    state["awaiting_confirm"] = False
    state["pending_action"] = None
    state["pending_checkpoint_upto"] = 0
    state["clarify_count"] = 0

    return True

# =========================
# COACH CORE (single final confirm via summary)
# =========================
def coach_turn_server_state(payload: Dict[str, Any], session_id: str, stream: bool = False):
    raw_query = _extract_user_text(payload)

    top_k = _clamp_int(payload.get("top_k"), TOP_K, 1, 30)
    reset = _as_bool(payload.get("reset"))
    start_template = _as_bool(payload.get("start_template"))
    user_name = _extract_user_name(payload)

    confirm_write = _as_bool(payload.get("confirm_write")) or _as_bool(payload.get("write_confirmed"))
    auto_confirm = _as_bool(payload.get("auto_confirm"))
    active_section = _active_section_from_payload(payload)

    _cleanup_sessions()

    if reset:
        _db_delete(session_id)

    if _is_smalltalk(raw_query):
        text = _smalltalk_reply(user_name) + " Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        return _with_debug({"text": text, "session_id": session_id, "done": False}, case="smalltalk")

    if raw_query and _START_AGAIN_RE.match(raw_query):
        mode = _extract_mode(payload, fallback_text=raw_query)
        st = _default_state(mode)
        _save_state(session_id, st)
        first = _current_step(mode, st)
        q_text = first["question"] if first else "Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        return _with_debug({"text": f"{opener} {q_text}".strip(), "session_id": session_id, "done": False}, mode=mode, step_index=0, start_again=True)

    mode = _extract_mode(payload, fallback_text=raw_query)
    state, existed = _load_state(session_id, mode)

    changed_section = _set_active_section(mode, state, active_section)
    if changed_section:
        _save_state(session_id, state)

    if raw_query:
        start_template = False

    if state.get("template_done") and not start_template and not reset:
        return _with_debug({"text": "Template complete. Say 'start again' or send start_template=true to restart.", "session_id": session_id, "done": True}, mode=mode, template_done=True)

    if start_template or (not existed):
        state = _default_state(mode)
        _set_active_section(mode, state, active_section)
        _save_state(session_id, state)
        first = _current_step(mode, state)
        q_text = first["question"] if first else "Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        return _with_debug({"text": f"{opener} {q_text}".strip(), "session_id": session_id, "done": False}, mode=mode, started=True)

    if not raw_query:
        cur = _current_step(mode, state)
        q_text = cur["question"] if cur else "Choose a shortcut: Build my confidence or Prepare for difficult behaviours."
        return _with_debug({"text": q_text, "session_id": session_id, "done": False}, mode=mode, empty_answer=True)

    # =========================
    # CHECKPOINT CONFIRM ONLY
    # =========================
    if state.get("awaiting_confirm"):
        yn = _parse_yes_no(raw_query)
        if yn is None and not (confirm_write or auto_confirm):
            return _with_debug({"text": "Yes or no — should we lock this in and continue?", "session_id": session_id, "done": False}, mode=mode, awaiting_confirm=True)

        # clear confirm state
        state["awaiting_confirm"] = False
        state["pending_action"] = None
        state["pending_checkpoint_upto"] = 0
        _save_state(session_id, state)

        # regardless yes/no -> continue
        nxt = _current_step(mode, state)
        q_text = nxt["question"] if nxt else "Okay — what do you want to do next?"
        text = f"{_reflect_line()} {q_text}".strip()
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        text = _rewrite_bad_opening(text, opener)
        return _with_debug({"text": text, "session_id": session_id, "done": False}, mode=mode, checkpoint_confirm_handled=True)

    # =========================
    # NORMAL ANSWER PATH
    # =========================
    cur = _current_step(mode, state)
    if cur is None:
        # Safety: if we somehow overshot, just generate
        return _with_debug(_generate_final(mode, state, top_k, user_name, session_id), mode=mode, overshot=True)

    cur_key = cur["key"]

    # SPECIAL: summary step = FINAL CONFIRM (single)
    if cur_key == "summary":
        yn = _parse_yes_no(raw_query)
        if yn is None and not (confirm_write or auto_confirm):
            opener = _pick_opener(session_id, user_name, "coach_last_opener")
            msg = _rewrite_bad_opening("Please reply Yes or No.", opener)
            return _with_debug({"text": msg, "session_id": session_id, "done": False}, mode=mode, need_yes_no_for_summary=True)

        if (yn is False) and not (confirm_write or auto_confirm):
            # user said no -> do not generate
            keys = [s["key"] for s in _steps(mode) if s.get("key") != "summary"]
            return _with_debug({"text": "No problem. Which part do you want to adjust? " + ", ".join(keys), "session_id": session_id, "done": False}, mode=mode, summary_no=True)

        # yes -> lock summary as yes + generate immediately (NO second confirm)
        state["answers"][cur_key] = "yes"
        state["step_index"] = int(state.get("step_index") or 0) + 1
        _save_state(session_id, state)
        return _with_debug(_generate_final(mode, state, top_k, user_name, session_id), mode=mode, summary_yes_generate=True)

    # normal steps: store answer
    state["answers"][cur_key] = raw_query.strip()
    state["step_index"] = int(state.get("step_index") or 0) + 1
    _save_state(session_id, state)

    # checkpoint confirm every N answers
    if _should_checkpoint_confirm(mode, state):
        state["awaiting_confirm"] = True
        state["pending_action"] = "checkpoint"
        state["pending_checkpoint_upto"] = int(state.get("step_index") or 0)
        _save_state(session_id, state)
        prompt = _checkpoint_prompt()
        opener = _pick_opener(session_id, user_name, "coach_last_opener")
        prompt = _rewrite_bad_opening(prompt, opener)
        return _with_debug({"text": prompt, "session_id": session_id, "done": False}, mode=mode, checkpoint_confirm=True)

    # ask next question
    nxt = _current_step(mode, state)
    q_text = nxt["question"] if nxt else "Okay — what do you want to do next?"
    text = f"{_reflect_line()} {q_text}".strip()
    opener = _pick_opener(session_id, user_name, "coach_last_opener")
    text = _rewrite_bad_opening(text, opener)
    return _with_debug({"text": text, "session_id": session_id, "done": False}, mode=mode, next_key=(nxt.get("key") if nxt else ""))


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
    request_id = str(uuid.uuid4())[:8]
    _cleanup_sessions()

    if not query:
        return JSONResponse({"answer": "", "session_id": session_id})

    if _is_smalltalk(query):
        return {"answer": _smalltalk_reply(user_name), "session_id": session_id}

    matches = get_matches(query, top_k, request_id=request_id)
    context = build_context(matches, request_id=request_id) if matches else ""

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
            temperature=0.2,
        )
        answer = strip_markdown_chars((resp.choices[0].message.content or "").strip())
        # IMPORTANT: if we are refusing, do not add openers.
        if answer and answer.strip() != "I can only help with questions related to the provided materials.":
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
    session_id = _get_or_create_session_id(payload)
    request_id = str(uuid.uuid4())[:8]
    _cleanup_sessions()

    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)
    user_name = _extract_user_name(payload)

    def gen():
        start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: start\ndata: {start_payload}\n\n"

        # Smalltalk: single chunk
        if not query:
            chunks = [""]
        elif _is_smalltalk(query):
            chunks = [_smalltalk_reply(user_name)]
        else:
            matches = get_matches(query, top_k, request_id=request_id)
            context = build_context(matches) if matches else ""

            if not matches:
                user = (
                    f"USER_NAME:\n{user_name}\n\n"
                    f"USER_MESSAGE:\n{query}\n\n"
                    f"INFORMATION:\n"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_CHAT},
                    {"role": "user", "content": user},
                ]
            else:
                user = (
                    f"USER_NAME:\n{user_name}\n\n"
                    f"QUESTION:\n{query}\n\n"
                    f"INFORMATION:\n{context}"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_QA},
                    {"role": "user", "content": user},
                ]

            # Stream from OpenAI
            chunks = _iter_text_as_sse_chunks(_openai_stream_text(messages, model=CHAT_MODEL, temperature=0.2), min_chars=28)

        # Emit chunks
        for part in chunks:
            part = strip_markdown_chars(part)
            data = json.dumps({"text": part}, ensure_ascii=False)
            yield f"event: chunk\ndata: {data}\n\n"

        done_payload = json.dumps({"done": True}, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_sse_headers())




# =========================
# COACH ENDPOINTS
# =========================
@app.post("/coach/chat")
def coach_chat(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    try:
        out = coach_turn_server_state(payload, session_id=session_id, stream=False)
        return JSONResponse(out)
    except Exception as e:
        _jlog("coach_chat_error", session_id=session_id, err=str(e))
        return JSONResponse({"text": "Server error.", "session_id": session_id, "done": False}, status_code=500)


@app.post("/coach/sse")
def coach_sse(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)

    def headers():
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

    def gen():
        try:
            start_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
            yield f"event: start\ndata: {start_payload}\n\n"

            try:
                result = coach_turn_server_state(payload, session_id=session_id, stream=False)
            except Exception as e:
                _jlog("coach_sse_error", session_id=session_id, err=str(e))
                result = {"text": "Server error.", "session_id": session_id, "done": False}

            # гарантируем dict
            if not isinstance(result, dict):
                result = {"text": "Internal error.", "done": False}

            chunk_payload = json.dumps({"text": result.get("text", "")}, ensure_ascii=False)
            yield f"event: chunk\ndata: {chunk_payload}\n\n"

            done_payload = json.dumps({"done": bool(result.get("done"))}, ensure_ascii=False)
            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception as e:
            _jlog("coach_sse_error", session_id=session_id, err=str(e)[:800])

            # SSE-friendly error
            err_chunk = json.dumps({"text": "Internal error. Please retry."}, ensure_ascii=False)
            yield f"event: chunk\ndata: {err_chunk}\n\n"

            err_done = json.dumps({"done": True}, ensure_ascii=False)
            yield f"event: done\ndata: {err_done}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers())

@app.post("/coach/reset")
def coach_reset(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    _jlog("coach_reset_endpoint", session_id=session_id)
    _db_delete(session_id)
    return JSONResponse({"ok": True, "session_id": session_id})



# =========================
# MASTER NEGOTIATOR TEMPLATE (TEXT-ONLY COACH)
# Mode: master_negotiator_template
# Notes:
# - AI NEVER writes to any tables / rows.
# - AI only gives guidance in plain text.
# - UI / user is responsible for entering data.
# =========================

MASTER_MODE = "master_negotiator_template"

MASTER_SYSTEM_PROMPT_TEXT = """You are a Diadem MASTER Negotiator assistant.
You help the user fill the MASTER negotiation template using Diadem language only.
Primary source: Master Negotiator Slides. Use INFORMATION. Do not invent.

Core behaviour:
- You are a live negotiation coach, not a teacher.
- Every reply must move the negotiation forward unless the user is asking for a definition.
- Work on ONE section at a time.
- Ask only ONE focused question per turn.
- Never present a full framework list.
- Never dump all preparation steps at once.
- Never reset or restart the framework unless the user explicitly asks to restart.
- Never greet again after the conversation has started.
- Never insert mid-session greetings.
- Always build directly on the user’s last choice, number, or statement.
- If the user selects A/B/C, continue developing that exact path.
- Never re-list generic preparation steps after a decision has been made.
- Do not repeat similar bullet lists across turns.
- Do not ignore the user’s actual question.

Input validation rule:
- If you present A/B/C options, accept ONLY "A", "B", or "C" (case-insensitive).
- If the user replies with anything else, do NOT guess.
- Do NOT reinterpret their answer.
- Correct them briefly and repeat the same A/B/C options.
- Ask again: "Which one: A, B, or C?"

Goal vs trade rule:
- Price is the GOAL in a price increase negotiation.
- Price is NOT a trade variable.
- If the user selects "price" as a trade variable, respond:
  "Price is your goal. Pick a non-price trade variable."
- Then repeat the same A/B/C options and ask again.

No invented numbers rule:
- Never invent durations, percentages, months, days, quantities, or commitments.
- Only use numbers explicitly provided by the user or found in the template state.
- If a number is required and missing, ask for it.
- Never generate examples like "24 months" unless the user gave it.

Anti-loop rule:
- If the user repeats their goal instead of answering the current decision,
  acknowledge in one short line.
- Then restate the exact current decision point.
- Ask one direct forward question.
- Never restart the process.

Concept integrity rule:
- If the user asks “what is X?” or “what does X mean?”, answer the definition directly.
- Do NOT convert a definition question into action steps.
- First clarify the concept in 2–4 short sentences.
- Then give one short practical example.
- Then ask one short direct question.

Anchor clarification rule:
- If the user questions the anchor (e.g. "why 13?" after choosing 10%),
  explain that the anchor is a starting position above the target.
- Clarify clearly that the target remains the original number (e.g. 10%).
- Explain that the anchor creates room to move during negotiation.
- Keep explanation short (2–3 sentences).
- Do NOT redefine the anchor as the new goal.
- Then ask one short forward-moving question.

Clarification rule:
- If the user challenges a recommendation ("why?", "I don’t understand", etc.),
  explain briefly and clearly (2–3 sentences).
- Clarify distinction between:
  target vs anchor,
  goal vs trade,
  position vs movement,
  variable vs priority.
- Do not become theoretical.
- Do not restart the process.
- After explaining, continue forward with one short question.

Hard rules:
- Plain text only. NO markdown.
- No emojis.
- Do NOT output tactic names (WATER, EARTH, FIRE, SILENCE).
- Do NOT output meta labels.
- No weak speak.
- No humour.
- No logic justification.
- Short declarative sentences only.
- Use negotiation-generic examples unless user specifies context.
- Reverse If/Then format only (IF = their commitment/money, THEN = our concession).
- Never reverse it incorrectly.
- If user gives a target %, open 3–5% higher.
- Always end with EXACTLY ONE short direct question.
- The final character of the response MUST be '?'.

Adaptive guidance:

If the user is uncertain:
- Present maximum 2 simple starting options.
- Provide one short paste-ready sentence for each.
- Ask the user to choose one.

If the user provides a number:
- Build directly from that number.
- Anchor if needed (+3–5%).
- Construct next move.
- Provide exact wording.
- Ask what they will trade.

If the user chooses an option:
- Continue that exact path.
- Convert into firm negotiation position.
- Add one correct Reverse If/Then.
- Move to next logical decision.
- Do not return to earlier stages.

Formatting:
- Use short bullets starting with "- " only when structured input is necessary.
- Separate sections with blank lines.
- Keep it sharp, practical, and forward-moving.
"""
def _mnt_default_state_text() -> Dict[str, Any]:
    return {
        "mode": MASTER_MODE,
        "help_offered": False,
        "help_accepted": None,   # True/False/None

        # template context (set by UI)
        "deal_value": None,      # float or None (optional; user enters)
        "active_section_id": "",
        "focus_field": "",

        # conversation memory (server-side)
        "stage": "",             # e.g. goal / variables / trade / script
        "slots": {},             # extracted facts from user (percent, price, terms, etc.)
        "last_question": "",     # last question we asked the user
        "history": [],           # last few turns [{"u": "...", "a": "..."}]

        "clarify_count": 0,
        "updated_at": _now(),
    }

def _mnt_load_state_text(session_id: str) -> Dict[str, Any]:
    entry = _db_get(session_id) or {}
    st = entry.get("master_state")
    if not isinstance(st, dict):
        st = _mnt_default_state_text()
        entry["master_state"] = st
        _db_set(session_id, entry)
        return st

    # defensive defaults
    for k, v in _mnt_default_state_text().items():
        if k not in st:
            st[k] = v

    st["mode"] = MASTER_MODE
    st["updated_at"] = _now()
    entry["master_state"] = st
    _db_set(session_id, entry)
    return st

def _mnt_save_state_text(session_id: str, st: Dict[str, Any]) -> None:
    entry = _db_get(session_id) or {}
    st = st or {}
    st["updated_at"] = _now()
    entry["master_state"] = st
    _db_set(session_id, entry)

def _mnt_reset_state_text(session_id: str) -> Dict[str, Any]:
    entry = _db_get(session_id) or {}
    entry["master_state"] = _mnt_default_state_text()
    _db_set(session_id, entry)
    return entry["master_state"]

def _mnt_extract_user_message(payload: Dict[str, Any]) -> str:
    # prefer template-specific keys, fallback to generic extractor
    for key in ("user_message", "message", "text", "query", "input"):
        s = _safe_str(payload.get(key))
        if s and s.lower() not in _FALSEY_STRS:
            return s
    return _extract_user_text(payload)

def _mnt_extract_focus(payload: Dict[str, Any]) -> Tuple[str, str]:
    active_section_id = _safe_str(payload.get("active_section_id") or payload.get("active_section") or payload.get("section_id"))
    focus_field = _safe_str(payload.get("focus_field") or payload.get("active_field") or payload.get("field_key"))
    return active_section_id, focus_field

def _mnt_extract_deal_value(payload: Dict[str, Any]) -> Optional[float]:
    dv = payload.get("deal_value")
    if dv is None:
        dv = payload.get("dealValue")
    if dv is None:
        return None
    if isinstance(dv, (int, float)):
        return float(dv)
    if isinstance(dv, str):
        s = dv.strip().replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    return None

def _mnt_slots(st: Dict[str, Any]) -> Dict[str, Any]:
    slots = st.get("slots")
    if not isinstance(slots, dict):
        slots = {}
        st["slots"] = slots
    return slots

def _mnt_history(st: Dict[str, Any]) -> List[Dict[str, str]]:
    hist = st.get("history")
    if not isinstance(hist, list):
        hist = []
        st["history"] = hist
    # keep it small
    if len(hist) > 12:
        st["history"] = hist[-12:]
        hist = st["history"]
    return hist

def _mnt_push_history(st: Dict[str, Any], user_msg: str, assistant_msg: str) -> None:
    hist = _mnt_history(st)
    u = (user_msg or "").strip()
    a = (assistant_msg or "").strip()
    if u or a:
        hist.append({"u": _truncate_words(u, 80), "a": _truncate_words(a, 90)})
    if len(hist) > 12:
        del hist[:-12]

def _mnt_parse_percent(text: str) -> Optional[float]:
    t = (text or "").lower()
    # match "10%" or "10 %"
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%\b", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # match "10 percent"
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:percent|per cent)\b", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _mnt_parse_money(text: str) -> Optional[float]:
    t = (text or "").strip()
    if not t:
        return None
    # $60000, 60000, 60,000, 60 000
    m = re.search(r"(?:\$|usd\s*)?(\d{2,3}(?:[\s,]\d{3})+|\d{2,})\b", t.lower())
    if not m:
        return None
    s = m.group(1).replace(" ", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def _mnt_parse_days_terms(text: str) -> Optional[int]:
    t = (text or "").lower()
    m = re.search(r"(\d{1,3})\s*(?:days|day)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _mnt_is_repeat_complaint(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in (
        "i have answered", "i already answered", "answered before", "you asked", "repeat", "again",
        "hello??", "hello ?", "you ignore", "ignoring"
    ))

def _mnt_extract_slots_from_user(st: Dict[str, Any], user_message: str) -> None:
    slots = _mnt_slots(st)
    t = (user_message or "").strip()
    tl = t.lower()

    # Topic: price increase / price decrease
    if "price increase" in tl or "increase price" in tl or "raise price" in tl:
        slots.setdefault("topic", "price_increase")
    if "price reduction" in tl or "price decrease" in tl or "discount" in tl:
        slots.setdefault("topic", "price_change")

    pct = _mnt_parse_percent(t)
    if pct is not None:
        slots["target_percent"] = pct
        # anchor 3–5% higher for opening (store once)
        if "anchor_percent" not in slots:
            slots["anchor_percent"] = min(100.0, pct + 3.0)

    money = _mnt_parse_money(t)
    if money is not None:
        # if we already have target_amount, keep both (current vs target)
        if "current_amount" not in slots and ("current" in tl or "now" in tl or "is" in tl):
            slots["current_amount"] = money
        else:
            slots["amount"] = money

    days = _mnt_parse_days_terms(t)
    if days is not None:
        # try to infer whether it's "offer" or "current"
        if "instead of" in tl or "from" in tl:
            slots["payment_terms_offer_days"] = days
        else:
            # if user says "30 days" early, treat as current; else treat as offer
            if "payment_terms_current_days" not in slots and days <= 45:
                slots["payment_terms_current_days"] = days
            else:
                slots["payment_terms_offer_days"] = days

    # Trade variable hint
    if "payment terms" in tl or "terms" in tl:
        slots.setdefault("trade_variable", "payment_terms")

    # Start date hint
    if "next month" in tl:
        slots["start_timing"] = "next_month"

def _mnt_build_state_memory_text(st: Dict[str, Any]) -> str:
    slots = _mnt_slots(st)
    parts = []
    if slots:
        # deterministic order
        keys = [
            "topic", "target_percent", "anchor_percent", "amount", "current_amount",
            "payment_terms_current_days", "payment_terms_offer_days", "trade_variable", "start_timing"
        ]
        for k in keys:
            if k in slots and slots.get(k) not in (None, "", [], {}):
                parts.append(f"{k}={slots.get(k)}")
        # any extras
        for k, v in slots.items():
            if k not in keys and v not in (None, "", [], {}):
                parts.append(f"{k}={v}")
    stage = (st.get("stage") or "").strip()
    lastq = (st.get("last_question") or "").strip()
    mem = ""
    if stage:
        mem += f"STAGE={stage}\n"
    if lastq:
        mem += f"LAST_QUESTION={lastq}\n"
    if parts:
        mem += "SLOTS=" + "; ".join(parts) + "\n"
    # recent history (last 4 turns)
    hist = _mnt_history(st)[-4:]
    if hist:
        mem += "RECENT_TURNS:\n"
        for h in hist:
            mem += f"- U: {h.get('u','')}\n  A: {h.get('a','')}\n"
    return mem.strip()

def _mnt_rule_based_response(user_message: str, st: Dict[str, Any]) -> Optional[str]:
    """Returns a ready response when we can answer deterministically; else None."""
    t = (user_message or "").strip()
    tl = t.lower()
    slots = _mnt_slots(st)

    # If the user complains about repetition, recap what we already have and ask the NEXT missing thing.
    if _mnt_is_repeat_complaint(t):
        lines = []
        if "target_percent" in slots:
            lines.append(f"You said your target is {slots['target_percent']}%.")
            if "anchor_percent" in slots:
                lines.append(f"Open at {slots['anchor_percent']}%.")
        if "amount" in slots:
            lines.append(f"You mentioned {int(slots['amount']) if float(slots['amount']).is_integer() else slots['amount']}.")
        if "payment_terms_offer_days" in slots:
            lines.append(f"You can trade payment terms: {slots['payment_terms_offer_days']} days.")
        if not lines:
            lines.append("I’m with you. We’ll go step by step.")
        # Decide next question
        if "payment_terms_offer_days" not in slots and ("payment terms" in tl or slots.get("trade_variable") == "payment_terms"):
            q = "What payment terms will you offer (e.g., 60 days instead of 30)?"
        elif "trade_variable" not in slots:
            q = "What variable will you trade first: payment terms, contract length, or scope?"
        else:
            q = "What is their concrete commitment you want in return?"
        st["last_question"] = q
        return "\n".join(lines + ["", q]).strip()

    # Where to start / which question first: pick next missing slot based on what we already know.
    if any(p in tl for p in ("where should i start", "which question", "help me", "i need help", "start", "i dont know", "i don't know")):
        # If goal already set, move forward, don't restart.
        if "target_percent" in slots or "amount" in slots:
            q = "What will you trade first: payment terms, contract length, or scope?"
            st["last_question"] = q
            st["stage"] = st.get("stage") or "trade"
            return "Stay on your goal. Now pick your first trade variable.\n\nA) Payment terms\nB) Contract length\nC) Scope\n\nWhich one: A, B, or C?"
        else:
            st["stage"] = "goal"
            q = "Is your goal a price increase, a discount, or a term change?"
            st["last_question"] = q
            return "Pick your goal.\n\nA) Price increase\nB) Price reduction\nC) Term / scope change\n\nWhich one: A, B, or C?"

    # Variables question: answer directly using slots.
    if "variable" in tl and ("what" in tl or "put" in tl or "write" in tl):
        # Provide a short list and ask user to choose 2 non-price variables (prevents jumping straight to price)
        lines = [
            "Variables are the things you can trade.",
            "- Payment terms",
            "- Contract length",
            "- Scope / deliverables",
            "- Volume / commitment",
            "- Support / service level",
            "- Start date / timing",
        ]
        q = "Pick two you can trade today (not price). Which two?"
        st["last_question"] = q
        st["stage"] = "variables"
        return "\n".join(lines + ["", q])

    return None

def _truncate_words(text: str, max_words: int = 140) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    words = t.split()
    if len(words) <= max_words:
        return t
    return " ".join(words[:max_words]).strip()


def _extract_last_question(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.endswith("?"):
            return ln
    return ""


def _master_llm_text(
    user_message: str,
    active_section_id: str,
    focus_field: str,
    deal_value: Optional[float],
    user_name: str,
    clarify_count: int,
    info: str,
    state_memory: str = "",
) -> str:
    # Compact user prompt. INFORMATION is retrieved from Pinecone.
    deal_line = "" if deal_value is None else f"DEAL_VALUE: {deal_value}\n"
    name_line = user_name.strip() if user_name else ""
    mem_line = (state_memory or "").strip()

    prompt_user = (
        f"USER_NAME: {name_line}\n"
        f"ACTIVE_SECTION_ID: {active_section_id}\n"
        f"FOCUS_FIELD: {focus_field}\n"
        f"{deal_line}"
        f"USER_MESSAGE: {user_message}\n\n"
        f"STATE_MEMORY:\n{mem_line}\n\n"
        f"INFORMATION:\n{info}\n\n"
        "TASK: Give Diadem-only, template-ready guidance for the MASTER template.\n"
        "If FOCUS_FIELD is set: tell the user exactly what to type there and give 1–2 paste-ready lines.\n"
        "If user asks what variables: propose 3–6 variables (short) and show how to anchor them in time or price.\n"
        "Do NOT refuse. If INFORMATION is thin, still give best-effort Diadem guidance.\n"
    )

    messages = [
        {"role": "system", "content": MASTER_SYSTEM_PROMPT_TEXT},
        {"role": "user", "content": prompt_user},
    ]
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    text = strip_markdown_chars((resp.choices[0].message.content or "").strip())

    # Never allow the generic refusal line in this mode
    if text.strip().lower() in ("i can't find this in the provided documents.", "i can’t find this in the provided documents."):
        # Minimal, still useful fallback
        if focus_field:
            text = (
                "EARTH\n"
                f"- Stay in the '{focus_field}' field.\n"
                "- Use short, declarative bullets.\n"
                "- Trade: If you... then I... (IF = their commitment, THEN = our concession).\n\n"
                "Template line:\n"
                "If you commit to [their concrete commitment], then I will [our concession]."
            )
        else:
            text = (
                "WATER\n"
                "- Tell me which field you’re filling (deal value / variables / goals / walk-away / concessions).\n"
                "- Then I’ll give you exact template wording."
            )

    return _truncate_words(text, 180)



def master_template_turn_text(payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    _cleanup_sessions()

    reset = _as_bool(payload.get("reset")) or _as_bool(payload.get("start_again")) or _as_bool(payload.get("restart"))
    if reset:
        st = _mnt_reset_state_text(session_id)
    else:
        st = _mnt_load_state_text(session_id)

    user_message = _mnt_extract_user_message(payload)
    active_section_id, focus_field = _mnt_extract_focus(payload)

    template_id = (payload.get("template_id") or payload.get("templateId") or payload.get("template") or "").strip()
    if template_id:
        st["template_id"] = template_id

    if active_section_id:
        st["active_section_id"] = active_section_id
    if focus_field:
        st["focus_field"] = focus_field

    dv = _mnt_extract_deal_value(payload)
    if dv is not None:
        st["deal_value"] = dv


    # help_accepted: explicit flag OR infer from user's "yes/no" text at the consent step
    help_accepted = payload.get("help_accepted")
    if help_accepted is None:
        help_accepted = payload.get("helpAccepted")
    if help_accepted is not None:
        st["help_accepted"] = _as_bool(help_accepted)
    elif st.get("help_accepted") is None:
        # Bubble may only send user_message="yes"/"no". Infer consent from text once.
        yn = _parse_yes_no(user_message)
        if yn is not None:
            st["help_accepted"] = yn


    user_name = _extract_user_name(payload)

    # First touch: always greet (no Yes/No gating)
    if st.get("help_accepted") is None:
        st["help_offered"] = True
        st["help_accepted"] = True  # user is already inside the MASTER template
        _mnt_save_state_text(session_id, st)
        return {
            "session_id": session_id,
            "mode": MASTER_MODE,
            "text": "Hello! How can I assist you with the MASTER negotiation template today?",
            "done": False,
        }

    # User explicitly disabled help
    if st.get("help_accepted") is False:
        _mnt_save_state_text(session_id, st)
        return {
            "session_id": session_id,
            "mode": MASTER_MODE,
            "text": "Okay. Tell me the section/field when you want help.",
            "done": False,
        }

    # Update slot memory from the user's message (percent, money, terms, etc.)
    _mnt_extract_slots_from_user(st, user_message)

    # Deterministic handlers to avoid repeating questions
    rb = _mnt_rule_based_response(user_message, st)
    if rb:
        _mnt_push_history(st, user_message, rb)
        _mnt_save_state_text(session_id, st)
        return {"session_id": session_id, "mode": MASTER_MODE, "text": rb, "done": False}

    # Smalltalk inside the template: greet + refocus
    if _is_smalltalk(user_message):
        _mnt_save_state_text(session_id, st)
        ff = (st.get("focus_field") or "").strip()
        if ff:
            return {"session_id": session_id, "mode": MASTER_MODE, "text": f"You’re on '{ff}'. What do you want to write there?", "done": False}
        sec = (st.get("active_section_id") or "").strip()
        if sec:
            return {"session_id": session_id, "mode": MASTER_MODE, "text": f"You’re in '{sec}'. What do you need to write?", "done": False}
        return {"session_id": session_id, "mode": MASTER_MODE, "text": "Which part are you filling right now (deal value, variables, goals, walk-away, concessions)?", "done": False}


    # If user hasn't sent anything, prompt gently — but NEVER lose focus
    if not user_message:
        _mnt_save_state_text(session_id, st)

        ff = (st.get("focus_field") or "").strip()
        sec = (st.get("active_section_id") or "").strip()

        if ff:
            return {
                "session_id": session_id,
                "mode": MASTER_MODE,
                "text": f"You’re working on '{ff}'. What are you unsure about in this field?",
                "done": False,
            }

        if sec:
            return {
                "session_id": session_id,
                "mode": MASTER_MODE,
                "text": f"You’re in the '{sec}' section. What decision are you trying to make here?",
                "done": False,
            }

        return {
            "session_id": session_id,
            "mode": MASTER_MODE,
            "text": "Which part are you filling right now (deal value, variables, goals, walk-away, concessions, etc.)?",
            "done": False,
        }

    # If deal value missing and user is in that field (or mentions it), nudge but still answer
    # (We keep this light because user may want help elsewhere.)
    needs_deal_value_hint = st.get("deal_value") is None and (focus_field.lower() in ("deal_value", "dealvalue", "value") or "deal value" in user_message.lower())


    # --- RAG retrieval for MASTER template (always) ---
    request_id = str(uuid.uuid4())[:8]
    rag_query = f"master_template {active_section_id} {focus_field}: {user_message}".strip()

    # 1) Prefer Master Negotiator Slides (primary source for this mode)
    raw = get_matches(rag_query, TOP_K, request_id=request_id)
    matches = [m for m in (raw or []) if "master negotiator slides" in str((m.get("metadata") or {}).get("file") or "").lower()
               or "master negotiator slides" in str((m.get("metadata") or {}).get("source") or "").lower()]

    # 2) If not enough, broaden to negotiation docs (fallback)
    if len(matches) < 2:
        matches = raw or []
        if len(matches) < 2:
            raw2 = get_matches(rag_query + " negotiation", TOP_K, request_id=request_id)
            matches = raw2 or []

    info = build_context(matches, request_id=request_id) if matches else ""
    clarify_count = int(st.get("clarify_count") or 0)


    # Fetch template state from Bubble (read-only) and inject into prompt
    template_state_text = ""
    try:
        cache = st.get("_tpl_cache") or {}
        cache_tid = str(cache.get("template_id") or "")
        cache_ts = float(cache.get("ts") or 0)
        if st.get("template_id"):
            now = time.time()
            if cache_tid == st.get("template_id") and (now - cache_ts) < 4 and cache.get("text"):
                template_state_text = str(cache.get("text") or "")
            else:
                template_state_text = _fetch_template_state_text(st.get("template_id"), request_id=request_id)
                st["_tpl_cache"] = {"template_id": st.get("template_id"), "ts": now, "text": template_state_text}
    except Exception as _e:
        template_state_text = ""

    # Generate guidance text (text-only)
    try:
        text = _master_llm_text(
            user_message=user_message,
            active_section_id=st.get("active_section_id") or "",
            focus_field=st.get("focus_field") or "",
            deal_value=st.get("deal_value"),
            user_name=user_name,
            clarify_count=clarify_count,
            state_memory=(_mnt_build_state_memory_text(st) + ("\n\n" + template_state_text if template_state_text else "")),
            info=info,
        )
    except Exception as e:
        _jlog("master_template_llm_error", session_id=session_id, err=str(e)[:800])
        text = "Server error."

    if needs_deal_value_hint and "deal value" not in (text or "").lower():
        # add one short line if not already covered
        text = (text + "\n\nDeal value: enter the total commercial value as a number (e.g., 120000).").strip()

    
    # Persist conversational memory so the assistant can reference what was already asked/answered.
    if user_message and text:
        _mnt_push_history(st, user_message, text)
    q = _extract_last_question(text)
    if q:
        st["last_question"] = q

    _mnt_save_state_text(session_id, st)
    return {"session_id": session_id, "mode": MASTER_MODE, "text": text, "done": False}


@app.post("/master/template")
def master_template(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    try:
        out = master_template_turn_text(payload, session_id=session_id)
        return JSONResponse(out)
    except Exception as e:
        _jlog("master_template_error", session_id=session_id, err=str(e))
        return JSONResponse({"text": "Server error.", "session_id": session_id, "mode": MASTER_MODE, "done": False}, status_code=500)


@app.post("/master/template/sse")
def master_template_sse(payload: Dict = Body(...)):
    """SSE for MASTER template (streams assistant text)."""
    session_id = _get_or_create_session_id(payload)

    def gen():
        start_payload = json.dumps({"session_id": session_id, "mode": MASTER_MODE}, ensure_ascii=False)
        yield f"event: start\ndata: {start_payload}\n\n"

        try:
            # Load/update state (same as non-SSE) but without the consent gate
            _cleanup_sessions()
            reset = _as_bool(payload.get("reset")) or _as_bool(payload.get("start_again")) or _as_bool(payload.get("restart"))
            st = _mnt_reset_state_text(session_id) if reset else _mnt_load_state_text(session_id)

            user_message = _mnt_extract_user_message(payload)
            active_section_id, focus_field = _mnt_extract_focus(payload)

            template_id = (payload.get("template_id") or payload.get("templateId") or payload.get("template") or "").strip()
            if template_id:
                st["template_id"] = template_id
            if active_section_id:
                st["active_section_id"] = active_section_id
            if focus_field:
                st["focus_field"] = focus_field
            dv = _mnt_extract_deal_value(payload)
            if dv is not None:
                st["deal_value"] = dv

            # First call greeting (if Bubble sends 'hi' as first message)
            if st.get("help_accepted") is None:
                st["help_offered"] = True
                st["help_accepted"] = True

            user_name = _extract_user_name(payload)

            # Update slot memory from the user's message
            _mnt_extract_slots_from_user(st, user_message)

            # Deterministic handlers to avoid repeating questions (SSE mode)
            rb = _mnt_rule_based_response(user_message, st)
            if rb:
                data = json.dumps({"text": rb}, ensure_ascii=False)
                yield f"event: chunk\ndata: {data}\n\n"
                _mnt_push_history(st, user_message, rb)
                _mnt_save_state_text(session_id, st)
                done_payload = json.dumps({"done": True, "session_id": session_id, "mode": MASTER_MODE}, ensure_ascii=False)
                yield f"event: done\ndata: {done_payload}\n\n"
                return

            if not user_message or _is_smalltalk(user_message):
                txt = "Hello! How can I assist you with the MASTER negotiation template today?"
                data = json.dumps({"text": txt}, ensure_ascii=False)
                yield f"event: chunk\ndata: {data}\n\n"
                _mnt_save_state_text(session_id, st)
                done_payload = json.dumps({"done": True, "session_id": session_id, "mode": MASTER_MODE}, ensure_ascii=False)
                yield f"event: done\ndata: {done_payload}\n\n"
                return

            # Retrieval (prefer Master Negotiator Slides)
            request_id = str(uuid.uuid4())[:8]
            rag_query = f"master_template {st.get('active_section_id','')} {st.get('focus_field','')}: {user_message}".strip()
            raw = get_matches(rag_query, TOP_K, request_id=request_id)
            matches = [m for m in (raw or []) if "master negotiator slides" in str((m.get("metadata") or {}).get("file") or "").lower()
                       or "master negotiator slides" in str((m.get("metadata") or {}).get("source") or "").lower()]
            if len(matches) < 2:
                matches = raw or []
            info = build_context(matches, request_id=request_id) if matches else ""

            # Build the same prompt as _master_llm_text, but stream
            deal_value = st.get("deal_value")
            deal_line = "" if deal_value is None else f"DEAL_VALUE: {deal_value}\n"
            name_line = user_name.strip() if user_name else ""

            # Fetch template state from Bubble (read-only) and inject into prompt
            template_state_text = ""
            try:
                cache = st.get("_tpl_cache") or {}
                cache_tid = str(cache.get("template_id") or "")
                cache_ts = float(cache.get("ts") or 0)
                if st.get("template_id"):
                    now = time.time()
                    if cache_tid == st.get("template_id") and (now - cache_ts) < 4 and cache.get("text"):
                        template_state_text = str(cache.get("text") or "")
                    else:
                        template_state_text = _fetch_template_state_text(st.get("template_id"), request_id=request_id)
                        st["_tpl_cache"] = {"template_id": st.get("template_id"), "ts": now, "text": template_state_text}
            except Exception:
                template_state_text = ""

            state_mem = _mnt_build_state_memory_text(st) + ("\n\n" + template_state_text if template_state_text else "")

            prompt_user = (
                f"USER_NAME: {name_line}\n"
                f"ACTIVE_SECTION_ID: {st.get('active_section_id','')}\n"
                f"FOCUS_FIELD: {st.get('focus_field','')}\n"
                f"{deal_line}"
                f"USER_MESSAGE: {user_message}\n\n"
                f"STATE_MEMORY:\n{state_mem}\n\n"
                f"INFORMATION:\n{info}\n\n"
                "TASK: Give Diadem-only, template-ready guidance for the MASTER template.\n"
                "Do NOT refuse.\n"
            )
            messages = [
                {"role": "system", "content": MASTER_SYSTEM_PROMPT_TEXT},
                {"role": "user", "content": prompt_user},
            ]

            full_parts: List[str] = []
            for part in _iter_text_as_sse_chunks(_openai_stream_text(messages, model=CHAT_MODEL, temperature=0.2), min_chars=28):
                part = strip_markdown_chars(part)
                if part:
                    full_parts.append(part)
                data = json.dumps({"text": part}, ensure_ascii=False)
                yield f"event: chunk\ndata: {data}\n\n"

            full_text = strip_markdown_chars("".join(full_parts)).strip()
            if user_message and full_text:
                _mnt_push_history(st, user_message, full_text)
            q = _extract_last_question(full_text)
            if q:
                st["last_question"] = q

            _mnt_save_state_text(session_id, st)
            done_payload = json.dumps({"done": True, "session_id": session_id, "mode": MASTER_MODE}, ensure_ascii=False)
            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception as e:
            _jlog("master_template_sse_error", session_id=session_id, err=str(e)[:800])
            err_chunk = json.dumps({"text": "Internal error. Please retry."}, ensure_ascii=False)
            yield f"event: chunk\ndata: {err_chunk}\n\n"
            err_done = json.dumps({"done": True, "session_id": session_id, "mode": MASTER_MODE}, ensure_ascii=False)
            yield f"event: done\ndata: {err_done}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_sse_headers())




@app.post("/master/template/reset")
def master_template_reset(payload: Dict = Body(...)):
    session_id = _get_or_create_session_id(payload)
    _jlog("master_template_reset", session_id=session_id)
    _mnt_reset_state_text(session_id)
    return JSONResponse({"ok": True, "session_id": session_id, "mode": MASTER_MODE})
