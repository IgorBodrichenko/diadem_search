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
PINECONE_TOPK_RAW = int(os.getenv("PINECONE_TOPK_RAW", "40"))  # raw candidates from pinecone (per query variant, capped below)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h default

# Retrieval quality knobs
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.30"))  # allow slightly lower, rerank will filter
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "600"))
MIN_OVERLAP_SCORE = float(os.getenv("MIN_OVERLAP_SCORE", "2.0"))  # stricter than before
MIN_KEYWORD_HITS = int(os.getenv("MIN_KEYWORD_HITS", "2"))  # top chunk must hit >= N keywords
DIVERSITY_JACCARD_MAX = float(os.getenv("DIVERSITY_JACCARD_MAX", "0.78"))  # lower = more diverse selection

RETRIEVAL_DEBUG = os.getenv("RETRIEVAL_DEBUG", "0") == "1"

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
    return (
        text.replace("*", "")
            .replace("`", "")
            .replace("_", "")
    )

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

# расширили "плохие старты" — чтобы не было одинаковых "Negotiation techniques are..."
_BAD_START_RE = re.compile(
    r"^\s*(it['’]s\s+(great|wonderful)|great|negotiation\s+techniques\s+are|negotiation\s+techniques\s+can)\b",
    flags=re.IGNORECASE,
)

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
    if user_name and random.random() < 0.45:
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

def _stream_opening_variation(deltas: Iterator[str], session_id: str, user_name: str, field: str) -> Iterator[str]:
    opener = _pick_opener(session_id, user_name, field)
    buf = ""
    decided = False
    for d in deltas:
        if not decided:
            buf += d
            if len(buf) >= 160 or re.search(r"[.!?]\s+", buf):
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
    {"match_any": ["deal with difficult questions", "handle difficult questions", "difficult questions"],
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
# TOKENIZATION / SCORING
# =========================
_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","without","is","are","was","were","be",
    "do","does","did","how","what","why","when","where","between","into","from","as","at","by","it","this","that",
    "your","you","i","we","they","their","our","can","could","should","would"
}

def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS and len(t) > 2]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

# Чем больше это встречается в тексте — тем вероятнее, что это “общий” фрагмент, не ответ на конкретный вопрос
_GENERIC_PHRASES = [
    "negotiation happens on a daily basis",
    "negotiation techniques",
    "self-awareness",
    "emotional intelligence",
    "win-win",
    "face-to-face",
    "master negotiators",
    "key techniques to consider",
    "manage and control your emotions",
]

# Небольшие “якоря” под ваши FAQ (можно расширять)
_KEY_PHRASES = [
    "balance of power",
    "power balance",
    "push back",
    "upsetting the relationship",
    "selling",
    "difference between selling",
    "difficult questions",
    "handle difficult questions",
    "tactics",
    "confident mindset",
    "win zone",
    "zone",
]

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

def _text_of_match(m: Dict) -> str:
    md = m.get("metadata") or {}
    return (md.get("text") or "").strip()

def build_context(matches: List[Dict]) -> str:
    parts: List[str] = []
    total = 0
    for m in matches:
        text = _text_of_match(m)
        if not text:
            continue
        snippet = text if len(text) <= 2500 else text[:2500] + "…"
        block = snippet + "\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)

def _keyword_hits(query: str, text: str) -> int:
    qt = _tokenize(query)
    tt = _tokenize(text)
    st = set(tt)
    return sum(1 for t in set(qt) if t in st)

def _generic_penalty(text_lower: str) -> float:
    p = 0.0
    for gp in _GENERIC_PHRASES:
        if gp in text_lower:
            p += 1.0
    return p

def _phrase_bonus(query_lower: str, text_lower: str) -> float:
    b = 0.0
    for ph in _KEY_PHRASES:
        if ph in query_lower and ph in text_lower:
            b += 1.6
    return b

def _score_candidate(query: str, m: Dict) -> Tuple[float, float, float, int]:
    """
    returns: (final_score, overlap_score, pinecone_score, keyword_hits)
    """
    text = _text_of_match(m)
    if not text:
        return (-1e9, 0.0, 0.0, 0)

    qlow = (query or "").lower()
    tlow = text.lower()

    qt = set(_tokenize(query))
    ttoks = _tokenize(text)

    overlap = sum(1.0 for t in ttoks if t in qt)
    hits = _keyword_hits(query, text)

    try:
        pscore = float(m.get("score") or 0.0)
    except Exception:
        pscore = 0.0

    bonus = _phrase_bonus(qlow, tlow)
    penalty = _generic_penalty(tlow)

    # Pinecone score не даём доминировать, иначе он тащит “общие” куски
    final = (overlap * 1.1) + (hits * 0.9) + bonus + (pscore * 0.35) - (penalty * 1.2)
    return (final, overlap, pscore, hits)

def _mmr_select(query: str, scored: List[Tuple[float, Dict]], k: int) -> List[Dict]:
    """
    Простая диверсификация:
    - берём лучший по score
    - дальше добавляем следующий, если он не слишком похож на уже выбранные (Jaccard по токенам)
    """
    selected: List[Dict] = []
    selected_toksets: List[List[str]] = []

    for score, m in scored:
        if len(selected) >= k:
            break
        text = _text_of_match(m)
        tt = _tokenize(text)
        if not selected:
            selected.append(m)
            selected_toksets.append(tt)
            continue

        too_similar = False
        for st in selected_toksets:
            if _jaccard(tt, st) >= DIVERSITY_JACCARD_MAX:
                too_similar = True
                break
        if too_similar:
            continue

        selected.append(m)
        selected_toksets.append(tt)

    # если диверсификация слишком “жёсткая” и не набрали k — добиваем оставшимися
    if len(selected) < k:
        for score, m in scored:
            if len(selected) >= k:
                break
            if m in selected:
                continue
            selected.append(m)

    return selected

def _query_variants(question: str) -> List[str]:
    """
    Multi-query retrieval: несколько формулировок для Pinecone,
    чтобы не застревать на одном и том же “общем” куске.
    """
    q = (question or "").strip()
    hint = _hint_for_question(q)
    kw = " ".join(_tokenize(q))[:220].strip()

    variants: List[str] = []
    variants.append(q)

    if kw:
        variants.append(f"keywords: {kw}")

    if hint:
        variants.append(f"{q}\nsearch_hint: {hint}")

    # focused variant заставляет эмбеддинг искать “про отличие/определение/тактику”
    variants.append(f"Find the specific section that answers: {q}")

    # убираем дубли
    out = []
    seen = set()
    for v in variants:
        vn = _norm_q(v)
        if vn not in seen:
            out.append(v)
            seen.add(vn)
    return out[:4]

def get_matches(question: str, top_k_final: int) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Improved retrieval pipeline:
    1) Multi-query to Pinecone (variants)
    2) Merge candidates by id
    3) Score candidates (overlap/hits/bonus/penalty + weak pinecone score)
    4) Require keyword hits / overlap
    5) Diversify selection
    """
    variants = _query_variants(question)

    # распределяем raw k по вариантам
    per_q = max(12, min(PINECONE_TOPK_RAW, 60))
    if len(variants) >= 2:
        per_q = max(10, per_q // len(variants) + 8)

    merged: Dict[str, Dict] = {}
    debug_rows = []

    for v in variants:
        qvec = embed_query(v)
        res = index.query(vector=qvec, top_k=per_q, include_metadata=True)
        raw = _filter_matches_by_score(res.get("matches") or [])
        for m in raw:
            mid = str(m.get("id") or "") or None
            if not mid:
                # если нет id — делаем псевдо id по тексту
                t = _text_of_match(m)[:80]
                mid = f"noid:{hash(t)}"
                m["id"] = mid

            # сохраняем лучший pinecone score для одного id
            old = merged.get(mid)
            if not old:
                merged[mid] = m
            else:
                try:
                    if float(m.get("score") or 0.0) > float(old.get("score") or 0.0):
                        merged[mid] = m
                except Exception:
                    pass

    candidates = list(merged.values())

    scored: List[Tuple[float, Dict]] = []
    for m in candidates:
        text = _text_of_match(m)
        if not text:
            continue

        final, overlap, pscore, hits = _score_candidate(question, m)

        # жёсткий барьер: кандидат должен реально “цепляться” за вопрос
        if hits < MIN_KEYWORD_HITS and overlap < MIN_OVERLAP_SCORE:
            continue

        scored.append((final, m))

        if RETRIEVAL_DEBUG:
            debug_rows.append({
                "id": str(m.get("id")),
                "pscore": float(m.get("score") or 0.0),
                "final": final,
                "overlap": overlap,
                "hits": hits,
                "text_head": text[:140].replace("\n", " "),
            })

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = _mmr_select(question, scored, top_k_final)

    ctx = build_context(selected)
    ok = len(ctx.strip()) >= MIN_CONTEXT_CHARS

    # Если всё плохо — возвращаем пусто, чтобы модель сказала "can't find"
    if not ok:
        selected = []
        ctx = ""

    meta = {
        "variants": variants,
        "candidates": len(candidates),
        "scored": len(scored),
        "selected": len(selected),
        "context_chars": len(ctx),
    }
    if RETRIEVAL_DEBUG:
        meta["top_debug"] = debug_rows[:12]
    return selected, meta

# =========================
# SYSTEM PROMPT (QA) — stronger “must be specific”
# =========================
SYSTEM_PROMPT_QA = (
    "You are a friendly, helpful assistant.\n"
    "You must answer ONLY using the provided INFORMATION.\n\n"
    "Hard rules:\n"
    "- Do NOT mention document names, page numbers, sources, citations, Pinecone, or the word 'context'.\n"
    "- Output plain text only. NO markdown.\n"
    "- If the answer is not clearly present in the INFORMATION, say exactly:\n"
    "  \"I can't find this in the provided documents.\".\n"
    "- The answer MUST directly address the QUESTION (not generic advice).\n"
    "- Prefer 2–5 concrete points that are clearly supported by INFORMATION.\n"
    "- Always end your message with a question.\n"
)

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)
    user_name = _extract_user_name(payload)

    session_id = _get_or_create_session_id(payload)
    _cleanup_sessions()

    if not query:
        return JSONResponse({"answer": "", "session_id": session_id})

    matches, meta = get_matches(query, top_k)
    context = build_context(matches)  # may be empty

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

    opener = _pick_opener(session_id, user_name, "qa_last_opener")
    answer = _rewrite_bad_opening(answer, opener)

    out = {"answer": answer, "session_id": session_id}
    if RETRIEVAL_DEBUG:
        out["retrieval"] = meta
    return out

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

    matches, meta = get_matches(query, top_k)
    context = build_context(matches)

    user = (
        f"USER_NAME:\n{user_name}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"INFORMATION:\n{context}"
    )

    def gen():
        start_payload = {"session_id": session_id}
        if RETRIEVAL_DEBUG:
            start_payload["retrieval"] = meta
        yield f"event: start\ndata: {json.dumps(start_payload, ensure_ascii=False)}\n\n"

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

        for delta in _stream_opening_variation(raw_deltas(), session_id=session_id, user_name=user_name, field="qa_last_opener"):
            data = json.dumps({"text": delta}, ensure_ascii=False)
            yield f"event: chunk\ndata: {data}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers())
