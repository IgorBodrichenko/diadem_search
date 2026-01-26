import os
from typing import List, Dict
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from openai import OpenAI
from pinecone import Pinecone

# ====== CONFIG ======
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))

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

# CORS (чтобы Bubble мог дергать API из браузера)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом можно ограничить доменом Bubble
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering ONLY using the provided information.\n"
    "Rules:\n"
    "- Do NOT mention document names, page numbers, sources, citations, or the word 'context'.\n"
    "- Write a natural chatbot answer as plain text.\n"
    "- If the answer is not present, say: \"I can't find this in the provided documents.\".\n"
    "- Be concise and practical.\n"
)

def embed_query(text: str) -> List[float]:
    resp = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        dimensions=512
    )
    return resp.data[0].embedding

def build_context(matches: List[Dict]) -> str:
    seen = set()
    parts = []
    total = 0

    for m in matches:
        md = (m.get("metadata") or {})
        text = (md.get("text") or "").strip()
        file_name = md.get("file_name") or md.get("doc_id") or "Unknown"
        page = md.get("page")

        key = (file_name, page)
        if key in seen:
            continue
        seen.add(key)

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

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: Dict = Body(...)):
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or TOP_K)

    if not query:
        return JSONResponse({"answer": ""})

    matches = get_matches(query, top_k)
    context = build_context(matches)

    user = f"QUESTION:\n{query}\n\nINFORMATION:\n{context}"

    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer = (chat.choices[0].message.content or "").strip()
    return {"answer": answer}

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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    # Простой text stream (не SSE). Bubble можно принимать как “текст по кускам”
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
