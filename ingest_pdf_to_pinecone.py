import os
import io
import json
import uuid
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import tiktoken
from pinecone import Pinecone
from anthropic import Anthropic
from openai import OpenAI

# -------------------------
# ENV CONFIG
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. us-east-1
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # diadem-ai

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 150
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

# Optional slide image enrichment (for Bubble/Drive-hosted images)
SLIDE_INDEX_PATH = os.getenv("SLIDE_INDEX_PATH", "slide_index.json")
SLIDE_URL_MAP_PATH = os.getenv("SLIDE_URL_MAP_PATH", "slide_image_urls.json")
SLIDE_IMAGE_BASE_URL = os.getenv("SLIDE_IMAGE_BASE_URL", "").strip().rstrip("/")
ATTACH_SLIDE_URLS_FOR_ALL_PDFS = os.getenv("ATTACH_SLIDE_URLS_FOR_ALL_PDFS", "0").strip().lower() in ("1", "true", "yes", "y", "on")

if not all([OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENV]):
    raise RuntimeError("Missing required environment variables")

# -------------------------
# CLIENT INIT
# -------------------------
openai = OpenAI(api_key=OPENAI_API_KEY)
anthropod = Anthropic(api_key=ANTHROPIC_API_KEY)


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Align embedding output dimension with the Pinecone index dimension.
# This prevents 400 errors when index dimension differs from model default.
INDEX_DIM = EMBED_DIM
try:
    index_info = pc.describe_index(PINECONE_INDEX_NAME)
    if isinstance(index_info, dict):
        INDEX_DIM = int(index_info.get("dimension") or INDEX_DIM)
    else:
        INDEX_DIM = int(getattr(index_info, "dimension", INDEX_DIM) or INDEX_DIM)
except Exception:
    INDEX_DIM = EMBED_DIM

ENC = tiktoken.get_encoding("cl100k_base")


def _safe_read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _parse_page_from_slide_id(slide_id: str) -> int:
    # Expects values like slide_001, slide_079
    if not isinstance(slide_id, str):
        return 0
    try:
        tail = slide_id.split("_")[-1]
        return int(tail)
    except Exception:
        return 0


def _load_slide_assets() -> Dict[str, Any]:
    """Build by-page image URL + metadata from optional local mapping files."""
    by_page: Dict[int, str] = {}
    meta_by_page: Dict[int, Dict[str, str]] = {}

    # 1) slide_index.json: supports building URLs from base + file names
    index_data = _safe_read_json(SLIDE_INDEX_PATH)
    slides = index_data.get("slides") if isinstance(index_data.get("slides"), dict) else {}
    for slide_id, slide in slides.items():
        page = _parse_page_from_slide_id(slide_id)
        if page <= 0 or not isinstance(slide, dict):
            continue

        file_name = str(slide.get("file") or "").strip()
        if file_name and SLIDE_IMAGE_BASE_URL:
            by_page[page] = f"{SLIDE_IMAGE_BASE_URL}/{file_name}"

        title = str(slide.get("title") or "").strip()
        topics = slide.get("topics") if isinstance(slide.get("topics"), list) else []
        tags = ", ".join(str(t).strip() for t in topics if str(t).strip())
        meta_by_page[page] = {
            "slide_id": slide_id,
            "name": title,
            "category": "negotiation_slide",
            "tags": tags,
        }

    # 2) slide_image_urls.json override mapping (if provided)
    # Supported shapes:
    # {"by_page": {"1": "https://...", "2": "https://..."}}
    # {"1": "https://...", "2": "https://..."}
    # {"slide_001": "https://..."}
    # {"slide_001": {"image_url": "https://..."}}
    map_data = _safe_read_json(SLIDE_URL_MAP_PATH)
    if map_data:
        scope = map_data.get("by_page") if isinstance(map_data.get("by_page"), dict) else map_data
        if isinstance(scope, dict):
            for k, v in scope.items():
                page = 0
                if isinstance(k, str) and k.startswith("slide_"):
                    page = _parse_page_from_slide_id(k)
                else:
                    try:
                        page = int(str(k))
                    except Exception:
                        page = 0
                if page <= 0:
                    continue

                if isinstance(v, str):
                    url = v.strip()
                elif isinstance(v, dict):
                    url = str(v.get("image_url") or v.get("url") or "").strip()
                else:
                    url = ""
                if url:
                    by_page[page] = url

    return {"by_page": by_page, "meta_by_page": meta_by_page}


SLIDE_ASSETS = _load_slide_assets()


def _should_attach_slide_assets(source_name: str) -> bool:
    if ATTACH_SLIDE_URLS_FOR_ALL_PDFS:
        return True
    s = (source_name or "").lower()
    return "master negotiator" in s


def _enrich_slide_metadata(base_md: Dict[str, Any], source_name: str, page: int) -> Dict[str, Any]:
    md = dict(base_md)
    if not _should_attach_slide_assets(source_name):
        return md

    by_page = SLIDE_ASSETS.get("by_page") if isinstance(SLIDE_ASSETS.get("by_page"), dict) else {}
    meta_by_page = SLIDE_ASSETS.get("meta_by_page") if isinstance(SLIDE_ASSETS.get("meta_by_page"), dict) else {}

    slide_url = by_page.get(page)
    if slide_url:
        md["image_url"] = slide_url

    s_meta = meta_by_page.get(page) or {}
    if s_meta.get("slide_id"):
        md["id"] = s_meta.get("slide_id")
    if s_meta.get("name"):
        md["name"] = s_meta.get("name")
    if s_meta.get("category"):
        md["category"] = s_meta.get("category")
    if s_meta.get("tags"):
        md["tags"] = s_meta.get("tags")

    # Keep a rich description field for vector traceability/UI payloads
    if md.get("text"):
        md["description"] = md.get("text")

    return md

# -------------------------
# PDF EXTRACTION
# -------------------------
def extract_pdf_items(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    items = []

    for page_number in range(len(doc)):
        page = doc[page_number]

        # Text blocks (slide text) - combine all blocks from same page
        blocks = page.get_text("blocks")
        text_blocks = []
        for block in blocks:
            text = block[4].strip()
            if text:
                # Filter out copyright/footer text that's not useful
                if "©" in text or "All Rights Reserved" in text:
                    continue
                text_blocks.append(text)
        
        # Combine all text blocks from the page into one text string
        # This ensures titles and their content stay together
        if text_blocks:
            combined_text = "\n".join(text_blocks)
            items.append({
                "type": "text",
                "page": page_number + 1,
                "text": combined_text
            })

        # Images (charts / diagrams)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
            pix = None

            items.append({
                "type": "image",
                "page": page_number + 1,
                "image": image,
                "img_index": img_index
            })

    doc.close()
    return items

# -------------------------
# OCR + IMAGE DESCRIPTION
# -------------------------
def ocr_image(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""

def describe_image(ocr_text: str) -> str:
    if not ocr_text:
        return ""

    prompt = (
        "This text was extracted from a negotiation training slide image.\n"
        "Summarise what the visual represents in 1–2 sentences and list 2 key negotiation takeaways.\n\n"
        f"OCR TEXT:\n{ocr_text}"
    )

    response = anthropod.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0
    )

    return response.content[0].text.strip()

# -------------------------
# CHUNKING
# -------------------------
def chunk_text(text: str) -> List[str]:
    tokens = ENC.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunks.append(ENC.decode(tokens[start:end]))
        start += CHUNK_TOKENS - CHUNK_OVERLAP

    return chunks

# -------------------------
# DOC PREPARATION
# -------------------------
# def prepare_documents(items: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
#     documents = []

#     for item in items:
#         page = item["page"]

#         if item["type"] == "text":
#             chunks = chunk_text(item["text"])
#             for idx, chunk in enumerate(chunks, 1):
#                 documents.append({
#                     "id": f"text-p{page}-{idx}-{uuid.uuid4().hex[:8]}",
#                     "text": chunk,
#                     "metadata": {
#                         "page": page,
#                         "type": "text",
#                         "source": source_name,
#                         "priority": 10  # HIGH – methodology rules
#                     }
#                 })

#         else:
#             ocr = ocr_image(item["image"])
#             description = describe_image(ocr)
#             content = f"[IMAGE – page {page}]\n{description}\n\nOCR:\n{ocr}"

#             documents.append({
#                 "id": f"img-p{page}-{item['img_index']}-{uuid.uuid4().hex[:8]}",
#                 "text": content,
#                 "metadata": {
#                     "page": page,
#                     "type": "image",
#                     "source": source_name,
#                     "priority": 7  # Visual frameworks
#                 }
#             })

#     return documents
def prepare_documents(items: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    documents = []

    for item in items:
        page = item["page"]

        if item["type"] == "text":
            chunks = chunk_text(item["text"])
            for idx, chunk in enumerate(chunks, 1):
                md = _enrich_slide_metadata({
                    "page": page,
                    "type": "text",
                    "source": source_name,
                    "priority": 10,   # HIGH – methodology rules
                    "text": chunk,    # store text for retrieval/use
                }, source_name, page)
                documents.append({
                    "id": f"text-p{page}-{idx}-{uuid.uuid4().hex[:8]}",
                    "text": chunk,
                    "metadata": md,
                })

        else:
            ocr = ocr_image(item["image"])
            description = describe_image(ocr)
            content = f"[IMAGE – page {page}]\n{description}\n\nOCR:\n{ocr}"

            md = _enrich_slide_metadata({
                "page": page,
                "type": "image",
                "source": source_name,
                "priority": 7,      # Visual frameworks
                "text": content,    # store text for retrieval/use
            }, source_name, page)

            documents.append({
                "id": f"img-p{page}-{item['img_index']}-{uuid.uuid4().hex[:8]}",
                "text": content,
                "metadata": md,
            })

    return documents

# -------------------------
# EMBEDDING + UPSERT
# -------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=INDEX_DIM,
    )
    return [item.embedding for item in response.data]

def upsert_documents(docs: List[Dict[str, Any]]):
    for i in range(0, len(docs), UPSERT_BATCH_SIZE):
        batch = docs[i:i + UPSERT_BATCH_SIZE]
        embeddings = embed_texts([d["text"] for d in batch])

        vectors = [
            (doc["id"], emb, doc["metadata"])
            for doc, emb in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)
        print(f"Upserted batch {i // UPSERT_BATCH_SIZE + 1} ({len(vectors)} vectors)")
        time.sleep(0.2)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest_pdf_to_pinecone.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    print("Extracting PDF content...")
    print(f"Embedding model: {EMBED_MODEL}, dimensions: {INDEX_DIM}")
    items = extract_pdf_items(pdf_path)

    print("Preparing documents...")
    documents = prepare_documents(items, os.path.basename(pdf_path))

    print(f"Embedding & uploading {len(documents)} chunks...")
    upsert_documents(documents)

    print("✅ Ingestion complete.")