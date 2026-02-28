import os
import io
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
from openai import OpenAI

# -------------------------
# ENV CONFIG
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. us-east-1
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # diadem-ai

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # REQUIRED for this model

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 150
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENV]):
    raise RuntimeError("Missing required environment variables")

# -------------------------
# CLIENT INIT
# -------------------------
openai = OpenAI(api_key=OPENAI_API_KEY)


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

ENC = tiktoken.get_encoding("cl100k_base")

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

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

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
                documents.append({
                    "id": f"text-p{page}-{idx}-{uuid.uuid4().hex[:8]}",
                    "text": chunk,
                    "metadata": {
                        "page": page,
                        "type": "text",
                        "source": source_name,
                        "priority": 10,   # HIGH – methodology rules
                        "text": chunk     # ✅ store text for retrieval/use
                    }
                })

        else:
            ocr = ocr_image(item["image"])
            description = describe_image(ocr)
            content = f"[IMAGE – page {page}]\n{description}\n\nOCR:\n{ocr}"

            documents.append({
                "id": f"img-p{page}-{item['img_index']}-{uuid.uuid4().hex[:8]}",
                "text": content,
                "metadata": {
                    "page": page,
                    "type": "image",
                    "source": source_name,
                    "priority": 7,      # Visual frameworks
                    "text": content     # ✅ store text for retrieval/use
                }
            })

    return documents

# -------------------------
# EMBEDDING + UPSERT
# -------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=texts
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
    items = extract_pdf_items(pdf_path)

    print("Preparing documents...")
    documents = prepare_documents(items, os.path.basename(pdf_path))

    print(f"Embedding & uploading {len(documents)} chunks...")
    upsert_documents(documents)

    print("✅ Ingestion complete.")