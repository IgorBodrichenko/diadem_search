# Diadem Search Test v3 - Detailed Technical Documentation

## 1. Purpose

This project is a Python FastAPI backend for a Diadem/Master negotiation coaching assistant. It exposes HTTP and Server-Sent Events endpoints that can be called by a frontend, most likely Bubble, to provide:

- general Diadem negotiation Q&A;
- streaming chat answers;
- guided coaching flows;
- Master Negotiator template help;
- template-oriented, paste-ready wording;
- related slide/template assets from retrieved metadata;
- session persistence across turns;
- retrieval from Diadem source material stored in Pinecone.

The application is not a standalone website. It is an API service.

## 2. High-Level Architecture

```text
Frontend / Bubble
      |
      | HTTP JSON or SSE
      v
FastAPI app.py
      |
      | user query -> OpenAI embedding
      v
Pinecone vector index
      |
      | relevant chunks + metadata
      v
Prompt assembly in app.py
      |
      | context + system prompt + user message
      v
Anthropic Claude
      |
      | answer text
      v
FastAPI response
      |
      | answer + session_id + optional assets
      v
Frontend / Bubble
```

## 3. Runtime Dependencies

From `requirements.txt`:

```text
fastapi==0.115.6
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
anthropic==0.49.0
openai==1.109.1
pinecone==5.4.0
requests==2.31.0
PyMuPDF>=1.26.0
Pillow==10.4.0
pytesseract==0.3.10
tiktoken==0.7.0
```

Key roles:

- `fastapi`: web API framework.
- `uvicorn`: ASGI server used to run the API.
- `python-dotenv`: loads `.env` locally.
- `openai`: creates embeddings.
- `anthropic`: creates Claude chat/coach responses.
- `pinecone`: vector database client.
- `PyMuPDF`: extracts PDF text/images.
- `Pillow`: image handling.
- `pytesseract`: OCR for extracted PDF images.
- `tiktoken`: token-based text chunking.
- `requests`: utility scripts and Bubble upload/integration.

## 4. Required Environment Variables

The main app validates required variables at import/startup time. If any required variable is missing, startup fails.

### Required for `app.py`

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_HOST
```

### Required for `ingest_pdf_to_pinecone.py`

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_ENV
```

Note: the ingestion script uses `PINECONE_ENV`, while `app.py` uses `PINECONE_HOST`.

### Optional runtime configuration

```text
DEBUG
EMBED_MODEL
CHAT_MODEL
TOP_K
PINECONE_TOPK_RAW
MAX_CONTEXT_CHARS
EMBED_DIM
SESSION_TTL_SECONDS
MIN_MATCH_SCORE
MIN_CONTEXT_CHARS
MIN_OVERLAP_SCORE
MULTI_QUERY_K
DIVERSITY_SAME_SOURCE_CAP
PRIORITY_BOOST
PRIORITY_MAX
SEARCH_DEBUG_LOGS
SEARCH_LOG_MAX_MATCHES
SEARCH_LOG_TEXT_PREVIEW
SLIDE_INDEX_PATH
COACH_CONFIRM_EVERY_N
MASTER_SYSTEM_PROMPT_PATH
SESSION_DB_PATH
```

### Optional Bubble integration

```text
BUBBLE_API_BASE
BUBBLE_API_KEY
```

If both are present, the app can read Bubble Data API objects. If either is absent, Bubble reads silently return empty data.

## 5. Model Usage

### OpenAI

OpenAI is used for embeddings:

```text
EMBED_MODEL default: text-embedding-3-small
EMBED_DIM default: 1536
```

The app calls:

```python
openai.embeddings.create(...)
```

This happens in `embed_query()` for runtime search and in `embed_texts()` inside the ingestion script.

### Anthropic Claude

Claude is used for chat/coaching generation:

```text
CHAT_MODEL default in app.py: claude-sonnet-4-6
```

The app calls:

```python
anthropod.messages.create(...)
anthropod.messages.stream(...)
```

Non-streaming endpoints use `messages.create`. SSE endpoints use `_openai_stream_text()`, which is named historically but currently streams through Anthropic.

Important naming note: `_openai_stream_text()` is misleading. Despite the name, the current implementation streams Claude output through the Anthropic client.

## 6. Main Application File: `app.py`

`app.py` contains:

- imports and environment loading;
- logging setup;
- configuration defaults;
- optional Bubble read helpers;
- required key checks;
- OpenAI, Anthropic, and Pinecone client initialization;
- SSE helper functions;
- request logging middleware;
- SQLite session store;
- utility parsing and normalization helpers;
- RAG search and reranking logic;
- system prompts and template policies;
- coach/template state machines;
- Master Negotiator template logic;
- endpoint definitions.

It is a large single-file backend.

## 7. API Endpoints

### `GET /health`

Purpose: simple health check.

Response shape:

```json
{
  "ok": true,
  "debug": false
}
```

### `POST /chat`

Purpose: normal non-streaming RAG chat.

Common payload fields:

```json
{
  "query": "How should I prepare my variables?",
  "top_k": 10,
  "history": [
    {"role": "user", "content": "Previous user message"},
    {"role": "assistant", "content": "Previous assistant answer"}
  ],
  "user_name": "Name",
  "session_id": "optional-existing-session"
}
```

Behavior:

1. Extracts `query`, `top_k`, `history`, user name, and admin settings.
2. Creates or reuses a `session_id`.
3. Cleans up old sessions.
4. Handles empty query and smalltalk directly.
5. Builds compact conversation context from recent history.
6. Retrieves matching Pinecone chunks using `get_matches()`.
7. Builds a context block using `build_context()`.
8. Extracts related assets using `_extract_chat_assets()`.
9. Sends context and question to Claude under `SYSTEM_PROMPT_CHAT`.
10. Finalizes the answer text.
11. Applies practical fallback if assets exist and the model deflects.
12. Rewrites weak/generic openers.

Response shape:

```json
{
  "answer": "Assistant answer",
  "session_id": "session-id",
  "assets": []
}
```

### `POST /chat/sse`

Purpose: SSE version of `/chat`.

Response event types:

```text
event: start
event: chunk
event: assets
event: done
```

Notes:

- It internally builds the full text from streaming output, finalizes it, then emits chunks.
- The `done` event includes `assets`.
- SSE headers include `Cache-Control: no-cache`, `Connection: keep-alive`, and `X-Accel-Buffering: no`.

### `POST /coach/chat`

Purpose: older guided coach/template flow, non-streaming.

It calls:

```python
coach_turn_server_state(...)
```

This flow supports templates defined in `TEMPLATES`, including:

- `build_confidence`;
- `prepare_difficult_behaviours`.

Response shape generally includes:

```json
{
  "text": "Coach response",
  "session_id": "session-id",
  "done": false
}
```

### `POST /coach/sse`

Purpose: SSE wrapper for the guided coach flow.

Response event types:

```text
event: start
event: chunk
event: done
```

### `POST /coach/reset`

Purpose: deletes the session state for the provided/current session.

Response:

```json
{
  "ok": true,
  "session_id": "session-id"
}
```

### `POST /master/template`

Purpose: non-streaming Master Negotiator template assistant.

This is the newer, more complex text-only coach for the Master Negotiator template.

Common payload fields:

```json
{
  "user_message": "Help me fill the variable name field",
  "query": "alternate message field",
  "template_id": "Bubble template id",
  "session_id": "optional-existing-session",
  "active_section_id": "my_list",
  "focus_field": "variable_name",
  "user_name": "Name",
  "help_accepted": true,
  "admin_prompt": "optional admin guidance",
  "summary_guidance_all": "optional extra guidance"
}
```

Behavior:

1. Loads or creates Master Negotiator state.
2. Reads optional current template state from Bubble if configured.
3. Tracks active section and focus field.
4. Detects whether the user is asking a fresh question or continuing template filling.
5. Extracts slots such as money, percentages, payment terms, and position values.
6. Uses retrieval from Pinecone, including recent conversation context.
7. Builds logic-grid guidance from `logic_grid.json`.
8. Applies rule-based direct responses for some structured cases.
9. Otherwise builds a detailed prompt with state memory, information, task instructions, active field, and admin addendum.
10. Calls Claude.
11. Finalizes and trims the text.
12. Returns assets if available.

Response shape:

```json
{
  "session_id": "session-id",
  "mode": "master_negotiator_template",
  "text": "Assistant response",
  "done": false,
  "assets": []
}
```

### `POST /master/template/sse`

Purpose: streaming Master Negotiator template assistant.

Response event types:

```text
event: start
event: chunk
event: assets
event: done
```

The `done` payload includes:

```json
{
  "done": true,
  "session_id": "session-id",
  "mode": "master_negotiator_template",
  "assets": []
}
```

### `POST /master/template/reset`

Purpose: resets Master Negotiator template state for the session.

## 8. Session Storage

The app uses SQLite through `sqlite3`.

Default database path:

```text
sessions.sqlite3
```

Override:

```text
SESSION_DB_PATH
```

The database has a `sessions` table with:

- `session_id`;
- serialized JSON payload;
- `updated_at`.

Session helpers:

- `_db()`: opens/reuses SQLite connection.
- `_db_get()`: reads session JSON.
- `_db_set()`: writes session JSON.
- `_db_delete()`: deletes a session.
- `_cleanup_sessions()`: deletes sessions older than `SESSION_TTL_SECONDS`.
- `_get_or_create_session_id()`: uses supplied session ID or creates a UUID.

Default session TTL:

```text
86400 seconds
```

## 9. Retrieval-Augmented Generation Pipeline

The RAG flow is centered on:

```python
get_matches()
build_context()
_extract_chat_assets()
```

### Step 1: Query preparation

`get_matches()` receives the user query and creates up to `MULTI_QUERY_K` query variants:

1. the original cleaned query;
2. query plus a curated hint from `_hint_for_question()`;
3. a keyword-only query from `_keyword_query()`.

Default:

```text
MULTI_QUERY_K=3
```

### Step 2: Embedding

Each query variant is embedded with:

```python
openai.embeddings.create(
    model=EMBED_MODEL,
    input=[text],
    dimensions=EMBED_QUERY_DIM,
)
```

The runtime embedding dimension is initially `EMBED_DIM`, then adjusted using Pinecone `describe_index()` if possible.

### Step 3: Pinecone search

Each embedding is sent to:

```python
index.query(vector=vec, top_k=PINECONE_TOPK_RAW, include_metadata=True)
```

Default raw top-k:

```text
PINECONE_TOPK_RAW=30
```

### Step 4: Merge and deduplicate

`_merge_dedup_matches()` combines result sets and keeps the best-scoring match per ID.

### Step 5: Minimum score filtering

`_filter_matches_by_score()` removes matches below:

```text
MIN_MATCH_SCORE=0.35
```

### Step 6: Reranking

`_rerank()` calculates a custom score using:

- query token overlap;
- hint token overlap;
- Pinecone similarity score;
- metadata priority;
- generic phrase penalties;
- source diversity cap.

Relevant config:

```text
PRIORITY_BOOST=0.6
PRIORITY_MAX=3
DIVERSITY_SAME_SOURCE_CAP=3
```

### Step 7: Context relevance gate

`is_context_relevant()` checks whether retrieved material is good enough. It considers:

- top match semantic score;
- token overlap;
- approximate context length;
- minimum context character threshold;
- minimum overlap threshold.

Relevant config:

```text
MIN_CONTEXT_CHARS=700
MIN_OVERLAP_SCORE=1.3
```

If `_hint_for_question()` produced a hint, the code can bypass the relevance gate when matches exist.

### Step 8: Context construction

`build_context()` creates the final `INFORMATION` block.

Behavior:

- separates text and image chunks;
- truncates very long chunks around sentence boundaries where possible;
- always tries to include at least one text block;
- joins chunks with `---`;
- respects `MAX_CONTEXT_CHARS`.

Default:

```text
MAX_CONTEXT_CHARS=14000
```

## 10. Asset Extraction

`_extract_chat_assets()` builds a list of related assets from Pinecone metadata.

It looks for metadata fields such as:

```text
file_name
source
file
doc_id
page
image_url
img_url
asset_url
url
text
id
slide_id
```

Asset types are inferred:

- source containing `slide` -> `slide_example`;
- source containing `template` -> `template_example`;
- otherwise -> `reference_example`.

Returned asset fields:

```json
{
  "type": "slide_example",
  "source": "Master Negotiator Slides.pdf",
  "page": 12,
  "image_url": "https://...",
  "preview": "short text preview",
  "has_image": true
}
```

Slide keyword boosting uses `slide_index.json` when available.

## 11. Prompting System

The code defines multiple prompt layers.

### Core prompt constants

- `SYSTEM_PROMPT_QA`
- `SYSTEM_PROMPT_EXPLAIN`
- `SYSTEM_PROMPT_CHAT`
- `SYSTEM_PROMPT_COACH_FINAL`
- `MASTER_SYSTEM_PROMPT_TEXT`

### External Master system prompt

The Master Negotiator assistant loads:

```text
Diadem_AI_System_Prompt_v1.txt
```

The path can be overridden:

```text
MASTER_SYSTEM_PROMPT_PATH
```

If the file cannot be read, the app falls back to a short safe default prompt.

### Admin prompt addendum

Payload/admin settings can add guidance through:

- `admin_prompt`;
- `summary_guidance_all`;
- related alternate key names handled by `_extract_admin_settings()`.

The code wraps admin text in a controlled addendum and tells the model never to mention admin guidance in user-facing output.

## 12. Master Negotiator Template Logic

The Master Negotiator path is designed as text-only guidance. It does not directly write rows or fields into Bubble tables.

Important concepts:

- `MASTER_MODE`: mode name for state.
- `_mnt_default_state_text()`: default Master Negotiator state.
- `_mnt_load_state_text()`: load state from SQLite.
- `_mnt_save_state_text()`: persist state.
- `_mnt_reset_state_text()`: reset state.
- `_mnt_extract_user_message()`: find user text from payload.
- `_mnt_extract_focus()`: active section/focus field.
- `_mnt_extract_slots_from_user()`: parse values from user text.
- `_mnt_build_state_memory_text()`: summarize state for prompting.
- `_mnt_rule_based_response()`: answer some structured cases without the LLM.
- `_build_logic_grid_guidance()`: create logic guidance from `logic_grid.json`.
- `_maybe_logic_grid_direct_response()`: direct response for certain logic-grid cases.
- `_master_llm_text()`: constructs and sends Master prompts.
- `master_template_turn_text()`: main non-SSE orchestration.

The assistant is heavily constrained to:

- use Diadem/Master methodology;
- stay practical;
- avoid generic negotiation advice;
- keep outputs plain text;
- keep answers concise;
- write paste-ready template wording when a field is active;
- not invent variable values without user confirmation;
- handle tactics/pressure questions as a coach rather than forcing every answer into template structure.

## 13. Guided Coach Templates

The older coach flow has two templates in `TEMPLATES`.

### `build_confidence`

Steps:

1. company;
2. situation;
3. relationship;
4. myself;
5. why confident;
6. summary.

### `prepare_difficult_behaviours`

Steps:

1. scenario;
2. anticipate tactics;
3. purpose;
4. response bullet;
5. move on air;
6. summary.

The state tracks:

- mode;
- step index;
- answers;
- active section;
- variables;
- confirmation state;
- completion state;
- clarification count.

## 14. Bubble Read Integration

Bubble support is read-only in `app.py`.

Environment variables:

```text
BUBBLE_API_BASE
BUBBLE_API_KEY
```

Helper functions:

- `_bubble_enabled()`
- `_bubble_headers()`
- `_bubble_url()`
- `_http_get_json()`
- `_bubble_get()`
- `_bubble_search()`
- `_fetch_template_state_text()`

`_fetch_template_state_text()` can read:

- `master_negotiation_template`;
- related `Deal` records;
- related `Variable_items`.

It formats the state into text for the AI prompt.

The app does not directly mutate Bubble template rows from the main endpoints.

## 15. PDF Ingestion Pipeline

File:

```text
ingest_pdf_to_pinecone.py
```

Purpose: convert a source PDF into Pinecone vectors.

### Required env for ingestion

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_ENV
```

### Main command

```bash
python ingest_pdf_to_pinecone.py path/to/file.pdf
```

### Extraction

`extract_pdf_items()`:

- opens the PDF with PyMuPDF;
- extracts text blocks per page;
- filters copyright/footer text;
- combines page text into a single page text item;
- extracts images as Pillow images;
- records page number and image index.

### OCR and image description

`ocr_image()` uses Tesseract OCR.

`describe_image()` sends OCR text to Claude and asks for:

- 1-2 sentence visual summary;
- 2 key negotiation takeaways.

### Chunking

`chunk_text()` uses `tiktoken`:

```text
CHUNK_TOKENS=800
CHUNK_OVERLAP=150
```

### Document preparation

`prepare_documents()`:

- creates text vector documents for text chunks;
- creates image vector documents using OCR and Claude description;
- stores `text` in metadata;
- enriches metadata with slide/image data where available;
- assigns higher priority to text methodology chunks than image chunks.

Text priority:

```text
10
```

Image priority:

```text
7
```

### Embedding and upload

`embed_texts()` embeds chunks with OpenAI.

`upsert_documents()` batches vectors and upserts them into Pinecone.

Batch size:

```text
UPSERT_BATCH_SIZE default: 100
```

## 16. Slide Asset Upload Pipeline

File:

```text
bubble_bulk_upload_slides.py
```

Purpose: upload slide image files to Bubble storage and optionally create Bubble `SlideAsset` records.

Typical inputs:

- folder of images via `--dir`;
- zip archive via `--archive`;
- Bubble base URL;
- Bubble API token;
- environment: `live` or `test`;
- data type name, default `SlideAsset`.

Example:

```bash
python bubble_bulk_upload_slides.py ^
  --dir path/to/slides ^
  --bubble-base-url https://yourapp.bubbleapps.io ^
  --bubble-api-token your-token ^
  --bubble-env test
```

Outputs:

- JSON mapping, default `slide_image_urls.json`;
- CSV, default `slide_upload.csv`.

The ingestion script can use these mappings to attach `image_url` metadata to vectors.

## 17. Utility and Smoke-Test Scripts

### `audit_pinecone_docs.py`

Audits Pinecone contents:

- total vector count;
- collected IDs;
- unique file names;
- doc IDs;
- concepts;
- query errors.

Requires:

```text
PINECONE_API_KEY
PINECONE_INDEX_NAME
```

### `test_pinecone_connection.py`

Basic Pinecone connection script.

Caution: it prints the API key and queries with a fixed 512-dimensional vector. If the Pinecone index is not 512-dimensional, the query will fail. This script is useful only as a rough connection check and should not print secrets in shared logs.

### `smoke_chat_local.py`

Local smoke test that:

- sets dummy API keys;
- imports `app.py`;
- monkeypatches external dependencies;
- tests `/chat` and `/chat/sse`;
- verifies answer text and SSE events.

### `smoke_chat_e2e_real.py`

Real end-to-end smoke test requiring real env variables.

Checks:

- required env values;
- `/chat`;
- `/chat/sse`;
- absence of known regression phrase;
- SSE chunk/done events.

### `smoke_assets_endpoints.py`

Checks asset contract across:

- `/chat`;
- `/chat/sse`;
- `/master/template`;
- `/master/template/sse`.

It verifies `assets` is present as a list in JSON responses and SSE done payloads.

### `smoke_master_prompt_update.py`

Tests Master prompt loading and propagation:

- confirms prompt file contains expected version/content markers;
- monkeypatches retrieval and Claude;
- calls `/master/template`;
- calls `/master/template/sse`;
- checks system prompt is used.

## 18. Deployment

File:

```text
render.yaml
```

Render configuration:

```yaml
services:
  - type: web
    name: rag-pinecone-api
    env: python
    plan: starter

    envVars:
      - key: PYTHON_VERSION
        value: 3.12.8

    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Required deployment setup:

1. Add all required env vars in Render.
2. Ensure Pinecone index exists and contains vectors.
3. Ensure the Pinecone index dimension matches `EMBED_DIM` or can be detected by `describe_index()`.
4. Ensure `Diadem_AI_System_Prompt_v1.txt` is included in the deployed filesystem.
5. Ensure optional slide JSON mappings are included if asset URLs are required.

## 19. Local Run Instructions

From the project folder:

```bash
pip install -r requirements.txt
```

Set environment variables. For PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:PINECONE_API_KEY="..."
$env:PINECONE_INDEX_NAME="..."
$env:PINECONE_HOST="..."
```

Run:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 20. Example Requests

### `/chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How should I prepare my negotiation variables?\",\"top_k\":10}"
```

### `/master/template`

```bash
curl -X POST http://127.0.0.1:8000/master/template \
  -H "Content-Type: application/json" \
  -d "{\"user_message\":\"Help me name a payment terms variable\",\"active_section_id\":\"my_list\",\"focus_field\":\"variable_name\"}"
```

### `/coach/reset`

```bash
curl -X POST http://127.0.0.1:8000/coach/reset \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"your-session-id\"}"
```

## 21. Common Failure Modes

### Startup fails with missing key error

Cause: one of the required environment variables is missing.

For `app.py`, check:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_HOST
```

### Search returns weak or empty answers

Possible causes:

- Pinecone index is empty.
- Wrong `PINECONE_INDEX_NAME`.
- Wrong `PINECONE_HOST`.
- Source PDFs were not ingested.
- Metadata lacks `text`.
- Score thresholds are too strict.
- Embedding dimensions do not match the index.

Helpful checks:

```bash
python audit_pinecone_docs.py
python smoke_chat_e2e_real.py
```

### Pinecone dimension errors

The app attempts to detect index dimension through `describe_index()`. If that fails, it falls back to `EMBED_DIM`.

Check:

```text
EMBED_DIM
PINECONE index dimension
OpenAI embedding model dimension support
```

### SSE works locally but not in production

Possible causes:

- proxy buffering;
- platform timeout;
- missing SSE headers;
- frontend not parsing SSE event format correctly.

The app sets:

```text
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### Assets missing

Possible causes:

- Pinecone metadata does not include image URLs;
- `slide_image_urls.json` was not generated or not used during ingestion;
- metadata field names do not match what `_extract_chat_assets()` expects;
- retrieved chunks are text-only and have no source/page/image metadata.

### Bubble template state missing

Possible causes:

- `BUBBLE_API_BASE` not set;
- `BUBBLE_API_KEY` not set;
- wrong Bubble API object names;
- wrong template ID;
- Bubble API privacy rules block access;
- live/test environment mismatch.

## 22. Security Notes

- Do not commit real API keys.
- `test_pinecone_connection.py` prints the Pinecone API key and should not be used in shared logs without editing.
- The main app logs structured events and may log query text. Be mindful if user inputs are sensitive.
- Bubble API tokens should be treated as secrets.
- `sessions.sqlite3` may contain user conversation state and should be handled as private data.

## 23. Current Provider Split

This codebase currently uses a hybrid provider setup:

```text
OpenAI -> embeddings
Anthropic Claude -> answer generation
Pinecone -> vector search
Bubble -> optional state/asset integration
SQLite -> session state
```

That means an OpenAI key alone is not sufficient for the current code. To run the app as written, Anthropic and Pinecone credentials are also required.

## 24. Important Code Observations

1. `_openai_stream_text()` is misnamed. It streams Claude responses through Anthropic.
2. `app.py` performs required env validation at import time. This means tests or scripts importing `app.py` must set dummy or real env vars first.
3. The main app and ingestion script use different Pinecone connection values: `PINECONE_HOST` for runtime app, `PINECONE_ENV` for ingestion.
4. The app has multiple overlapping coaching systems: general chat, older coach templates, and Master Negotiator template assistant.
5. The Master template assistant includes significant rule-based logic before falling back to the LLM.
6. Asset support depends heavily on correct metadata during ingestion.
7. The retrieval pipeline is more than semantic search; it includes hints, keyword fallback, reranking, source diversity, metadata priority, and relevance gating.

## 25. Recommended Operational Checklist

Before deploying or testing with real users:

1. Confirm all required env vars are set.
2. Confirm Pinecone index exists.
3. Confirm index dimension matches the embedding dimension.
4. Run a Pinecone audit.
5. Run local smoke tests.
6. Run real e2e smoke test against the deployed API.
7. Verify `/health`.
8. Verify `/chat` returns an answer and assets list.
9. Verify `/chat/sse` emits `start`, `chunk`, and `done`.
10. Verify `/master/template` returns `text`, `session_id`, `mode`, `done`, and `assets`.
11. Verify `/master/template/sse` emits `assets` and includes assets in `done`.
12. Check logs for Pinecone query errors.
13. Confirm Bubble env, object names, and API permissions if Bubble state is expected.

## 26. Short Mental Model

Think of the system as five layers:

1. API layer: FastAPI endpoints receive requests and return JSON/SSE.
2. Memory layer: SQLite stores session state.
3. Retrieval layer: OpenAI embeddings plus Pinecone find relevant Diadem material.
4. Reasoning layer: Claude turns retrieved material and state into coaching text.
5. Integration layer: Bubble and slide asset metadata connect the backend to the frontend experience.

When debugging, identify which layer is failing first.
