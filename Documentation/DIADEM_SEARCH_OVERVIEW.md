# Diadem Search Test v3 - Plain-English Overview

## What This Project Is

This project is a web API for a Diadem negotiation coach. It lets another app, a Bubble frontend, send a user's negotiation question or template request to a Python backend. The backend searches Diadem training material, builds a useful context pack, asks an AI model to answer as a negotiation coach, and returns the answer plus any related slide or template assets.

In simple terms:

1. The user asks a negotiation question.
2. The app searches Diadem source material stored in Pinecone.
3. The best matching snippets are sent to an AI model.
4. The AI responds as a Diadem/Master methodology coach.
5. The API sends back text, session details, and optional supporting assets.

## Main Parts

### FastAPI backend

The main application is `app.py`. It defines the web server, API endpoints, retrieval logic, session memory, and coaching logic.

### OpenAI

OpenAI is used for embeddings. Embeddings turn text into vectors so Pinecone can search by meaning rather than only by keywords.

The app currently expects:

```text
OPENAI_API_KEY
```

### Anthropic Claude

Claude is used for the chat/coaching responses. The code uses the Anthropic client for normal and streaming answers.

The app currently expects:

```text
ANTHROPIC_API_KEY
```

### Pinecone

Pinecone stores the Diadem training material as searchable vectors. When a user asks something, the app creates an embedding for the question, sends it to Pinecone, and gets back the closest matching chunks.

The app expects:

```text
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_HOST
```

### SQLite session store

The app stores session state in a local SQLite file:

```text
sessions.sqlite3
```

This keeps track of things like previous answers, template progress, conversation state, and last-used openers.

### Bubble integration

There is optional Bubble Data API support. If Bubble settings are provided, the app can read template state from Bubble. It does not directly write template values into Bubble from `app.py`; the assistant gives text guidance and the frontend/user is responsible for applying it.

Optional Bubble environment variables:

```text
BUBBLE_API_BASE
BUBBLE_API_KEY
```

## What Users Can Do

### Ask a normal Diadem question

Use:

```text
POST /chat
POST /chat/sse
```

These endpoints answer general negotiation questions using Diadem material.

### Use a guided coach flow

Use:

```text
POST /coach/chat
POST /coach/sse
POST /coach/reset
```

These endpoints run older guided template-style flows, such as confidence-building and difficult-behaviour preparation.

### Use the Master Negotiator template assistant

Use:

```text
POST /master/template
POST /master/template/sse
POST /master/template/reset
```

These endpoints support the Master Negotiator template. They are designed to help users fill fields, think through variables, understand negotiation positions, and get paste-ready wording.

### Check whether the API is alive

Use:

```text
GET /health
```

It returns a simple JSON response showing the server is running.

## How Search Works

The app does not simply pass the user's question straight to the AI model. It first searches Diadem material:

1. It creates one or more search queries from the user's message.
2. It embeds those queries with OpenAI.
3. It queries Pinecone for matching chunks.
4. It filters out weak matches.
5. It reranks matches using semantic score, keyword overlap, source priority, and diversity.
6. It builds a context block from the best chunks.
7. It gives that context to Claude with a Diadem-specific system prompt.

This is called RAG: retrieval-augmented generation.

## How Data Gets Into Pinecone

The script `ingest_pdf_to_pinecone.py` takes a PDF and uploads it into Pinecone:

1. It opens the PDF.
2. It extracts page text.
3. It extracts images.
4. It OCRs image content.
5. It asks Claude to describe images from OCR text.
6. It chunks the resulting text.
7. It embeds chunks with OpenAI.
8. It upserts those vectors into Pinecone.

That ingestion step must be done before the chatbot can retrieve useful Diadem content.

## Deployment

The included `render.yaml` is configured for Render:

```text
buildCommand: pip install -r requirements.txt
startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Render needs the required environment variables configured in its dashboard.

## Important Thing To Know About API Keys

At the moment, this project requires more than an OpenAI key:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_HOST
```

OpenAI is used for embeddings. Claude is used for answer generation. Pinecone is used for search. If any required variable is missing, `app.py` raises an error during startup.

## Main Files

```text
app.py
```

The main backend API.

```text
ingest_pdf_to_pinecone.py
```

Loads PDFs into Pinecone.

```text
bubble_bulk_upload_slides.py
```

Uploads slide images to Bubble storage and creates Bubble slide asset records.

```text
audit_pinecone_docs.py
```

Audits what is currently stored in Pinecone.

```text
smoke_chat_local.py
smoke_chat_e2e_real.py
smoke_assets_endpoints.py
smoke_master_prompt_update.py
```

Small test scripts for checking local behavior, real API behavior, assets, and prompt wiring.

```text
Diadem_AI_System_Prompt_v1.txt
```

External master system prompt loaded by the Master Negotiator template assistant.

```text
logic_grid.json
```

Rules used for structured variable/position logic.

## One-Sentence Summary

This is a FastAPI-based Diadem negotiation coaching API that uses OpenAI embeddings, Pinecone retrieval, Claude answer generation, SQLite sessions, optional Bubble integration, and PDF ingestion scripts to turn Diadem training material into a searchable coaching assistant.
