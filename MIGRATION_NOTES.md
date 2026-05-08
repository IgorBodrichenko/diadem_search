# OpenAI → Claude Migration Summary

## Changes Made

### 1. **requirements.txt**
- Added `anthropic==0.32.0` 
- Kept `openai==1.109.1` for embeddings API

### 2. **app.py**
- **Import:** Added `from anthropic import Anthropic`
- **Config:** Changed `CHAT_MODEL` to `claude-3-5-sonnet-20241022`
- **Environment:** Added `ANTHROPIC_API_KEY` requirement
- **Initialization:** Created `anthropod = Anthropic(api_key=ANTHROPIC_API_KEY)`
- **Streaming Function:** Rewrote `_openai_stream_text()` to use Claude streaming API
- **Chat Calls:** Replaced 3 instances of `openai.chat.completions.create()` with `anthropod.messages.create()`:
  - `_generate_final()` (line ~1606)
  - Chat handler (line ~1862)
  - Master template handler (line ~3689)

### 3. **ingest_pdf_to_pinecone.py**
- **Import:** Added `from anthropic import Anthropic`
- **Config:** Added `ANTHROPIC_API_KEY` requirement
- **Initialization:** Created `anthropod = Anthropic(api_key=ANTHROPIC_API_KEY)`
- **Image Description:** Updated `describe_image()` to use Claude instead of GPT-4o-mini

## Architecture After Migration

```
✅ Claude 3.5 Sonnet:
   - Streaming chat responses
   - Image description (OCR text analysis)
   - Template generation
   - Final coaching responses

✅ OpenAI (text-embedding-3-small):
   - Query embeddings for Pinecone RAG retrieval
   - Still required - Claude has no embedding API

✅ Pinecone:
   - Vector storage (unchanged)
```

## Environment Variables Required

```bash
# Existing (still needed)
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
PINECONE_HOST=...

# NEW - Must add for Claude
ANTHROPIC_API_KEY=sk-ant-...
```

## Testing Checklist

- [ ] Set `ANTHROPIC_API_KEY` in `.env` or deployment config
- [ ] Run `pip install -r requirements.txt` (or update packages)
- [ ] Test streaming chat endpoint with `/chat`
- [ ] Test template generation with `/coach`
- [ ] Test PDF ingestion: `python ingest_pdf_to_pinecone.py <path-to-pdf>`
- [ ] Verify RAG still works (embeddings from OpenAI)

## Cost Comparison

| Operation | Before (GPT) | After (Claude) | Notes |
|-----------|--------------|----------------|-------|
| Chat streaming | gpt-4o-mini | Claude 3.5 Sonnet | Sonnet has better context handling |
| Image description | gpt-4o-mini | Claude 3.5 Sonnet | Same functionality, likely better quality |
| Query embeddings | text-embedding-3-small | text-embedding-3-small | Unchanged (OpenAI only) |

## Potential Issues & Notes

1. **Claude API Rate Limits:** Monitor usage, Sonnet has ~90k tokens/min limit
2. **Response Format:** Claude returns `response.content[0].text` vs OpenAI's `response.choices[0].message.content`
3. **System Prompt Handling:** Claude uses `system=` parameter instead of a message role
4. **Max Tokens:** Claude requires explicit `max_tokens` parameter (set to 4096 for flexibility)
