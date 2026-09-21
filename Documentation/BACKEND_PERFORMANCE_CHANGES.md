# Backend Performance Changes

This document describes the response-time improvements introduced in commit
`6f78722` on the `test_v3` branch.

The changes reduce repeated retrieval and prompt-processing work without
changing the AI model, prompts, API response shapes, document memory, resource
selection, or MASTER coaching functionality.

## Why This Work Was Needed

Performance timing showed two distinct parts of a typical request:

- Retrieval from OpenAI embeddings and Pinecone generally completed in under
  one second.
- Claude response generation commonly took approximately 14 to 15 seconds for
  a full answer.

The backend already batched embedding requests and ran independent Pinecone
queries concurrently. The latest changes therefore focus on avoiding repeated
work around retrieval and Claude's system instructions while retaining the
existing Sonnet model and answer quality.

## What Changed

### 1. Anthropic prompt caching

Repeated system instructions are now sent to Anthropic with an ephemeral cache
breakpoint.

This applies to:

- Standard chat responses.
- Streaming chat responses.
- Coach final summaries.
- MASTER template coaching.
- Document reviews.

The system-prompt text itself is unchanged. The backend only changes the
request representation so Anthropic can reuse an eligible prompt prefix on
subsequent calls.

Prompt caching is enabled by default and can be disabled through:

```env
ENABLE_PROMPT_CACHE=0
```

Prompt caching is most useful when requests repeatedly use the same large
system instructions. The first request may create the cache; later matching
requests can reuse it. Dynamic admin instructions must also match for the full
combined system block to receive a cache hit.

### 2. Completed retrieval caching

The backend now temporarily caches the final, filtered and reranked Pinecone
results for each normalized query and `top_k` value.

On a cache hit, the backend skips:

- Embedding generation.
- Pinecone queries.
- Match merging.
- Score filtering.
- Reranking and relevance checks.

Both successful results and empty results are cached. This prevents repeated
queries that have no relevant context from repeatedly calling external
services.

Default configuration:

```env
RETRIEVAL_CACHE_TTL_SECONDS=120
RETRIEVAL_CACHE_MAX=256
```

Set `RETRIEVAL_CACHE_TTL_SECONDS=0` to disable this cache.

### 3. Existing retrieval optimizations retained

The backend continues to use the earlier performance improvements:

- Multiple semantic query variants are embedded in one OpenAI request.
- Pinecone searches for those variants run concurrently.
- Individual query embeddings are retained in a bounded in-memory cache.
- Search and model timing can be logged independently.

The relevant existing controls are:

```env
RETRIEVAL_MAX_WORKERS=3
EMBED_CACHE_MAX=512
PERF_DEBUG_LOGS=0
```

## Request Flow

A standard `/chat` request now follows this path:

1. Validate and bound the request data.
2. Load conversation and uploaded-document memory.
3. Build the contextual retrieval query.
4. Check the completed retrieval cache.
5. On a miss, batch embeddings and query Pinecone concurrently.
6. Filter and rerank the matches, then cache the final result.
7. Build the answer context and select relevant assets.
8. Send the unchanged system instructions with a prompt-cache breakpoint.
9. Finalize the answer and save the conversation turn.

## API Compatibility

No Bubble integration changes are required for this backend update.

The following remain unchanged:

- Endpoint paths.
- Request parameters.
- JSON response fields.
- SSE event names and chunk format.
- Session and user identifiers.
- Uploaded-document persistence.
- Resource and image payloads.
- AI model selection.

## Cache Scope and Limitations

Both caches are in memory and local to one running backend process.

Consequences:

- Caches are cleared when Render restarts or redeploys the service.
- Separate Render workers do not share cache entries.
- The retrieval cache may serve the previous search result for up to the
  configured TTL after Pinecone content is updated.
- Caching reduces repeated input and retrieval work. It does not eliminate the
  time Claude needs to generate a new answer.
- Unique questions may not benefit from the completed retrieval cache, although
  they still use batched embeddings and concurrent Pinecone searches.

The two-minute retrieval TTL is intentionally short so newly indexed training
content becomes visible quickly.

## Deployment Configuration

Recommended Render values:

```env
ENABLE_PROMPT_CACHE=1
RETRIEVAL_CACHE_TTL_SECONDS=120
RETRIEVAL_CACHE_MAX=256
RETRIEVAL_MAX_WORKERS=3
EMBED_CACHE_MAX=512
PERF_DEBUG_LOGS=0
```

Enable `PERF_DEBUG_LOGS=1` temporarily when measuring live performance. Disable
it again after diagnosis to keep production logs concise.

## Verification

### Functional checks

After deployment, verify:

1. `/health` returns a successful response.
2. Standard chat still returns `answer`, `session_id`, and `assets`.
3. Follow-up questions retain conversation context.
4. Uploaded-document follow-ups still use document memory.
5. MASTER coaching and document review still return their existing formats.
6. Streaming chat still emits `start`, `chunk`, and `done` events.

### Performance checks

With `PERF_DEBUG_LOGS=1`, compare these events:

- `retrieval_timing`
- `chat_stage_timing` with `step=retrieval`
- `chat_stage_timing` with `step=model`
- `http_timing`

For a repeated query inside the cache window, `retrieval_timing` should report
`reason=cache_hit`, with no new embedding or Pinecone request for that query.

Anthropic response usage may report cache creation or cache-read input tokens
when the system prompt meets the provider's caching requirements.

## Rollback

The optimizations can be disabled without reverting code:

```env
ENABLE_PROMPT_CACHE=0
RETRIEVAL_CACHE_TTL_SECONDS=0
```

This returns runtime behavior to uncached prompt and completed-retrieval
processing while preserving all other backend functionality.

## Further Improvements

The remaining major delay is full model generation. Safe next steps are:

- Ensure Bubble uses `/chat/sse` where incremental output is supported, so the
  user sees the answer as it is generated.
- Measure time to first token separately from total generation time.
- Evaluate shorter answer contracts using the production quality test set.
- Consider a faster model only after side-by-side quality evaluation; do not
  switch models solely for latency.
