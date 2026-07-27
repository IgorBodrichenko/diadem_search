# Production Readiness Review

This document explains the local production-hardening changes added for review before pushing.

## What Changed

The backend now has a safer production wrapper around the existing AI functionality:

- Configurable CORS through `ALLOWED_ORIGINS`.
- Request size protection through `MAX_REQUEST_BYTES`.
- Simple in-memory per-endpoint rate limiting through `RATE_LIMIT_PER_MINUTE`.
- Request IDs on responses through `X-Request-ID`.
- Basic security headers on all HTTP responses.
- Redacted logging for API keys, bearer tokens, credentials, secrets and long payloads.
- Health and readiness endpoints that show configuration status without exposing secret values.
- Safer input bounds for `query`, `history`, `top_k` and `session_id`.
- Generic 500 error handling so production errors do not expose stack details to users.
- A safe `.env.example` showing required variables without real keys.

## Why This Matters

These changes do not fundamentally change the AI behaviour. They make the existing backend safer to run in front of real users.

The main production risks addressed are:

- Accidental key leakage in logs.
- Unexpected huge requests causing cost or stability issues.
- Overly broad retrieval requests from large `top_k` values.
- Dirty or oversized session IDs reaching SQLite.
- Browser/API-origin confusion between Bubble test/live domains.
- Hard-to-debug failures with no request identifier.
- Error messages exposing implementation detail.

## Environment Variables

Required:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_HOST`

Recommended production values:

- `ENVIRONMENT=production`
- `DEBUG=0`
- `ALLOWED_ORIGINS=https://ai.diademperformance.com,https://diadem-51532.bubbleapps.io`
- `RATE_LIMIT_PER_MINUTE=120`
- `MAX_REQUEST_BYTES=262144`
- `MAX_QUERY_CHARS=4000`
- `MAX_TOP_K=20`

Optional:

- `BUBBLE_API_BASE`
- `BUBBLE_API_KEY`
- `SEARCH_DEBUG_LOGS=0`

## Health Checks

`GET /health`

Returns:

- Whether required config is present.
- Current environment.
- Debug mode.
- Runtime safety limits.
- Only boolean config status, never secret values.

`GET /ready`

Returns `200` if required config exists.

Returns `503` if required config is missing.

## Privacy Notes

The current product should still be treated as a privacy-sensitive AI backend.

These changes reduce risk, but full production privacy still requires:

- A formal retention policy for uploaded or pasted documents.
- Clear consent wording in Bubble before document review or upload.
- Organisation-level data separation if multiple clients use the same backend.
- A decision on whether session memory should be ephemeral, per user, or per organisation.
- Monitoring that avoids logging private document contents.
- A deletion process for user content and session records.

## What This Does Not Yet Solve

This is not a full enterprise security programme. The following remain as next production steps:

- Replace local SQLite session storage with managed storage if usage grows.
- Add proper authenticated API access between Bubble and the backend.
- Add user/org-level permissions for custom AI settings.
- Add persistent audit logs that are privacy-safe.
- Add automated CI checks for tests, secret scanning and evals before deployment.
- Add a real file upload pipeline with virus scanning, content extraction, retention rules and deletion.
- Add monitoring/alerting for error rate, latency and model/provider failures.

## Review Checklist Before Push

- Confirm `ALLOWED_ORIGINS` includes the correct Bubble live and test URLs.
- Confirm Render has all required environment variables.
- Confirm no real keys are present in `.env.example` or documentation.
- Run smoke tests locally with dummy keys where appropriate.
- Run the production eval before pushing to live.
- Review whether `RATE_LIMIT_PER_MINUTE=120` is appropriate for Bubble traffic.
