# Diadem AI Production Release Runbook

## What Is Automated

- Every GitHub push and pull request compiles the production Python modules and runs the feedback regression suite.
- Render deploys only after repository checks pass.
- Render calls `/ready` during deployment. The endpoint verifies required configuration and the session database before accepting traffic.
- Production starts with `DEBUG=0`, one application instance, and a 60-second graceful shutdown window.

## Pre-Release Checklist

1. Confirm GitHub Actions is green for the release commit.
2. Confirm all values from `.env.example` marked required are configured in Render.
3. Set `ALLOWED_ORIGINS` to the live Bubble domain and any approved preview domain only.
4. Open `/health` and verify `ok` is `true`, `environment` is `production`, and `debug` is `false`.
5. Open `/ready` and verify both `config` and `session_store` are `true`.
6. Run `python smoke_assets_endpoints.py --base-url https://YOUR-RENDER-SERVICE.onrender.com` from a trusted machine.
7. Test Bubble's normal chat, document upload, document follow-up, resource cards, and password-reset journey with a non-admin account.
8. Confirm logs contain request IDs but no document text, passwords, API keys, or authorization headers.

## External Work Still Required

### Bubble

- Ensure the normal send workflow passes the user's current chat input as `query`, the stable chat ID as `session_id`, and the Bubble user's unique ID as `user_id`.
- Ensure document upload and later chat messages reuse the same `session_id` and `user_id` so document context persists.
- Hide or reset the upload controls after a successful upload while retaining the server-side session.
- Make the temporary-password message explicit and force a password change after first login.
- Test privacy rules with separate admin, tester, and standard-user accounts.

### Email And Domain

- Correct the Bubble domain DNS records and wait for Bubble to validate HTTPS.
- Send transactional mail from a branded address such as `support@diademperformance.com`.
- Configure SPF, DKIM, and DMARC for the selected mail provider, then test delivery to Gmail and Outlook.

### Content And Retrieval

- Add or correct the missing source assets identified during testing, including CARD/SCOTSMAN and the selling transition image.
- Re-ingest changed resources and run representative queries before release.

## Known Production Limitation

Session and uploaded-document context currently use a local SQLite database. The Render service is intentionally limited to one instance because this store is not shared between instances, and an instance replacement can remove local state. Before horizontal scaling or stronger durability guarantees, migrate session state to managed Postgres or Redis and set `SESSION_DB_PATH` only for local development.

## Rollback

1. In Render, redeploy the last known-good commit.
2. Confirm `/ready` returns HTTP 200 before reopening access.
3. Run the smoke test against the rolled-back service.
4. Record the failed commit, request ID, endpoint, and user-visible symptom before attempting another release.
