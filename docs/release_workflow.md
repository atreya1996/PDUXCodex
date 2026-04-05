# Release Workflow (Source of Truth: `main`)

This repository uses `main` as the single source-of-truth branch for deployment.

## Branching and deployment policy

1. All production deployments must originate from a commit already merged into `main`.
2. No direct deployment from feature branches.
3. Every release should use an immutable commit SHA and expose it in the running Streamlit UI.

## Required CI checks

CI must pass both checks before deploying:

- full test suite (`pytest`), and
- runtime SHA UI verification (`tests/test_app.py -k commit_sha`) with `PAYDAY_RELEASE_SHA` set to the current commit.

This guarantees that the sidebar runtime metadata renders the deployment SHA in the app.

## Deployment checklist

Use this checklist for each release:

- [ ] **Merged commit SHA**: confirm commit merged in `main`.
- [ ] **Deployed commit SHA**: confirm commit SHA used by deployment runtime.
- [ ] **Runtime sidebar SHA**: open app sidebar and verify `Runtime commit SHA` matches deployed SHA.
- [ ] Trigger sidebar **Refresh status + release metadata** once post-deploy.
- [ ] **Upload guardrails configured**: set preview runtime env vars `PAYDAY_ENV_MAX_UPLOAD_FILE_MB`, `PAYDAY_ENV_MAX_UPLOAD_BATCH_MB`, and `PAYDAY_UPLOAD_BATCH_CHUNK_SIZE`.
- [ ] Add/update release notes in `docs/release_notes.md` mapping shipped UX/features to commit IDs.

## Preview upload guardrails

For the deployed preview target, use:

- `PAYDAY_ENV_MAX_UPLOAD_FILE_MB=20`
- `PAYDAY_ENV_MAX_UPLOAD_BATCH_MB=45`
- `PAYDAY_UPLOAD_BATCH_CHUNK_SIZE=3` (app clamps 1–3)

These values keep uploads below common proxy/browser request-size ceilings and reduce 413 risk for multipart audio batches.

### Required transport/runtime alignment

Before release, verify these deployment controls are aligned with PayDay guardrails:

- reverse-proxy max request body size (HTTP multipart uploads),
- runtime/app-server max request body size, and
- websocket/frame limits used by Streamlit transport paths.

If any upstream cap is below PayDay batch limits, users will hit `413 Payload Too Large` before app-side validation can help them split batches.

## Runtime metadata refresh

In the Streamlit sidebar use **Refresh status + release metadata** to force:

- service/runtime metadata reload, and
- durable status refresh from SQLite.

Set `PAYDAY_RELEASE_SHA` in runtime environment to surface the deployed SHA.

## Render service startup and health checks

Source-of-truth blueprint: `render.yaml`.

### Web service (`payday-web`)

- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Health check path: `/_stcore/health`

### Worker service (`payday-worker`)

- Start command: `python -m payday.worker`
- Health check path: N/A (background worker, no HTTP endpoint)

## Render environment variable matrix

Set these env vars in Render (secrets as secret env vars):

- `PAYDAY_APP_ENV=production`
- `PAYDAY_RELEASE_SHA=<deployed_commit_sha>`
- `PAYDAY_USE_SAMPLE_MODE=false`
- `PAYDAY_ENABLE_UPLOADS=true` (web), `false` (worker)
- `PAYDAY_ENABLE_DASHBOARD=true` (web), `false` (worker)
- `PAYDAY_ENABLE_ANALYSIS=true`
- `PAYDAY_ENV_MAX_UPLOAD_FILE_MB=20`
- `PAYDAY_ENV_MAX_UPLOAD_BATCH_MB=45`
- `PAYDAY_UPLOAD_BATCH_CHUNK_SIZE=3`
- `PAYDAY_SQLITE_PATH=/var/data/payday.db`
- `PAYDAY_UPLOADS_ROOT=/var/data/uploads`
- `LLM_PROVIDER=openai`
- `LLM_MODEL=gpt-4.1-mini`
- `LLM_API_KEY=<secret>`
- `TRANSCRIPTION_PROVIDER=openai`
- `TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe`
- `TRANSCRIPTION_API_KEY=<secret>`
- `SUPABASE_URL=<secret>`
- `SUPABASE_ANON_KEY=<secret>`
- `SUPABASE_SERVICE_ROLE_KEY=<secret>`
- `SUPABASE_STORAGE_BUCKET=payday-assets`

## Local-state persistence policy on Render

Current architecture still relies on local SQLite + local upload path for queue/repository operations. Keep both paths on a persistent disk mounted at `/var/data`.

- Durable DB path: `/var/data/payday.db`
- Durable upload root: `/var/data/uploads`

If/when repository and queue state are migrated fully to Supabase/Postgres and artifacts to Supabase Storage, this disk dependency can be removed.
