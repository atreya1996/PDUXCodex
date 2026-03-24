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
