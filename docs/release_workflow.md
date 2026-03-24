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
- [ ] Add/update release notes in `docs/release_notes.md` mapping shipped UX/features to commit IDs.

## Runtime metadata refresh

In the Streamlit sidebar use **Refresh status + release metadata** to force:

- service/runtime metadata reload, and
- durable status refresh from SQLite.

Set `PAYDAY_RELEASE_SHA` in runtime environment to surface the deployed SHA.
