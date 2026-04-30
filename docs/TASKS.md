# TASKS

> **Maintenance rule:** every merged PR must update `docs/TASKS.md` with status deltas (what moved, what changed, what is newly blocked/unblocked).

## In Progress

- **Stabilize Supabase as primary persistence backend while keeping SQLite parity**
  - Continue wiring and validating `PAYDAY_DB_BACKEND=supabase` for end-to-end queue + dashboard reads.
  - Complete parity checks between SQLite repository paths and Postgres/Supabase equivalents.
  - Why now: recent merges added selectable repository/storage backends and Postgres migration scaffolding; parity hardening remains active work.

- **Production readiness for split web + worker runtime on Render**
  - Validate operational runbook, env var completeness, and worker/web coordination in deployed environments.
  - Why now: `render.yaml` and worker process model are in place, but ongoing stabilization is still implied by recent fix cadence.

## Blocked

- **Live-provider confidence testing in non-sample mode**
  - Blocker: requires provisioned production-safe credentials and environment access for LLM/transcription providers and Supabase service role usage.
  - Unblock by: setting up deployment secrets and running smoke/regression checks in live mode (`PAYDAY_USE_SAMPLE_MODE=false`).

## Backlog

- **Automate migration/backfill pipeline from local SQLite to Supabase Postgres**
  - Current docs describe a manual export/import path; add scripted tooling and verification helpers.

- **Add CI checks that enforce docs + task hygiene**
  - Include a lightweight check that PRs touching feature code also update `docs/TASKS.md` and relevant runbook/docs.

- **Expand operational observability for queue lifecycle**
  - Add clearer operational dashboards/queries for job aging, retries, and failure clusters.

- **Harden startup/runtime validation UX for operators**
  - Improve operator-facing diagnostics for provider misconfiguration and failed readiness checks.

- **Broaden automated test coverage around queue + repository cross-backend behavior**
  - Strengthen integration tests for jobs/events flows and failure recovery paths.

## Done (recent)

Backfilled from `git log --oneline` (most recent first):

- **Merged PR #124** — scan codebase for backend setup status (`39f4b0c`).
- **Fixed Render worker startup crash** from stale app-service argument (`5ba36f0`).
- **Merged PR #123** — backend setup status sweep (`6c6fe95`).
- **Fixed flaky queue upload `KeyError`** while creating jobs (`1a2434d`).
- **Merged PR #122** — aligned defaults and docs for `PAYDAY_USE_SAMPLE_MODE` (`c186504`).
- **Merged PR #121** — added Postgres migration files + runbook (`64535f3`).
- **Clarified sample-mode defaults** and added startup mode banner (`7445d3b`).
- **Added Postgres-first migrations + Supabase runbook** (`abe20f2`).
- **Merged PR #120** — added `render.yaml` for web/worker services (`db2f16a`).
- **Added Render web/worker blueprint + deployment runbook details** (`9709bf2`).
- **Merged PR #119** — backend service package + app updates (`a0cc1f0`).
- **Added backend API package + standalone worker loop** (`f9c0ac0`).
- **Merged PR #118** — Supabase-selectable repository/storage services (`d9babdc`).
- **Added Supabase-selectable repository + storage backends** (`5324543`).
- **Merged PR #117** — local-storage pipeline refactor (`c5792cd`).
- **Restored live processing helper in Streamlit app** (`602743e`).
- **Merged PR #116** — pipeline/local storage refactor (`49e5399`).
- **Fixed repository processing events table + APIs** (`9e8ea51`).
- **Merged PR #115** — jobs table and upload enqueue flow (`dbb75bd`).
- **Merged PR #114** — live failure tracking + UI updates (`3b04929`).
- **Added queued jobs worker + non-blocking dashboard actions** (`980c405`).
- **Added live sidebar failures with interview-detail links** (`36bac87`).
- **Merged PR #113** — lightweight jobs/events SQL tables (`e6a5b42`).
- **Added live processing events + polling progress UI** (`3966c85`).
- **Merged PR #112** — safety checkpoint + repository refactor (`1a2a96f`).
- **Fixed repository interview storage contract/syntax blockers** (`fb6a150`).

## Next 7 days

1. **Run live-mode deployment smoke tests** on Render web + worker with real secrets in a controlled environment.
2. **Document a concise deploy verification checklist** (startup validation, queue processing, dashboard read parity, failure handling).
3. **Implement a scripted SQLite -> Supabase backfill utility** plus post-migration sanity checks.
4. **Add targeted integration tests** for queued batch uploads, retry/failure states, and repository parity across backends.
5. **Define operational SLO-style metrics** for queue latency, job success rate, and failure triage turnaround.
6. **Introduce CI guardrails** that fail when core docs (including `docs/TASKS.md`) are stale for feature-changing PRs.
