# Architecture Decision Records (ADRs)

This document records project-level decisions for PayDay.

## Requested backfill items and scope check

The following requested ADR topics appear unrelated to this repository’s actual domain and code history (PayDay interview-analysis MVP), so they are marked **out-of-scope** to avoid fictional history:

1. chat-based onboarding — **Out-of-scope for this repo history**
2. hybrid nutrition approach — **Out-of-scope for this repo history**
3. callAgent caching — **Out-of-scope for this repo history**
4. fail-closed limiter — **Out-of-scope for this repo history**
5. deferred CSP — **Out-of-scope for this repo history**
6. server-action wrapper — **Out-of-scope for this repo history**

Per request, these are replaced with six PayDay-relevant ADRs grounded in the current repository structure and product docs.

---

## ADR-001
- **Decision ID:** ADR-001
- **Date:** 2026-04-30
- **Status:** Accepted
- **Context:** PayDay analyzes interview transcripts to produce structured, reviewable outputs and persona assignments. Unstructured free-text model output is hard to validate and brittle for dashboards.
- **Decision:** Require structured JSON outputs for interview analysis, aligned to repository schemas and API serialization contracts.
- **Consequences:**
  - Pros: predictable parsing, easier validation/testing, safer dashboard rendering.
  - Cons: prompt/schema coupling must be maintained during iteration.
- **Supersedes:** None
- **Superseded-by:** None

## ADR-002
- **Decision ID:** ADR-002
- **Date:** 2026-04-30
- **Status:** Accepted
- **Context:** Product/research rules require strict targeting behavior to avoid misclassification for excluded users.
- **Decision:** Enforce non-target override: **no smartphone => Persona 3**, **no bank account => Persona 3**.
- **Consequences:**
  - Pros: consistent policy, safer downstream decisions for excluded users.
  - Cons: edge cases cannot override this rule even if other signals suggest a different persona.
- **Supersedes:** None
- **Superseded-by:** None

## ADR-003
- **Decision ID:** ADR-003
- **Date:** 2026-04-30
- **Status:** Accepted
- **Context:** Batch uploads and AI analysis can take variable time and should not block user-facing operations.
- **Decision:** Use queue-based, non-blocking pipeline/workers for interview processing.
- **Consequences:**
  - Pros: better reliability for batch jobs, improved responsiveness, clearer retry boundaries.
  - Cons: operational complexity (queue state, retries, worker monitoring).
- **Supersedes:** None
- **Superseded-by:** None

## ADR-004
- **Decision ID:** ADR-004
- **Date:** 2026-04-30
- **Status:** Accepted
- **Context:** Trustworthiness is critical for research workflows; unsupported inference can degrade data quality and researcher confidence.
- **Decision:** Prefer direct transcript evidence and avoid unsupported inference in analysis/persona logic.
- **Consequences:**
  - Pros: higher auditability and research integrity.
  - Cons: outputs may be conservative when transcript evidence is sparse.
- **Supersedes:** None
- **Superseded-by:** None

## ADR-005
- **Decision ID:** ADR-005
- **Date:** 2026-04-30
- **Status:** Accepted
- **Context:** The project supports local development and demo workflows where production integrations may be unavailable.
- **Decision:** Maintain sample/mock interview and structured output data for reproducible local testing and dashboard demos.
- **Consequences:**
  - Pros: faster onboarding, stable tests, easier product demos.
  - Cons: requires periodic refresh so samples stay representative.
- **Supersedes:** None
- **Superseded-by:** None

## ADR-006
- **Decision ID:** ADR-006
- **Date:** 2026-04-30
- **Status:** Accepted
- **Context:** PayDay spans UI, pipeline orchestration, storage/repositories, analysis, and persona logic.
- **Decision:** Keep strict modular separation across UI, orchestration, storage/repository, analysis, and persona modules.
- **Consequences:**
  - Pros: clearer ownership, easier testing, safer future migrations.
  - Cons: cross-module changes require explicit interface updates.
- **Supersedes:** None
- **Superseded-by:** None
