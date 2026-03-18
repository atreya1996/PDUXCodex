# System Reliability Skill

## Purpose
Keep PayDay resilient under batch processing, partial failures, and repeated runs.

## Reliability constraints
- Prefer queue-based, non-blocking execution for batch uploads and downstream processing.
- Design long-running stages so they can be retried without corrupting state.
- Make job status, partial completion, and failures observable.
- Avoid designs that require the UI request cycle to hold open for full pipeline completion.

## Operational guidance
- Isolate side effects and external calls behind clear interfaces.
- Use durable identifiers for interviews, jobs, and artifacts.
- Make state transitions explicit and auditable.
- Handle missing files, malformed transcripts, and analysis failures gracefully.

## Data and recovery guidance
- Favor idempotent writes where practical.
- Preserve enough metadata to support reprocessing and debugging.
- Ensure failures in one interview do not block an entire batch unnecessarily.
- Keep validation near ingestion boundaries to reduce downstream error spread.

## Guardrails
- Reliability improvements must not weaken targeting rules or evidence requirements.
- If a skill instruction conflicts with product targeting rules, the targeting rules win.
