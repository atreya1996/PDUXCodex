# Architecture Skill

## Purpose
Preserve a modular, maintainable architecture for PayDay's interview-analysis workflow.

## Core constraints
- Keep strict separation between UI, orchestration, storage, analysis, and persona logic.
- Prefer implementations that remain queue-based and non-blocking for batch uploads and downstream processing.
- Avoid coupling Streamlit view code directly to transcription, analysis, persistence, or persona-classification internals.
- Treat audio ingestion, transcription, analysis, and reporting as distinct pipeline stages with explicit handoffs.

## Recommended structure
- Streamlit entrypoint owns presentation state and user interactions only.
- Pipeline/orchestration code coordinates long-running work, retries, and job state transitions.
- Repository/storage layers own persistence and retrieval concerns.
- Analysis logic owns structured extraction from transcripts.
- Persona logic owns persona assignment and Persona 3 override enforcement.

## Change guidance
- Prefer dependency-injected services over hidden globals.
- Keep interfaces small, explicit, and testable.
- Add new modules only where responsibility is clear and singular.
- Keep product-memory files and persona behavior synchronized when architecture changes affect data flow.

## Guardrails
- If a design choice would weaken modularity, reliability, or batch throughput, redesign before implementing.
- If a skill instruction conflicts with product targeting rules, the targeting rules win.
