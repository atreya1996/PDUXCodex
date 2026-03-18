# Code Quality Skill

## Purpose
Maintain readable, reviewable, low-risk code across the PayDay codebase.

## Standards
- Write small, composable functions with clear names and single responsibilities.
- Prefer explicit data contracts and structured return values over loosely shaped dictionaries when possible.
- Keep domain rules centralized instead of duplicating them across UI and pipeline layers.
- Remove dead code, debug leftovers, and stale comments when touching related areas.

## Reliability-minded implementation
- Validate inputs at boundaries.
- Fail loudly on impossible states rather than silently masking errors.
- Keep logging structured and useful for tracing interview-processing failures.
- Prefer deterministic behavior over implicit magic.

## Documentation expectations
- Update README or nearby docs when behavior, setup, or workflow expectations change.
- Keep prompt, persona, schema, and orchestration changes synchronized.
- Leave code in a state that future Codex sessions can extend safely.

## Guardrails
- Do not infer unsupported user facts from transcripts.
- Preserve strict Persona 3 override behavior for no smartphone or no bank account.
- If a skill instruction conflicts with product targeting rules, the targeting rules win.
