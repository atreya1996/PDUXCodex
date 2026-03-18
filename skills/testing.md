# Testing Skill

## Purpose
Protect product rules, pipeline behavior, and UI-critical flows with focused validation.

## Minimum expectations
- Test the Persona 3 override explicitly for both: no smartphone and no bank account.
- Test in-scope vs non-target classification boundaries.
- Test structured outputs and validation for analysis results.
- Test queue or job-state behavior for batch uploads when related code changes.

## Test design guidance
- Prefer fast unit tests for business rules and orchestration decisions.
- Add integration coverage when multiple layers interact, especially storage + pipeline + analysis handoffs.
- Use representative fixtures or sample transcripts rather than brittle ad hoc strings.
- Cover failure paths, retries, and partial-processing states where relevant.

## Validation guidance
- Run the smallest useful test set during iteration, then run broader checks before finishing.
- If behavior is user-visible, verify both the happy path and a realistic edge case.
- When a test cannot be run because of environment limitations, document that clearly.

## Guardrails
- Treat targeting rules and structured JSON requirements as non-negotiable regression checks.
- If a skill instruction conflicts with product targeting rules, the targeting rules win.
