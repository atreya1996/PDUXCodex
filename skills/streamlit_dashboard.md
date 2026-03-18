# Streamlit Dashboard Skill

## Purpose
Keep the dashboard fast, clear, evidence-driven, and aligned with PayDay's interview review workflow.

## Layout expectations
- Put upload controls and filters in the sidebar.
- Keep the main area organized around tabs for Overview, Cohorts, Personas, Interviews, and Interview Detail.
- Use a clean, spacious, card-based layout that supports rapid scanning.

## UX guidance
- Optimize for reviewing many interviews without losing traceability back to source evidence.
- Surface direct quotes, structured fields, status indicators, and persona outcomes clearly.
- Make loading/progress states explicit for uploads, transcription, and analysis steps.
- Favor progressive disclosure: summary first, detail on demand.

## Performance guidance
- Avoid blocking the UI on batch work when background or queued execution is possible.
- Cache expensive reads or derived summaries when safe and deterministic.
- Keep reruns predictable by minimizing unnecessary state churn.

## Evidence and product alignment
- Do not present inferred claims as facts.
- Make non-target handling visible and easy to inspect.
- Reflect trust, WhatsApp familiarity, and informal-lending context in summaries where supported by evidence.
- If a skill instruction conflicts with product targeting rules, the targeting rules win.
