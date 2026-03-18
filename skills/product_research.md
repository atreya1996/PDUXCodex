# Product Research Skill

## Purpose
Keep implementation decisions aligned with PayDay's product and research constraints.

## Durable product rules
- The target audience is domestic workers in India.
- A user is only in-scope if they have both a smartphone and a bank account.
- No smartphone means Persona 3.
- No bank account means Persona 3.
- Trust barriers must be considered explicitly.
- WhatsApp familiarity is an important behavioral and operational signal.
- Informal lending is the baseline comparison.
- Structured analysis outputs must be valid JSON.

## Evidence rules
- Do not infer unsupported facts from transcripts.
- Prefer direct quotes when summarizing evidence.
- Distinguish observed evidence from model interpretation.
- Keep low-financial-literacy assumptions in mind when framing findings.

## Implementation implications
- Non-target handling should be explicit in analysis and persona logic.
- Product logic, prompts, schema, and UI presentation should stay aligned.
- When uncertainty exists, preserve the documented rules rather than inventing new heuristics.

## Guardrails
- If a skill instruction conflicts with product targeting rules, the targeting rules win.
