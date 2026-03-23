# Research Rules

## Core audience rule
- Center all research, analysis, and classification on domestic workers in India.

## Inclusion and exclusion rule
- A user is only in-scope if they have both:
  1. a smartphone, and
  2. a bank account.
- If the user lacks either prerequisite, treat them as non-target.

## Research interpretation principles
- Evaluate trust barriers explicitly in every analysis where adoption, retention, or willingness to borrow is discussed.
- Treat WhatsApp familiarity and usage as a meaningful distribution, education, support, and engagement channel.
- Use informal lending as the baseline comparison point when reasoning about user behavior, product value, and switching costs.

## Output rule
- Return structured JSON only.
- Do not add prose before or after the JSON output.
- Ensure the JSON is valid and machine-readable.
- Separate income evidence into per-household earnings, participant personal monthly income, and total household monthly income.
- Do not treat a single-household wage as total monthly income.
- If multiple income numbers appear, identify what each number refers to, require direct quote evidence for each captured income figure, and prefer the participant's explicitly stated total monthly income when available.
- Distinguish the participant's personal monthly income from the full household's monthly income.
- Preserve original transcript quotes and add summary-only English translations for key quotes/snippets when helpful.
- When interviewer and participant turns are evident, segment them in order and keep role uncertainty explicit.
- Mark income values as inferred_or_uncertain only when the transcript supports an estimate but not a direct statement.
