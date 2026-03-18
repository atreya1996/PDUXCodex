# AGENTS.md

This repository contains PayDay, an MVP for analyzing user interview recordings from domestic workers in India. Before changing product logic, analysis prompts, persona rules, or dashboard behavior, read:

* `context/product.md`
* `context/research_rules.md`
* `prompts/analysis_prompt.md`

Non-target override is strict:

* No smartphone => Persona 3
* No bank account => Persona 3

Do not infer unsupported facts from transcripts. Prefer direct quotes and structured JSON outputs.

## Project identity

PayDay is an MVP for analyzing user interview recordings. The core workflow is:

1. batch audio upload,
2. transcription,
3. structured LLM analysis,
4. persona classification, and
5. a Streamlit dashboard for reviewing findings.

## Canonical memory files to load first

These are the canonical project-memory files. Always read them before changing analysis, persona, or dashboard logic:

* `context/product.md`
* `context/research_rules.md`
* `prompts/analysis_prompt.md`

Use these exact file names consistently across the repository.

## Targeting and persona hard rules

The strict non-target override always applies:

* if no smartphone -> always Persona 3
* if no bank account -> always Persona 3

Final personas:

* Persona 1: Employer-Dependent Digital Borrower
* Persona 2: Digitally Ready but Fearful
* Persona 3: Offline / Excluded
* Persona 4: Self-Reliant Non-Borrower
* Persona 5: High-Stress Cyclical Borrower

## Research and evidence rules

* Do not infer without evidence.
* Prefer direct quotes.
* Assume financial literacy may be low.
* Trust matters more than pricing.
* Informal lending is the baseline.
* Structured analysis must return JSON.

## Architecture guidance

Keep a clear separation between:

* UI,
* pipeline orchestration,
* storage,
* analysis, and
* persona logic.

Once implementation exists, likely paths include:

* `app.py` or `streamlit_app.py` for the Streamlit entrypoint
* `src/payday/pipeline.py`
* `src/payday/analysis.py`
* `src/payday/personas.py`
* `src/payday/repository.py`
* `src/payday/storage.py`

## UX expectations

* Sidebar should contain upload + filters.
* Main view should use tabs for Overview, Cohorts, Personas, Interviews, and Interview Detail.
* UI should be clean, card-based, spacious, and fast.

## Developer workflow

* Update `README.md` whenever setup steps change.
* Prefer mock/sample data support for local development.
* Keep prompt, persona, and schema changes synchronized.
