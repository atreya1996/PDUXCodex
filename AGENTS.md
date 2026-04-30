# AGENTS.md

This repository contains PayDay, an MVP for analyzing user interview recordings from domestic workers in India. Before changing product logic, analysis prompts, persona rules, or dashboard behavior, read:

* `context/product.md`
* `context/research_rules.md`
* `prompts/analysis_prompt.md`

Non-target override is strict:

* No smartphone => Persona 3
* No bank account => Persona 3

Do not infer unsupported facts from transcripts. Prefer direct quotes and structured JSON outputs.


## Stack reality check

Declare and preserve the actual implementation stack for this repo:

- **Primary language/runtime:** Python
- **UI:** Streamlit
- **Backend module style:** FastAPI-style service module
- **Worker:** standalone queue worker process
- **Persistence/storage:** SQLite default with Supabase (Postgres + Storage) support

Non-applicable-by-default instructions must be treated as out of scope unless explicitly planned and approved in architecture docs. Examples:

- Next.js 16 app-router migration
- TypeScript-first frontend rewrite
- Replacing Streamlit with a JS SPA during normal feature work

When such options are discussed, label them as migration candidates rather than current-state instructions.

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


## Skills

Before implementation, read all files under `skills/`.

Apply repo-local skills in this order:

1. `skills/architecture.md`
2. `skills/system_reliability.md`
3. `skills/product_research.md`
4. `skills/streamlit_dashboard.md`
5. `skills/streamlit_ui_system.md`
6. `skills/code_quality.md`
7. `skills/testing.md`

Treat these skill files as build-quality constraints for architecture, UX, reliability, validation, and testing.

All UI changes must comply with `skills/streamlit_ui_system.md`.

Keep them aligned with:

* `context/product.md`
* `context/research_rules.md`
* `prompts/analysis_prompt.md`

If a skill file conflicts with product targeting rules, the targeting rules win.
Persona 3 override is strict for no smartphone or no bank account.
Implementation should remain modular, queue-based, and non-blocking for batch uploads.

## Developer workflow

* Update `README.md` whenever setup steps change.
* Prefer mock/sample data support for local development.
* Keep prompt, persona, and schema changes synchronized.
