# PayDay MVP

PayDay is an MVP for analyzing interview recordings from domestic workers in India. The product is designed to support a workflow of batch upload, transcription, structured LLM analysis, persona assignment, and dashboard review in Streamlit. The product and research rules are strict: participants without a smartphone or without a bank account are non-target and must be assigned Persona 3. The analysis flow should prefer direct quotes, avoid unsupported inference, and keep outputs machine-readable JSON.

## Project purpose

PayDay helps a product or research team:

- upload single interviews first and then scale to batches,
- transcribe each file,
- turn transcripts into structured evidence,
- classify interviews into the canonical personas, and
- review outcomes in a dashboard built for quick cohort and interview inspection.

The current repository is a scaffold for that workflow. It already separates UI, pipeline orchestration, transcription, analysis, storage, repository access, and persona logic so live integrations can be swapped in without rewriting the dashboard.

## Architecture overview

### High-level flow

1. **Streamlit UI** accepts uploads and displays results.
2. **Application service** wires configuration and dependencies together.
3. **Pipeline orchestration** processes each file independently.
4. **Transcription service** converts uploaded audio or text into a transcript.
5. **Analysis service** converts the transcript into structured JSON-like evidence.
6. **Persona service** applies deterministic persona rules, including the strict Persona 3 override.
7. **Storage + repository layers** persist audio paths, structured responses, insights, and dashboard reads.

### Repository structure

```text
app.py                         # Streamlit entrypoint
src/payday/app.py              # Streamlit app composition
src/payday/service.py          # Backend facade used by the UI
src/payday/pipeline.py         # Per-file and batch processing orchestration
src/payday/transcription.py    # Transcription provider abstraction
src/payday/analysis.py         # Structured analysis provider abstraction
src/payday/personas.py         # Persona rules and Persona 3 override
src/payday/storage.py          # Supabase/local storage abstraction
src/payday/repository.py       # Persistence + dashboard reads
src/payday/dashboard/views.py  # Dashboard rendering
sample_data/                   # Demo-ready mock interview inputs and outputs
sql/                           # Database schema and migrations
```

### Architectural notes

- The dashboard layer should remain presentation-focused.
- The pipeline is designed so each interview can succeed or fail independently.
- Provider settings are injected through `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `TRANSCRIPTION_PROVIDER`, `TRANSCRIPTION_MODEL`, and `TRANSCRIPTION_API_KEY` instead of being hard-coded in the UI.
- Sample mode lets developers exercise the end-to-end flow without live APIs or external credentials.
- Live mode now runs a single startup validation pass for provider selection and credentials, then fails fast with field-specific errors if `LLM_API_KEY`, `TRANSCRIPTION_API_KEY`, or a provider name is invalid.
- The current implementation is still an MVP scaffold: provider factories are in place for OpenAI-first live integrations, and the analysis branch already reserves a clean extension point for Anthropic later.

## Required environment variables

Copy `.env.example` to `.env` before running locally.

```bash
cp .env.example .env
```

### Core application flags

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `PAYDAY_APP_ENV` | No | `development` | Labels the runtime environment in the UI. |
| `PAYDAY_RELEASE_SHA` | No | `""` | Commit SHA rendered in sidebar runtime metadata (`Runtime commit SHA`). |
| `PAYDAY_DATABASE_PATH` | No | `data/payday.db` | Local SQLite file used for repository persistence and dashboard reads. |
| `PAYDAY_USE_SAMPLE_MODE` | No | `true` | Enables mock/demo processing without live APIs. |
| `PAYDAY_ENABLE_UPLOADS` | No | `true` | Toggles the upload widget. |
| `PAYDAY_ENABLE_DASHBOARD` | No | `true` | Toggles dashboard rendering. |
| `PAYDAY_ENABLE_ANALYSIS` | No | `true` | Toggles the pipeline run button. |
| `PAYDAY_SQLITE_PATH` | No | `data/payday.db` | Local SQLite path used for durable dashboard reads across app restarts. |

### Supabase variables

| Variable | Required when using live storage | Default | Purpose |
| --- | --- | --- | --- |
| `SUPABASE_URL` | Yes | `""` | Supabase project URL. |
| `SUPABASE_ANON_KEY` | Usually yes | `""` | Client-facing key for standard Supabase access patterns. |
| `SUPABASE_SERVICE_ROLE_KEY` | Optional but recommended for server-side jobs | `""` | Elevated key for backend-only storage/database operations. |
| `SUPABASE_STORAGE_BUCKET` | No | `payday-assets` | Bucket name used for uploaded artifacts. |

### LLM analysis variables

| Variable | Required when using a live LLM | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_PROVIDER` | No | `openai` | Logical provider name shown in app metadata. |
| `LLM_MODEL` | No | `gpt-4.1-mini` | Model identifier for structured analysis. |
| `LLM_API_KEY` | Yes for live analysis (`PAYDAY_USE_SAMPLE_MODE=false`) | `""` | API key for the selected LLM provider. The app fails fast at startup if it is missing in live mode. |

### Transcription variables

| Variable | Required when using a live transcription API | Default | Purpose |
| --- | --- | --- | --- |
| `TRANSCRIPTION_PROVIDER` | No | `openai` | Logical provider name used by the transcription service. |
| `TRANSCRIPTION_MODEL` | No | `gpt-4o-mini-transcribe` | Model identifier for transcription. |
| `TRANSCRIPTION_API_KEY` | Yes for live transcription (`PAYDAY_USE_SAMPLE_MODE=false`) | `""` | API key for the selected transcription provider. The app fails fast at startup if it is missing in live mode. |

## How to configure Supabase

Supabase is optional in sample mode and recommended for live uploads.

1. Create a Supabase project.
2. Copy the project URL into `SUPABASE_URL`.
3. Copy the API keys into `SUPABASE_ANON_KEY` and/or `SUPABASE_SERVICE_ROLE_KEY`.
4. Create a storage bucket that matches `SUPABASE_STORAGE_BUCKET` (default: `payday-assets`).
5. Apply the schema from `sql/schema.sql` or the migration under `sql/migrations/001_initial_schema.sql` to your database.
6. Keep the service role key server-side only; do not expose it in a client bundle.

### Suggested Supabase setup checklist

- Create bucket: `payday-assets`
- Confirm the app can write objects under `audio/<interview_id>/<filename>`
- Load the SQL schema before wiring dashboard reads
- Use row-level security and storage policies appropriate for interview data

## How to choose LLM and transcription providers

The code is provider-configured only through environment variables: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `TRANSCRIPTION_PROVIDER`, `TRANSCRIPTION_MODEL`, and `TRANSCRIPTION_API_KEY`.

### LLM provider selection

Set:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-key
```

Use sample mode if you do not want a live LLM yet. In sample mode, no external credentials are required. In live mode, startup validation rejects missing `LLM_API_KEY` with an explicit `LLM_API_KEY is required ...` message and also rejects unsupported `LLM_PROVIDER` values before the app service is constructed. The analysis layer is abstracted behind a provider factory so OpenAI can be implemented first and Anthropic can be added later without changing the dashboard or pipeline call sites.

### Transcription provider selection

Set:

```bash
TRANSCRIPTION_PROVIDER=openai
TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
TRANSCRIPTION_API_KEY=your-key
```

When `PAYDAY_USE_SAMPLE_MODE=true`, the mock transcription flow reads uploaded file bytes as text when possible and does not require `TRANSCRIPTION_API_KEY`. That means the `.txt` files in `sample_data/mock_uploads/` can be uploaded directly to demonstrate the dashboard without any live transcription integration. When `PAYDAY_USE_SAMPLE_MODE=false`, the same startup validation path rejects missing `TRANSCRIPTION_API_KEY` with an explicit `TRANSCRIPTION_API_KEY is required ...` message and also rejects unsupported `TRANSCRIPTION_PROVIDER` values before the app service is constructed.

## How to run Streamlit locally

1. Create and activate a Python 3.11+ virtual environment.
2. Install the package in editable mode.
3. Copy `.env.example` to `.env` and adjust settings, including `PAYDAY_DATABASE_PATH` if you want a custom local SQLite location.
4. Start Streamlit.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
streamlit run app.py
```

Open the local URL printed by Streamlit, then:

- leave sample mode on for a demo without live providers, or
- switch sample mode off and provide live Supabase/LLM/transcription credentials.

## Release workflow and deployment checklist

Deployment source-of-truth branch is **`main`**. Use `docs/release_workflow.md` for the required release process, CI gates, and deployment checklist, including SHA matching across:

- merged commit SHA,
- deployed commit SHA, and
- runtime sidebar SHA.

After deployment, use the sidebar action **Refresh status + release metadata** once to reload durable status and runtime release metadata. Track shipped UX/features and commit mappings in `docs/release_notes.md`.

## How to use sample/mock mode without live APIs

Sample mode is the fastest way to demo the product before live integrations are ready.

### Enable sample mode

Make sure your `.env` includes:

```bash
PAYDAY_USE_SAMPLE_MODE=true
PAYDAY_ENABLE_UPLOADS=true
PAYDAY_ENABLE_DASHBOARD=true
PAYDAY_ENABLE_ANALYSIS=true
```

No live API keys are required in this mode.

The dashboard also persists interview metadata, structured fields, and summaries into the local SQLite database at `PAYDAY_SQLITE_PATH`, so restarting Streamlit does not clear previously processed interviews as long as that file still exists.

When prompts or `src/payday/analysis.py` change, do not judge new behavior from stale rows. Use dashboard re-analysis actions so SQLite rows are actively rebuilt:

- **Re-analyze selected interview** (Interview Detail overlay) reruns analysis + persona for one row and overwrites that interview's structured response/insight records.
- **Re-analyze all interviews** (Interviews tab) reruns analysis + persona for every durable interview and overwrites all structured response/insight records.

Each structured/insight row now stores `analysis_version` and `analyzed_at`, so Overview/Cohorts can surface mixed-version warnings and reviewers can trace which prompt/model revision produced each row.

### Demo workflow using `sample_data/`

The repository now includes ten mock uploads and matching metadata/expected outputs:

- `sample_data/mock_uploads/` contains ten representative interview files you can upload in the app.
- `sample_data/interview_metadata.json` lists interview-level metadata and the expected persona for each upload.
- `sample_data/structured_outputs/` contains representative structured analysis outputs for the same ten interviews.
- `sample_data/README.md` explains how the files map together.

### Why the mock uploads work

In sample mode, the current transcription service decodes uploaded bytes as text. Because the mock uploads are text files, each file acts as both the uploaded asset and the transcript source. This makes it possible to run the upload → transcription → analysis → persona flow locally with no external providers.

When `PAYDAY_USE_SAMPLE_MODE=false` and `TRANSCRIPTION_PROVIDER=openai`, the transcription service now sends the uploaded audio bytes to OpenAI using the configured model, returns the provider transcript text, and attaches useful metadata such as the filename, content type, byte size, and sample-mode flag. Timeouts are configurable through `TRANSCRIPTION_TIMEOUT_SECONDS`, and provider errors are allowed to bubble so the pipeline retry logic can re-attempt failed transcription jobs.

## Verification checklist

Use the checklist below before considering the local demo ready.

### 1. Upload 1 file first, then 10 files

- Start Streamlit in sample mode.
- Upload one small file first and confirm it processes end to end.
- Then upload the ten files from `sample_data/mock_uploads/`.
- Confirm all selected files are accepted by the uploader.

### 2. All files process independently

- Run the pipeline for each mock upload.
- Confirm one file failing or being retried does not block the others.
- Confirm the dashboard shows separate result cards/statuses for each upload.

### 3. Personas assign correctly

Validate the outputs against `sample_data/interview_metadata.json` and `sample_data/structured_outputs/`:

- `demo_01_employer_digital_borrower.txt` → Persona 1
- `demo_02_fearful_but_ready.txt` → Persona 2
- `demo_03_no_smartphone.txt` → Persona 3
- `demo_04_no_bank_account.txt` → Persona 3
- `demo_05_self_reliant.txt` → Persona 4
- `demo_06_cyclical_stress.txt` → Persona 5
- `demo_07_whatsapp_but_fearful.txt` → Persona 2
- `demo_08_employer_supported_app_user.txt` → Persona 1
- `demo_09_fully_excluded.txt` → Persona 3
- `demo_10_savings_first.txt` → Persona 4

Pay special attention to the hard override rule:

- no smartphone → Persona 3
- no bank account → Persona 3

### 4. Dashboard metrics update

- Confirm the dashboard refreshes after each processed file.
- Confirm summary cards, counts, and interview lists reflect the newly processed mock uploads.
- Confirm status, evidence, and persona outputs remain traceable back to the sample transcripts and structured outputs.

## Sample data inventory

```text
sample_data/
├── README.md
├── interview_metadata.json
├── mock_uploads/
│   ├── demo_01_employer_digital_borrower.txt
│   ├── demo_02_fearful_but_ready.txt
│   ├── demo_03_no_smartphone.txt
│   ├── demo_04_no_bank_account.txt
│   ├── demo_05_self_reliant.txt
│   ├── demo_06_cyclical_stress.txt
│   ├── demo_07_whatsapp_but_fearful.txt
│   ├── demo_08_employer_supported_app_user.txt
│   ├── demo_09_fully_excluded.txt
│   └── demo_10_savings_first.txt
└── structured_outputs/
    ├── demo_01_employer_digital_borrower.json
    ├── demo_02_fearful_but_ready.json
    ├── demo_03_no_smartphone.json
    ├── demo_04_no_bank_account.json
    ├── demo_05_self_reliant.json
    ├── demo_06_cyclical_stress.json
    ├── demo_07_whatsapp_but_fearful.json
    ├── demo_08_employer_supported_app_user.json
    ├── demo_09_fully_excluded.json
    └── demo_10_savings_first.json
```
