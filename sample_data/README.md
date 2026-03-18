# Sample data

This directory provides a demo dataset for running PayDay in sample/mock mode.

## Contents

- `mock_uploads/`: ten representative interview files that can be uploaded directly in Streamlit while `PAYDAY_USE_SAMPLE_MODE=true`.
- `interview_metadata.json`: interview-level metadata, expected personas, and processing notes.
- `structured_outputs/`: representative structured outputs for each mock upload.

## Usage

1. Keep `PAYDAY_USE_SAMPLE_MODE=true` in `.env`.
2. Start Streamlit with `streamlit run app.py`.
3. Upload the files from `mock_uploads/` one by one.
4. Compare the app output with `interview_metadata.json` and the paired file in `structured_outputs/`.

## Persona coverage

The sample set intentionally covers all five personas and both strict Persona 3 override paths:

- no smartphone -> Persona 3
- no bank account -> Persona 3
