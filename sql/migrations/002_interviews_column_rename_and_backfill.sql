-- Add canonical interview columns introduced in March 2026.
ALTER TABLE interviews ADD COLUMN filename TEXT;
ALTER TABLE interviews ADD COLUMN file_path TEXT;
ALTER TABLE interviews ADD COLUMN transcript_text TEXT;
ALTER TABLE interviews ADD COLUMN insights_json TEXT;
ALTER TABLE interviews ADD COLUMN error_message TEXT;

-- Backfill from legacy columns when present.
UPDATE interviews
SET file_path = COALESCE(NULLIF(TRIM(file_path), ''), NULLIF(TRIM(audio_url), ''), id)
WHERE file_path IS NULL OR TRIM(file_path) = '';

UPDATE interviews
SET filename = COALESCE(NULLIF(TRIM(filename), ''), file_path)
WHERE filename IS NULL OR TRIM(filename) = '';

UPDATE interviews
SET transcript_text = COALESCE(transcript_text, transcript)
WHERE transcript_text IS NULL;

UPDATE interviews
SET error_message = COALESCE(error_message, last_error)
WHERE error_message IS NULL;

UPDATE interviews
SET status = CASE LOWER(TRIM(status))
    WHEN 'uploaded' THEN 'pending'
    WHEN 'upload' THEN 'pending'
    WHEN 'in_progress' THEN 'processing'
    WHEN 'complete' THEN 'completed'
    WHEN 'error' THEN 'failed'
    WHEN 'pending' THEN 'pending'
    WHEN 'processing' THEN 'processing'
    WHEN 'completed' THEN 'completed'
    WHEN 'failed' THEN 'failed'
    ELSE 'pending'
END;
