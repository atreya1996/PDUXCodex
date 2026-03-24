CREATE TABLE IF NOT EXISTS interviews (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    transcript TEXT,
    transcript_text TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS structured_responses (
    interview_id TEXT PRIMARY KEY,
    smartphone_user INTEGER,
    has_bank_account INTEGER,
    income_range TEXT,
    borrowing_history TEXT,
    repayment_preference TEXT,
    loan_interest TEXT,
    FOREIGN KEY (interview_id) REFERENCES interviews (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS insights (
    interview_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    key_quotes TEXT NOT NULL,
    persona TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    FOREIGN KEY (interview_id) REFERENCES interviews (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_interviews_status_created_at
    ON interviews (status, created_at DESC);
