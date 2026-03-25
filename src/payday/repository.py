from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from payday.analysis import DEFAULT_UNKNOWN_VALUE
from payday.models import AnalysisResult, PersonaClassification, PipelineResult
from payday.schema_versions import ANALYSIS_SCHEMA_VERSION, PERSONA_RULESET_VERSION

_UNSET = object()
_VALID_STATUS = {"pending", "processing", "completed", "failed"}
_VALID_JOB_STATUS = {"pending", "processing", "completed", "failed", "cancelled"}
_LEGACY_STATUS_MAP = {
    "uploaded": "pending",
    "upload": "pending",
    "pending": "pending",
    "processing": "processing",
    "in_progress": "processing",
    "completed": "completed",
    "complete": "completed",
    "failed": "failed",
    "error": "failed",
}


@dataclass(frozen=True, slots=True)
class InterviewRecord:
    id: str
    filename: str
    file_path: str
    status: str
    transcript_text: str | None
    error_message: str | None
    created_at: str
    latest_stage: str = "upload"
    analysis_version: str | None = None
    transcript: str | None = None
    last_error: str | None = None
    audio_url: str | None = None
    insights_json: str | None = None

    def __post_init__(self) -> None:
        if not self.filename:
            object.__setattr__(self, "filename", PaydayRepository._filename_from_audio_url(self.file_path))
        object.__setattr__(self, "audio_url", self.file_path)
        object.__setattr__(self, "transcript_text", self.transcript_text if self.transcript_text is not None else self.transcript)
        object.__setattr__(self, "transcript", self.transcript if self.transcript is not None else self.transcript_text)
        object.__setattr__(self, "error_message", self.error_message if self.error_message is not None else self.last_error)
        object.__setattr__(self, "last_error", self.last_error if self.last_error is not None else self.error_message)


@dataclass(frozen=True, slots=True)
class StructuredResponseRecord:
    interview_id: str
    smartphone_user: bool | None
    has_bank_account: bool | None
    per_household_earnings: str | None
    participant_personal_monthly_income: str | None
    total_household_monthly_income: str | None
    income_range: str | None
    borrowing_history: str | None
    repayment_preference: str | None
    loan_interest: str | None
    segmented_dialogue: list[dict[str, Any]] = field(default_factory=list)
    analysis_version: str | None = None
    analyzed_at: str | None = None


@dataclass(frozen=True, slots=True)
class InsightRecord:
    interview_id: str
    summary: str
    key_quotes: list[str]
    persona: str
    confidence_score: float
    analysis_version: str | None = None
    analyzed_at: str | None = None


@dataclass(frozen=True, slots=True)
class InterviewListItem:
    id: str
    filename: str
    file_path: str
    status: str
    created_at: str
    persona: str | None
    confidence_score: float | None
    summary: str | None
    audio_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_url", self.file_path)


@dataclass(frozen=True, slots=True)
class InterviewDetail:
    interview: InterviewRecord
    structured_response: StructuredResponseRecord | None
    insight: InsightRecord | None


@dataclass(frozen=True, slots=True)
class DashboardInterviewRecord:
    id: str
    filename: str
    file_path: str
    transcript: str | None
    status: str
    latest_stage: str
    last_error: str | None
    created_at: str
    smartphone_user: bool | None
    has_bank_account: bool | None
    per_household_earnings: str | None
    participant_personal_monthly_income: str | None
    total_household_monthly_income: str | None
    income_range: str | None
    borrowing_history: str | None
    repayment_preference: str | None
    loan_interest: str | None
    summary: str | None
    key_quotes: list[str]
    persona: str | None
    confidence_score: float | None
    transcript_quality: str | None = None
    analysis_version: str | None = None
    analyzed_at: str | None = None
    segmented_dialogue: list[dict[str, Any]] = field(default_factory=list)
    data_malformed: bool = False
    data_malformed_details: list[str] = field(default_factory=list)
    audio_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_url", self.file_path)


@dataclass(frozen=True, slots=True)
class DashboardStatusOverview:
    total_interviews: int
    status_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    batch_id: str
    interview_id: str
    filename: str
    content_type: str
    status: str
    attempts: int
    max_attempts: int
    error_message: str | None
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class ProcessingEventRecord:
    id: int
    file_id: str
    stage: str
    status: str
    message: str | None
    created_at: str


class PaydayRepository:
    """Persistence facade for analysis artifacts and dashboard reads."""

    def __init__(self, database_path: str = ":memory:", schema_path: str | None = None) -> None:
        self.database_path = self._normalize_database_path(database_path)
        self.schema_path = schema_path or str(Path(__file__).resolve().parents[2] / "sql" / "schema.sql")
        self._items: dict[str, PipelineResult] = {}
        self._ensure_database_directory()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()


    def _ensure_database_directory(self) -> None:
        if self.database_path == ":memory:":
            return

        Path(self.database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return self._connection

    def _initialize(self) -> None:
        schema_sql = Path(self.schema_path).read_text(encoding="utf-8")
        with self._connect() as connection:
            connection.executescript(schema_sql)
            self._ensure_interview_columns(connection)
            self._ensure_job_columns(connection)
            self._migrate_legacy_interview_data(connection)
            self._ensure_structured_response_columns(connection)
            self._ensure_insight_columns(connection)
            self._ensure_processing_events_table(connection)

    def _ensure_interview_columns(self, connection: sqlite3.Connection) -> None:
        interview_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(interviews)").fetchall()
        }
        if "latest_stage" not in interview_columns:
            connection.execute(
                "ALTER TABLE interviews ADD COLUMN latest_stage TEXT NOT NULL DEFAULT 'upload'"
            )
        if "audio_url" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN audio_url TEXT")
        if "transcript" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN transcript TEXT")
        if "last_error" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN last_error TEXT")
        if "transcript_text" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN transcript_text TEXT")
        if "error_message" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN error_message TEXT")
        if "analysis_schema_version" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN analysis_schema_version INTEGER")
        if "persona_ruleset_version" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN persona_ruleset_version INTEGER")
        if "analysis_version" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN analysis_version TEXT")
        if "filename" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN filename TEXT")
        if "file_path" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN file_path TEXT")
        if "insights_json" not in interview_columns:
            connection.execute("ALTER TABLE interviews ADD COLUMN insights_json TEXT")

    def _migrate_legacy_interview_data(self, connection: sqlite3.Connection) -> None:
        interview_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(interviews)").fetchall()
        }
        def _select_expr(column_name: str) -> str:
            return column_name if column_name in interview_columns else f"NULL AS {column_name}"

        rows = connection.execute(
            f"""
            SELECT
                id,
                {_select_expr("audio_url")},
                {_select_expr("transcript")},
                {_select_expr("last_error")},
                status,
                {_select_expr("filename")},
                {_select_expr("file_path")},
                {_select_expr("transcript_text")},
                {_select_expr("error_message")}
            FROM interviews
            """
        ).fetchall()
        for row in rows:
            legacy_path = str(row["audio_url"] or "").strip()
            derived_file_path = str(row["file_path"] or "").strip() or legacy_path
            derived_filename = str(row["filename"] or "").strip() or self._filename_from_audio_url(derived_file_path)
            transcript_text = row["transcript_text"] if row["transcript_text"] is not None else row["transcript"]
            error_message = row["error_message"] if row["error_message"] is not None else row["last_error"]
            status = self._normalize_status(str(row["status"] or "pending"))
            connection.execute(
                """
                UPDATE interviews
                SET filename = ?,
                    file_path = ?,
                    transcript_text = ?,
                    error_message = ?,
                    status = ?
                WHERE id = ?
                """,
                (derived_filename, derived_file_path, transcript_text, error_message, status, row["id"]),
            )

    def _ensure_structured_response_columns(self, connection: sqlite3.Connection) -> None:
        structured_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(structured_responses)").fetchall()
        }
        if "segmented_dialogue" not in structured_columns:
            connection.execute("ALTER TABLE structured_responses ADD COLUMN segmented_dialogue TEXT")
        if "analysis_version" not in structured_columns:
            connection.execute("ALTER TABLE structured_responses ADD COLUMN analysis_version TEXT")
        if "analyzed_at" not in structured_columns:
            connection.execute("ALTER TABLE structured_responses ADD COLUMN analyzed_at TEXT")

    def _ensure_insight_columns(self, connection: sqlite3.Connection) -> None:
        insight_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(insights)").fetchall()
        }
        if "analysis_version" not in insight_columns:
            connection.execute("ALTER TABLE insights ADD COLUMN analysis_version TEXT")
        if "analyzed_at" not in insight_columns:
            connection.execute("ALTER TABLE insights ADD COLUMN analyzed_at TEXT")

    def _ensure_job_columns(self, connection: sqlite3.Connection) -> None:
        table_row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if table_row is None:
            connection.execute(
                """
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    interview_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (interview_id) REFERENCES interviews (id) ON DELETE CASCADE
                )
                """
            )
            connection.execute("CREATE INDEX idx_jobs_status_created_at ON jobs (status, created_at ASC)")
            connection.execute("CREATE INDEX idx_jobs_batch_id ON jobs (batch_id)")
            return

        job_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "cancel_requested" not in job_columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
        if "attempts" not in job_columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "max_attempts" not in job_columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 3")
        if "updated_at" not in job_columns:
            now = datetime.now(timezone.utc).isoformat()
            connection.execute("ALTER TABLE jobs ADD COLUMN updated_at TEXT")
            connection.execute("UPDATE jobs SET updated_at = ? WHERE updated_at IS NULL", (now,))

    def _ensure_processing_events_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES interviews (id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_processing_events_file_created
                ON processing_events (file_id, created_at DESC, id DESC)
            """
        )

    def create_interview(
        self,
        *,
        file_path: str | None = None,
        audio_url: str | None = None,
        transcript: str | None = None,
        status: str = "pending",
        latest_stage: str = "upload",
        last_error: str | None = None,
        interview_id: str | None = None,
        created_at: str | None = None,
    ) -> InterviewRecord:
        resolved_file_path = (file_path or audio_url or "").strip()
        if not resolved_file_path:
            raise ValueError("create_interview requires file_path or audio_url.")

        record = InterviewRecord(
            id=interview_id or str(uuid4()),
            filename=self._filename_from_audio_url(resolved_file_path),
            file_path=resolved_file_path,
            status=self._normalize_status(status),
            transcript_text=transcript,
            error_message=last_error,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            latest_stage=latest_stage,
            analysis_version=None,
            transcript=transcript,
            last_error=last_error,
            insights_json=None,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interviews (
                    id, audio_url, filename, file_path, transcript, transcript_text, status, latest_stage, last_error, error_message, created_at, analysis_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.file_path,
                    record.filename,
                    record.file_path,
                    record.transcript,
                    record.transcript_text,
                    record.status,
                    record.latest_stage,
                    record.last_error,
                    record.error_message,
                    record.created_at,
                    record.analysis_version,
                ),
            )
        return record

    def update_interview(
        self,
        interview_id: str,
        *,
        file_path: str | None = None,
        audio_url: str | None = None,
        transcript: str | None = None,
        status: str | None = None,
        latest_stage: str | None = None,
        last_error: str | object = _UNSET,
    ) -> InterviewRecord:
        existing = self.get_interview(interview_id)
        next_file_path = file_path if file_path is not None else audio_url if audio_url is not None else existing.file_path
        next_transcript = transcript if transcript is not None else existing.transcript_text
        next_error_message = existing.error_message if last_error is _UNSET else last_error

        updated = InterviewRecord(
            id=existing.id,
            filename=self._filename_from_audio_url(next_file_path),
            file_path=next_file_path,
            status=self._normalize_status(status) if status is not None else existing.status,
            transcript_text=next_transcript,
            error_message=next_error_message if isinstance(next_error_message, str) or next_error_message is None else str(next_error_message),
            created_at=existing.created_at,
            latest_stage=latest_stage if latest_stage is not None else existing.latest_stage,
            analysis_version=existing.analysis_version,
            transcript=next_transcript,
            last_error=next_error_message if isinstance(next_error_message, str) or next_error_message is None else str(next_error_message),
            insights_json=existing.insights_json,
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE interviews
                SET audio_url = ?,
                    filename = ?,
                    file_path = ?,
                    transcript = ?,
                    transcript_text = ?,
                    status = ?,
                    last_error = ?,
                    error_message = ?,
                    latest_stage = ?,
                    analysis_version = ?
                WHERE id = ?
                """,
                (
                    updated.file_path,
                    updated.filename,
                    updated.file_path,
                    updated.transcript,
                    updated.transcript_text,
                    updated.status,
                    updated.last_error,
                    updated.error_message,
                    updated.latest_stage,
                    updated.analysis_version,
                    interview_id,
                ),
            )
        return updated

    def get_interview(self, interview_id: str) -> InterviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    COALESCE(filename, '') AS filename,
                    COALESCE(file_path, audio_url) AS file_path,
                    status,
                    COALESCE(transcript_text, transcript) AS transcript_text,
                    COALESCE(error_message, last_error) AS error_message,
                    created_at,
                    latest_stage,
                    analysis_version
                FROM interviews
                WHERE id = ?
                """,
                (interview_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Interview {interview_id} was not found.")
        return InterviewRecord(**dict(row))

    def delete_interview(self, interview_id: str) -> bool:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM interviews WHERE id = ?",
                (interview_id,),
            ).fetchone()
            if existing is None:
                return False
            connection.execute(
                """
                DELETE FROM interviews
                WHERE id = ?
                """,
                (interview_id,),
            )
        self._items.pop(interview_id, None)
        return True

    def upsert_structured_response(
        self,
        interview_id: str,
        *,
        smartphone_user: bool | None,
        has_bank_account: bool | None,
        per_household_earnings: str | None,
        participant_personal_monthly_income: str | None,
        total_household_monthly_income: str | None,
        income_range: str | None,
        borrowing_history: str | None,
        repayment_preference: str | None,
        loan_interest: str | None,
        segmented_dialogue: list[dict[str, Any]] | None = None,
        analysis_version: str | None = None,
        analyzed_at: str | None = None,
    ) -> StructuredResponseRecord:
        record = StructuredResponseRecord(
            interview_id=interview_id,
            smartphone_user=smartphone_user,
            has_bank_account=has_bank_account,
            per_household_earnings=per_household_earnings,
            participant_personal_monthly_income=participant_personal_monthly_income,
            total_household_monthly_income=total_household_monthly_income,
            income_range=income_range,
            borrowing_history=borrowing_history,
            repayment_preference=repayment_preference,
            loan_interest=loan_interest,
            segmented_dialogue=segmented_dialogue or [],
            analysis_version=analysis_version,
            analyzed_at=analyzed_at or datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO structured_responses (
                    interview_id,
                    smartphone_user,
                    has_bank_account,
                    per_household_earnings,
                    participant_personal_monthly_income,
                    total_household_monthly_income,
                    income_range,
                    borrowing_history,
                    repayment_preference,
                    loan_interest,
                    segmented_dialogue,
                    analysis_version,
                    analyzed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    smartphone_user = excluded.smartphone_user,
                    has_bank_account = excluded.has_bank_account,
                    per_household_earnings = excluded.per_household_earnings,
                    participant_personal_monthly_income = excluded.participant_personal_monthly_income,
                    total_household_monthly_income = excluded.total_household_monthly_income,
                    income_range = excluded.income_range,
                    borrowing_history = excluded.borrowing_history,
                    repayment_preference = excluded.repayment_preference,
                    loan_interest = excluded.loan_interest,
                    segmented_dialogue = excluded.segmented_dialogue,
                    analysis_version = excluded.analysis_version,
                    analyzed_at = excluded.analyzed_at
                """,
                (
                    record.interview_id,
                    self._bool_to_int(record.smartphone_user),
                    self._bool_to_int(record.has_bank_account),
                    record.per_household_earnings,
                    record.participant_personal_monthly_income,
                    record.total_household_monthly_income,
                    record.income_range,
                    record.borrowing_history,
                    record.repayment_preference,
                    record.loan_interest,
                    json.dumps(record.segmented_dialogue, ensure_ascii=False),
                    record.analysis_version,
                    record.analyzed_at,
                ),
            )
            connection.execute(
                """
                UPDATE interviews
                SET analysis_schema_version = ?, analysis_version = ?, insights_json = ?
                WHERE id = ?
                """,
                (
                    ANALYSIS_SCHEMA_VERSION,
                    analysis_version,
                    json.dumps(
                        {
                            "smartphone_user": record.smartphone_user,
                            "has_bank_account": record.has_bank_account,
                            "per_household_earnings": record.per_household_earnings,
                            "participant_personal_monthly_income": record.participant_personal_monthly_income,
                            "total_household_monthly_income": record.total_household_monthly_income,
                            "income_range": record.income_range,
                            "borrowing_history": record.borrowing_history,
                            "repayment_preference": record.repayment_preference,
                            "loan_interest": record.loan_interest,
                            "segmented_dialogue": record.segmented_dialogue,
                        },
                        ensure_ascii=False,
                    ),
                    interview_id,
                ),
            )
        return record

    def upsert_insight(
        self,
        interview_id: str,
        *,
        summary: str,
        key_quotes: list[str],
        persona: str,
        confidence_score: float,
        analysis_version: str | None = None,
        analyzed_at: str | None = None,
    ) -> InsightRecord:
        record = InsightRecord(
            interview_id=interview_id,
            summary=summary,
            key_quotes=key_quotes,
            persona=persona,
            confidence_score=confidence_score,
            analysis_version=analysis_version,
            analyzed_at=analyzed_at or datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO insights (interview_id, summary, key_quotes, persona, confidence_score, analysis_version, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    summary = excluded.summary,
                    key_quotes = excluded.key_quotes,
                    persona = excluded.persona,
                    confidence_score = excluded.confidence_score,
                    analysis_version = excluded.analysis_version,
                    analyzed_at = excluded.analyzed_at
                """,
                (
                    record.interview_id,
                    record.summary,
                    json.dumps(record.key_quotes),
                    record.persona,
                    record.confidence_score,
                    record.analysis_version,
                    record.analyzed_at,
                ),
            )
            connection.execute(
                """
                UPDATE interviews
                SET persona_ruleset_version = ?, analysis_version = COALESCE(?, analysis_version)
                WHERE id = ?
                """,
                (
                    PERSONA_RULESET_VERSION,
                    analysis_version,
                    interview_id,
                ),
            )
        return record

    def persist_reprocessed_interview(
        self,
        interview_id: str,
        *,
        transcript: str,
        status: str,
        latest_stage: str,
        last_error: str | None,
        structured_response: dict[str, object],
        insight: dict[str, object],
        analysis_version: str,
        analyzed_at: str,
    ) -> DashboardInterviewRecord:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE interviews
                SET transcript_text = ?, status = ?, latest_stage = ?, error_message = ?
                WHERE id = ?
                """,
                (transcript, status, latest_stage, last_error, interview_id),
            )
            connection.execute(
                """
                INSERT INTO structured_responses (
                    interview_id,
                    smartphone_user,
                    has_bank_account,
                    per_household_earnings,
                    participant_personal_monthly_income,
                    total_household_monthly_income,
                    income_range,
                    borrowing_history,
                    repayment_preference,
                    loan_interest,
                    segmented_dialogue,
                    analysis_version,
                    analyzed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    smartphone_user = excluded.smartphone_user,
                    has_bank_account = excluded.has_bank_account,
                    per_household_earnings = excluded.per_household_earnings,
                    participant_personal_monthly_income = excluded.participant_personal_monthly_income,
                    total_household_monthly_income = excluded.total_household_monthly_income,
                    income_range = excluded.income_range,
                    borrowing_history = excluded.borrowing_history,
                    repayment_preference = excluded.repayment_preference,
                    loan_interest = excluded.loan_interest,
                    segmented_dialogue = excluded.segmented_dialogue,
                    analysis_version = excluded.analysis_version,
                    analyzed_at = excluded.analyzed_at
                """,
                (
                    interview_id,
                    self._bool_to_int(structured_response.get("smartphone_user")),
                    self._bool_to_int(structured_response.get("has_bank_account")),
                    structured_response.get("per_household_earnings"),
                    structured_response.get("participant_personal_monthly_income"),
                    structured_response.get("total_household_monthly_income"),
                    structured_response.get("income_range"),
                    structured_response.get("borrowing_history"),
                    structured_response.get("repayment_preference"),
                    structured_response.get("loan_interest"),
                    json.dumps(structured_response.get("segmented_dialogue") or [], ensure_ascii=False),
                    analysis_version,
                    analyzed_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO insights (interview_id, summary, key_quotes, persona, confidence_score, analysis_version, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    summary = excluded.summary,
                    key_quotes = excluded.key_quotes,
                    persona = excluded.persona,
                    confidence_score = excluded.confidence_score,
                    analysis_version = excluded.analysis_version,
                    analyzed_at = excluded.analyzed_at
                """,
                (
                    interview_id,
                    insight["summary"],
                    json.dumps(insight["key_quotes"]),
                    insight["persona"],
                    insight["confidence_score"],
                    analysis_version,
                    analyzed_at,
                ),
            )
            connection.execute(
                """
                UPDATE interviews
                SET analysis_schema_version = ?, persona_ruleset_version = ?, analysis_version = ?, insights_json = ?
                WHERE id = ?
                """,
                (
                    ANALYSIS_SCHEMA_VERSION,
                    PERSONA_RULESET_VERSION,
                    analysis_version,
                    json.dumps(structured_response, ensure_ascii=False),
                    interview_id,
                ),
            )
        return self.get_dashboard_interview_detail(interview_id)

    def list_stale_interview_ids(self, *, latest_analysis_version: str | None = None, limit: int = 500) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT interviews.id
                FROM interviews
                LEFT JOIN insights ON insights.interview_id = interviews.id
                LEFT JOIN structured_responses ON structured_responses.interview_id = interviews.id
                WHERE COALESCE(interviews.transcript_text, interviews.transcript) IS NOT NULL
                  AND TRIM(COALESCE(interviews.transcript_text, interviews.transcript)) <> ''
                  AND (
                    interviews.analysis_schema_version IS NULL
                    OR interviews.analysis_schema_version < ?
                    OR interviews.persona_ruleset_version IS NULL
                    OR interviews.persona_ruleset_version < ?
                    OR insights.persona IS NULL
                    OR structured_responses.interview_id IS NULL
                    OR structured_responses.analysis_version IS NULL
                    OR insights.analysis_version IS NULL
                    OR structured_responses.analysis_version <> insights.analysis_version
                    OR (? IS NOT NULL AND (interviews.analysis_version IS NULL OR interviews.analysis_version <> ?))
                  )
                ORDER BY interviews.created_at DESC, interviews.id DESC
                LIMIT ?
                """,
                (ANALYSIS_SCHEMA_VERSION, PERSONA_RULESET_VERSION, latest_analysis_version, latest_analysis_version, limit),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def list_interviews(self, *, status: str | None = None, limit: int = 100) -> list[InterviewListItem]:
        query = """
            SELECT
                interviews.id,
                COALESCE(interviews.filename, '') AS filename,
                COALESCE(interviews.file_path, interviews.audio_url) AS file_path,
                interviews.status,
                interviews.created_at,
                insights.persona,
                insights.confidence_score,
                insights.summary
            FROM interviews
            LEFT JOIN insights ON insights.interview_id = interviews.id
        """
        params: list[object] = []
        if status is not None:
            query += " WHERE interviews.status = ?"
            params.append(self._normalize_status(status))
        query += " ORDER BY interviews.created_at DESC, interviews.id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [InterviewListItem(**dict(row)) for row in rows]

    def list_recent_interviews(self, *, limit: int = 100) -> list[DashboardInterviewRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    interviews.id,
                    COALESCE(interviews.file_path, interviews.audio_url) AS file_path,
                    COALESCE(interviews.transcript_text, interviews.transcript) AS transcript,
                    interviews.status,
                    interviews.latest_stage,
                    COALESCE(interviews.error_message, interviews.last_error) AS error_message,
                    interviews.created_at,
                    interviews.analysis_version AS interview_analysis_version,
                    structured_responses.smartphone_user,
                    structured_responses.has_bank_account,
                    structured_responses.per_household_earnings,
                    structured_responses.participant_personal_monthly_income,
                    structured_responses.total_household_monthly_income,
                    structured_responses.income_range,
                    structured_responses.borrowing_history,
                    structured_responses.repayment_preference,
                    structured_responses.loan_interest,
                    structured_responses.segmented_dialogue,
                    structured_responses.analysis_version AS structured_analysis_version,
                    structured_responses.analyzed_at AS structured_analyzed_at,
                    insights.summary,
                    insights.key_quotes,
                    insights.persona,
                    insights.confidence_score,
                    insights.analysis_version AS insight_analysis_version,
                    insights.analyzed_at AS insight_analyzed_at
                FROM interviews
                LEFT JOIN structured_responses ON structured_responses.interview_id = interviews.id
                LEFT JOIN insights ON insights.interview_id = interviews.id
                ORDER BY interviews.created_at DESC, interviews.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            migrated_ids = {
                str(row["id"])
                for row in rows
                if self._migrate_legacy_html_fragments(connection, row)
            }
            if migrated_ids:
                refreshed_rows = connection.execute(
                    """
                    SELECT
                        interviews.id,
                        COALESCE(interviews.file_path, interviews.audio_url) AS file_path,
                        COALESCE(interviews.transcript_text, interviews.transcript) AS transcript,
                        interviews.status,
                        interviews.latest_stage,
                        COALESCE(interviews.error_message, interviews.last_error) AS error_message,
                        interviews.created_at,
                        interviews.analysis_version AS interview_analysis_version,
                        structured_responses.smartphone_user,
                        structured_responses.has_bank_account,
                        structured_responses.per_household_earnings,
                        structured_responses.participant_personal_monthly_income,
                        structured_responses.total_household_monthly_income,
                        structured_responses.income_range,
                        structured_responses.borrowing_history,
                        structured_responses.repayment_preference,
                        structured_responses.loan_interest,
                        structured_responses.segmented_dialogue,
                        structured_responses.analysis_version AS structured_analysis_version,
                        structured_responses.analyzed_at AS structured_analyzed_at,
                        insights.summary,
                        insights.key_quotes,
                        insights.persona,
                        insights.confidence_score,
                        insights.analysis_version AS insight_analysis_version,
                        insights.analyzed_at AS insight_analyzed_at
                    FROM interviews
                    LEFT JOIN structured_responses ON structured_responses.interview_id = interviews.id
                    LEFT JOIN insights ON insights.interview_id = interviews.id
                    WHERE interviews.id IN ({placeholders})
                    """.format(placeholders=",".join("?" for _ in migrated_ids)),
                    tuple(migrated_ids),
                ).fetchall()
                refreshed_lookup = {str(row["id"]): row for row in refreshed_rows}
                rows = [refreshed_lookup.get(str(row["id"]), row) for row in rows]
        return [self._dashboard_record_from_row(row) for row in rows]

    def list_failed_or_malformed_interview_ids(self, *, limit: int = 500) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, status, COALESCE(error_message, last_error) AS last_error, COALESCE(transcript_text, transcript) AS transcript
                FROM interviews
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            str(row["id"])
            for row in rows
            if self._infer_transcript_quality(
                status=row["status"],
                transcript=row["transcript"],
                last_error=row["last_error"],
            )
            in {"failed", "malformed"}
        ]

    def list_stale_corrupted_interview_ids(self, *, latest_analysis_version: str | None = None, limit: int = 500) -> list[str]:
        stale_ids = set(self.list_stale_interview_ids(latest_analysis_version=latest_analysis_version, limit=limit))
        if not stale_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, status, COALESCE(error_message, last_error) AS last_error, COALESCE(transcript_text, transcript) AS transcript
                FROM interviews
                WHERE id IN ({placeholders})
                ORDER BY created_at DESC, id DESC
                """.format(placeholders=",".join("?" for _ in stale_ids)),
                tuple(stale_ids),
            ).fetchall()
        return [
            str(row["id"])
            for row in rows
            if self._infer_transcript_quality(
                status=row["status"],
                transcript=row["transcript"],
                last_error=row["last_error"],
            )
            in {"failed", "malformed"}
        ]

    def get_interview_detail(self, interview_id: str) -> InterviewDetail:
        interview = self.get_interview(interview_id)
        with self._connect() as connection:
            structured_row = connection.execute(
                """
                SELECT
                    interview_id,
                    smartphone_user,
                    has_bank_account,
                    per_household_earnings,
                    participant_personal_monthly_income,
                    total_household_monthly_income,
                    income_range,
                    borrowing_history,
                    repayment_preference,
                    loan_interest,
                    segmented_dialogue,
                    analysis_version,
                    analyzed_at
                FROM structured_responses
                WHERE interview_id = ?
                """,
                (interview_id,),
            ).fetchone()
            insight_row = connection.execute(
                """
                SELECT interview_id, summary, key_quotes, persona, confidence_score, analysis_version, analyzed_at
                FROM insights
                WHERE interview_id = ?
                """,
                (interview_id,),
            ).fetchone()
        structured_response = (
            StructuredResponseRecord(
                interview_id=structured_row["interview_id"],
                smartphone_user=self._int_to_bool(structured_row["smartphone_user"]),
                has_bank_account=self._int_to_bool(structured_row["has_bank_account"]),
                per_household_earnings=structured_row["per_household_earnings"],
                participant_personal_monthly_income=structured_row["participant_personal_monthly_income"],
                total_household_monthly_income=structured_row["total_household_monthly_income"],
                income_range=structured_row["income_range"],
                borrowing_history=structured_row["borrowing_history"],
                repayment_preference=structured_row["repayment_preference"],
                loan_interest=structured_row["loan_interest"],
                segmented_dialogue=json.loads(structured_row["segmented_dialogue"]) if structured_row["segmented_dialogue"] else [],
                analysis_version=structured_row["analysis_version"],
                analyzed_at=structured_row["analyzed_at"],
            )
            if structured_row is not None
            else None
        )
        insight = (
            InsightRecord(
                interview_id=insight_row["interview_id"],
                summary=insight_row["summary"],
                key_quotes=json.loads(insight_row["key_quotes"]),
                persona=insight_row["persona"],
                confidence_score=insight_row["confidence_score"],
                analysis_version=insight_row["analysis_version"],
                analyzed_at=insight_row["analyzed_at"],
            )
            if insight_row is not None
            else None
        )
        return InterviewDetail(
            interview=interview,
            structured_response=structured_response,
            insight=insight,
        )

    def get_dashboard_interview_detail(self, interview_id: str) -> DashboardInterviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    interviews.id,
                    COALESCE(interviews.file_path, interviews.audio_url) AS file_path,
                    COALESCE(interviews.transcript_text, interviews.transcript) AS transcript,
                    interviews.status,
                    interviews.latest_stage,
                    COALESCE(interviews.error_message, interviews.last_error) AS error_message,
                    interviews.created_at,
                    interviews.analysis_version AS interview_analysis_version,
                    structured_responses.smartphone_user,
                    structured_responses.has_bank_account,
                    structured_responses.per_household_earnings,
                    structured_responses.participant_personal_monthly_income,
                    structured_responses.total_household_monthly_income,
                    structured_responses.income_range,
                    structured_responses.borrowing_history,
                    structured_responses.repayment_preference,
                    structured_responses.loan_interest,
                    structured_responses.segmented_dialogue,
                    structured_responses.analysis_version AS structured_analysis_version,
                    structured_responses.analyzed_at AS structured_analyzed_at,
                    insights.summary,
                    insights.key_quotes,
                    insights.persona,
                    insights.confidence_score,
                    insights.analysis_version AS insight_analysis_version,
                    insights.analyzed_at AS insight_analyzed_at
                FROM interviews
                LEFT JOIN structured_responses ON structured_responses.interview_id = interviews.id
                LEFT JOIN insights ON insights.interview_id = interviews.id
                WHERE interviews.id = ?
                """,
                (interview_id,),
            ).fetchone()
            if row is not None and self._migrate_legacy_html_fragments(connection, row):
                row = connection.execute(
                    """
                    SELECT
                        interviews.id,
                        COALESCE(interviews.file_path, interviews.audio_url) AS file_path,
                        COALESCE(interviews.transcript_text, interviews.transcript) AS transcript,
                        interviews.status,
                        interviews.latest_stage,
                        COALESCE(interviews.error_message, interviews.last_error) AS error_message,
                        interviews.created_at,
                        interviews.analysis_version AS interview_analysis_version,
                        structured_responses.smartphone_user,
                        structured_responses.has_bank_account,
                        structured_responses.per_household_earnings,
                        structured_responses.participant_personal_monthly_income,
                        structured_responses.total_household_monthly_income,
                        structured_responses.income_range,
                        structured_responses.borrowing_history,
                        structured_responses.repayment_preference,
                        structured_responses.loan_interest,
                        structured_responses.segmented_dialogue,
                        structured_responses.analysis_version AS structured_analysis_version,
                        structured_responses.analyzed_at AS structured_analyzed_at,
                        insights.summary,
                        insights.key_quotes,
                        insights.persona,
                        insights.confidence_score,
                        insights.analysis_version AS insight_analysis_version,
                        insights.analyzed_at AS insight_analyzed_at
                    FROM interviews
                    LEFT JOIN structured_responses ON structured_responses.interview_id = interviews.id
                    LEFT JOIN insights ON insights.interview_id = interviews.id
                    WHERE interviews.id = ?
                    """,
                    (interview_id,),
                ).fetchone()
        if row is None:
            raise KeyError(f"Interview {interview_id} was not found.")
        return self._dashboard_record_from_row(row)

    def get_status_overview(self) -> DashboardStatusOverview:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM interviews
                GROUP BY status
                ORDER BY status ASC
                """
            ).fetchall()
        status_counts = {row["status"]: row["count"] for row in rows}
        return DashboardStatusOverview(
            total_interviews=sum(status_counts.values()),
            status_counts=status_counts,
        )

    def create_job(
        self,
        *,
        batch_id: str,
        interview_id: str,
        filename: str,
        content_type: str,
        payload: bytes,
        max_attempts: int = 3,
    ) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        job_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, batch_id, interview_id, filename, content_type, payload, status, attempts, max_attempts,
                    error_message, cancel_requested, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, NULL, 0, ?, ?)
                """,
                (job_id, batch_id, interview_id, filename, content_type, payload, max_attempts, now, now),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, batch_id, interview_id, filename, content_type, status, attempts, max_attempts, error_message,
                       cancel_requested, created_at, updated_at, started_at, completed_at
                FROM jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Job {job_id} was not found.")
        return self._job_from_row(row)

    def list_jobs(self, *, limit: int = 200, statuses: tuple[str, ...] | None = None) -> list[JobRecord]:
        query = """
            SELECT id, batch_id, interview_id, filename, content_type, status, attempts, max_attempts, error_message,
                   cancel_requested, created_at, updated_at, started_at, completed_at
            FROM jobs
        """
        params: list[object] = []
        if statuses:
            normalized_statuses = [self._normalize_job_status(status) for status in statuses]
            placeholders = ",".join("?" for _ in normalized_statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(normalized_statuses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._job_from_row(row) for row in rows]

    def get_next_pending_job(self) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, batch_id, interview_id, filename, content_type, status, attempts, max_attempts, error_message,
                       cancel_requested, created_at, updated_at, started_at, completed_at
                FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """
            ).fetchone()
        return None if row is None else self._job_from_row(row)

    def get_job_payload(self, job_id: str) -> bytes:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Job {job_id} was not found.")
        return bytes(row["payload"])

    def claim_job(self, job_id: str) -> JobRecord | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, cancel_requested FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            if self._normalize_job_status(row["status"]) != "pending":
                return None
            if bool(row["cancel_requested"]):
                connection.execute(
                    "UPDATE jobs SET status = 'cancelled', updated_at = ?, completed_at = ? WHERE id = ?",
                    (now, now, job_id),
                )
                return self.get_job(job_id)
            connection.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    attempts = attempts + 1,
                    updated_at = ?,
                    started_at = COALESCE(started_at, ?),
                    error_message = NULL
                WHERE id = ?
                """,
                (now, now, job_id),
            )
        return self.get_job(job_id)

    def complete_job(self, job_id: str) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'completed', updated_at = ?, completed_at = ?, error_message = NULL WHERE id = ?",
                (now, now, job_id),
            )
        return self.get_job(job_id)

    def fail_job(self, job_id: str, *, error_message: str) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute("SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Job {job_id} was not found.")
            attempts = int(row["attempts"])
            max_attempts = int(row["max_attempts"])
            next_status = "failed" if attempts >= max_attempts else "pending"
            completed_at = now if next_status == "failed" else None
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, error_message = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (next_status, error_message, now, completed_at, job_id),
            )
        return self.get_job(job_id)

    def cancel_job(self, job_id: str) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Job {job_id} was not found.")
            status = self._normalize_job_status(row["status"])
            if status == "pending":
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'cancelled', cancel_requested = 1, updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (now, now, job_id),
                )
            elif status == "processing":
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (now, job_id),
                )
            return self.get_job(job_id)

    def retry_job(self, job_id: str) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    error_message = NULL,
                    cancel_requested = 0,
                    updated_at = ?,
                    started_at = NULL,
                    completed_at = NULL
                WHERE id = ?
                """,
                (now, job_id),
            )
        return self.get_job(job_id)

    def cancel_batch(self, batch_id: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            pending_rows = connection.execute(
                "SELECT id, status FROM jobs WHERE batch_id = ? AND status IN ('pending', 'processing')",
                (batch_id,),
            ).fetchall()
            for row in pending_rows:
                status = self._normalize_job_status(row["status"])
                if status == "pending":
                    connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'cancelled', cancel_requested = 1, updated_at = ?, completed_at = ?
                        WHERE id = ?
                        """,
                        (now, now, row["id"]),
                    )
                else:
                    connection.execute(
                        "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
        return len(pending_rows)

    def retry_batch(self, batch_id: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    error_message = NULL,
                    cancel_requested = 0,
                    updated_at = ?,
                    started_at = NULL,
                    completed_at = NULL
                WHERE batch_id = ? AND status IN ('failed', 'cancelled')
                """,
                (now, batch_id),
            )
        return int(updated.rowcount)

    def save_result(self, result: PipelineResult) -> PipelineResult:
        self._items[result.file_id] = result
        return result

    def sync_pipeline_result(self, result: PipelineResult, *, file_path: str) -> None:
        interview = self._upsert_interview_from_result(result, file_path=file_path)
        if result.analysis is not None:
            self._upsert_structured_response_from_analysis(interview.id, result.analysis)
        if result.analysis is not None:
            self._upsert_insight_from_result(interview.id, result.analysis, result.persona)

    def get_result(self, file_id: str) -> PipelineResult | None:
        return self._items.get(file_id)

    def list_results(self) -> list[PipelineResult]:
        """Return stored pipeline results in stable insertion order for the dashboard."""
        return list(self._items.values())

    def add_processing_event(
        self,
        *,
        file_id: str,
        stage: str,
        status: str,
        message: str | None = None,
    ) -> ProcessingEventRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO processing_events (file_id, stage, status, message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    str(stage),
                    self._normalize_status(status),
                    message,
                    created_at,
                ),
            )
        return ProcessingEventRecord(
            id=int(cursor.lastrowid),
            file_id=file_id,
            stage=str(stage),
            status=self._normalize_status(status),
            message=message,
            created_at=created_at,
        )

    def list_processing_events(
        self,
        *,
        file_ids: list[str] | None = None,
        limit: int = 200,
    ) -> list[ProcessingEventRecord]:
        query = """
            SELECT id, file_id, stage, status, message, created_at
            FROM processing_events
        """
        params: list[object] = []
        if file_ids:
            placeholders = ",".join("?" for _ in file_ids)
            query += f" WHERE file_id IN ({placeholders})"
            params.extend(file_ids)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [ProcessingEventRecord(**dict(row)) for row in rows]

    @staticmethod
    def _bool_to_int(value: bool | None) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _int_to_bool(value: int | None) -> bool | None:
        if value is None:
            return None
        return bool(value)

    @staticmethod
    def _normalize_job_status(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in _VALID_JOB_STATUS:
            return normalized
        raise ValueError(f"Unsupported job status '{value}'. Expected one of {sorted(_VALID_JOB_STATUS)}.")

    @staticmethod
    def _normalize_database_path(database_path: str) -> str:
        if database_path == ":memory:":
            return database_path
        path = Path(database_path).expanduser()
        return str(path)

    def _upsert_interview_from_result(self, result: PipelineResult, *, file_path: str) -> InterviewRecord:
        transcript_text = result.transcript.text if result.transcript is not None else None
        try:
            return self.update_interview(
                result.file_id,
                file_path=file_path,
                transcript=transcript_text,
                status=result.status.value,
                latest_stage=result.current_stage.value,
                last_error=result.errors[-1] if result.errors else None,
            )
        except KeyError:
            return self.create_interview(
                interview_id=result.file_id,
                file_path=file_path,
                transcript=transcript_text,
                status=result.status.value,
                latest_stage=result.current_stage.value,
                last_error=result.errors[-1] if result.errors else None,
            )

    def _upsert_structured_response_from_analysis(self, interview_id: str, analysis: AnalysisResult) -> None:
        structured = analysis.structured_output
        self.upsert_structured_response(
            interview_id,
            smartphone_user=self._extract_bool(structured, "participant_profile", "smartphone_user"),
            has_bank_account=self._extract_bool(structured, "participant_profile", "has_bank_account"),
            per_household_earnings=self._extract_value(structured, "per_household_earnings"),
            participant_personal_monthly_income=self._extract_value(structured, "participant_personal_monthly_income"),
            total_household_monthly_income=self._extract_value(structured, "total_household_monthly_income"),
            income_range=self._preferred_income_value(structured),
            borrowing_history=self._extract_value(structured, "borrowing_history"),
            repayment_preference=self._extract_value(structured, "repayment_preference"),
            loan_interest=self._extract_value(structured, "loan_interest"),
            segmented_dialogue=self._extract_segmented_dialogue(structured),
        )

    def _upsert_insight_from_result(
        self,
        interview_id: str,
        analysis: AnalysisResult,
        persona: PersonaClassification | None,
    ) -> None:
        persona_name = persona.persona_name if persona is not None else "Unknown"
        confidence_score = analysis.metrics.get("confidence_score")
        if isinstance(confidence_score, int | float):
            normalized_confidence = float(confidence_score)
        else:
            json_attempts = int(analysis.metrics.get("json_attempts", 1))
            normalized_confidence = max(0.0, 1.0 - 0.1 * (json_attempts - 1))
        self.upsert_insight(
            interview_id,
            summary=analysis.summary,
            key_quotes=list(analysis.evidence_quotes),
            persona=persona_name,
            confidence_score=normalized_confidence,
            analysis_version=analysis.metrics.get("analysis_version") if isinstance(analysis.metrics.get("analysis_version"), str) else None,
            analyzed_at=analysis.metrics.get("analyzed_at") if isinstance(analysis.metrics.get("analyzed_at"), str) else None,
        )

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        payload = dict(row)
        return JobRecord(
            id=str(payload["id"]),
            batch_id=str(payload["batch_id"]),
            interview_id=str(payload["interview_id"]),
            filename=str(payload["filename"]),
            content_type=str(payload["content_type"]),
            status=self._normalize_job_status(payload["status"]),
            attempts=int(payload["attempts"]),
            max_attempts=int(payload["max_attempts"]),
            error_message=payload.get("error_message"),
            cancel_requested=bool(payload.get("cancel_requested")),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
        )

    def _dashboard_record_from_row(self, row: sqlite3.Row) -> DashboardInterviewRecord:
        payload = dict(row)
        normalization = self._normalize_dashboard_payload(payload)
        transcript_quality = self._infer_transcript_quality(
            status=payload["status"],
            transcript=normalization["transcript"],
            last_error=payload["error_message"],
        )
        if normalization["data_malformed"] and transcript_quality == "good":
            transcript_quality = "malformed"
        return DashboardInterviewRecord(
            id=payload["id"],
            file_path=payload["file_path"],
            filename=self._filename_from_audio_url(payload["file_path"]),
            transcript=normalization["transcript"],
            status=payload["status"],
            latest_stage=payload["latest_stage"],
            last_error=payload.get("error_message", payload.get("last_error")),
            created_at=payload["created_at"],
            smartphone_user=self._int_to_bool(payload["smartphone_user"]),
            has_bank_account=self._int_to_bool(payload["has_bank_account"]),
            per_household_earnings=payload["per_household_earnings"],
            participant_personal_monthly_income=payload["participant_personal_monthly_income"],
            total_household_monthly_income=payload["total_household_monthly_income"],
            income_range=payload["income_range"],
            borrowing_history=payload["borrowing_history"],
            repayment_preference=payload["repayment_preference"],
            loan_interest=payload["loan_interest"],
            segmented_dialogue=json.loads(payload["segmented_dialogue"]) if payload["segmented_dialogue"] else [],
            summary=normalization["summary"],
            key_quotes=json.loads(payload["key_quotes"]) if payload["key_quotes"] else [],
            persona=payload["persona"],
            confidence_score=payload["confidence_score"],
            transcript_quality=transcript_quality,
            analysis_version=payload.get("interview_analysis_version") or payload.get("insight_analysis_version") or payload.get("structured_analysis_version"),
            analyzed_at=payload.get("insight_analyzed_at") or payload.get("structured_analyzed_at"),
            data_malformed=normalization["data_malformed"],
            data_malformed_details=normalization["data_malformed_details"],
        )

    def _normalize_dashboard_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = payload.get("summary")
        transcript = payload.get("transcript")
        details: list[str] = []
        parsed_summary = self._parse_legacy_html_fragment(summary)
        parsed_transcript = self._parse_legacy_html_fragment(transcript)

        normalized_summary = self._safe_text_value(summary)
        normalized_transcript = self._safe_text_value(transcript)

        if parsed_summary is not None and not normalized_summary:
            normalized_summary = parsed_summary.get("summary") or parsed_summary.get("text")
        if parsed_transcript is not None and not normalized_transcript:
            normalized_transcript = parsed_transcript.get("transcript") or parsed_transcript.get("text")
        if parsed_summary is not None and parsed_summary.get("summary"):
            normalized_summary = parsed_summary["summary"]
        if parsed_transcript is not None and parsed_transcript.get("transcript"):
            normalized_transcript = parsed_transcript["transcript"]

        if self._looks_like_html_blob(summary):
            details.append("insights.summary contained HTML-like content")
        if self._looks_like_html_blob(transcript):
            details.append("interviews.transcript contained HTML-like content")
        if parsed_summary is not None or parsed_transcript is not None:
            details.append("legacy dashboard HTML fragment detected (meta-label/interview-summary)")

        if not normalized_summary:
            normalized_summary = "Analysis pending."
            if summary not in (None, ""):
                details.append("summary could not be normalized; using safe fallback text")
        if not normalized_transcript:
            normalized_transcript = "Transcript unavailable due to malformed stored data."
            if transcript not in (None, ""):
                details.append("transcript could not be normalized; using safe fallback text")

        return {
            "summary": normalized_summary,
            "transcript": normalized_transcript,
            "data_malformed": bool(details),
            "data_malformed_details": details,
        }

    def _migrate_legacy_html_fragments(self, connection: sqlite3.Connection, row: sqlite3.Row) -> bool:
        payload = dict(row)
        summary_fragment = self._parse_legacy_html_fragment(payload.get("summary"))
        transcript_fragment = self._parse_legacy_html_fragment(payload.get("transcript"))
        merged_fragment = self._merge_legacy_fragments(summary_fragment, transcript_fragment)
        if merged_fragment is None:
            return False

        updated = False
        interview_id = str(payload["id"])
        normalized_summary = merged_fragment.get("summary")
        normalized_transcript = merged_fragment.get("transcript")
        if normalized_summary and (self._looks_like_html_blob(payload.get("summary")) or not self._safe_text_value(payload.get("summary"))):
            connection.execute(
                "UPDATE insights SET summary = ? WHERE interview_id = ?",
                (normalized_summary, interview_id),
            )
            updated = True
        if normalized_transcript and (self._looks_like_html_blob(payload.get("transcript")) or not self._safe_text_value(payload.get("transcript"))):
            connection.execute(
                "UPDATE interviews SET transcript_text = ? WHERE id = ?",
                (normalized_transcript, interview_id),
            )
            updated = True

        if merged_fragment.get("borrowing_history") and not payload.get("borrowing_history"):
            connection.execute(
                "UPDATE structured_responses SET borrowing_history = ? WHERE interview_id = ?",
                (merged_fragment["borrowing_history"], interview_id),
            )
            updated = True
        if merged_fragment.get("income_range") and not payload.get("income_range"):
            connection.execute(
                "UPDATE structured_responses SET income_range = ? WHERE interview_id = ?",
                (merged_fragment["income_range"], interview_id),
            )
            updated = True
        return updated

    @staticmethod
    def _looks_like_html_blob(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower()
        if not normalized:
            return False
        return bool(re.search(r"<[^>]+>", normalized))

    @staticmethod
    def _safe_text_value(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if "\x00" in normalized:
            return None
        if re.search(r"<[^>]+>", normalized):
            return None
        return normalized

    def _parse_legacy_html_fragment(self, value: Any) -> dict[str, str] | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized or not self._looks_like_html_blob(normalized):
            return None
        if "meta-label" not in normalized and "interview-summary" not in normalized:
            return None

        summary_match = re.search(r"interview-summary[^>]*>\s*(.*?)\s*</", normalized, flags=re.IGNORECASE | re.DOTALL)
        transcript_match = re.search(
            r"Transcript preview:\s*</strong>\s*(.*?)\s*</",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        labels = re.findall(
            r"meta-label[^>]*>\s*([^<]+?)\s*</span>\s*<span[^>]*meta-value[^>]*>\s*([^<]+?)\s*</span>",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        cleaned_labels = {self._strip_html(label).lower(): self._strip_html(raw_value) for label, raw_value in labels}
        parsed: dict[str, str] = {}
        if summary_match is not None:
            summary_text = self._strip_html(summary_match.group(1))
            if summary_text and "transcript preview" not in summary_text.lower():
                parsed["summary"] = summary_text
        if transcript_match is not None:
            transcript_text = self._strip_html(transcript_match.group(1))
            if transcript_text:
                parsed["transcript"] = transcript_text
        if "borrowing" in cleaned_labels:
            parsed["borrowing_history"] = self._legacy_borrowing_value(cleaned_labels["borrowing"])
        if "income" in cleaned_labels and cleaned_labels["income"]:
            parsed["income_range"] = cleaned_labels["income"]
        text_only = self._strip_html(normalized)
        if text_only:
            parsed["text"] = text_only
        return parsed or None

    @staticmethod
    def _merge_legacy_fragments(
        first: dict[str, str] | None,
        second: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if first is None and second is None:
            return None
        merged: dict[str, str] = {}
        for fragment in (first, second):
            if fragment is None:
                continue
            for key, value in fragment.items():
                if value and key not in merged:
                    merged[key] = value
        return merged

    @staticmethod
    def _strip_html(value: str) -> str:
        no_tags = re.sub(r"<[^>]+>", " ", value)
        normalized = re.sub(r"\s+", " ", no_tags).strip()
        return normalized

    @staticmethod
    def _legacy_borrowing_value(value: str) -> str:
        lowered = value.strip().lower()
        if "non-borrow" in lowered or "no borrowing" in lowered:
            return "has_not_borrowed_recently"
        if "borrow" in lowered:
            return "has_borrowed"
        return "unknown"


    def _preferred_income_value(self, structured: dict[str, object]) -> str | None:
        for field_name, label in (
            ("participant_personal_monthly_income", "Participant monthly income"),
            ("total_household_monthly_income", "Household monthly income"),
            ("per_household_earnings", "Per-household earnings"),
        ):
            value = self._extract_value(structured, field_name)
            if value and value != DEFAULT_UNKNOWN_VALUE:
                return f"{label}: {value}"
        legacy_value = self._extract_value(structured, "income_range")
        if legacy_value and legacy_value != DEFAULT_UNKNOWN_VALUE:
            return legacy_value
        return None

    @staticmethod
    def _extract_value(structured: dict[str, object], field_name: str) -> str | None:
        value = structured.get(field_name)
        if isinstance(value, dict):
            raw = value.get("value")
            return str(raw) if raw is not None else None
        participant_profile = structured.get("participant_profile")
        if isinstance(participant_profile, dict):
            nested = participant_profile.get(field_name)
            if isinstance(nested, dict):
                raw = nested.get("value")
                return str(raw) if raw is not None else None
        persona_signals = structured.get("persona_signals")
        if isinstance(persona_signals, dict):
            nested = persona_signals.get(field_name)
            if isinstance(nested, dict):
                raw = nested.get("value")
                return str(raw) if raw is not None else None
        return None

    def _extract_bool(self, structured: dict[str, object], section: str, field_name: str) -> bool | None:
        parent = structured.get(section)
        if isinstance(parent, dict):
            nested = parent.get(field_name)
            if isinstance(nested, dict):
                value = nested.get("value")
                if isinstance(value, bool):
                    return value

        if section == "participant_profile" and field_name == "smartphone_user":
            return self._extract_bool_from_flat_field(
                structured,
                field_name="smartphone_usage",
                positive="has_smartphone",
                negative="no_smartphone",
            )

        if section == "participant_profile" and field_name == "has_bank_account":
            return self._extract_bool_from_flat_field(
                structured,
                field_name="bank_account_status",
                positive="has_bank_account",
                negative="no_bank_account",
            )

        return None

    @staticmethod
    def _extract_bool_from_flat_field(
        structured: dict[str, object],
        *,
        field_name: str,
        positive: str,
        negative: str,
    ) -> bool | None:
        value = PaydayRepository._extract_value(structured, field_name)
        if value == positive:
            return True
        if value == negative:
            return False
        return None

    @staticmethod
    def _extract_segmented_dialogue(structured: dict[str, object]) -> list[dict[str, Any]]:
        value = structured.get("segmented_dialogue")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _normalize_status(value: str | None) -> str:
        normalized = str(value or "").strip().lower()
        mapped = _LEGACY_STATUS_MAP.get(normalized, normalized)
        if mapped in _VALID_STATUS:
            return mapped
        return "pending"

    @staticmethod
    def _filename_from_file_path(file_path: str) -> str:
        parsed = urlparse(file_path)
        path = parsed.path or file_path
        return Path(path).name

    @staticmethod
    def _filename_from_audio_url(audio_url: str) -> str:
        return PaydayRepository._filename_from_file_path(audio_url)

    @staticmethod
    def _infer_transcript_quality(*, status: str, transcript: str | None, last_error: str | None) -> str:
        normalized_transcript = (transcript or "").strip()
        normalized_error = (last_error or "").strip().lower()
        if status == "failed":
            return "failed"
        if not normalized_transcript:
            return "malformed"
        if len(normalized_transcript.split()) < 5:
            return "malformed"
        if any(token in normalized_error for token in ("transcription", "malformed", "invalid", "schema", "decode")):
            return "malformed"
        return "good"
