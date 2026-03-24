from __future__ import annotations

import json
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


@dataclass(frozen=True, slots=True)
class InterviewRecord:
    id: str
    audio_url: str
    transcript: str | None
    status: str
    latest_stage: str
    last_error: str | None
    created_at: str
    analysis_version: str | None = None


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
    audio_url: str
    status: str
    created_at: str
    persona: str | None
    confidence_score: float | None
    summary: str | None


@dataclass(frozen=True, slots=True)
class InterviewDetail:
    interview: InterviewRecord
    structured_response: StructuredResponseRecord | None
    insight: InsightRecord | None


@dataclass(frozen=True, slots=True)
class DashboardInterviewRecord:
    id: str
    audio_url: str
    filename: str
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


@dataclass(frozen=True, slots=True)
class DashboardStatusOverview:
    total_interviews: int
    status_counts: dict[str, int]


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
            self._ensure_structured_response_columns(connection)
            self._ensure_insight_columns(connection)

    def _ensure_interview_columns(self, connection: sqlite3.Connection) -> None:
        interview_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(interviews)").fetchall()
        }
        if "latest_stage" not in interview_columns:
            connection.execute(
                "ALTER TABLE interviews ADD COLUMN latest_stage TEXT NOT NULL DEFAULT 'upload'"
            )
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

    def create_interview(
        self,
        *,
        audio_url: str,
        transcript: str | None = None,
        status: str = "uploaded",
        latest_stage: str = "upload",
        last_error: str | None = None,
        interview_id: str | None = None,
        created_at: str | None = None,
    ) -> InterviewRecord:
        record = InterviewRecord(
            id=interview_id or str(uuid4()),
            audio_url=audio_url,
            transcript=transcript,
            status=status,
            latest_stage=latest_stage,
            last_error=last_error,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interviews (
                    id, audio_url, transcript, transcript_text, status, latest_stage, last_error, error_message, created_at, analysis_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.audio_url,
                    record.transcript,
                    record.transcript,
                    record.status,
                    record.latest_stage,
                    record.last_error,
                    record.last_error,
                    record.created_at,
                    record.analysis_version,
                ),
            )
        return record

    def update_interview(
        self,
        interview_id: str,
        *,
        audio_url: str | None = None,
        transcript: str | None = None,
        status: str | None = None,
        latest_stage: str | None = None,
        last_error: str | object = _UNSET,
    ) -> InterviewRecord:
        existing = self.get_interview(interview_id)
        updated = InterviewRecord(
            id=existing.id,
            audio_url=audio_url if audio_url is not None else existing.audio_url,
            transcript=transcript if transcript is not None else existing.transcript,
            status=status if status is not None else existing.status,
            latest_stage=latest_stage if latest_stage is not None else existing.latest_stage,
            last_error=existing.last_error if last_error is _UNSET else last_error,
            created_at=existing.created_at,
            analysis_version=existing.analysis_version,
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE interviews
                SET audio_url = ?,
                    transcript = ?,
                    transcript_text = ?,
                    status = ?,
                    latest_stage = ?,
                    last_error = ?,
                    error_message = ?,
                    analysis_version = ?
                WHERE id = ?
                """,
                (
                    updated.audio_url,
                    updated.transcript,
                    updated.transcript,
                    updated.status,
                    updated.latest_stage,
                    updated.last_error,
                    updated.last_error,
                    updated.analysis_version,
                    interview_id,
                ),
            )
        return updated

    def get_interview(self, interview_id: str) -> InterviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, audio_url, COALESCE(transcript_text, transcript) AS transcript, status, latest_stage, COALESCE(error_message, last_error) AS last_error, created_at, analysis_version
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
                SET analysis_schema_version = ?, analysis_version = ?
                WHERE id = ?
                """,
                (ANALYSIS_SCHEMA_VERSION, analysis_version, interview_id),
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
                SET transcript = ?, transcript_text = ?, status = ?, latest_stage = ?, last_error = ?, error_message = ?
                WHERE id = ?
                """,
                (transcript, transcript, status, latest_stage, last_error, last_error, interview_id),
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
                SET analysis_schema_version = ?, persona_ruleset_version = ?, analysis_version = ?
                WHERE id = ?
                """,
                (ANALYSIS_SCHEMA_VERSION, PERSONA_RULESET_VERSION, analysis_version, interview_id),
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
                interviews.audio_url,
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
            params.append(status)
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
                    interviews.audio_url,
                    COALESCE(interviews.transcript_text, interviews.transcript) AS transcript,
                    interviews.status,
                    interviews.latest_stage,
                    COALESCE(interviews.error_message, interviews.last_error) AS last_error,
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
                    interviews.audio_url,
                    COALESCE(interviews.transcript_text, interviews.transcript) AS transcript,
                    interviews.status,
                    interviews.latest_stage,
                    COALESCE(interviews.error_message, interviews.last_error) AS last_error,
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

    def save_result(self, result: PipelineResult) -> PipelineResult:
        self._items[result.file_id] = result
        return result

    def sync_pipeline_result(self, result: PipelineResult, *, audio_url: str) -> None:
        interview = self._upsert_interview_from_result(result, audio_url=audio_url)
        if result.analysis is not None:
            self._upsert_structured_response_from_analysis(interview.id, result.analysis)
        if result.analysis is not None:
            self._upsert_insight_from_result(interview.id, result.analysis, result.persona)

    def get_result(self, file_id: str) -> PipelineResult | None:
        return self._items.get(file_id)

    def list_results(self) -> list[PipelineResult]:
        """Return stored pipeline results in stable insertion order for the dashboard."""
        return list(self._items.values())

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
    def _normalize_database_path(database_path: str) -> str:
        if database_path == ":memory:":
            return database_path
        path = Path(database_path).expanduser()
        return str(path)

    def _upsert_interview_from_result(self, result: PipelineResult, *, audio_url: str) -> InterviewRecord:
        transcript_text = result.transcript.text if result.transcript is not None else None
        try:
            return self.update_interview(
                result.file_id,
                audio_url=audio_url,
                transcript=transcript_text,
                status=result.status.value,
                latest_stage=result.current_stage.value,
                last_error=result.errors[-1] if result.errors else None,
            )
        except KeyError:
            return self.create_interview(
                interview_id=result.file_id,
                audio_url=audio_url,
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

    def _dashboard_record_from_row(self, row: sqlite3.Row) -> DashboardInterviewRecord:
        payload = dict(row)
        transcript_quality = self._infer_transcript_quality(
            status=payload["status"],
            transcript=payload["transcript"],
            last_error=payload["last_error"],
        )
        return DashboardInterviewRecord(
            id=payload["id"],
            audio_url=payload["audio_url"],
            filename=self._filename_from_audio_url(payload["audio_url"]),
            transcript=payload["transcript"],
            status=payload["status"],
            latest_stage=payload["latest_stage"],
            last_error=payload["last_error"],
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
            summary=payload["summary"],
            key_quotes=json.loads(payload["key_quotes"]) if payload["key_quotes"] else [],
            persona=payload["persona"],
            confidence_score=payload["confidence_score"],
            transcript_quality=transcript_quality,
            analysis_version=payload.get("interview_analysis_version") or payload.get("insight_analysis_version") or payload.get("structured_analysis_version"),
            analyzed_at=payload.get("insight_analyzed_at") or payload.get("structured_analyzed_at"),
        )


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
    def _filename_from_audio_url(audio_url: str) -> str:
        parsed = urlparse(audio_url)
        path = parsed.path or audio_url
        return Path(path).name

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
