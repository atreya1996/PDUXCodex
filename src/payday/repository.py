from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from payday.models import PipelineResult


@dataclass(frozen=True, slots=True)
class InterviewRecord:
    id: str
    audio_url: str
    transcript: str | None
    status: str
    created_at: str


@dataclass(frozen=True, slots=True)
class StructuredResponseRecord:
    interview_id: str
    smartphone_user: bool | None
    has_bank_account: bool | None
    income_range: str | None
    borrowing_history: str | None
    repayment_preference: str | None
    loan_interest: str | None


@dataclass(frozen=True, slots=True)
class InsightRecord:
    interview_id: str
    summary: str
    key_quotes: list[str]
    persona: str
    confidence_score: float


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


class PaydayRepository:
    """Persistence facade for analysis artifacts and dashboard reads."""

    def __init__(self, database_path: str = ":memory:", schema_path: str | None = None) -> None:
        self.database_path = database_path
        self.schema_path = schema_path or str(Path(__file__).resolve().parents[2] / "sql" / "schema.sql")
        self._items: dict[str, PipelineResult] = {}
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return self._connection

    def _initialize(self) -> None:
        schema_sql = Path(self.schema_path).read_text(encoding="utf-8")
        with self._connect() as connection:
            connection.executescript(schema_sql)

    def create_interview(
        self,
        *,
        audio_url: str,
        transcript: str | None = None,
        status: str = "uploaded",
        interview_id: str | None = None,
        created_at: str | None = None,
    ) -> InterviewRecord:
        record = InterviewRecord(
            id=interview_id or str(uuid4()),
            audio_url=audio_url,
            transcript=transcript,
            status=status,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interviews (id, audio_url, transcript, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record.id, record.audio_url, record.transcript, record.status, record.created_at),
            )
        return record

    def update_interview(
        self,
        interview_id: str,
        *,
        audio_url: str | None = None,
        transcript: str | None = None,
        status: str | None = None,
    ) -> InterviewRecord:
        existing = self.get_interview(interview_id)
        updated = InterviewRecord(
            id=existing.id,
            audio_url=audio_url if audio_url is not None else existing.audio_url,
            transcript=transcript if transcript is not None else existing.transcript,
            status=status if status is not None else existing.status,
            created_at=existing.created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE interviews
                SET audio_url = ?, transcript = ?, status = ?
                WHERE id = ?
                """,
                (updated.audio_url, updated.transcript, updated.status, interview_id),
            )
        return updated

    def get_interview(self, interview_id: str) -> InterviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, audio_url, transcript, status, created_at
                FROM interviews
                WHERE id = ?
                """,
                (interview_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Interview {interview_id} was not found.")
        return InterviewRecord(**dict(row))

    def upsert_structured_response(
        self,
        interview_id: str,
        *,
        smartphone_user: bool | None,
        has_bank_account: bool | None,
        income_range: str | None,
        borrowing_history: str | None,
        repayment_preference: str | None,
        loan_interest: str | None,
    ) -> StructuredResponseRecord:
        record = StructuredResponseRecord(
            interview_id=interview_id,
            smartphone_user=smartphone_user,
            has_bank_account=has_bank_account,
            income_range=income_range,
            borrowing_history=borrowing_history,
            repayment_preference=repayment_preference,
            loan_interest=loan_interest,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO structured_responses (
                    interview_id,
                    smartphone_user,
                    has_bank_account,
                    income_range,
                    borrowing_history,
                    repayment_preference,
                    loan_interest
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    smartphone_user = excluded.smartphone_user,
                    has_bank_account = excluded.has_bank_account,
                    income_range = excluded.income_range,
                    borrowing_history = excluded.borrowing_history,
                    repayment_preference = excluded.repayment_preference,
                    loan_interest = excluded.loan_interest
                """,
                (
                    record.interview_id,
                    self._bool_to_int(record.smartphone_user),
                    self._bool_to_int(record.has_bank_account),
                    record.income_range,
                    record.borrowing_history,
                    record.repayment_preference,
                    record.loan_interest,
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
    ) -> InsightRecord:
        record = InsightRecord(
            interview_id=interview_id,
            summary=summary,
            key_quotes=key_quotes,
            persona=persona,
            confidence_score=confidence_score,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO insights (interview_id, summary, key_quotes, persona, confidence_score)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    summary = excluded.summary,
                    key_quotes = excluded.key_quotes,
                    persona = excluded.persona,
                    confidence_score = excluded.confidence_score
                """,
                (
                    record.interview_id,
                    record.summary,
                    json.dumps(record.key_quotes),
                    record.persona,
                    record.confidence_score,
                ),
            )
        return record

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
        query += " ORDER BY interviews.created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [InterviewListItem(**dict(row)) for row in rows]

    def get_interview_detail(self, interview_id: str) -> InterviewDetail:
        interview = self.get_interview(interview_id)
        with self._connect() as connection:
            structured_row = connection.execute(
                """
                SELECT
                    interview_id,
                    smartphone_user,
                    has_bank_account,
                    income_range,
                    borrowing_history,
                    repayment_preference,
                    loan_interest
                FROM structured_responses
                WHERE interview_id = ?
                """,
                (interview_id,),
            ).fetchone()
            insight_row = connection.execute(
                """
                SELECT interview_id, summary, key_quotes, persona, confidence_score
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
                income_range=structured_row["income_range"],
                borrowing_history=structured_row["borrowing_history"],
                repayment_preference=structured_row["repayment_preference"],
                loan_interest=structured_row["loan_interest"],
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
            )
            if insight_row is not None
            else None
        )
        return InterviewDetail(
            interview=interview,
            structured_response=structured_response,
            insight=insight,
        )

    def save_result(self, result: PipelineResult) -> PipelineResult:
        self._items[result.file_id] = result
        return result

    def get_result(self, file_id: str) -> PipelineResult | None:
        return self._items.get(file_id)

    def list_results(self) -> list[PipelineResult]:
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
