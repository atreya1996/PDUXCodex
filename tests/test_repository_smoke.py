from __future__ import annotations

from payday.repository import PaydayRepository


def test_repository_constructor_smoke_with_in_memory_db() -> None:
    repository = PaydayRepository(database_path=":memory:")

    assert repository.database_path == ":memory:"


def test_create_interview_persists_canonical_fields_to_sqlite() -> None:
    repository = PaydayRepository(database_path=":memory:")

    interview = repository.create_interview(
        file_path="audio/interview-123/worker-voice.wav",
        transcript="I borrow from my employer in emergencies.",
        status="processing",
        latest_stage="transcription",
        last_error=None,
    )

    row = repository._connect().execute(
        """
        SELECT id, audio_url, filename, file_path, transcript, transcript_text, status, latest_stage, last_error, error_message
        FROM interviews
        WHERE id = ?
        """,
        (interview.id,),
    ).fetchone()

    assert row is not None
    assert row["id"] == interview.id
    assert row["audio_url"] == "audio/interview-123/worker-voice.wav"
    assert row["filename"] == "worker-voice.wav"
    assert row["file_path"] == "audio/interview-123/worker-voice.wav"
    assert row["transcript"] == "I borrow from my employer in emergencies."
    assert row["transcript_text"] == "I borrow from my employer in emergencies."
    assert row["status"] == "processing"
    assert row["latest_stage"] == "transcription"
    assert row["last_error"] is None
    assert row["error_message"] is None


def test_processing_events_round_trip_for_file() -> None:
    repository = PaydayRepository(database_path=":memory:")
    interview = repository.create_interview(
        file_path="audio/interview-456/events.wav",
        status="pending",
    )

    repository.add_processing_event(
        file_id=interview.id,
        stage="upload",
        status="processing",
        message="upload accepted",
    )
    repository.add_processing_event(
        file_id=interview.id,
        stage="analysis",
        status="failed",
        message="analysis failed",
    )

    events = repository.list_processing_events(file_ids=[interview.id], limit=10)

    assert len(events) == 2
    assert events[0].stage == "analysis"
    assert events[0].status == "failed"
    assert events[0].message == "analysis failed"
    assert events[1].stage == "upload"
    assert events[1].status == "processing"


def test_create_job_returns_inserted_job_record() -> None:
    repository = PaydayRepository(database_path=":memory:")
    interview = repository.create_interview(
        file_path="audio/interview-789/queue.wav",
        status="pending",
    )

    job = repository.create_job(
        batch_id="batch-123",
        interview_id=interview.id,
        filename="queue.wav",
        content_type="audio/wav",
        payload=b"payload",
    )

    assert job.id
    assert job.batch_id == "batch-123"
    assert job.interview_id == interview.id
    assert job.filename == "queue.wav"
    assert job.status == "pending"
