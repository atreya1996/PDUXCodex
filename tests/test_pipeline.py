from __future__ import annotations

import sqlite3
import types

import pytest

from payday.analysis import AnalysisService, OpenAIAnalysisAdapter
from payday.config import DatabaseSettings, FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.models import AnalysisResult, BatchUploadItem, PipelineResult, PipelineStage, ProcessingStatus, Transcript, UploadedAsset
from payday.personas import PersonaService
from payday.repository import PaydayRepository
from payday.service import PaydayAppService
from payday.storage import StorageService
from payday.transcription import (
    OPENAI_TRANSCRIPTION_SAFE_FILE_LIMIT_BYTES,
    OpenAITranscriptionAdapter,
    TranscriptionService,
    describe_transcription_file_size_limit,
)
from payday.upload import SUPPORTED_UPLOAD_EXTENSIONS, UploadService, UploadValidationError


class FlakyTranscriptionService(TranscriptionService):
    def __init__(self, settings: TranscriptionSettings) -> None:
        super().__init__(settings)
        self.calls: dict[str, int] = {}

    def transcribe(self, asset: UploadedAsset, sample_mode: bool = False) -> Transcript:
        self.calls.setdefault(asset.file_id, 0)
        self.calls[asset.file_id] += 1
        if asset.filename == "retry.wav" and self.calls[asset.file_id] == 1:
            raise RuntimeError("temporary transcription issue")
        if asset.filename == "fail.wav":
            raise RuntimeError("hard transcription failure")
        return super().transcribe(asset, sample_mode=sample_mode)


class StaticTranscriptionService(TranscriptionService):
    def __init__(self, settings: TranscriptionSettings) -> None:
        super().__init__(settings)
        self.calls: list[str] = []

    def transcribe(self, asset: UploadedAsset, sample_mode: bool = False) -> Transcript:
        del sample_mode
        self.calls.append(asset.filename)
        return Transcript(
            text=(
                "I use WhatsApp every day. My bank account is active. "
                "I borrow from neighbors when money is short."
            ),
            provider="openai",
            model=self.settings.model,
            metadata={"file_id": asset.file_id, "size_bytes": asset.size_bytes},
        )


class ErroringTranscriptionTransport:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def __call__(self, *, asset: UploadedAsset, model: str, api_key: str) -> dict[str, object]:
        del asset, model, api_key
        self.calls += 1
        return {"error": {"message": self.message}}


class StaticAnalysisTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def __call__(self, *, prompt: str, model: str, api_key: str) -> object:
        del prompt, model, api_key
        self.calls += 1
        return self.response


class RecordingBucket:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []

    def upload(self, path: str, payload: bytes, file_options: dict[str, object] | None = None) -> None:
        self.upload_calls.append(
            {
                "path": path,
                "payload": payload,
                "file_options": file_options or {},
            }
        )


class RecordingSupabaseStorageClient:
    def __init__(self) -> None:
        self.bucket_names: list[str] = []
        self.bucket = RecordingBucket()

    def from_(self, bucket_name: str) -> RecordingBucket:
        self.bucket_names.append(bucket_name)
        return self.bucket


VALID_ANALYSIS_JSON = """
{
  "smartphone_usage": {"value": "has_smartphone", "status": "observed", "evidence_quotes": ["I use WhatsApp every day"], "notes": "Directly stated."},
  "bank_account_status": {"value": "has_bank_account", "status": "observed", "evidence_quotes": ["My bank account is active"], "notes": "Directly stated."},
  "income_range": {"value": "₹12,000", "status": "observed", "evidence_quotes": ["I earn ₹12,000 per month"], "notes": "Directly stated."},
  "borrowing_history": {"value": "has_borrowed", "status": "observed", "evidence_quotes": ["I borrow from neighbors sometimes"], "notes": "Directly stated."},
  "repayment_preference": {"value": "monthly", "status": "observed", "evidence_quotes": ["I repay monthly after salary"], "notes": "Directly stated."},
  "loan_interest": {"value": "fearful_or_uncertain", "status": "observed", "evidence_quotes": ["I am worried about scams"], "notes": "Trust barrier present."},
  "summary": {"value": "The participant uses WhatsApp, has a bank account, earns ₹12,000, and worries about scams.", "status": "observed", "evidence_quotes": ["I use WhatsApp every day", "I am worried about scams"], "notes": "Grounded in transcript."},
  "key_quotes": ["I use WhatsApp every day", "I am worried about scams"],
  "confidence_signals": {"observed_evidence": ["trust barrier mentioned"], "missing_or_unknown": []}
}
""".strip()


def build_settings(sqlite_path: str = ":memory:", *, sample_mode: bool = True) -> Settings:
    return Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=sqlite_path),
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(),
        features=FeatureFlags(use_sample_mode=sample_mode),
    )


def build_transcript(text: str = "test transcript") -> Transcript:
    return Transcript(text=text, provider="test", model="test")


def build_analysis(structured_output: dict) -> AnalysisResult:
    return AnalysisResult(summary="summary", structured_output=structured_output)


def test_transcription_service_sample_mode_preserves_demo_behavior() -> None:
    service = TranscriptionService(TranscriptionSettings())
    asset = UploadedAsset(
        filename="demo.txt",
        content_type="text/plain",
        size_bytes=11,
        raw_bytes=b"hello world",
        file_id="file-123",
    )

    transcript = service.transcribe(asset, sample_mode=True)

    assert transcript.text == "hello world"
    assert transcript.provider == "openai"
    assert transcript.model == "gpt-4o-mini-transcribe"
    assert transcript.metadata == {
        "sample_mode": True,
        "filename": "demo.txt",
        "file_id": "file-123",
        "content_type": "text/plain",
        "size_bytes": 11,
        "source": "demo.txt",
    }


def test_transcription_service_non_sample_mode_uses_openai_client_and_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai_module(monkeypatch)
    transcriptions = StubOpenAITranscriptions(response=StubTranscriptionResponse("Real transcript", language="hi", duration=8.25))
    client = StubOpenAIClient(transcriptions)
    settings = TranscriptionSettings(api_key="test-key", timeout_seconds=42.0)
    service = TranscriptionService(settings, client=client)
    asset = UploadedAsset(
        filename="call.wav",
        content_type="audio/wav",
        size_bytes=4,
        raw_bytes=b"RIFF",
        file_id="file-456",
    )

    transcript = service.transcribe(asset, sample_mode=False)

    assert len(transcriptions.calls) == 1
    call = transcriptions.calls[0]
    assert call["model"] == "gpt-4o-mini-transcribe"
    assert call["file"] == ("call.wav", b"RIFF", "audio/wav")
    assert transcript.text == "Real transcript"
    assert transcript.provider == "openai"
    assert transcript.model == "gpt-4o-mini-transcribe"
    assert transcript.metadata == {
        "sample_mode": False,
        "filename": "call.wav",
        "file_id": "file-456",
        "content_type": "audio/wav",
        "size_bytes": 4,
        "source": "call.wav",
        "language": "hi",
        "duration_seconds": 8.25,
    }


def test_transcription_service_non_sample_mode_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai_module(monkeypatch)
    service = TranscriptionService(TranscriptionSettings(api_key=""), client=StubOpenAIClient(StubOpenAITranscriptions()))
    asset = UploadedAsset(filename="call.wav", content_type="audio/wav", size_bytes=4, raw_bytes=b"RIFF")

    with pytest.raises(ValueError, match="TRANSCRIPTION_API_KEY"):
        service.transcribe(asset, sample_mode=False)


def test_transcription_service_timeout_propagates_for_pipeline_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai_module(monkeypatch)
    service = TranscriptionService(
        TranscriptionSettings(api_key="test-key", timeout_seconds=7.0),
        client=StubOpenAIClient(StubOpenAITranscriptions(error=FakeOpenAITimeoutError("request timed out"))),
    )
    asset = UploadedAsset(filename="call.wav", content_type="audio/wav", size_bytes=4, raw_bytes=b"RIFF")

    with pytest.raises(TimeoutError, match="timed out"):
        service.transcribe(asset, sample_mode=False)


def test_upload_service_normalizes_supported_openai_formats() -> None:
    service = UploadService()

    mp4_asset = service.create_asset("interview.mp4", "application/octet-stream", b"mp4-bytes")
    mpeg_asset = service.create_asset("interview.mpeg", "", b"mpeg-bytes")
    webm_asset = service.create_asset("interview.webm", "audio/webm", b"webm-bytes")

    assert mp4_asset.content_type == "video/mp4"
    assert mpeg_asset.content_type == "audio/mpeg"
    assert webm_asset.content_type == "audio/webm"
    assert SUPPORTED_UPLOAD_EXTENSIONS == ("mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "aac")


def test_upload_service_rejects_unsupported_extensions() -> None:
    service = UploadService()

    with pytest.raises(UploadValidationError, match="Unsupported upload format"):
        service.create_asset("notes.txt", "text/plain", b"hello")


def test_pipeline_accepts_mp4_and_mpeg_family_uploads() -> None:
    service = PaydayAppService(build_settings())

    mp4_result = service.process_upload(
        "interview.mp4",
        "application/octet-stream",
        b"I use WhatsApp and my bank account is active.",
    )
    mpga_result = service.process_upload(
        "interview.mpga",
        "",
        b"I have a smartphone and my bank account is active.",
    )

    assert mp4_result.status is ProcessingStatus.COMPLETED
    assert mp4_result.asset is not None
    assert mp4_result.asset.content_type == "video/mp4"
    assert mpga_result.status is ProcessingStatus.COMPLETED
    assert mpga_result.asset is not None
    assert mpga_result.asset.content_type == "audio/mpeg"


def test_pipeline_process_upload_returns_completed_result() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload(
        "demo.wav",
        "audio/wav",
        b"I use WhatsApp and my bank account is active. I earn \\xe2\\x82\\xb912,000 per month.",
    )

    assert result.asset is not None
    assert result.asset.filename == "demo.wav"
    assert result.persisted is True
    assert result.analysis is not None
    assert result.analysis.metrics["word_count"] > 0
    assert result.status is ProcessingStatus.COMPLETED
    assert result.current_stage is PipelineStage.STORAGE
    assert service.list_results()


def test_pipeline_process_upload_persists_interview_structured_response_and_insight_rows(tmp_path) -> None:
    database_path = tmp_path / "payday.sqlite3"
    repository = PaydayRepository(database_path=str(database_path))
    service = PaydayAppService(build_settings(), repository=repository)

    result = service.process_upload(
        "demo.wav",
        "audio/wav",
        (
            b"I use WhatsApp every day. My bank account is active. I earn \xe2\x82\xb912,000 per month. "
            b"I borrow from neighbors sometimes. I repay monthly after salary. I am worried about scams."
        ),
    )

    assert result.status is ProcessingStatus.COMPLETED

    with sqlite3.connect(database_path) as connection:
        interview_row = connection.execute(
            "SELECT id, audio_url, transcript, status, latest_stage, last_error FROM interviews WHERE id = ?",
            (result.file_id,),
        ).fetchone()
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
            (result.file_id,),
        ).fetchone()
        insight_row = connection.execute(
            "SELECT interview_id, summary, persona, confidence_score FROM insights WHERE interview_id = ?",
            (result.file_id,),
        ).fetchone()

    assert interview_row is not None
    assert interview_row[0] == result.file_id
    assert interview_row[1] == f"audio/{result.file_id}/demo.wav"
    assert "WhatsApp every day" in interview_row[2]
    assert interview_row[3] == ProcessingStatus.COMPLETED.value
    assert interview_row[4] == PipelineStage.STORAGE.value
    assert interview_row[5] is None

    assert structured_row is not None
    assert structured_row[0] == result.file_id
    assert structured_row[1] == 1
    assert structured_row[2] == 1
    assert structured_row[3] == "₹12,000"
    assert structured_row[4] == result.analysis.structured_output["borrowing_history"]["value"]
    assert structured_row[5] == result.analysis.structured_output["repayment_preference"]["value"]
    assert structured_row[6] == result.analysis.structured_output["loan_interest"]["value"]

    assert insight_row is not None
    assert insight_row[0] == result.file_id
    assert "WhatsApp every day" in insight_row[1]
    assert insight_row[2] == "Self-Reliant Non-Borrower"
    assert insight_row[3] == 1.0

    reloaded_repository = PaydayRepository(database_path=str(database_path))
    detail = reloaded_repository.get_interview_detail(result.file_id)

    assert detail.interview.id == result.file_id
    assert detail.interview.status == ProcessingStatus.COMPLETED.value
    assert detail.interview.latest_stage == PipelineStage.STORAGE.value
    assert detail.interview.last_error is None
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.structured_response.has_bank_account is True
    assert detail.structured_response.income_range == "₹12,000"
    assert detail.structured_response.borrowing_history == result.analysis.structured_output["borrowing_history"]["value"]
    assert (
        detail.structured_response.repayment_preference
        == result.analysis.structured_output["repayment_preference"]["value"]
    )
    assert detail.structured_response.loan_interest == result.analysis.structured_output["loan_interest"]["value"]
    assert detail.insight is not None
    assert detail.insight.summary == result.analysis.summary
    assert detail.insight.persona == "Self-Reliant Non-Borrower"
    assert detail.insight.confidence_score == 1.0


def test_pipeline_process_upload_persists_failed_interview_status_to_file_backed_sqlite(tmp_path) -> None:
    database_path = tmp_path / "payday.sqlite3"
    repository = PaydayRepository(database_path=str(database_path))
    pipeline = PaydayAppService(build_settings(), repository=repository).pipeline
    pipeline.transcription_service = FlakyTranscriptionService(build_settings().transcription)

    result = pipeline.process_upload("fail.wav", "audio/wav", b"content does not matter")

    assert result.status is ProcessingStatus.FAILED
    assert result.errors == ["transcription failed after 3 attempts: hard transcription failure"]

    with sqlite3.connect(database_path) as connection:
        interview_row = connection.execute(
            "SELECT id, transcript, status, latest_stage, last_error FROM interviews WHERE id = ?",
            (result.file_id,),
        ).fetchone()
        structured_count = connection.execute(
            "SELECT COUNT(*) FROM structured_responses WHERE interview_id = ?",
            (result.file_id,),
        ).fetchone()
        insight_count = connection.execute(
            "SELECT COUNT(*) FROM insights WHERE interview_id = ?",
            (result.file_id,),
        ).fetchone()

    assert interview_row is not None
    assert interview_row[0] == result.file_id
    assert interview_row[1] is None
    assert interview_row[2] == ProcessingStatus.FAILED.value
    assert interview_row[3] == PipelineStage.TRANSCRIPTION.value
    assert interview_row[4] == "transcription failed after 3 attempts: hard transcription failure"
    assert structured_count == (0,)
    assert insight_count == (0,)

    reloaded_repository = PaydayRepository(database_path=str(database_path))
    detail = reloaded_repository.get_interview_detail(result.file_id)

    assert detail.interview.status == ProcessingStatus.FAILED.value
    assert detail.interview.transcript is None
    assert detail.interview.latest_stage == PipelineStage.TRANSCRIPTION.value
    assert detail.interview.last_error == "transcription failed after 3 attempts: hard transcription failure"
    assert detail.structured_response is None
    assert detail.insight is None


def test_pipeline_marks_external_analysis_provider_errors_as_failed_results() -> None:
    pipeline = PaydayAppService(build_settings()).pipeline
    pipeline.sample_mode = False
    pipeline.transcription_service = TranscriptionService(
        TranscriptionSettings(provider="openai", model="whisper-test", api_key="transcription-key"),
        adapter=OpenAITranscriptionAdapter(
            TranscriptionSettings(provider="openai", model="whisper-test", api_key="transcription-key"),
            transport=lambda *, asset, model, api_key: {"text": "I use WhatsApp every day. My bank account is active."},
        ),
    )
    pipeline.analysis_service = AnalysisService(
        adapter=OpenAIAnalysisAdapter(
            LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
            transport=StaticAnalysisTransport({"output": [{"content": [{"type": "output_text", "text": ""}]}]}),
        ),
        settings=LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
    )

    result = pipeline.process_upload("provider-error.wav", "audio/wav", b"audio payload")

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.ANALYSIS
    assert result.errors == [
        "analysis failed after 3 attempts: OpenAI analysis request succeeded but returned no text output."
    ]


def test_app_service_fails_fast_when_live_transcription_api_key_is_missing() -> None:
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
        transcription=TranscriptionSettings(provider="openai", model="whisper-test", api_key=""),
        features=FeatureFlags(use_sample_mode=False),
    )

    with pytest.raises(ValueError, match="TRANSCRIPTION_API_KEY is required"):
        PaydayAppService(settings)


def test_openai_transcription_adapter_uses_mocked_provider_response() -> None:
    transport = ErroringTranscriptionTransport("provider timeout")
    adapter = OpenAITranscriptionAdapter(
        TranscriptionSettings(provider="openai", model="whisper-test", api_key="secret-key"),
        transport=lambda *, asset, model, api_key: {"text": "I use WhatsApp every day."},
    )

    transcript = adapter.transcribe(
        UploadedAsset(
            filename="sample.wav",
            content_type="audio/wav",
            size_bytes=4,
            raw_bytes=b"demo",
            file_id="file-123",
        )
    )

    assert transcript.text == "I use WhatsApp every day."
    assert transcript.provider == "openai"
    assert transcript.model == "whisper-test"

    failing_adapter = OpenAITranscriptionAdapter(
        TranscriptionSettings(provider="openai", model="whisper-test", api_key="secret-key"),
        transport=transport,
    )

    try:
        failing_adapter.transcribe(
            UploadedAsset(
                filename="sample.wav",
                content_type="audio/wav",
                size_bytes=4,
                raw_bytes=b"demo",
                file_id="file-124",
            )
        )
    except RuntimeError as exc:
        assert "provider timeout" in str(exc)
    else:  # pragma: no cover - defensive guard for the test itself
        raise AssertionError("Expected provider error was not raised.")

    assert transport.calls == 1


def test_repository_list_results_returns_pipeline_results_in_insertion_order() -> None:
    repository = PaydayRepository()
    first = repository.save_result(
        PipelineResult(file_id="file-1", filename="first.wav", status=ProcessingStatus.PENDING)
    )
    repository.save_result(PipelineResult(file_id="file-2", filename="second.wav", status=ProcessingStatus.COMPLETED))
    replacement = repository.save_result(
        PipelineResult(file_id="file-1", filename="first-retry.wav", status=ProcessingStatus.FAILED)
    )

    results = repository.list_results()

    assert [result.file_id for result in results] == ["file-1", "file-2"]
    assert results == [replacement, repository.get_result("file-2")]
    assert results[0] is not first
    assert repository.get_result("file-1") == replacement


def test_repository_crud_supports_interview_related_tables(tmp_path) -> None:
    repository = PaydayRepository(database_path=str(tmp_path / "payday.db"))
    interview = repository.create_interview(audio_url="audio/interview-1/demo.wav", status="uploaded")

    repository.update_interview(interview.id, transcript="A direct quote from the interview.", status="completed")
    repository.upsert_structured_response(
        interview.id,
        smartphone_user=True,
        has_bank_account=True,
        income_range="10k-20k",
        borrowing_history="Borrowed from employer during emergencies.",
        repayment_preference="Weekly",
        loan_interest="Interested if trust is high.",
    )
    repository.upsert_insight(
        interview.id,
        summary="Trust matters more than pricing.",
        key_quotes=["I ask my employer first when money is short."],
        persona="Persona 2",
        confidence_score=0.86,
    )

    listing = repository.list_interviews()
    detail = repository.get_interview_detail(interview.id)
    recent = repository.list_recent_interviews()
    status_overview = repository.get_status_overview()

    assert listing[0].id == interview.id
    assert detail.interview.status == "completed"
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.insight is not None
    assert detail.insight.key_quotes == ["I ask my employer first when money is short."]
    assert recent[0].filename == "demo.wav"
    assert status_overview.total_interviews == 1
    assert status_overview.status_counts == {"completed": 1}


def test_repository_initialization_creates_missing_database_directory(tmp_path) -> None:
    database_path = tmp_path / "nested" / "storage" / "payday.db"

    repository = PaydayRepository(database_path=str(database_path))
    interview = repository.create_interview(audio_url="audio/interview-1/demo.wav")

    assert database_path.exists()
    assert interview.audio_url == "audio/interview-1/demo.wav"


def test_app_service_uses_sqlite_for_durable_dashboard_reads(tmp_path) -> None:
    database_path = str(tmp_path / "payday-dashboard.db")
    first_service = PaydayAppService(build_settings(database_path))

    result = first_service.process_upload(
        "durable.wav",
        "audio/wav",
        b"I use WhatsApp, I have a bank account, and I ask my employer when money is short.",
    )

    assert result.status is ProcessingStatus.COMPLETED
    assert first_service.list_recent_interviews()[0].id == result.file_id
    assert first_service.get_status_overview().status_counts["completed"] == 1

    second_service = PaydayAppService(build_settings(database_path))
    recent = second_service.list_recent_interviews()
    detail = second_service.get_interview_detail(result.file_id)

    assert second_service.list_results() == []
    assert recent[0].id == result.file_id
    assert detail.summary is not None
    assert detail.filename == "durable.wav"


def test_app_service_runtime_summary_sanitizes_provider_configuration() -> None:
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
        transcription=TranscriptionSettings(provider="openai", model="whisper-test", api_key=""),
        features=FeatureFlags(use_sample_mode=True),
    )

    summary = PaydayAppService(settings).runtime_summary()

    assert summary == {
        "sample_mode": True,
        "transcription": {
            "provider": "openai",
            "model": "whisper-test",
            "required_key_present": False,
        },
        "analysis": {
            "provider": "openai",
            "model": "gpt-test",
            "required_key_present": True,
        },
    }


def test_batch_uploads_persist_mixed_statuses_for_durable_dashboard_refresh(tmp_path) -> None:
    database_path = str(tmp_path / "payday-batch-dashboard.db")
    service = PaydayAppService(build_settings(database_path))
    service.pipeline.transcription_service = FlakyTranscriptionService(build_settings().transcription)

    batch_result = service.process_batch_uploads(
        [
            BatchUploadItem(
                filename="success-1.wav",
                content_type="audio/wav",
                data=b"I use WhatsApp and my bank account is active.",
            ),
            BatchUploadItem(
                filename="fail.wav",
                content_type="audio/wav",
                data=b"This file should fail transcription.",
            ),
            BatchUploadItem(
                filename="success-2.wav",
                content_type="audio/wav",
                data=b"I have a smartphone, a bank account, and I avoid loans.",
            ),
        ]
    )

    assert batch_result.completed_count == 2
    assert batch_result.failed_count == 1

    refreshed_service = PaydayAppService(build_settings(database_path))
    refreshed_recent = refreshed_service.list_recent_interviews()
    refreshed_overview = refreshed_service.get_status_overview()

    assert refreshed_service.list_results() == []
    assert {record.filename: record.status for record in refreshed_recent} == {
        "success-2.wav": ProcessingStatus.COMPLETED.value,
        "fail.wav": ProcessingStatus.FAILED.value,
        "success-1.wav": ProcessingStatus.COMPLETED.value,
    }
    assert refreshed_overview.total_interviews == 3
    assert refreshed_overview.status_counts == {
        ProcessingStatus.COMPLETED.value: 2,
        ProcessingStatus.FAILED.value: 1,
    }


def test_storage_service_builds_predictable_audio_paths() -> None:
    storage = StorageService(SupabaseSettings())
    asset = UploadedAsset(
        filename="../demo clip.wav",
        content_type="audio/wav",
        size_bytes=3,
        raw_bytes=b"123",
    )

    assert storage.build_audio_path("interview-123", asset.filename) == "audio/interview-123/demo_clip.wav"
    assert storage.store_asset(asset, sample_mode=True) is True


def test_storage_service_sample_mode_succeeds_without_storage_client() -> None:
    storage = StorageService(SupabaseSettings())
    asset = UploadedAsset(
        filename="sample.wav",
        content_type="audio/wav",
        size_bytes=4,
        raw_bytes=b"demo",
    )

    object_path = storage.upload_audio(asset, interview_id="interview-123", sample_mode=True)

    assert object_path == "audio/interview-123/sample.wav"
    assert storage.store_asset(asset, sample_mode=True, interview_id="interview-123") is True


def test_storage_service_live_mode_without_storage_client_fails() -> None:
    storage = StorageService(SupabaseSettings())
    asset = UploadedAsset(
        filename="live.wav",
        content_type="audio/wav",
        size_bytes=4,
        raw_bytes=b"demo",
    )

    with pytest.raises(RuntimeError, match="configured storage client or explicit upload implementation"):
        storage.upload_audio(asset, interview_id="interview-123", sample_mode=False)

    with pytest.raises(RuntimeError, match="configured storage client or explicit upload implementation"):
        storage.store_asset(asset, sample_mode=False, interview_id="interview-123")


def test_pipeline_marks_storage_persistence_failed_without_live_storage_client(tmp_path) -> None:
    database_path = tmp_path / "storage-failure.sqlite3"
    repository = PaydayRepository(database_path=str(database_path))
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=str(database_path)),
        supabase=SupabaseSettings(),
        llm=LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
        transcription=TranscriptionSettings(provider="openai", model="whisper-test", api_key="transcription-key"),
        features=FeatureFlags(use_sample_mode=False),
    )
    pipeline = PaydayAppService(settings, repository=repository).pipeline
    pipeline.transcription_service = StaticTranscriptionService(settings.transcription)
    pipeline.analysis_service = AnalysisService(
        adapter=OpenAIAnalysisAdapter(
            settings.llm,
            transport=StaticAnalysisTransport(
                {"output": [{"content": [{"type": "output_text", "text": VALID_ANALYSIS_JSON}]}]}
            ),
        ),
        settings=settings.llm,
    )

    result = pipeline.process_upload("live.wav", "audio/wav", b"audio payload")

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.STORAGE
    assert result.persisted is False
    assert result.errors == [
        "Live storage upload requires a configured storage client or explicit upload implementation."
    ]

    with sqlite3.connect(database_path) as connection:
        interview_row = connection.execute(
            "SELECT id, status, latest_stage, last_error FROM interviews WHERE id = ?",
            (result.file_id,),
        ).fetchone()

    assert interview_row == (
        result.file_id,
        ProcessingStatus.FAILED.value,
        PipelineStage.STORAGE.value,
        "Live storage upload requires a configured storage client or explicit upload implementation.",
    )


def test_pipeline_live_storage_persists_only_after_successful_upload(tmp_path) -> None:
    database_path = tmp_path / "storage-success.sqlite3"
    repository = PaydayRepository(database_path=str(database_path))
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=str(database_path)),
        supabase=SupabaseSettings(storage_bucket="test-assets"),
        llm=LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
        transcription=TranscriptionSettings(provider="openai", model="whisper-test", api_key="transcription-key"),
        features=FeatureFlags(use_sample_mode=False),
    )
    storage_client = RecordingSupabaseStorageClient()
    pipeline = PaydayAppService(settings, repository=repository).pipeline
    pipeline.storage_service = StorageService(settings.supabase, storage_client=storage_client)
    pipeline.transcription_service = StaticTranscriptionService(settings.transcription)
    pipeline.analysis_service = AnalysisService(
        adapter=OpenAIAnalysisAdapter(
            settings.llm,
            transport=StaticAnalysisTransport(
                {"output": [{"content": [{"type": "output_text", "text": VALID_ANALYSIS_JSON}]}]}
            ),
        ),
        settings=settings.llm,
    )

    result = pipeline.process_upload("live.wav", "audio/wav", b"audio payload")

    assert result.status is ProcessingStatus.COMPLETED
    assert result.current_stage is PipelineStage.STORAGE
    assert result.persisted is True
    assert storage_client.bucket_names == ["test-assets"]
    assert storage_client.bucket.upload_calls == [
        {
            "path": f"audio/{result.file_id}/live.wav",
            "payload": b"audio payload",
            "file_options": {"content-type": "audio/wav"},
        }
    ]

    with sqlite3.connect(database_path) as connection:
        interview_row = connection.execute(
            "SELECT id, status, latest_stage, last_error FROM interviews WHERE id = ?",
            (result.file_id,),
        ).fetchone()

    assert interview_row == (
        result.file_id,
        ProcessingStatus.COMPLETED.value,
        PipelineStage.STORAGE.value,
        None,
    )


def test_persona_classifier_uses_bank_account_override() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": True, "evidence": ["whatsapp"]},
                    "has_bank_account": {"value": False, "evidence": ["no bank account"]},
                },
                "persona_signals": {
                    "digital_readiness": {"value": True, "evidence": ["uses apps"]},
                    "trust_fear_barrier": {"value": True, "evidence": ["worried"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_3"
    assert persona.is_non_target is True
    assert persona.explanation_payload["triggered_fields"] == ["participant_profile.has_bank_account"]


def test_persona_classifier_uses_smartphone_override() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": False, "evidence": ["basic phone"]},
                    "has_bank_account": {"value": True, "evidence": ["bank account"]},
                },
                "persona_signals": {
                    "cyclical_borrowing": {"value": True, "evidence": ["every month"]},
                    "repayment_stress": {"value": True, "evidence": ["repay pressure"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_3"
    assert persona.is_non_target is True
    assert persona.explanation_payload["triggered_fields"] == ["participant_profile.smartphone_user"]


def test_persona_classifier_uses_structured_fields_for_persona_one() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": True, "evidence": ["whatsapp"]},
                    "has_bank_account": {"value": True, "evidence": ["bank account"]},
                },
                "persona_signals": {
                    "employer_dependency": {"value": True, "evidence": ["madam helped"]},
                    "digital_borrowing": {"value": True, "evidence": ["loan app"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_1"
    assert persona.explanation_payload["triggered_fields"] == [
        "persona_signals.employer_dependency",
        "persona_signals.digital_borrowing",
    ]
    assert persona.evidence_quotes == ("madam helped", "loan app")


def test_persona_classifier_matches_persona_five_before_lower_priority_rules() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": True, "evidence": ["whatsapp"]},
                    "has_bank_account": {"value": True, "evidence": ["bank account"]},
                },
                "persona_signals": {
                    "cyclical_borrowing": {"value": True, "evidence": ["every month"]},
                    "repayment_stress": {"value": True, "evidence": ["repayment pressure"]},
                    "self_reliance_non_borrowing": {"value": True, "evidence": ["use my savings"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_5"
    assert persona.explanation_payload["triggered_fields"] == [
        "persona_signals.cyclical_borrowing",
        "persona_signals.repayment_stress",
    ]


def test_app_service_uses_live_openai_analysis_adapter_when_sample_mode_disabled() -> None:
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(provider="openai", model="gpt-4.1-mini", api_key="live-key"),
        transcription=TranscriptionSettings(api_key="tx-key"),
        features=FeatureFlags(use_sample_mode=False),
    )

    service = PaydayAppService(settings)

    assert service.pipeline.analysis_service.adapter.provider_name == "openai"
    assert service.pipeline.analysis_service.adapter.model_name == "gpt-4.1-mini"


def test_app_service_keeps_heuristic_analysis_adapter_in_sample_mode() -> None:
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(provider="openai", model="gpt-4.1-mini", api_key=""),
        transcription=TranscriptionSettings(),
        features=FeatureFlags(use_sample_mode=True),
    )

    service = PaydayAppService(settings)

    assert service.pipeline.analysis_service.adapter.provider_name == "heuristic"
    assert service.pipeline.analysis_service.adapter.model_name == "heuristic-json"


def test_pipeline_preflights_openai_batch_upload_sizes_and_keeps_other_mp3s_processing() -> None:
    service = PaydayAppService(build_settings())
    transcription_service = StaticTranscriptionService(TranscriptionSettings(provider="openai"))
    service.pipeline.transcription_service = transcription_service

    small_mp3 = b"ID3small-audio"
    near_limit_mp3 = b"0" * OPENAI_TRANSCRIPTION_SAFE_FILE_LIMIT_BYTES
    oversized_mp3 = b"1" * (OPENAI_TRANSCRIPTION_SAFE_FILE_LIMIT_BYTES + 1)

    batch_result = service.process_batch_uploads(
        [
            BatchUploadItem(filename="small.mp3", content_type="audio/mpeg", data=small_mp3),
            BatchUploadItem(filename="near-limit.mp3", content_type="audio/mpeg", data=near_limit_mp3),
            BatchUploadItem(filename="oversized.mp3", content_type="audio/mpeg", data=oversized_mp3),
        ]
    )

    assert batch_result.completed_count == 2
    assert batch_result.failed_count == 1
    assert transcription_service.calls == ["small.mp3", "near-limit.mp3"]

    results_by_filename = {result.filename: result for result in batch_result.results}
    assert results_by_filename["small.mp3"].status is ProcessingStatus.COMPLETED
    assert results_by_filename["near-limit.mp3"].status is ProcessingStatus.COMPLETED

    oversized_result = results_by_filename["oversized.mp3"]
    assert oversized_result.status is ProcessingStatus.FAILED
    assert oversized_result.current_stage is PipelineStage.UPLOAD
    assert oversized_result.asset is None
    assert oversized_result.errors == [
        "oversized.mp3 is 24.0 MB, which exceeds PayDay's 24 MB OpenAI transcription safety limit. "
        "OpenAI caps transcription requests at 25 MB including multipart upload overhead, so please "
        "compress or split this recording and upload it again."
    ]


def test_describe_transcription_file_size_limit_matches_sidebar_guidance() -> None:
    assert describe_transcription_file_size_limit(TranscriptionSettings(provider="openai")) == (
        "OpenAI transcriptions cap requests at 25 MB, so PayDay preflights uploads at 24 MB to leave "
        "room for multipart overhead. Compress or split larger recordings before processing."
    )
    assert describe_transcription_file_size_limit(TranscriptionSettings(provider="other")) is None
