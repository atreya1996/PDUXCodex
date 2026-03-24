from __future__ import annotations

import json
import sqlite3
import types
from pathlib import Path

import pytest

from payday.analysis import AnalysisService, OpenAIAnalysisAdapter
from payday.config import DatabaseSettings, FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.dashboard.views import DashboardRenderer
from payday.models import AnalysisResult, BatchUploadItem, PipelineResult, PipelineStage, ProcessingStatus, Transcript, UploadedAsset
from payday.personas import PersonaService
from payday.repository import PaydayRepository
from payday.service import PaydayAppService
from payday.storage import StorageService
from payday.transcription import (
    OPENAI_TRANSCRIPTION_SAFE_FILE_LIMIT_BYTES,
    OpenAITranscriptionAdapter,
    SampleModeDisabledError,
    TranscriptionService,
    detect_malformed_transcript_reason,
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
                "I earn ₹12,000 per month. I borrow from neighbors sometimes. "
                "I repay monthly after salary. I am worried about scams."
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
        self.remove_calls: list[list[str]] = []

    def upload(self, path: str, payload: bytes, file_options: dict[str, object] | None = None) -> None:
        self.upload_calls.append(
            {
                "path": path,
                "payload": payload,
                "file_options": file_options or {},
            }
        )

    def remove(self, paths: list[str]) -> None:
        self.remove_calls.append(paths)


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
  "per_household_earnings": {"value": "unknown", "status": "unknown", "evidence_quotes": [], "notes": "", "evidence_type": "unknown"},
  "participant_personal_monthly_income": {"value": "₹12,000", "status": "observed", "evidence_quotes": ["I earn ₹12,000 per month"], "notes": "Directly stated.", "evidence_type": "direct"},
  "total_household_monthly_income": {"value": "unknown", "status": "unknown", "evidence_quotes": [], "notes": "", "evidence_type": "unknown"},
  "borrowing_history": {"value": "has_borrowed", "status": "observed", "evidence_quotes": ["I borrow from neighbors sometimes"], "notes": "Directly stated."},
  "repayment_preference": {"value": "monthly", "status": "observed", "evidence_quotes": ["I repay monthly after salary"], "notes": "Directly stated."},
  "loan_interest": {"value": "fearful_or_uncertain", "status": "observed", "evidence_quotes": ["I am worried about scams"], "notes": "Trust barrier present."},
  "summary": {"value": "The participant uses WhatsApp, has a bank account, earns ₹12,000, and worries about scams.", "status": "observed", "evidence_quotes": ["I use WhatsApp every day", "I am worried about scams"], "notes": "Grounded in transcript."},
  "key_quotes": ["I use WhatsApp every day", "I am worried about scams"],
  "confidence_signals": {"observed_evidence": ["trust barrier mentioned"], "missing_or_unknown": []},
  "segmented_dialogue": [
    {"speaker_label": "participant", "utterance_text": "I use WhatsApp every day", "speaker_confidence": "medium", "speaker_uncertainty": "First-person phrasing suggests the participant is speaking."},
    {"speaker_label": "participant", "utterance_text": "I am worried about scams", "speaker_confidence": "medium", "speaker_uncertainty": "First-person phrasing suggests the participant is speaking."}
  ]
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


def load_sample_case(case_filename: str) -> tuple[str, dict[str, object]]:
    transcript = (Path("sample_data/mock_uploads") / case_filename).read_text(encoding="utf-8")
    payload = json.loads((Path("sample_data/structured_outputs") / case_filename.replace(".txt", ".json")).read_text(encoding="utf-8"))
    return transcript, payload["structured_output"]


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
        "file_path": "",
        "content_type": "text/plain",
        "size_bytes": 11,
        "source": "demo.txt",
    }


def test_transcription_service_sample_mode_requires_explicit_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAYDAY_USE_SAMPLE_MODE", "false")
    service = TranscriptionService(TranscriptionSettings())
    asset = UploadedAsset(
        filename="demo.txt",
        content_type="text/plain",
        size_bytes=11,
        raw_bytes=b"hello world",
        file_id="file-123",
    )

    with pytest.raises(SampleModeDisabledError, match="PAYDAY_USE_SAMPLE_MODE=true"):
        service.transcribe(asset, sample_mode=True)


def test_transcription_service_sample_mode_accepts_truthy_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAYDAY_USE_SAMPLE_MODE", "1")
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
    assert transcript.metadata["sample_mode"] is True


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
        "file_path": "",
        "content_type": "audio/wav",
        "size_bytes": 4,
        "source": "call.wav",
        "language": "hi",
        "duration_seconds": 8.25,
    }


@pytest.mark.parametrize(
    ("filename", "content_type", "payload"),
    [
        ("sample_lame.mp3", "audio/mpeg", b"ID3\x04\x00\x00\x00\x00\x00\x15LAME3.100\x00\x00\x00\x00\x00\x00\x00audio"),
        ("sample_ftyp.mp4", "video/mp4", b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2\x00\x00\x00\x00\x00\x00\x00\x00payload"),
    ],
)
def test_transcription_service_non_sample_mode_uses_provider_output_not_raw_decode(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content_type: str,
    payload: bytes,
) -> None:
    install_fake_openai_module(monkeypatch)
    transcriptions = StubOpenAITranscriptions(response=StubTranscriptionResponse("Provider transcript only"))
    client = StubOpenAIClient(transcriptions)
    service = TranscriptionService(TranscriptionSettings(api_key="test-key"), client=client)
    asset = UploadedAsset(
        filename=filename,
        content_type=content_type,
        size_bytes=len(payload),
        raw_bytes=payload,
        file_id=f"file-{filename}",
    )

    transcript = service.transcribe(asset, sample_mode=False)

    assert transcript.text == "Provider transcript only"
    assert "ftyp" not in transcript.text.lower()
    assert "lame" not in transcript.text.lower()
    assert len(transcriptions.calls) == 1


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


def test_upload_service_normalizes_supported_openai_formats(tmp_path: Path) -> None:
    service = UploadService(upload_dir=tmp_path)

    mp4_asset = service.create_asset("interview.mp4", "application/octet-stream", b"mp4-bytes")
    mpeg_asset = service.create_asset("interview.mpeg", "", b"mpeg-bytes")
    webm_asset = service.create_asset("interview.webm", "audio/webm", b"webm-bytes")

    assert mp4_asset.content_type == "video/mp4"
    assert mpeg_asset.content_type == "audio/mpeg"
    assert webm_asset.content_type == "audio/webm"
    assert Path(mp4_asset.file_path).exists()
    assert SUPPORTED_UPLOAD_EXTENSIONS == ("mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "aac")


def test_upload_service_rejects_unsupported_extensions(tmp_path: Path) -> None:
    service = UploadService(upload_dir=tmp_path)

    with pytest.raises(UploadValidationError, match="Unsupported upload format"):
        service.create_asset("notes.txt", "text/plain", b"hello")


def test_upload_service_sanitizes_filename_and_prefixes_file_id(tmp_path: Path) -> None:
    service = UploadService(upload_dir=tmp_path)

    asset = service.create_asset("../voice note (1).wav", "audio/wav", b"RIFF", file_id="abc123")

    assert asset.filename == "voice_note_1_.wav"
    assert asset.file_path.endswith("abc123_voice_note_1_.wav")
    assert Path(asset.file_path).read_bytes() == b"RIFF"


def test_upload_service_rejects_content_type_extension_mismatch(tmp_path: Path) -> None:
    service = UploadService(upload_dir=tmp_path)

    with pytest.raises(UploadValidationError, match="extension '.wav' but content type 'video/mp4'"):
        service.create_asset("clip.wav", "video/mp4", b"RIFF")


def test_upload_service_rejects_empty_and_oversized_payloads(tmp_path: Path) -> None:
    service = UploadService(upload_dir=tmp_path, max_file_size_bytes=5)

    with pytest.raises(UploadValidationError, match="is empty"):
        service.create_asset("empty.wav", "audio/wav", b"")

    with pytest.raises(UploadValidationError, match="exceeds the 0 MB upload limit"):
        service.create_asset("big.wav", "audio/wav", b"123456")


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


def test_pipeline_fails_before_transcription_when_persisted_upload_path_is_missing() -> None:
    service = PaydayAppService(build_settings())
    service.pipeline.upload_service.create_asset = lambda **_: UploadedAsset(  # type: ignore[method-assign]
        filename="demo.wav",
        content_type="audio/wav",
        size_bytes=4,
        file_path="data/uploads/missing_demo.wav",
        raw_bytes=b"RIFF",
        file_id="missing-file-id",
    )

    result = service.process_upload("demo.wav", "audio/wav", b"RIFF")

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.UPLOAD
    assert result.transcript is None
    assert result.last_error is not None
    assert "was not persisted" in result.last_error


def test_pipeline_rejects_binary_signature_transcript_before_analysis() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload(
        "bad.mp4",
        "video/mp4",
        b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00\x00\x00\x00\x00",
    )

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.TRANSCRIPTION
    assert result.analysis is None
    assert result.persona is None
    assert result.last_error == "Transcription failed: binary payload detected, please retry or verify API key/provider."


def test_pipeline_rejects_malformed_mp4_binary_fixture_before_analysis() -> None:
    service = PaydayAppService(build_settings())
    binary_payload = bytes.fromhex((Path("tests/fixtures") / "malformed_binary_mp4_hex.txt").read_text().strip())

    result = service.process_upload("fixture-bad.mp4", "video/mp4", binary_payload)

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.TRANSCRIPTION
    assert result.analysis is None
    assert result.persona is None
    assert result.last_error == "Transcription failed: binary payload detected, please retry or verify API key/provider."


def test_pipeline_rejects_malformed_mp3_binary_fixture_before_analysis() -> None:
    service = PaydayAppService(build_settings())
    binary_payload = bytes.fromhex((Path("tests/fixtures") / "malformed_binary_mp3_hex.txt").read_text().strip())

    result = service.process_upload("fixture-bad.mp3", "audio/mpeg", binary_payload)

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.TRANSCRIPTION
    assert result.analysis is None
    assert result.persona is None
    assert result.last_error == "Transcription failed: binary payload detected, please retry or verify API key/provider."


def test_pipeline_rejects_transcript_with_excessive_non_text_ratio_before_analysis() -> None:
    service = PaydayAppService(build_settings())
    malformed_payload = (b"\x01\x02\x03\x04\x05\x06\x07\x08" * 8) + b"ok"

    result = service.process_upload("control-heavy.wav", "audio/wav", malformed_payload)

    assert result.status is ProcessingStatus.FAILED
    assert result.current_stage is PipelineStage.TRANSCRIPTION
    assert result.analysis is None
    assert result.persona is None
    assert result.last_error is not None
    assert result.last_error.startswith("Transcription failed: malformed transcript detected")


def test_detect_malformed_transcript_reason_flags_binary_signatures_and_non_text_ratio() -> None:
    assert detect_malformed_transcript_reason("\x00\x00\x00\x18ftypisom") == "binary signature detected in transcript output"
    assert detect_malformed_transcript_reason("\x01\x02\x03\x04\x05\x06\x07\x08" * 8).startswith(
        "excessive non-text character ratio"
    )


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
                per_household_earnings,
                participant_personal_monthly_income,
                total_household_monthly_income,
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
    assert structured_row[3] is None
    assert structured_row[4] == "₹12,000"
    assert structured_row[5] is None
    assert structured_row[6] == "Participant monthly income: ₹12,000"
    assert structured_row[7] == result.analysis.structured_output["borrowing_history"]["value"]
    assert structured_row[8] == result.analysis.structured_output["repayment_preference"]["value"]
    assert structured_row[9] == result.analysis.structured_output["loan_interest"]["value"]

    assert insight_row is not None
    assert insight_row[0] == result.file_id
    assert "WhatsApp every day" in insight_row[1]
    assert insight_row[2] == "High-Stress Cyclical Borrower"
    assert insight_row[3] == 0.78

    reloaded_repository = PaydayRepository(database_path=str(database_path))
    detail = reloaded_repository.get_interview_detail(result.file_id)

    assert detail.interview.id == result.file_id
    assert detail.interview.status == ProcessingStatus.COMPLETED.value
    assert detail.interview.latest_stage == PipelineStage.STORAGE.value
    assert detail.interview.last_error is None
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.structured_response.has_bank_account is True
    assert detail.structured_response.per_household_earnings is None
    assert detail.structured_response.participant_personal_monthly_income == "₹12,000"
    assert detail.structured_response.total_household_monthly_income is None
    assert detail.structured_response.income_range == "Participant monthly income: ₹12,000"
    assert detail.structured_response.borrowing_history == result.analysis.structured_output["borrowing_history"]["value"]
    assert (
        detail.structured_response.repayment_preference
        == result.analysis.structured_output["repayment_preference"]["value"]
    )
    assert detail.structured_response.loan_interest == result.analysis.structured_output["loan_interest"]["value"]
    assert detail.insight is not None
    assert detail.insight.summary == result.analysis.summary
    assert detail.insight.persona == "High-Stress Cyclical Borrower"
    assert detail.insight.confidence_score == 0.78


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
        per_household_earnings=None,
        participant_personal_monthly_income="10k-20k",
        total_household_monthly_income=None,
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


def test_repository_delete_interview_cascades_related_rows(tmp_path) -> None:
    repository = PaydayRepository(database_path=str(tmp_path / "cascade.db"))
    interview = repository.create_interview(audio_url="audio/interview-1/demo.wav", status="completed")
    repository.upsert_structured_response(
        interview.id,
        smartphone_user=True,
        has_bank_account=True,
        per_household_earnings=None,
        participant_personal_monthly_income="10k-20k",
        total_household_monthly_income=None,
        income_range="10k-20k",
        borrowing_history="has_borrowed",
        repayment_preference="monthly",
        loan_interest="interested",
    )
    repository.upsert_insight(
        interview.id,
        summary="Summary",
        key_quotes=["Direct quote"],
        persona="Self-Reliant Non-Borrower",
        confidence_score=0.8,
    )

    deleted = repository.delete_interview(interview.id)

    assert deleted is True
    assert repository.list_recent_interviews() == []
    with pytest.raises(KeyError, match=interview.id):
        repository.get_interview(interview.id)

    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM interviews").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM structured_responses").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM insights").fetchone()[0] == 0


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


def test_app_service_save_interview_edits_persists_transcript_and_refreshes_outputs(tmp_path) -> None:
    database_path = str(tmp_path / "edit-save.db")
    service = PaydayAppService(build_settings(database_path))
    result = service.process_upload(
        "editable.wav",
        "audio/wav",
        b"I use WhatsApp and my bank account is active.",
    )

    current_detail = service.get_interview_detail(result.file_id)
    current_payload = {
        "smartphone_usage": {
            "value": "has_smartphone" if current_detail.smartphone_user else "no_smartphone",
            "status": "observed",
            "evidence_quotes": [],
            "notes": "",
        },
        "per_household_earnings": {"value": current_detail.per_household_earnings},
        "participant_personal_monthly_income": {"value": current_detail.participant_personal_monthly_income or current_detail.income_range},
        "total_household_monthly_income": {"value": current_detail.total_household_monthly_income},
        "borrowing_history": {"value": current_detail.borrowing_history},
        "repayment_preference": {"value": current_detail.repayment_preference},
        "loan_interest": {"value": current_detail.loan_interest},
        "insight": {
            "summary": current_detail.summary,
            "key_quotes": current_detail.key_quotes,
        },
        "income_range": {"value": current_detail.income_range or "unknown", "status": "unknown", "evidence_quotes": [], "notes": ""},
        "borrowing_history": {"value": current_detail.borrowing_history or "unknown", "status": "unknown", "evidence_quotes": [], "notes": ""},
        "repayment_preference": {"value": current_detail.repayment_preference or "unknown", "status": "unknown", "evidence_quotes": [], "notes": ""},
        "loan_interest": {"value": current_detail.loan_interest or "unknown", "status": "unknown", "evidence_quotes": [], "notes": ""},
        "summary": {"value": current_detail.summary or "unknown", "status": "observed", "evidence_quotes": current_detail.key_quotes, "notes": ""},
        "key_quotes": current_detail.key_quotes,
        "confidence_signals": {"observed_evidence": [], "missing_or_unknown": []},
    }

    saved_detail = service.save_interview_edits(
        result.file_id,
        transcript="I use WhatsApp but I do not have a bank account.",
        extracted_json=json.dumps(current_payload, ensure_ascii=False),
        transcript_changed=True,
        structured_json_changed=False,
    )

    assert saved_detail.transcript == "I use WhatsApp but I do not have a bank account."
    assert saved_detail.status == ProcessingStatus.COMPLETED.value
    assert saved_detail.has_bank_account is False
    assert saved_detail.persona == "Offline / Excluded"
    assert service.get_status_overview().status_counts == {ProcessingStatus.COMPLETED.value: 1}

    reloaded_service = PaydayAppService(build_settings(database_path))
    reloaded_detail = reloaded_service.get_interview_detail(result.file_id)

    assert reloaded_detail.transcript == "I use WhatsApp but I do not have a bank account."
    assert reloaded_detail.has_bank_account is False
    assert reloaded_detail.persona == "Offline / Excluded"
    assert reloaded_service.get_status_overview().status_counts == {ProcessingStatus.COMPLETED.value: 1}


def test_app_service_save_interview_edits_accepts_dashboard_json_and_rederives_persona(tmp_path) -> None:
    database_path = str(tmp_path / "edit-json.db")
    service = PaydayAppService(build_settings(database_path))
    result = service.process_upload(
        "json-edit.wav",
        "audio/wav",
        b"I use WhatsApp and my bank account is active.",
    )

    edited_payload = {
        "audio_url": service.get_interview_detail(result.file_id).audio_url,
        "participant_profile": {
            "smartphone_user": {"value": False},
            "has_bank_account": {"value": True},
        },
        "participant_personal_monthly_income": {"value": "₹9,000", "status": "observed", "evidence_quotes": ["₹9,000"], "notes": "", "evidence_type": "direct"},
        "per_household_earnings": {"value": "unknown", "status": "unknown", "evidence_quotes": [], "notes": "", "evidence_type": "unknown"},
        "total_household_monthly_income": {"value": "unknown", "status": "unknown", "evidence_quotes": [], "notes": "", "evidence_type": "unknown"},
        "borrowing_history": {"value": "has_not_borrowed_recently", "status": "observed", "evidence_quotes": ["I avoid loans"], "notes": ""},
        "repayment_preference": {"value": "monthly", "status": "observed", "evidence_quotes": ["monthly"], "notes": ""},
        "loan_interest": {"value": "not_interested", "status": "observed", "evidence_quotes": ["not interested"], "notes": ""},
        "summary": {"value": "Participant lacks a smartphone and avoids borrowing.", "status": "observed", "evidence_quotes": ["I use a basic phone now."], "notes": ""},
        "key_quotes": ["I use a basic phone now."],
        "confidence_signals": {"observed_evidence": [], "missing_or_unknown": []},
    }

    saved_detail = service.save_interview_edits(
        result.file_id,
        transcript="I use a basic phone now.",
        extracted_json=json.dumps(edited_payload, ensure_ascii=False),
        transcript_changed=False,
        structured_json_changed=True,
    )

    assert saved_detail.smartphone_user is False
    assert saved_detail.persona == "Offline / Excluded"
    assert saved_detail.summary == "Participant lacks a smartphone and avoids borrowing."
    assert saved_detail.key_quotes == ["I use a basic phone now."]
    assert service.get_status_overview().status_counts == {ProcessingStatus.COMPLETED.value: 1}

    reloaded_service = PaydayAppService(build_settings(database_path))
    reloaded_detail = reloaded_service.get_interview_detail(result.file_id)

    assert reloaded_detail.smartphone_user is False
    assert reloaded_detail.persona == "Offline / Excluded"
    assert reloaded_detail.summary == "Participant lacks a smartphone and avoids borrowing."
    assert reloaded_detail.key_quotes == ["I use a basic phone now."]
    assert reloaded_service.get_status_overview().status_counts == {ProcessingStatus.COMPLETED.value: 1}


def test_app_service_save_interview_edits_preserves_manual_income_evidence_quotes(tmp_path) -> None:
    database_path = str(tmp_path / "edit-json-income-manual.db")
    service = PaydayAppService(build_settings(database_path))
    result = service.process_upload(
        "json-income-edit.wav",
        "audio/wav",
        b"I use WhatsApp and my bank account is active.",
    )

    edited_payload = {
        "audio_url": service.get_interview_detail(result.file_id).audio_url,
        "participant_profile": {
            "smartphone_user": {"value": True},
            "has_bank_account": {"value": True},
        },
        "participant_personal_monthly_income": {
            "value": "₹9,000",
            "status": "observed",
            "evidence_quotes": ["Monthly salary around 9k from all homes"],
            "notes": "Curated from manual review notes.",
            "evidence_type": "direct",
        },
        "per_household_earnings": {"value": "unknown", "status": "unknown", "evidence_quotes": [], "notes": "", "evidence_type": "unknown"},
        "total_household_monthly_income": {"value": "unknown", "status": "unknown", "evidence_quotes": [], "notes": "", "evidence_type": "unknown"},
        "borrowing_history": {"value": "has_not_borrowed_recently", "status": "observed", "evidence_quotes": ["I avoid loans"], "notes": ""},
        "repayment_preference": {"value": "monthly", "status": "observed", "evidence_quotes": ["monthly"], "notes": ""},
        "loan_interest": {"value": "not_interested", "status": "observed", "evidence_quotes": ["not interested"], "notes": ""},
        "summary": {"value": "Manual correction retains curated income quote.", "status": "observed", "evidence_quotes": ["Monthly salary around 9k from all homes"], "notes": ""},
        "key_quotes": ["Monthly salary around 9k from all homes"],
        "confidence_signals": {"observed_evidence": [], "missing_or_unknown": []},
    }

    saved_detail = service.save_interview_edits(
        result.file_id,
        transcript="I use WhatsApp and my bank account is active.",
        extracted_json=json.dumps(edited_payload, ensure_ascii=False),
        transcript_changed=False,
        structured_json_changed=True,
    )

    assert saved_detail.participant_personal_monthly_income == "₹9,000"
    assert saved_detail.income_range == "Participant monthly income: ₹9,000"
    assert saved_detail.key_quotes == ["Monthly salary around 9k from all homes"]

    persisted = service.repository.get_result(result.file_id)
    assert persisted is not None
    assert persisted.analysis is not None
    assert persisted.analysis.structured_output["participant_personal_monthly_income"]["evidence_quotes"] == [
        "Monthly salary around 9k from all homes"
    ]
    assert persisted.analysis.metrics["evidence_mode"] == "manual_edit"


def test_dashboard_tabs_reflect_legacy_reprocessed_persona_examples_one_to_five(tmp_path) -> None:
    database_path = str(tmp_path / "legacy-dashboard.db")
    service = PaydayAppService(build_settings(database_path))

    cases = [
        "demo_01_employer_digital_borrower.txt",
        "demo_02_fearful_but_ready.txt",
        "demo_03_no_smartphone.txt",
        "demo_05_self_reliant.txt",
        "demo_06_cyclical_stress.txt",
    ]

    for index, case_filename in enumerate(cases, start=1):
        transcript_text, structured_output = load_sample_case(case_filename)
        result = service.process_upload(
            f"legacy-case-{index}.wav",
            "audio/wav",
            transcript_text.encode("utf-8"),
        )

        service.save_interview_edits(
            result.file_id,
            transcript=transcript_text,
            extracted_json=json.dumps(structured_output, ensure_ascii=False),
            transcript_changed=False,
            structured_json_changed=True,
        )

    reloaded_service = PaydayAppService(build_settings(database_path))
    renderer = DashboardRenderer()
    dashboard_interviews = renderer._build_dashboard_interviews([], reloaded_service.list_recent_interviews())
    completed_interviews = [item for item in dashboard_interviews if item.status == ProcessingStatus.COMPLETED.value]

    persona_rows = {
        label: count for label, count, _percent in renderer._build_cohort_rows(completed_interviews, "persona_name")
    }
    digital_access_rows = {
        label: count for label, count, _percent in renderer._build_cohort_rows(completed_interviews, "digital_access")
    }

    assert persona_rows == {
        "Employer-Dependent Digital Borrower": 1,
        "Digitally Ready but Fearful": 1,
        "Offline / Excluded": 1,
        "Self-Reliant Non-Borrower": 1,
        "High-Stress Cyclical Borrower": 1,
    }
    assert digital_access_rows == {
        "Smartphone + bank account": 4,
        "Excluded / offline": 1,
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


def test_processed_interviews_populate_dashboard_cohorts_personas_and_status(tmp_path) -> None:
    database_path = str(tmp_path / "dashboard-cohorts.db")
    service = PaydayAppService(build_settings(database_path))
    service.process_upload(
        "fearful-borrower.wav",
        "audio/wav",
        (
            b"I use WhatsApp every day. My bank account is active. "
            b"I borrow from neighbors sometimes. I repay monthly after salary. I am worried about scams."
        ),
    )
    service.process_upload(
        "excluded.wav",
        "audio/wav",
        b"I use a basic phone. I do not have a bank account.",
    )

    renderer = DashboardRenderer()
    dashboard_interviews = renderer._build_dashboard_interviews([], service.list_recent_interviews())
    completed_interviews = [item for item in dashboard_interviews if item.status == ProcessingStatus.COMPLETED.value]

    digital_access_rows = {
        label: count for label, count, _percent in renderer._build_cohort_rows(completed_interviews, "digital_access")
    }
    borrowing_rows = {
        label: count for label, count, _percent in renderer._build_cohort_rows(completed_interviews, "borrowing_label")
    }
    persona_rows = {
        label: count for label, count, _percent in renderer._build_cohort_rows(completed_interviews, "persona_name")
    }
    status_counts = renderer._status_counts(dashboard_interviews, service.get_status_overview())

    assert digital_access_rows == {
        "Smartphone + bank account": 1,
        "Excluded / offline": 1,
    }
    assert borrowing_rows == {
        "Borrower": 1,
        "Non-borrower": 1,
    }
    assert persona_rows == {
        "High-Stress Cyclical Borrower": 1,
        "Offline / Excluded": 1,
    }
    assert status_counts == {
        ProcessingStatus.PENDING.value: 0,
        ProcessingStatus.PROCESSING.value: 0,
        ProcessingStatus.COMPLETED.value: 2,
        ProcessingStatus.FAILED.value: 0,
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

    with pytest.raises(RuntimeError, match="configured storage client or explicit delete implementation"):
        storage.delete_asset("audio/interview-123/live.wav", sample_mode=False)


def test_storage_service_live_mode_deletes_uploaded_asset_from_bucket() -> None:
    storage_client = RecordingSupabaseStorageClient()
    storage = StorageService(SupabaseSettings(storage_bucket="test-assets"), storage_client=storage_client)

    deleted = storage.delete_asset(
        "https://example.supabase.co/storage/v1/object/public/test-assets/audio/interview-123/live.wav",
        sample_mode=False,
    )

    assert deleted is True
    assert storage_client.bucket_names == ["test-assets"]
    assert storage_client.bucket.remove_calls == [["audio/interview-123/live.wav"]]


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


def test_app_service_delete_interview_removes_durable_rows_and_stored_audio(tmp_path) -> None:
    database_path = str(tmp_path / "delete-service.sqlite3")
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=database_path),
        supabase=SupabaseSettings(storage_bucket="test-assets"),
        llm=LLMSettings(provider="openai", model="gpt-test", api_key="analysis-key"),
        transcription=TranscriptionSettings(provider="openai", model="whisper-test", api_key="transcription-key"),
        features=FeatureFlags(use_sample_mode=False),
    )
    storage_client = RecordingSupabaseStorageClient()
    service = PaydayAppService(settings)
    service.pipeline.storage_service = StorageService(settings.supabase, storage_client=storage_client)
    interview = service.repository.create_interview(
        interview_id="delete-me-id",
        audio_url="audio/delete-me-id/delete-me.wav",
        transcript="I use WhatsApp and have a bank account.",
        status=ProcessingStatus.COMPLETED.value,
        latest_stage=PipelineStage.STORAGE.value,
    )

    service.repository.upsert_insight(
        interview.id,
        summary="Completed interview ready for deletion.",
        key_quotes=["I use WhatsApp and have a bank account."],
        persona="Digitally Ready but Fearful",
        confidence_score=0.9,
    )

    deleted = service.delete_interview(interview.id)

    assert deleted is True
    assert service.list_recent_interviews() == []
    assert service.get_status_overview().total_interviews == 0
    assert storage_client.bucket.remove_calls == [["audio/delete-me-id/delete-me.wav"]]
    with pytest.raises(KeyError, match=interview.id):
        service.get_interview_detail(interview.id)


def test_persona_classifier_uses_bank_account_override() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "smartphone_usage": {"value": "has_smartphone", "status": "observed", "evidence_quotes": ["whatsapp"], "notes": ""},
                "bank_account_status": {"value": "no_bank_account", "status": "observed", "evidence_quotes": ["no bank account"], "notes": ""},
            }
        ),
    )

    assert persona.persona_id == "persona_3"
    assert persona.is_non_target is True
    assert persona.explanation_payload["triggered_fields"] == ["bank_account_status"]


def test_persona_classifier_uses_smartphone_override() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "smartphone_usage": {"value": "no_smartphone", "status": "observed", "evidence_quotes": ["basic phone"], "notes": ""},
                "bank_account_status": {"value": "has_bank_account", "status": "observed", "evidence_quotes": ["bank account"], "notes": ""},
            }
        ),
    )

    assert persona.persona_id == "persona_3"
    assert persona.is_non_target is True
    assert persona.explanation_payload["triggered_fields"] == ["smartphone_usage"]


def test_persona_classifier_uses_structured_fields_for_persona_one() -> None:
    persona = PersonaService().classify(
        build_transcript("My employer helps when I need money. I use WhatsApp. My bank account is active. I borrowed last month."),
        build_analysis(
            {
                "smartphone_usage": {"value": "has_smartphone", "status": "observed", "evidence_quotes": ["whatsapp"], "notes": ""},
                "bank_account_status": {"value": "has_bank_account", "status": "observed", "evidence_quotes": ["bank account"], "notes": ""},
                "borrowing_history": {"value": "has_borrowed", "status": "observed", "evidence_quotes": ["I borrowed last month"], "notes": ""},
            }
        ),
    )

    assert persona.persona_id == "persona_1"
    assert persona.explanation_payload["triggered_fields"] == [
        "borrowing_history",
        "smartphone_usage",
    ]
    assert persona.evidence_quotes[0] == "I borrowed last month"


def test_persona_classifier_matches_persona_five_before_lower_priority_rules() -> None:
    persona = PersonaService().classify(
        build_transcript("I use WhatsApp. My bank account is active. I borrow every month and feel repayment pressure."),
        build_analysis(
            {
                "smartphone_usage": {"value": "has_smartphone", "status": "observed", "evidence_quotes": ["whatsapp"], "notes": ""},
                "bank_account_status": {"value": "has_bank_account", "status": "observed", "evidence_quotes": ["bank account"], "notes": ""},
                "borrowing_history": {"value": "has_borrowed", "status": "observed", "evidence_quotes": ["I borrow every month"], "notes": ""},
                "repayment_preference": {"value": "monthly", "status": "observed", "evidence_quotes": ["every month"], "notes": ""},
                "loan_interest": {"value": "fearful_or_uncertain", "status": "observed", "evidence_quotes": ["repayment pressure"], "notes": ""},
            }
        ),
    )

    assert persona.persona_id == "persona_5"
    assert persona.explanation_payload["triggered_fields"] == [
        "borrowing_history",
        "repayment_preference",
        "loan_interest",
    ]


def test_persona_classifier_never_assigns_persona_four_when_borrowing_evidence_exists() -> None:
    persona = PersonaService().classify(
        build_transcript("Sujata says she has taken a loan and repays it monthly."),
        build_analysis(
            {
                "smartphone_usage": {"value": "has_smartphone", "status": "observed", "evidence_quotes": ["I use WhatsApp"], "notes": ""},
                "bank_account_status": {"value": "has_bank_account", "status": "observed", "evidence_quotes": ["I have a bank account"], "notes": ""},
                "borrowing_history": {
                    "value": "has_borrowed",
                    "status": "observed",
                    "evidence_quotes": ["Sujata says she has taken a loan"],
                    "notes": "",
                },
                "persona_signals": {
                    "self_reliance_non_borrowing": {
                        "value": True,
                        "evidence_quotes": ["I manage with savings when possible"],
                    }
                },
            }
        ),
    )

    assert persona.persona_id != "persona_4"


def test_persona_classifier_requires_explicit_non_borrowing_evidence_for_persona_four() -> None:
    persona = PersonaService().classify(
        build_transcript("I use WhatsApp and have a bank account."),
        build_analysis(
            {
                "smartphone_usage": {"value": "has_smartphone", "status": "observed", "evidence_quotes": ["I use WhatsApp"], "notes": ""},
                "bank_account_status": {"value": "has_bank_account", "status": "observed", "evidence_quotes": ["I have a bank account"], "notes": ""},
                "borrowing_history": {"value": "has_not_borrowed_recently", "status": "observed", "evidence_quotes": [], "notes": ""},
            }
        ),
    )

    assert persona.persona_id != "persona_4"


def test_app_service_reprocess_stale_interviews_recomputes_legacy_rows(tmp_path) -> None:
    database_path = str(tmp_path / "stale-reprocess.sqlite3")
    service = PaydayAppService(build_settings(sqlite_path=database_path))
    interview = service.repository.create_interview(
        interview_id="legacy-row-1",
        audio_url="audio/legacy-row-1/demo.wav",
        transcript="I use WhatsApp. I have a bank account. I have taken a loan.",
        status=ProcessingStatus.COMPLETED.value,
        latest_stage=PipelineStage.STORAGE.value,
    )
    with service.repository._connect() as connection:  # noqa: SLF001
        connection.execute(
            """
            UPDATE interviews
            SET analysis_schema_version = NULL, persona_ruleset_version = NULL
            WHERE id = ?
            """,
            (interview.id,),
        )

    summary = service.reprocess_stale_interviews()
    refreshed = service.repository.get_dashboard_interview_detail(interview.id)
    stale_after = service.repository.list_stale_interview_ids()

    assert summary["stale_count"] == 1
    assert summary["failed"] == {}
    assert interview.id in summary["reprocessed_ids"]
    assert refreshed.persona is not None
    assert interview.id not in stale_after


def test_app_service_lists_and_deletes_stale_corrupted_rows(tmp_path) -> None:
    database_path = str(tmp_path / "stale-corrupted.sqlite3")
    service = PaydayAppService(build_settings(sqlite_path=database_path))
    interview = service.repository.create_interview(
        interview_id="stale-bad-1",
        audio_url="audio/stale-bad-1/demo.wav",
        transcript="bad",
        status=ProcessingStatus.FAILED.value,
        latest_stage=PipelineStage.TRANSCRIPTION.value,
        last_error="transcription malformed payload",
    )
    with service.repository._connect() as connection:  # noqa: SLF001
        connection.execute(
            """
            UPDATE interviews
            SET analysis_schema_version = NULL, persona_ruleset_version = NULL
            WHERE id = ?
            """,
            (interview.id,),
        )

    stale_corrupted_before = service.list_stale_corrupted_interview_ids()
    deletion_summary = service.delete_stale_corrupted_interviews()

    assert interview.id in stale_corrupted_before
    assert deletion_summary["stale_corrupted_count"] == 1
    assert interview.id in deletion_summary["deleted_ids"]
    assert deletion_summary["failed"] == {}


def test_persona_classifier_supports_legacy_nested_structured_output_for_personas_one_to_five() -> None:
    expected_personas = {
        "demo_01_employer_digital_borrower.txt": "persona_1",
        "demo_02_fearful_but_ready.txt": "persona_2",
        "demo_03_no_smartphone.txt": "persona_3",
        "demo_05_self_reliant.txt": "persona_4",
        "demo_06_cyclical_stress.txt": "persona_5",
    }

    service = PersonaService()

    for filename, expected_persona in expected_personas.items():
        transcript_text, structured_output = load_sample_case(filename)
        persona = service.classify(build_transcript(transcript_text), build_analysis(structured_output))

        assert persona.persona_id == expected_persona


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
