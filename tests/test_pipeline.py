from payday.analysis import AnalysisService
from payday.config import FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.models import BatchUploadItem, PipelineStage, ProcessingStatus
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import PaydayRepository
from payday.service import PaydayAppService
from payday.storage import StorageService
from payday.transcription import TranscriptionService
from payday.upload import UploadService


class FlakyTranscriptionService(TranscriptionService):
    def __init__(self, settings: TranscriptionSettings) -> None:
        super().__init__(settings)
        self.calls: dict[str, int] = {}

    def transcribe(self, asset, sample_mode: bool = False):
        self.calls.setdefault(asset.file_id, 0)
        self.calls[asset.file_id] += 1
        if asset.filename == "retry.wav" and self.calls[asset.file_id] == 1:
            raise RuntimeError("temporary transcription issue")
        if asset.filename == "fail.wav":
            raise RuntimeError("hard transcription failure")
        return super().transcribe(asset, sample_mode=sample_mode)


def build_settings() -> Settings:
    return Settings(
        app_env="test",
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(),
        features=FeatureFlags(use_sample_mode=True),
    )


def test_pipeline_process_upload_returns_completed_result() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload("demo.wav", "audio/wav", b"fake-bytes")

    assert result.asset is not None
    assert result.asset.filename == "demo.wav"
    assert result.persisted is True
    assert result.analysis is not None
    assert result.analysis.metrics["word_count"] > 0
    assert result.status is ProcessingStatus.COMPLETED
    assert result.current_stage is PipelineStage.STORAGE
    assert service.list_results()


def test_pipeline_batch_retries_and_isolates_failures() -> None:
    settings = build_settings()
    repository = PaydayRepository()
    transcription_service = FlakyTranscriptionService(settings.transcription)
    pipeline = PaydayPipeline(
        upload_service=UploadService(),
        transcription_service=transcription_service,
        analysis_service=AnalysisService(),
        persona_service=PersonaService(),
        storage_service=StorageService(settings.supabase),
        repository=repository,
        sample_mode=True,
        max_retries=1,
    )

    batch = [
        BatchUploadItem(filename="ok-1.wav", content_type="audio/wav", data=b"1"),
        BatchUploadItem(filename="retry.wav", content_type="audio/wav", data=b"2"),
        BatchUploadItem(filename="fail.wav", content_type="audio/wav", data=b"3"),
        BatchUploadItem(filename="ok-2.wav", content_type="audio/wav", data=b"4"),
        BatchUploadItem(filename="ok-3.wav", content_type="audio/wav", data=b"5"),
    ]

    result = pipeline.process_batch_uploads(batch)

    assert len(result.results) == 5
    assert result.completed_count == 4
    assert result.failed_count == 1

    retry_result = next(item for item in result.results if item.filename == "retry.wav")
    failed_result = next(item for item in result.results if item.filename == "fail.wav")

    assert retry_result.status is ProcessingStatus.COMPLETED
    assert retry_result.attempts[PipelineStage.TRANSCRIPTION.value] == 2
    assert failed_result.status is ProcessingStatus.FAILED
    assert failed_result.errors
    assert repository.get_result(failed_result.file_id) is failed_result


def test_persona_three_override_for_no_bank_account() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload(
        "persona.wav",
        "audio/wav",
        b"I do not have a bank account and I borrow from neighbors when needed.",
    )

    assert result.persona is not None
    assert result.persona.persona_id == "persona_3"
    assert result.persona.is_non_target is True



def test_persona_three_override_for_no_smartphone() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload(
        "persona-phone.wav",
        "audio/wav",
        b"I use a basic phone, so I cannot use WhatsApp loan apps.",
    )

    assert result.persona is not None
    assert result.persona.persona_id == "persona_3"
    assert result.persona.is_non_target is True
