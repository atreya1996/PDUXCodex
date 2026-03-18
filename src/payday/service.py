from __future__ import annotations

from payday.analysis import AnalysisService, HeuristicAnalysisAdapter
from payday.config import Settings
from payday.models import BatchPipelineResult, BatchUploadItem, PipelineResult
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import PaydayRepository
from payday.storage import StorageService
from payday.transcription import TranscriptionService
from payday.upload import UploadService


class PaydayAppService:
    """Thin backend facade used by the Streamlit UI."""

    def __init__(self, settings: Settings) -> None:
        self.repository = PaydayRepository()
        persona_service = PersonaService()
        analysis_service = AnalysisService(
            adapter=HeuristicAnalysisAdapter(settings.llm),
            settings=settings.llm,
        )
        self.pipeline = PaydayPipeline(
            upload_service=UploadService(),
            transcription_service=TranscriptionService(settings.transcription),
            analysis_service=analysis_service,
            persona_service=persona_service,
            storage_service=StorageService(settings.supabase),
            repository=self.repository,
            sample_mode=settings.features.use_sample_mode,
        )

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        return self.pipeline.process_upload(filename=filename, content_type=content_type, data=data)

    def process_batch_uploads(self, items: list[BatchUploadItem]) -> BatchPipelineResult:
        return self.pipeline.process_batch_uploads(items)

    def list_results(self) -> list[PipelineResult]:
        return self.repository.list_results()
