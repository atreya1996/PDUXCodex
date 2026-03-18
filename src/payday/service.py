from __future__ import annotations

from payday.analysis import AnalysisService, build_analysis_adapter
from payday.config import Settings, validate_runtime_settings
from payday.models import BatchPipelineResult, BatchUploadItem, PipelineResult
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview, PaydayRepository
from payday.storage import StorageService
from payday.transcription import build_transcription_service
from payday.upload import UploadService


class PaydayAppService:
    """Thin backend facade used by the Streamlit UI."""

    def __init__(self, settings: Settings, repository: PaydayRepository | None = None) -> None:
        validate_runtime_settings(settings)
        self.repository = repository or PaydayRepository(database_path=settings.database.sqlite_path)
        persona_service = PersonaService()
        analysis_service = AnalysisService(
            adapter=self._build_analysis_adapter(settings),
            settings=settings.llm,
        )
        self.pipeline = PaydayPipeline(
            upload_service=UploadService(),
            transcription_service=build_transcription_service(
                settings.transcription,
                sample_mode=settings.features.use_sample_mode,
            ),
            analysis_service=analysis_service,
            persona_service=persona_service,
            storage_service=StorageService(settings.supabase),
            repository=self.repository,
            sample_mode=settings.features.use_sample_mode,
        )

    def _build_analysis_adapter(self, settings: Settings):
        return build_analysis_adapter(settings.llm, sample_mode=settings.features.use_sample_mode)

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        return self.pipeline.process_upload(filename=filename, content_type=content_type, data=data)

    def process_batch_uploads(self, items: list[BatchUploadItem]) -> BatchPipelineResult:
        return self.pipeline.process_batch_uploads(items)

    def list_results(self) -> list[PipelineResult]:
        return self.repository.list_results()

    def list_recent_interviews(self, *, limit: int = 100) -> list[DashboardInterviewRecord]:
        return self.repository.list_recent_interviews(limit=limit)

    def get_interview_detail(self, interview_id: str) -> DashboardInterviewRecord:
        return self.repository.get_dashboard_interview_detail(interview_id)

    def get_status_overview(self) -> DashboardStatusOverview:
        return self.repository.get_status_overview()
