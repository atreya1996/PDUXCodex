from __future__ import annotations

import logging

from payday.analysis import AnalysisService, build_analysis_adapter
from payday.config import Settings, validate_runtime_settings
from payday.models import BatchPipelineResult, BatchUploadItem, PipelineResult
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview, PaydayRepository
from payday.storage import StorageService
from payday.transcription import build_transcription_service
from payday.upload import UploadService

logger = logging.getLogger(__name__)


class PaydayAppService:
    """Thin backend facade used by the Streamlit UI."""

    def __init__(self, settings: Settings, repository: PaydayRepository | None = None) -> None:
        validate_runtime_settings(settings)
        self.settings = settings
        self.repository = repository or PaydayRepository(database_path=settings.database.sqlite_path)
        persona_service = PersonaService()
        analysis_service = AnalysisService(
            adapter=build_analysis_adapter(settings.llm, sample_mode=settings.features.use_sample_mode),
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
        logger.info("PayDay runtime configuration loaded: %s", self.runtime_summary())

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        return self.pipeline.process_upload(filename=filename, content_type=content_type, data=data)

    def process_batch_uploads(self, items: list[BatchUploadItem]) -> BatchPipelineResult:
        return self.pipeline.process_batch_uploads(items)

    def runtime_summary(self) -> dict[str, object]:
        return {
            "sample_mode": self.settings.features.use_sample_mode,
            "transcription": {
                "provider": self.settings.transcription.provider,
                "model": self.settings.transcription.model,
                "required_key_present": bool(self.settings.transcription.api_key),
            },
            "analysis": {
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "required_key_present": bool(self.settings.llm.api_key),
            },
        }

    def list_results(self) -> list[PipelineResult]:
        return self.repository.list_results()

    def list_recent_interviews(self, *, limit: int = 100) -> list[DashboardInterviewRecord]:
        return self.repository.list_recent_interviews(limit=limit)

    def get_interview_detail(self, interview_id: str) -> DashboardInterviewRecord:
        return self.repository.get_dashboard_interview_detail(interview_id)

    def get_status_overview(self) -> DashboardStatusOverview:
        return self.repository.get_status_overview()

    def save_interview_edits(
        self,
        interview_id: str,
        *,
        transcript: str,
        extracted_json: str,
        transcript_changed: bool,
        structured_json_changed: bool,
    ) -> DashboardInterviewRecord:
        self.pipeline.reprocess_interview_detail(
            interview_id,
            transcript_text=transcript,
            extracted_json=extracted_json,
            transcript_changed=transcript_changed,
            structured_json_changed=structured_json_changed,
        )
        return self.repository.get_dashboard_interview_detail(interview_id)

    def delete_interview(self, interview_id: str) -> bool:
        return self.repository.delete_interview(interview_id)
