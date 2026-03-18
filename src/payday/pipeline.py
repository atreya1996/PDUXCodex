from __future__ import annotations

from payday.analysis import AnalysisService
from payday.models import PipelineResult
from payday.repository import PaydayRepository
from payday.storage import StorageService
from payday.transcription import TranscriptionService
from payday.upload import UploadService


class PaydayPipeline:
    """Application service orchestrating upload, transcription, analysis, and persistence."""

    def __init__(
        self,
        upload_service: UploadService,
        transcription_service: TranscriptionService,
        analysis_service: AnalysisService,
        storage_service: StorageService,
        repository: PaydayRepository,
        sample_mode: bool = True,
    ) -> None:
        self.upload_service = upload_service
        self.transcription_service = transcription_service
        self.analysis_service = analysis_service
        self.storage_service = storage_service
        self.repository = repository
        self.sample_mode = sample_mode

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        asset = self.upload_service.create_asset(filename=filename, content_type=content_type, data=data)
        interview = self.repository.create_interview(
            audio_url=self.storage_service.build_audio_path(interview_id="pending", filename=filename),
            status="processing",
        )
        audio_url = self.storage_service.upload_audio(
            asset=asset,
            interview_id=interview.id,
            sample_mode=self.sample_mode,
        )
        self.repository.update_interview(interview.id, audio_url=audio_url)

        transcript = self.transcription_service.transcribe(asset, sample_mode=self.sample_mode)
        analysis = self.analysis_service.analyze(transcript)
        persisted = bool(audio_url)

        self.repository.update_interview(
            interview.id,
            transcript=transcript.text,
            status="completed" if persisted else "processed",
        )
        self.repository.upsert_insight(
            interview.id,
            summary=analysis.summary,
            key_quotes=[],
            persona=analysis.persona_matches[0],
            confidence_score=1.0 if analysis.persona_matches else 0.0,
        )

        result = PipelineResult(
            asset=asset,
            transcript=transcript,
            analysis=analysis,
            persisted=persisted,
        )
        self.repository.save_result(result)
        return result
