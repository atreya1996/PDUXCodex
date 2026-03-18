from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar
from uuid import uuid4

from payday.analysis import AnalysisService
from payday.models import (
    AnalysisResult,
    BatchPipelineResult,
    BatchUploadItem,
    PersonaClassification,
    PipelineResult,
    PipelineStage,
    ProcessingStatus,
    Transcript,
    UploadedAsset,
)
from payday.personas import PersonaService
from payday.repository import PaydayRepository
from payday.storage import StorageService
from payday.transcription import TranscriptionService
from payday.upload import UploadService

T = TypeVar("T")


class PaydayPipeline:
    """Coordinates resilient per-file processing for uploads, transcription, analysis, and storage."""

    def __init__(
        self,
        upload_service: UploadService,
        transcription_service: TranscriptionService,
        analysis_service: AnalysisService,
        persona_service: PersonaService,
        storage_service: StorageService,
        repository: PaydayRepository,
        sample_mode: bool = True,
        max_retries: int = 2,
    ) -> None:
        self.upload_service = upload_service
        self.transcription_service = transcription_service
        self.analysis_service = analysis_service
        self.persona_service = persona_service
        self.storage_service = storage_service
        self.repository = repository
        self.sample_mode = sample_mode
        self.max_retries = max_retries

    def upload_audio(self, item: BatchUploadItem) -> UploadedAsset:
        return self.upload_service.create_asset(
            filename=item.filename,
            content_type=item.content_type,
            data=item.data,
            file_id=item.file_id,
        )

    def transcribe_audio(self, asset: UploadedAsset) -> tuple[Transcript, int]:
        return self._run_with_retries(
            stage=PipelineStage.TRANSCRIPTION,
            operation=lambda: self.transcription_service.transcribe(asset, sample_mode=self.sample_mode),
        )

    def analyze_transcript(self, transcript: Transcript) -> tuple[AnalysisResult, int]:
        return self._run_with_retries(
            stage=PipelineStage.ANALYSIS,
            operation=lambda: self.analysis_service.analyze(transcript),
        )

    def classify_persona(
        self,
        transcript: Transcript,
        analysis: AnalysisResult,
    ) -> PersonaClassification:
        return self.persona_service.classify(transcript, analysis)

    def store_results(self, result: PipelineResult) -> PipelineResult:
        if result.asset is None:
            raise ValueError("Cannot store results without an uploaded asset.")
        result.current_stage = PipelineStage.STORAGE
        result.status = ProcessingStatus.PROCESSING
        result.persisted = self.storage_service.store_asset(result.asset, sample_mode=self.sample_mode)
        if not result.persisted:
            raise RuntimeError(f"Failed to persist asset for {result.filename}.")
        result.status = ProcessingStatus.COMPLETED
        self.repository.save_result(result)
        return result

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        item = BatchUploadItem(filename=filename, content_type=content_type, data=data)
        return self._process_item(item)

    def process_batch_uploads(self, items: Iterable[BatchUploadItem]) -> BatchPipelineResult:
        batch_items = list(items)
        if not 5 <= len(batch_items) <= 10:
            raise ValueError("Batch uploads must contain between 5 and 10 files.")
        results = [self._process_item(item) for item in batch_items]
        return BatchPipelineResult(batch_id=uuid4().hex, results=results)

    def _process_item(self, item: BatchUploadItem) -> PipelineResult:
        result = PipelineResult(file_id=item.file_id, filename=item.filename)
        self.repository.save_result(result)

        try:
            result.status = ProcessingStatus.PROCESSING
            result.current_stage = PipelineStage.UPLOAD
            result.asset = self.upload_audio(item)
            self.repository.save_result(result)

            result.current_stage = PipelineStage.TRANSCRIPTION
            result.transcript, transcription_attempts = self.transcribe_audio(result.asset)
            result.attempts[PipelineStage.TRANSCRIPTION.value] = transcription_attempts
            self.repository.save_result(result)

            result.current_stage = PipelineStage.ANALYSIS
            result.analysis, analysis_attempts = self.analyze_transcript(result.transcript)
            result.attempts[PipelineStage.ANALYSIS.value] = analysis_attempts
            self.repository.save_result(result)

            result.current_stage = PipelineStage.PERSONA
            result.persona = self.classify_persona(result.transcript, result.analysis)
            self.repository.save_result(result)

            return self.store_results(result)
        except Exception as exc:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(exc))
            self.repository.save_result(result)
            return result

    def _run_with_retries(self, stage: PipelineStage, operation: Callable[[], T]) -> tuple[T, int]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return operation(), attempt
            except Exception as exc:  # pragma: no cover - exercised through tests
                last_error = exc
        if last_error is None:
            raise RuntimeError(f"{stage.value} failed without an exception.")
        raise RuntimeError(
            f"{stage.value} failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error
