from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar
from uuid import uuid4

from payday.analysis import DEFAULT_UNKNOWN_VALUE
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
        self._sync_interview(result, status=ProcessingStatus.PROCESSING)
        result.persisted = self.storage_service.store_asset(
            result.asset,
            sample_mode=self.sample_mode,
            interview_id=result.file_id,
        )
        if not result.persisted:
            raise RuntimeError(f"Failed to persist asset for {result.filename}.")
        result.status = ProcessingStatus.COMPLETED
        self._sync_interview(result, status=ProcessingStatus.COMPLETED)
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
        self._sync_interview(result, status=ProcessingStatus.PENDING)

        try:
            result.status = ProcessingStatus.PROCESSING
            result.current_stage = PipelineStage.UPLOAD
            self._sync_interview(result, status=ProcessingStatus.PROCESSING)
            result.asset = self.upload_audio(item)
            self._sync_interview(result, status=ProcessingStatus.PROCESSING)
            self.repository.save_result(result)

            result.current_stage = PipelineStage.TRANSCRIPTION
            result.transcript, transcription_attempts = self.transcribe_audio(result.asset)
            result.attempts[PipelineStage.TRANSCRIPTION.value] = transcription_attempts
            self._sync_interview(result, transcript=result.transcript.text, status=ProcessingStatus.PROCESSING)
            self.repository.save_result(result)

            result.current_stage = PipelineStage.ANALYSIS
            result.analysis, analysis_attempts = self.analyze_transcript(result.transcript)
            result.attempts[PipelineStage.ANALYSIS.value] = analysis_attempts
            self._persist_analysis_artifacts(result)
            self.repository.save_result(result)

            result.current_stage = PipelineStage.PERSONA
            result.persona = self.classify_persona(result.transcript, result.analysis)
            self._persist_insight(result)
            self.repository.save_result(result)

            return self.store_results(result)
        except Exception as exc:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(exc))
            self._sync_interview(result, status=ProcessingStatus.FAILED)
            self.repository.save_result(result)
            return result

    def _sync_interview(
        self,
        result: PipelineResult,
        *,
        transcript: str | None = None,
        status: ProcessingStatus | None = None,
    ) -> None:
        audio_url = self.storage_service.build_audio_path(result.file_id, result.filename)
        try:
            self.repository.update_interview(
                result.file_id,
                audio_url=audio_url,
                transcript=transcript,
                status=status.value if status is not None else None,
            )
        except KeyError:
            self.repository.create_interview(
                interview_id=result.file_id,
                audio_url=audio_url,
                transcript=transcript,
                status=status.value if status is not None else ProcessingStatus.PENDING.value,
            )

    def _persist_analysis_artifacts(self, result: PipelineResult) -> None:
        if result.analysis is None:
            raise ValueError("Cannot persist analysis artifacts without analysis output.")
        self.repository.upsert_structured_response(
            result.file_id,
            **self._structured_response_fields(result.analysis.structured_output),
        )

    def _persist_insight(self, result: PipelineResult) -> None:
        if result.analysis is None or result.persona is None:
            raise ValueError("Cannot persist insight without analysis and persona data.")
        self.repository.upsert_insight(
            result.file_id,
            summary=result.analysis.summary,
            key_quotes=list(result.analysis.evidence_quotes),
            persona=result.persona.persona_name,
            confidence_score=self._confidence_score(result.analysis.structured_output),
        )

    def _structured_response_fields(self, structured_output: dict[str, Any]) -> dict[str, Any]:
        smartphone_user = self._read_bool(structured_output, "participant_profile.smartphone_user")
        if smartphone_user is None:
            smartphone_user = self._smartphone_bool_from_flat(structured_output)

        has_bank_account = self._read_bool(structured_output, "participant_profile.has_bank_account")
        if has_bank_account is None:
            has_bank_account = self._bank_account_bool_from_flat(structured_output)

        return {
            "smartphone_user": smartphone_user,
            "has_bank_account": has_bank_account,
            "income_range": self._read_value(structured_output, "income_range"),
            "borrowing_history": self._read_value(structured_output, "borrowing_history"),
            "repayment_preference": self._read_value(structured_output, "repayment_preference"),
            "loan_interest": self._read_value(structured_output, "loan_interest"),
        }

    def _confidence_score(self, structured_output: dict[str, Any]) -> float:
        confidence_signals = structured_output.get("confidence_signals")
        if isinstance(confidence_signals, dict):
            observed = len(confidence_signals.get("observed_evidence", []))
            missing = len(confidence_signals.get("missing_or_unknown", []))
            total = observed + missing
            if total > 0:
                return round(observed / total, 2)

        candidate_fields = (
            "smartphone_usage",
            "bank_account_status",
            "income_range",
            "borrowing_history",
            "repayment_preference",
            "loan_interest",
            "summary",
        )
        observed = 0
        total = 0
        for field_name in candidate_fields:
            field = structured_output.get(field_name)
            if isinstance(field, dict):
                total += 1
                if field.get("status") == "observed":
                    observed += 1
        if total == 0:
            return 0.0
        return round(observed / total, 2)

    def _read_value(self, structured_output: dict[str, Any], key: str) -> str | None:
        value = structured_output.get(key)
        if isinstance(value, dict):
            candidate = value.get("value")
        else:
            candidate = value
        if candidate is None:
            return None
        normalized = str(candidate).strip()
        if not normalized or normalized == DEFAULT_UNKNOWN_VALUE:
            return None
        return normalized

    def _read_bool(self, structured_output: dict[str, Any], dotted_path: str) -> bool | None:
        current: Any = structured_output
        for key in dotted_path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        if isinstance(current, dict) and isinstance(current.get("value"), bool):
            return current["value"]
        if isinstance(current, bool):
            return current
        return None

    def _smartphone_bool_from_flat(self, structured_output: dict[str, Any]) -> bool | None:
        value = self._read_value(structured_output, "smartphone_usage")
        if value == "has_smartphone":
            return True
        if value == "no_smartphone":
            return False
        return None

    def _bank_account_bool_from_flat(self, structured_output: dict[str, Any]) -> bool | None:
        value = self._read_value(structured_output, "bank_account_status")
        if value == "has_bank_account":
            return True
        if value == "no_bank_account":
            return False
        return None

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
