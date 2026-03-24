from __future__ import annotations

import json
import logging
from hashlib import sha1
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable, Iterable
from typing import Any, TypeVar
from uuid import uuid4

from payday.analysis import (
    BANK_ACCOUNT_HAS_VALUE,
    BANK_ACCOUNT_NO_VALUE,
    DEFAULT_UNKNOWN_VALUE,
    SMARTPHONE_HAS_VALUE,
    SMARTPHONE_NO_VALUE,
    AnalysisSchemaError,
    AnalysisService,
    bank_account_user_from_analysis,
    get_income_display_value,
    get_analysis_value,
    smartphone_user_from_analysis,
)
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
from payday.repository import DashboardInterviewRecord, PaydayRepository
from payday.storage import StorageService
from payday.transcription import TranscriptionService
from payday.upload import UploadService

T = TypeVar("T")
logger = logging.getLogger(__name__)


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
        self._set_stage(
            result,
            stage=PipelineStage.STORAGE,
            status=ProcessingStatus.PROCESSING,
            message="storage started",
        )
        result.persisted = False
        try:
            result.persisted = self.storage_service.store_asset(
                result.asset,
                sample_mode=self.sample_mode,
                interview_id=result.file_id,
            )
            if not result.persisted:
                raise RuntimeError(f"Failed to persist asset for {result.filename}.")
        except Exception as exc:
            self._record_failure(result, stage=PipelineStage.STORAGE, error=str(exc), message="storage failed")
            raise
        result.status = ProcessingStatus.COMPLETED
        result.last_error = None
        self._log_stage("storage succeeded", result)
        self._sync_result(result)
        return result

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        item = BatchUploadItem(filename=filename, content_type=content_type, data=data)
        return self._process_item(item)

    def process_batch_uploads(self, items: Iterable[BatchUploadItem]) -> BatchPipelineResult:
        batch_items = list(items)
        if not 1 <= len(batch_items) <= 10:
            raise ValueError("Batch uploads must contain between 1 and 10 files.")
        results = [self._process_item(item) for item in batch_items]
        return BatchPipelineResult(batch_id=uuid4().hex, results=results)

    def reprocess_interview_detail(
        self,
        interview_id: str,
        *,
        transcript_text: str,
        extracted_json: str,
        transcript_changed: bool,
        structured_json_changed: bool,
    ) -> PipelineResult:
        detail = self.repository.get_dashboard_interview_detail(interview_id)
        result = self.repository.get_result(interview_id) or PipelineResult(
            file_id=detail.id,
            filename=detail.filename,
            persisted=True,
        )
        result.status = ProcessingStatus.PROCESSING
        result.current_stage = PipelineStage.ANALYSIS
        result.errors = []
        transcript = Transcript(
            text=transcript_text,
            provider="dashboard-edit",
            model="manual-edit",
            metadata={"interview_id": interview_id, "edit_source": "dashboard"},
        )
        result.transcript = transcript
        self.repository.save_result(result)
        self.repository.update_interview(
            interview_id,
            transcript=transcript_text,
            status=ProcessingStatus.PROCESSING.value,
            latest_stage=PipelineStage.ANALYSIS.value,
            last_error=None,
        )

        try:
            analysis_metadata = self._analysis_metadata()
            analysis = self._analysis_from_edit(
                transcript=transcript,
                extracted_json=extracted_json,
                detail=detail,
                transcript_changed=transcript_changed,
                structured_json_changed=structured_json_changed,
            )
            analysis.metrics.update(analysis_metadata)
            result.analysis = analysis
            result.current_stage = PipelineStage.PERSONA
            result.persona = self.classify_persona(transcript, analysis)
            result.status = ProcessingStatus.COMPLETED
            result.persisted = True
            self.repository.persist_reprocessed_interview(
                interview_id,
                transcript=transcript_text,
                status=ProcessingStatus.COMPLETED.value,
                latest_stage=result.current_stage.value,
                last_error=None,
                structured_response=self._structured_response_fields(result.analysis.structured_output),
                insight={
                    "summary": result.analysis.summary,
                    "key_quotes": list(result.analysis.evidence_quotes),
                    "persona": result.persona.persona_name,
                    "confidence_score": self._confidence_score(result.analysis.structured_output),
                },
                analysis_version=analysis_metadata["analysis_version"],
                analyzed_at=analysis_metadata["analyzed_at"],
            )
            self.repository.save_result(result)
            return result
        except Exception as exc:
            result.status = ProcessingStatus.FAILED
            result.last_error = str(exc)
            result.errors.append(str(exc))
            self.repository.update_interview(
                interview_id,
                transcript=transcript_text,
                status=ProcessingStatus.FAILED.value,
                latest_stage=result.current_stage.value,
                last_error=str(exc),
            )
            self.repository.save_result(result)
            raise

    def _process_item(self, item: BatchUploadItem) -> PipelineResult:
        result = PipelineResult(file_id=item.file_id, filename=item.filename)
        self.repository.save_result(result)
        interview = self.repository.create_interview(
            interview_id=item.file_id,
            audio_url=self.storage_service.build_audio_path(item.file_id, item.filename),
            status=ProcessingStatus.PENDING.value,
            latest_stage=result.current_stage.value,
        )

        preflight_error = self._preflight_upload_error(item)
        if preflight_error is not None:
            result.status = ProcessingStatus.FAILED
            result.current_stage = PipelineStage.UPLOAD
            self._record_failure(result, stage=PipelineStage.UPLOAD, error=preflight_error, message="upload rejected")
            return result

        try:
            self._set_stage(
                result,
                stage=PipelineStage.UPLOAD,
                status=ProcessingStatus.PROCESSING,
                message="upload accepted",
            )
            result.asset = self.upload_audio(item)

            self._set_stage(
                result,
                stage=PipelineStage.TRANSCRIPTION,
                status=ProcessingStatus.PROCESSING,
                message="transcription started",
            )
            result.transcript, transcription_attempts = self.transcribe_audio(result.asset)
            result.attempts[PipelineStage.TRANSCRIPTION.value] = transcription_attempts
            self._log_stage("transcription succeeded", result, attempts=transcription_attempts)
            self._sync_result(result)

            self._set_stage(
                result,
                stage=PipelineStage.ANALYSIS,
                status=ProcessingStatus.PROCESSING,
                message="analysis started",
            )
            result.analysis, analysis_attempts = self.analyze_transcript(result.transcript)
            result.attempts[PipelineStage.ANALYSIS.value] = analysis_attempts
            self._log_stage("analysis succeeded", result, attempts=analysis_attempts)
            self._sync_result(result)

            self._set_stage(result, stage=PipelineStage.PERSONA, message="persona classification completed")
            result.persona = self.classify_persona(result.transcript, result.analysis)
            self._persist_analysis_outputs(result)

            stored = self.store_results(result)
            self.repository.update_interview(interview.id, status=stored.status.value)
            return stored
        except Exception as exc:
            if result.last_error != str(exc):
                self._record_failure(result, stage=result.current_stage, error=str(exc))
            return result

    def _analysis_from_edit(
        self,
        *,
        transcript: Transcript,
        extracted_json: str,
        detail: DashboardInterviewRecord,
        transcript_changed: bool,
        structured_json_changed: bool,
    ) -> AnalysisResult:
        if structured_json_changed:
            try:
                payload = json.loads(extracted_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Extracted JSON is invalid: {exc.msg}") from exc
            return self._analysis_from_payload(payload, transcript)

        if transcript_changed:
            analysis, attempts = self.analyze_transcript(transcript)
            analysis.metrics["json_attempts"] = attempts
            analysis.metrics["edit_source"] = "transcript"
            return analysis

        return self._analysis_from_payload(self._dashboard_payload_from_record(detail), transcript)

    def _analysis_from_payload(self, payload: dict[str, Any], transcript: Transcript) -> AnalysisResult:
        normalized_payload = self._normalize_edit_payload(payload)
        try:
            structured_output = self.analysis_service._parse_and_validate(  # noqa: SLF001
                json.dumps(normalized_payload, ensure_ascii=False),
                transcript.text,
            )
        except AnalysisSchemaError as exc:
            raise ValueError(str(exc)) from exc

        return AnalysisResult(
            summary=structured_output["summary"]["value"],
            metrics={
                "word_count": len(transcript.text.split()),
                "character_count": len(transcript.text),
                "transcript_word_count": len(transcript.text.split()),
                "transcript_character_count": len(transcript.text),
                "analysis_provider": "dashboard-edit",
                "analysis_model": "manual-json",
                "json_attempts": 1,
                "edit_source": "structured_json",
            },
            structured_output=structured_output,
            evidence_quotes=structured_output["key_quotes"],
        )

    def _normalize_edit_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        expected_keys = set(self.analysis_service.expected_schema())
        if set(payload).issubset(expected_keys):
            return payload

        insight = payload.get("insight") if isinstance(payload.get("insight"), dict) else {}
        participant_profile = (
            payload.get("participant_profile") if isinstance(payload.get("participant_profile"), dict) else {}
        )
        return {
            "smartphone_usage": self._field_from_bool(
                value=self._bool_from_value(participant_profile.get("smartphone_user")),
                true_value=SMARTPHONE_HAS_VALUE,
                false_value=SMARTPHONE_NO_VALUE,
                fallback=payload.get("smartphone_usage"),
            ),
            "bank_account_status": self._field_from_bool(
                value=self._bool_from_value(participant_profile.get("has_bank_account")),
                true_value=BANK_ACCOUNT_HAS_VALUE,
                false_value=BANK_ACCOUNT_NO_VALUE,
                fallback=payload.get("bank_account_status"),
            ),
            "per_household_earnings": self._field_from_unknownable(payload.get("per_household_earnings")),
            "participant_personal_monthly_income": self._field_from_unknownable(
                payload.get("participant_personal_monthly_income") or payload.get("income_range")
            ),
            "total_household_monthly_income": self._field_from_unknownable(payload.get("total_household_monthly_income")),
            "borrowing_history": self._field_from_unknownable(
                payload.get("borrowing_history") or self._legacy_borrowing_history(payload)
            ),
            "repayment_preference": self._field_from_unknownable(payload.get("repayment_preference")),
            "loan_interest": self._field_from_unknownable(
                payload.get("loan_interest") or self._legacy_loan_interest(payload)
            ),
            "summary": self._field_from_unknownable(insight.get("summary") or payload.get("summary")),
            "key_quotes": self._string_list(insight.get("key_quotes") or payload.get("key_quotes")),
            "confidence_signals": self._confidence_signal_object(payload.get("confidence_signals")),
        }

    def _field_from_bool(
        self,
        *,
        value: bool | None,
        true_value: str,
        false_value: str,
        fallback: Any,
    ) -> Any:
        if value is True:
            return {"value": true_value, "status": "observed", "evidence_quotes": [], "notes": ""}
        if value is False:
            return {"value": false_value, "status": "observed", "evidence_quotes": [], "notes": ""}
        return self._field_from_unknownable(fallback)

    def _field_from_unknownable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        status = "unknown" if normalized.lower() == DEFAULT_UNKNOWN_VALUE else "observed"
        return {"value": normalized, "status": status, "evidence_quotes": [], "notes": ""}

    def _confidence_signal_object(self, value: Any) -> dict[str, list[str]] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("confidence_signals must be an object.")
        return {
            "observed_evidence": self._string_list(value.get("observed_evidence")),
            "missing_or_unknown": self._string_list(value.get("missing_or_unknown")),
        }

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Expected an array of strings.")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Expected an array of strings.")
            candidate = item.strip()
            if candidate and candidate not in normalized:
                normalized.append(candidate)
        return normalized

    def _bool_from_value(self, value: Any) -> bool | None:
        if isinstance(value, dict):
            candidate = value.get("value")
            if isinstance(candidate, bool):
                return candidate
        if isinstance(value, bool):
            return value
        return None

    def _legacy_borrowing_history(self, payload: dict[str, Any]) -> str | None:
        persona_signals = payload.get("persona_signals")
        if not isinstance(persona_signals, dict):
            return None
        if self._bool_from_value(persona_signals.get("cyclical_borrowing")) is True:
            return "has_borrowed"
        if self._bool_from_value(persona_signals.get("digital_borrowing")) is True:
            return "has_borrowed"
        if self._bool_from_value(persona_signals.get("self_reliance_non_borrowing")) is True:
            return "has_not_borrowed_recently"
        return None

    def _legacy_loan_interest(self, payload: dict[str, Any]) -> str | None:
        persona_signals = payload.get("persona_signals")
        if not isinstance(persona_signals, dict):
            return None
        if self._bool_from_value(persona_signals.get("trust_fear_barrier")) is True:
            return "fearful_or_uncertain"
        if self._bool_from_value(persona_signals.get("repayment_stress")) is True:
            return "fearful_or_uncertain"
        return None

    def _dashboard_payload_from_record(self, detail: DashboardInterviewRecord) -> dict[str, Any]:
        return {
            "audio_url": detail.audio_url,
            "participant_profile": {
                "smartphone_user": {"value": detail.smartphone_user},
                "has_bank_account": {"value": detail.has_bank_account},
            },
            "per_household_earnings": {"value": detail.per_household_earnings},
            "participant_personal_monthly_income": {"value": detail.participant_personal_monthly_income},
            "total_household_monthly_income": {"value": detail.total_household_monthly_income},
            "borrowing_history": {"value": detail.borrowing_history},
            "repayment_preference": {"value": detail.repayment_preference},
            "loan_interest": {"value": detail.loan_interest},
            "insight": {
                "summary": detail.summary,
                "persona": detail.persona,
                "confidence_score": detail.confidence_score,
                "key_quotes": detail.key_quotes,
            },
        }

    def _sync_result(self, result: PipelineResult) -> None:
        self.repository.save_result(result)
        self._sync_interview(
            result,
            transcript=result.transcript.text if result.transcript is not None else None,
            status=result.status,
            latest_stage=result.current_stage,
            last_error=result.last_error,
        )

    def _preflight_upload_error(self, item: BatchUploadItem) -> str | None:
        asset = UploadedAsset(
            filename=item.filename,
            content_type=item.content_type,
            size_bytes=len(item.data),
            file_id=item.file_id,
        )
        return self.transcription_service.validate_asset(asset)

    def _sync_interview(
        self,
        result: PipelineResult,
        *,
        transcript: str | None = None,
        status: ProcessingStatus | None = None,
        latest_stage: PipelineStage | None = None,
        last_error: str | object = None,
    ) -> None:
        audio_url = self.storage_service.build_audio_path(result.file_id, result.filename)
        try:
            self.repository.update_interview(
                result.file_id,
                audio_url=audio_url,
                transcript=transcript,
                status=status.value if status is not None else None,
                latest_stage=latest_stage.value if latest_stage is not None else None,
                last_error=last_error,
            )
        except KeyError:
            self.repository.create_interview(
                interview_id=result.file_id,
                audio_url=audio_url,
                transcript=transcript,
                status=status.value if status is not None else ProcessingStatus.PENDING.value,
                latest_stage=latest_stage.value if latest_stage is not None else result.current_stage.value,
                last_error=result.last_error if last_error is not None else None,
            )

    def _set_stage(
        self,
        result: PipelineResult,
        *,
        stage: PipelineStage,
        status: ProcessingStatus | None = None,
        message: str | None = None,
    ) -> None:
        result.current_stage = stage
        if status is not None:
            result.status = status
        if status is not ProcessingStatus.FAILED:
            result.last_error = None
        if message:
            self._log_stage(message, result)
        self._sync_result(result)

    def _record_failure(
        self,
        result: PipelineResult,
        *,
        stage: PipelineStage,
        error: str,
        message: str | None = None,
    ) -> None:
        result.current_stage = stage
        result.status = ProcessingStatus.FAILED
        result.last_error = error
        if not result.errors or result.errors[-1] != error:
            result.errors.append(error)
        if message:
            self._log_stage(message, result)
        self._sync_result(result)

    def _log_stage(self, message: str, result: PipelineResult, *, attempts: int | None = None) -> None:
        payload = {
            "file_id": result.file_id,
            "filename": result.filename,
            "stage": result.current_stage.value,
            "status": result.status.value,
        }
        if attempts is not None:
            payload["attempts"] = attempts
        if result.last_error:
            payload["last_error"] = result.last_error
        logger.info("%s: %s", message, payload)

    def _persist_analysis_outputs(self, result: PipelineResult) -> None:
        if result.analysis is None or result.persona is None:
            raise ValueError("Cannot persist analysis outputs without both analysis and persona results.")

        structured_output = result.analysis.structured_output
        metadata = self._analysis_metadata()
        result.analysis.metrics.update(metadata)
        self.repository.upsert_structured_response(
            result.file_id,
            **self._structured_response_fields(structured_output),
            analysis_version=metadata["analysis_version"],
            analyzed_at=metadata["analyzed_at"],
        )
        self.repository.upsert_insight(
            result.file_id,
            summary=result.analysis.summary,
            key_quotes=list(result.analysis.evidence_quotes),
            persona=result.persona.persona_name,
            confidence_score=self._confidence_score(structured_output),
            analysis_version=metadata["analysis_version"],
            analyzed_at=metadata["analyzed_at"],
        )

    def _analysis_metadata(self) -> dict[str, str]:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "analysis_prompt.md"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        prompt_hash = sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
        provider = self.analysis_service.adapter.provider_name
        model = self.analysis_service.adapter.model_name
        return {
            "analysis_version": f"{provider}:{model}:{prompt_hash}",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _structured_response_fields(self, structured_output: dict[str, Any]) -> dict[str, Any]:
        smartphone_user = smartphone_user_from_analysis(structured_output)
        has_bank_account = bank_account_user_from_analysis(structured_output)
        return {
            "smartphone_user": smartphone_user,
            "has_bank_account": has_bank_account,
            "per_household_earnings": self._read_value(structured_output, "per_household_earnings"),
            "participant_personal_monthly_income": self._read_value(structured_output, "participant_personal_monthly_income"),
            "total_household_monthly_income": self._read_value(structured_output, "total_household_monthly_income"),
            "income_range": self._preferred_dashboard_income(structured_output),
            "borrowing_history": self._read_value(structured_output, "borrowing_history"),
            "repayment_preference": self._read_value(structured_output, "repayment_preference"),
            "loan_interest": self._read_value(structured_output, "loan_interest"),
            "segmented_dialogue": self._read_segmented_dialogue(structured_output),
        }

    @staticmethod
    def _read_segmented_dialogue(structured_output: dict[str, Any]) -> list[dict[str, Any]]:
        segmented_dialogue = structured_output.get("segmented_dialogue")
        if isinstance(segmented_dialogue, list):
            return [turn for turn in segmented_dialogue if isinstance(turn, dict)]
        return []

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
            "per_household_earnings",
            "participant_personal_monthly_income",
            "total_household_monthly_income",
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


    def _preferred_dashboard_income(self, structured_output: dict[str, Any]) -> str | None:
        corrected_value = get_income_display_value(structured_output)
        if corrected_value is not None:
            return corrected_value
        return self._read_value(structured_output, "income_range")

    def _read_value(self, structured_output: dict[str, Any], key: str) -> str | None:
        normalized = get_analysis_value(structured_output, key).strip()
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


    def _set_stage(
        self,
        result: PipelineResult,
        *,
        stage: PipelineStage,
        status: ProcessingStatus | None = None,
        message: str | None = None,
    ) -> None:
        result.current_stage = stage
        if status is not None:
            result.status = status
        if message:
            logger.info("%s: %s (%s)", result.file_id, message, stage.value)

    def _record_failure(
        self,
        result: PipelineResult,
        *,
        stage: PipelineStage,
        error: str,
        message: str | None = None,
    ) -> None:
        result.current_stage = stage
        result.status = ProcessingStatus.FAILED
        result.last_error = error
        if not result.errors or result.errors[-1] != error:
            result.errors.append(error)
        logger.error("%s: %s (%s)", result.file_id, message or error, stage.value)
        self._sync_result(result)

    def _log_stage(self, message: str, result: PipelineResult, *, attempts: int | None = None) -> None:
        suffix = f" after {attempts} attempt(s)" if attempts is not None else ""
        logger.info("%s: %s%s", result.file_id, message, suffix)

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
