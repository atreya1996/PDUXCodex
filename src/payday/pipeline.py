from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, TypeVar
from uuid import uuid4

from payday.analysis import DEFAULT_UNKNOWN_VALUE
from payday.analysis import AnalysisSchemaError, AnalysisService
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
        result.persisted = False
        result.persisted = self.storage_service.store_asset(
            result.asset,
            sample_mode=self.sample_mode,
            interview_id=result.file_id,
        )
        if not result.persisted:
            raise RuntimeError(f"Failed to persist asset for {result.filename}.")
        result.status = ProcessingStatus.COMPLETED
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
        )

        try:
            analysis = self._analysis_from_edit(
                transcript=transcript,
                extracted_json=extracted_json,
                detail=detail,
                transcript_changed=transcript_changed,
                structured_json_changed=structured_json_changed,
            )
            analysis.structured_output = self._augment_structured_output(analysis.structured_output)
            result.analysis = analysis
            result.current_stage = PipelineStage.PERSONA
            result.persona = self.classify_persona(transcript, analysis)
            result.status = ProcessingStatus.COMPLETED
            result.persisted = True
            self._sync_result(result)
            self._persist_analysis_outputs(result)
            self.repository.update_interview(
                interview_id,
                transcript=transcript_text,
                status=ProcessingStatus.COMPLETED.value,
            )
            return result
        except Exception as exc:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(exc))
            self._sync_result(result)
            self.repository.update_interview(
                interview_id,
                transcript=transcript_text,
                status=ProcessingStatus.FAILED.value,
            )
            raise

    def _process_item(self, item: BatchUploadItem) -> PipelineResult:
        result = PipelineResult(file_id=item.file_id, filename=item.filename)
        self.repository.save_result(result)
        interview = self.repository.create_interview(
            interview_id=item.file_id,
            audio_url=self.storage_service.build_audio_path(item.file_id, item.filename),
            status=ProcessingStatus.PENDING.value,
        )

        preflight_error = self._preflight_upload_error(item)
        if preflight_error is not None:
            result.status = ProcessingStatus.FAILED
            result.current_stage = PipelineStage.UPLOAD
            result.errors.append(preflight_error)
            self.repository.save_result(result)
            self.repository.update_interview(interview.id, status=result.status.value)
            return result

        try:
            result.status = ProcessingStatus.PROCESSING
            result.current_stage = PipelineStage.UPLOAD
            self._sync_interview(result, status=ProcessingStatus.PROCESSING)
            result.asset = self.upload_audio(item)
            self.repository.save_result(result)
            self.repository.update_interview(interview.id, status=result.status.value)

            result.current_stage = PipelineStage.TRANSCRIPTION
            result.transcript, transcription_attempts = self.transcribe_audio(result.asset)
            result.attempts[PipelineStage.TRANSCRIPTION.value] = transcription_attempts
            self.repository.save_result(result)
            self.repository.update_interview(
                interview.id,
                transcript=result.transcript.text,
                status=result.status.value,
            )

            result.current_stage = PipelineStage.ANALYSIS
            result.analysis, analysis_attempts = self.analyze_transcript(result.transcript)
            result.analysis.structured_output = self._augment_structured_output(result.analysis.structured_output)
            result.attempts[PipelineStage.ANALYSIS.value] = analysis_attempts
            self._sync_result(result)

            result.current_stage = PipelineStage.PERSONA
            result.persona = self.classify_persona(result.transcript, result.analysis)
            self.repository.save_result(result)
            self._persist_analysis_outputs(result)

            stored = self.store_results(result)
            self.repository.update_interview(interview.id, status=stored.status.value)
            return stored
        except Exception as exc:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(exc))
            self.repository.save_result(result)
            self.repository.update_interview(
                interview.id,
                transcript=result.transcript.text if result.transcript is not None else None,
                status=result.status.value,
            )
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

        structured_output = self._augment_structured_output(structured_output)
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
                true_value="has_smartphone",
                false_value="no_smartphone",
                fallback=payload.get("smartphone_usage"),
            ),
            "bank_account_status": self._field_from_bool(
                value=self._bool_from_value(participant_profile.get("has_bank_account")),
                true_value="has_bank_account",
                false_value="no_bank_account",
                fallback=payload.get("bank_account_status"),
            ),
            "income_range": self._field_from_unknownable(payload.get("income_range")),
            "borrowing_history": self._field_from_unknownable(payload.get("borrowing_history")),
            "repayment_preference": self._field_from_unknownable(payload.get("repayment_preference")),
            "loan_interest": self._field_from_unknownable(payload.get("loan_interest")),
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

    def _dashboard_payload_from_record(self, detail: DashboardInterviewRecord) -> dict[str, Any]:
        return {
            "audio_url": detail.audio_url,
            "participant_profile": {
                "smartphone_user": {"value": detail.smartphone_user},
                "has_bank_account": {"value": detail.has_bank_account},
            },
            "income_range": {"value": detail.income_range},
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

    def _persist_analysis_outputs(self, result: PipelineResult) -> None:
        if result.analysis is None or result.persona is None:
            raise ValueError("Cannot persist analysis outputs without both analysis and persona results.")

        structured_output = result.analysis.structured_output
        self.repository.upsert_structured_response(
            result.file_id,
            **self._structured_response_fields(structured_output),
        )
        self.repository.upsert_insight(
            result.file_id,
            summary=result.analysis.summary,
            key_quotes=list(result.analysis.evidence_quotes),
            persona=result.persona.persona_name,
            confidence_score=self._confidence_score(structured_output),
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

    def _augment_structured_output(self, structured_output: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(structured_output)
        participant_profile = dict(normalized.get("participant_profile", {})) if isinstance(normalized.get("participant_profile"), dict) else {}

        smartphone_user = self._read_bool(normalized, "participant_profile.smartphone_user")
        if smartphone_user is None:
            smartphone_user = self._smartphone_bool_from_flat(normalized)
        if smartphone_user is not None:
            participant_profile["smartphone_user"] = {
                "value": smartphone_user,
                "evidence": self._read_evidence_quotes(normalized, "smartphone_usage"),
            }

        has_bank_account = self._read_bool(normalized, "participant_profile.has_bank_account")
        if has_bank_account is None:
            has_bank_account = self._bank_account_bool_from_flat(normalized)
        if has_bank_account is not None:
            participant_profile["has_bank_account"] = {
                "value": has_bank_account,
                "evidence": self._read_evidence_quotes(normalized, "bank_account_status"),
            }

        if participant_profile:
            normalized["participant_profile"] = participant_profile
        return normalized

    def _read_evidence_quotes(self, structured_output: dict[str, Any], key: str) -> list[str]:
        value = structured_output.get(key)
        if isinstance(value, dict):
            evidence_quotes = value.get("evidence_quotes")
            if isinstance(evidence_quotes, list):
                return [str(item) for item in evidence_quotes if item]
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
