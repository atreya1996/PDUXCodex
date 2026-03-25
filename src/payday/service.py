from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from payday.analysis import AnalysisService, build_analysis_adapter
from payday.config import Settings, resolve_runtime_commit_sha, validate_runtime_settings
from payday.models import BatchPipelineResult, BatchUploadItem, PipelineResult
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview, PaydayRepository, ProcessingEventRecord
from payday.storage import StorageService
from payday.transcription import build_transcription_service
from payday.upload import UploadService

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


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
            storage_service=StorageService(settings.storage.uploads_root),
            repository=self.repository,
            sample_mode=settings.features.use_sample_mode,
        )
        logger.info("PayDay runtime diagnostics: %s", self.runtime_diagnostics())
        logger.info("PayDay runtime configuration loaded: %s", self.runtime_summary())

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

    def list_processing_events(
        self,
        *,
        file_ids: list[str] | None = None,
        limit: int = 200,
    ) -> list[ProcessingEventRecord]:
        return self.repository.list_processing_events(file_ids=file_ids, limit=limit)

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

    def reprocess_interview(self, interview_id: str) -> DashboardInterviewRecord:
        detail = self.repository.get_dashboard_interview_detail(interview_id)
        transcript_text = (detail.transcript or "").strip()
        if not transcript_text:
            raise ValueError("Cannot reprocess an interview without a saved transcript.")

        self.pipeline.reprocess_interview_detail(
            interview_id,
            transcript_text=transcript_text,
            extracted_json="{}",
            transcript_changed=True,
            structured_json_changed=False,
        )
        return self.repository.get_dashboard_interview_detail(interview_id)

    def reanalyze_interviews(self, interview_ids: list[str]) -> list[DashboardInterviewRecord]:
        refreshed: list[DashboardInterviewRecord] = []
        for interview_id in interview_ids:
            refreshed.append(self.reprocess_interview(interview_id))
        return refreshed

    def _reprocess_ids(self, interview_ids: list[str]) -> dict[str, object]:
        failed: dict[str, str] = {}
        reprocessed_ids: list[str] = []
        for interview_id in interview_ids:
            try:
                self.reprocess_interview(interview_id)
            except Exception as exc:  # noqa: BLE001
                failed[interview_id] = str(exc)
            else:
                reprocessed_ids.append(interview_id)
        return {
            "count": len(interview_ids),
            "reprocessed_ids": reprocessed_ids,
            "failed": failed,
        }

    def reprocess_failed_or_malformed_interviews(self, *, limit: int = 500) -> dict[str, object]:
        target_ids = self.repository.list_failed_or_malformed_interview_ids(limit=limit)
        summary = self._reprocess_ids(target_ids)
        summary["failed_or_malformed_count"] = summary.pop("count")
        return summary

    def list_failed_or_malformed_interview_ids(self, *, limit: int = 500) -> list[str]:
        return self.repository.list_failed_or_malformed_interview_ids(limit=limit)

    def reanalyze_all_interviews(self) -> list[DashboardInterviewRecord]:
        interview_ids = [record.id for record in self.repository.list_recent_interviews(limit=10_000)]
        return self.reanalyze_interviews(interview_ids)

    def reprocess_stale_interviews(self, *, limit: int = 500) -> dict[str, object]:
        stale_ids = self.repository.list_stale_interview_ids(limit=limit)
        summary = self._reprocess_ids(stale_ids)
        summary["stale_count"] = summary.pop("count")
        return summary

    def delete_stale_corrupted_interviews(self, *, limit: int = 500) -> dict[str, object]:
        stale_corrupted_ids = self.repository.list_stale_corrupted_interview_ids(limit=limit)
        deleted_ids: list[str] = []
        failed: dict[str, str] = {}
        for interview_id in stale_corrupted_ids:
            try:
                deleted = self.delete_interview(interview_id)
            except Exception as exc:  # noqa: BLE001
                failed[interview_id] = str(exc)
                continue
            if deleted:
                deleted_ids.append(interview_id)
            else:
                failed[interview_id] = "Interview was already deleted."
        return {
            "stale_corrupted_count": len(stale_corrupted_ids),
            "deleted_ids": deleted_ids,
            "failed": failed,
        }

    def list_stale_corrupted_interview_ids(self, *, limit: int = 500) -> list[str]:
        return self.repository.list_stale_corrupted_interview_ids(limit=limit)

    def delete_interview(self, interview_id: str) -> bool:
        try:
            interview = self.repository.get_interview(interview_id)
        except KeyError:
            return False

        deleted = self.repository.delete_interview(interview_id)
        if not deleted:
            return False

        self._delete_stored_audio_if_needed(interview_id=interview.id, file_path=interview.file_path)
        return True

    def runtime_summary(self) -> dict[str, object]:
        return {
            "sample_mode": self.settings.features.use_sample_mode,
            "analysis_provider": self.pipeline.analysis_service.adapter.provider_name,
            "analysis_model": self.pipeline.analysis_service.adapter.model_name,
            "transcription_provider": self.pipeline.transcription_service.settings.provider,
            "transcription_model": self.pipeline.transcription_service.settings.model,
            "database_path": self.repository.database_path,
            "runtime_commit_sha": resolve_runtime_commit_sha(),
        }

    def runtime_diagnostics(self) -> dict[str, object]:
        git_branch, git_commit = self._resolve_git_metadata()
        return {
            "git_branch": git_branch,
            "git_commit": git_commit,
            "database_path": self.repository.database_path,
            "sample_mode": self.settings.features.use_sample_mode,
        }

    def _resolve_git_metadata(self) -> tuple[str, str]:
        branch = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        commit = self._run_git_command(["rev-parse", "--short", "HEAD"])
        return branch or "unknown", commit or "unknown"

    def _run_git_command(self, args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    def _delete_stored_audio_if_needed(self, *, interview_id: str, file_path: str) -> None:
        if self.settings.features.use_sample_mode:
            return
        if not file_path.strip():
            return

        try:
            self.pipeline.storage_service.delete_asset(file_path, sample_mode=False)
        except FileNotFoundError:
            logger.warning("Stored audio for interview %s was already missing at %s.", interview_id, file_path)
        except Exception:
            logger.exception("Failed to delete stored audio for interview %s at %s.", interview_id, file_path)
