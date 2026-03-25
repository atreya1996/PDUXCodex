from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from payday.analysis import AnalysisService, build_analysis_adapter
from payday.config import Settings, resolve_runtime_commit_sha, validate_runtime_settings
from payday.models import BatchPipelineResult, BatchUploadItem, PipelineResult
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview, JobRecord, PaydayRepository
from payday.storage import StorageService
from payday.transcription import build_transcription_service
from payday.upload import UploadService

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_POLL_INTERVAL_SECONDS = 1.0


class PaydayAppService:
    """Thin backend facade used by the Streamlit UI."""

    def __init__(
        self,
        settings: Settings,
        repository: PaydayRepository | None = None,
        *,
        start_worker: bool = True,
    ) -> None:
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
        self._worker_stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        if start_worker:
            self._worker_thread = threading.Thread(target=self._worker_loop, name="payday-job-worker", daemon=True)
            self._worker_thread.start()
        logger.info("PayDay runtime diagnostics: %s", self.runtime_diagnostics())
        logger.info("PayDay runtime configuration loaded: %s", self.runtime_summary())

    def process_upload(self, filename: str, content_type: str, data: bytes) -> PipelineResult:
        return self.pipeline.process_upload(filename=filename, content_type=content_type, data=data)

    def process_batch_uploads(self, items: list[BatchUploadItem]) -> BatchPipelineResult:
        return self.pipeline.process_batch_uploads(items)

    def enqueue_batch_uploads(self, items: list[BatchUploadItem]) -> dict[str, object]:
        if not 1 <= len(items) <= 10:
            raise ValueError("Batch uploads must contain between 1 and 10 files.")
        batch_id = uuid4().hex
        job_ids: list[str] = []
        for item in items:
            self.repository.create_interview(
                interview_id=item.file_id,
                file_path=self.pipeline.storage_service.build_file_path(item.file_id, item.filename),
                status="pending",
                latest_stage="upload",
            )
            job = self.repository.create_job(
                batch_id=batch_id,
                interview_id=item.file_id,
                filename=item.filename,
                content_type=item.content_type,
                payload=item.data,
            )
            job_ids.append(job.id)
        return {"batch_id": batch_id, "job_ids": job_ids, "count": len(job_ids)}

    def list_jobs(self, *, limit: int = 200) -> list[JobRecord]:
        return self.repository.list_jobs(limit=limit)

    def cancel_job(self, job_id: str) -> JobRecord:
        return self.repository.cancel_job(job_id)

    def retry_job(self, job_id: str) -> JobRecord:
        return self.repository.retry_job(job_id)

    def cancel_batch_jobs(self, batch_id: str) -> int:
        return self.repository.cancel_batch(batch_id)

    def retry_batch_jobs(self, batch_id: str) -> int:
        return self.repository.retry_batch(batch_id)

    def run_worker_cycle(self, *, max_jobs: int = 1) -> int:
        processed = 0
        for _ in range(max_jobs):
            next_job = self.repository.get_next_pending_job()
            if next_job is None:
                break
            claimed = self.repository.claim_job(next_job.id)
            if claimed is None:
                continue
            if claimed.status == "cancelled":
                processed += 1
                continue
            payload = self.repository.get_job_payload(claimed.id)
            try:
                result = self.pipeline.process_upload(
                    filename=claimed.filename,
                    content_type=claimed.content_type,
                    data=payload,
                )
            except Exception as exc:  # noqa: BLE001
                self.repository.fail_job(claimed.id, error_message=str(exc))
                processed += 1
                continue
            if result.status.value == "completed":
                self.repository.complete_job(claimed.id)
            else:
                error = result.last_error or "Pipeline failed without a detailed error."
                self.repository.fail_job(claimed.id, error_message=error)
            processed += 1
        return processed

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

    def shutdown(self) -> None:
        self._worker_stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

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

    def _worker_loop(self) -> None:
        while not self._worker_stop_event.is_set():
            processed = self.run_worker_cycle(max_jobs=1)
            if processed == 0:
                time.sleep(WORKER_POLL_INTERVAL_SECONDS)
