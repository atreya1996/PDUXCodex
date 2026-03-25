from __future__ import annotations

from payday.config import DatabaseSettings, FeatureFlags, LLMSettings, Settings, StorageSettings, SupabaseSettings, TranscriptionSettings
from payday.models import BatchUploadItem
from payday.service import PaydayAppService


def _settings() -> Settings:
    return Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(),
        features=FeatureFlags(use_sample_mode=True, enable_analysis=True),
        storage=StorageSettings(uploads_root="./data/uploads"),
    )


def test_enqueue_and_worker_cycle_processes_pending_job() -> None:
    service = PaydayAppService(_settings(), start_worker=False)
    item = BatchUploadItem(filename="demo.txt", content_type="text/plain", data=b"hello from queue")

    enqueue_result = service.enqueue_batch_uploads([item])

    assert enqueue_result["count"] == 1
    jobs = service.list_jobs(limit=10)
    assert len(jobs) == 1
    assert jobs[0].status == "pending"

    processed = service.run_worker_cycle(max_jobs=1)

    assert processed == 1
    refreshed_jobs = service.list_jobs(limit=10)
    assert refreshed_jobs[0].attempts == 1
    assert refreshed_jobs[0].status in {"pending", "completed", "failed"}
    assert service.get_interview_detail(item.file_id).id == item.file_id
    service.shutdown()


def test_cancel_and_retry_batch_controls_job_state() -> None:
    service = PaydayAppService(_settings(), start_worker=False)
    item = BatchUploadItem(filename="cancel.txt", content_type="text/plain", data=b"queued")

    enqueue_result = service.enqueue_batch_uploads([item])
    batch_id = str(enqueue_result["batch_id"])

    cancelled = service.cancel_batch_jobs(batch_id)
    assert cancelled == 1
    job = service.list_jobs(limit=10)[0]
    assert job.status == "cancelled"

    retried = service.retry_batch_jobs(batch_id)
    assert retried == 1
    retried_job = service.list_jobs(limit=10)[0]
    assert retried_job.status == "pending"
    service.shutdown()
