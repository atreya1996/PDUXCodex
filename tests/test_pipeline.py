from payday.config import FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.models import UploadedAsset
from payday.repository import PaydayRepository
from payday.service import PaydayAppService
from payday.storage import StorageService


def test_pipeline_process_upload_returns_result() -> None:
    settings = Settings(
        app_env="test",
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(),
        features=FeatureFlags(use_sample_mode=True),
    )
    service = PaydayAppService(settings)

    result = service.process_upload("demo.wav", "audio/wav", b"fake-bytes")

    assert result.asset.filename == "demo.wav"
    assert result.persisted is True
    assert result.analysis.metrics["word_count"] > 0
    assert service.list_results()


def test_repository_crud_supports_interview_related_tables(tmp_path) -> None:
    repository = PaydayRepository(database_path=str(tmp_path / "payday.db"))
    interview = repository.create_interview(audio_url="audio/interview-1/demo.wav", status="uploaded")

    repository.update_interview(interview.id, transcript="A direct quote from the interview.", status="completed")
    repository.upsert_structured_response(
        interview.id,
        smartphone_user=True,
        has_bank_account=True,
        income_range="10k-20k",
        borrowing_history="Borrowed from employer during emergencies.",
        repayment_preference="Weekly",
        loan_interest="Interested if trust is high.",
    )
    repository.upsert_insight(
        interview.id,
        summary="Trust matters more than pricing.",
        key_quotes=["I ask my employer first when money is short."],
        persona="Persona 2",
        confidence_score=0.86,
    )

    listing = repository.list_interviews()
    detail = repository.get_interview_detail(interview.id)

    assert listing[0].id == interview.id
    assert detail.interview.status == "completed"
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.insight is not None
    assert detail.insight.key_quotes == ["I ask my employer first when money is short."]


def test_storage_service_builds_predictable_audio_paths() -> None:
    storage = StorageService(SupabaseSettings())
    asset = UploadedAsset(
        filename="../demo clip.wav",
        content_type="audio/wav",
        size_bytes=3,
        raw_bytes=b"123",
    )

    path = storage.upload_audio(asset=asset, interview_id="interview-123", sample_mode=True)

    assert path == "audio/interview-123/demo_clip.wav"
