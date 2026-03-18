from payday.config import FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.service import PaydayAppService


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
