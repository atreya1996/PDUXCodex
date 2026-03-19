from __future__ import annotations

import pytest

from payday.analysis import AnthropicAnalysisAdapter, HeuristicAnalysisAdapter, OpenAIAnalysisAdapter, build_analysis_adapter
from payday.config import (
    DatabaseSettings,
    FeatureFlags,
    LLMSettings,
    Settings,
    SettingsConfigurationError,
    SupabaseSettings,
    TranscriptionSettings,
    validate_runtime_settings,
)
from payday.service import PaydayAppService
from payday.transcription import OpenAITranscriptionService, TranscriptionService, build_transcription_service


def build_settings(
    *,
    sample_mode: bool,
    llm_provider: str = "openai",
    llm_api_key: str = "",
    transcription_provider: str = "openai",
    transcription_api_key: str = "",
) -> Settings:
    return Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(provider=llm_provider, model="analysis-model", api_key=llm_api_key),
        transcription=TranscriptionSettings(
            provider=transcription_provider,
            model="transcription-model",
            api_key=transcription_api_key,
        ),
        features=FeatureFlags(use_sample_mode=sample_mode),
    )


def test_sample_mode_allows_missing_external_credentials() -> None:
    settings = build_settings(sample_mode=True)

    validate_runtime_settings(settings)

    analysis_adapter = build_analysis_adapter(settings.llm, sample_mode=settings.features.use_sample_mode)
    transcription_service = build_transcription_service(
        settings.transcription,
        sample_mode=settings.features.use_sample_mode,
    )

    assert isinstance(analysis_adapter, HeuristicAnalysisAdapter)
    assert isinstance(transcription_service, TranscriptionService)


def test_non_sample_mode_requires_both_provider_keys() -> None:
    settings = build_settings(sample_mode=False)

    with pytest.raises(SettingsConfigurationError) as exc_info:
        validate_runtime_settings(settings)

    message = str(exc_info.value)
    assert "LLM_API_KEY is required" in message
    assert "TRANSCRIPTION_API_KEY is required" in message


def test_non_sample_mode_reports_missing_llm_key_without_transcription_key_error() -> None:
    settings = build_settings(sample_mode=False, transcription_api_key="tx-key")

    with pytest.raises(SettingsConfigurationError) as exc_info:
        validate_runtime_settings(settings)

    message = str(exc_info.value)
    assert "LLM_API_KEY is required" in message
    assert "TRANSCRIPTION_API_KEY is required" not in message


def test_non_sample_mode_reports_missing_transcription_key_without_llm_key_error() -> None:
    settings = build_settings(sample_mode=False, llm_api_key="llm-key")

    with pytest.raises(SettingsConfigurationError) as exc_info:
        validate_runtime_settings(settings)

    message = str(exc_info.value)
    assert "TRANSCRIPTION_API_KEY is required" in message
    assert "LLM_API_KEY is required" not in message


def test_validate_runtime_settings_reports_invalid_provider_names_clearly() -> None:
    settings = build_settings(
        sample_mode=False,
        llm_provider="invalid-llm",
        transcription_provider="invalid-transcription",
        llm_api_key="llm-key",
        transcription_api_key="tx-key",
    )

    with pytest.raises(SettingsConfigurationError) as exc_info:
        validate_runtime_settings(settings)

    message = str(exc_info.value)
    assert "LLM_PROVIDER must be one of: anthropic, openai. Got 'invalid-llm'." in message
    assert "TRANSCRIPTION_PROVIDER must be one of: openai. Got 'invalid-transcription'." in message


def test_analysis_provider_factory_reserves_anthropic_for_future_live_support() -> None:
    sample_settings = build_settings(sample_mode=True, llm_provider="anthropic")
    live_settings = build_settings(
        sample_mode=False,
        llm_provider="anthropic",
        llm_api_key="test-key",
        transcription_api_key="test-key",
    )

    sample_adapter = build_analysis_adapter(
        sample_settings.llm,
        sample_mode=sample_settings.features.use_sample_mode,
    )
    live_adapter = build_analysis_adapter(
        live_settings.llm,
        sample_mode=live_settings.features.use_sample_mode,
    )

    assert isinstance(sample_adapter, HeuristicAnalysisAdapter)
    assert isinstance(live_adapter, AnthropicAnalysisAdapter)


def test_live_openai_services_are_selected_when_sample_mode_is_disabled() -> None:
    settings = build_settings(
        sample_mode=False,
        llm_provider="openai",
        llm_api_key="llm-key",
        transcription_provider="openai",
        transcription_api_key="tx-key",
    )

    analysis_adapter = build_analysis_adapter(settings.llm, sample_mode=settings.features.use_sample_mode)
    transcription_service = build_transcription_service(
        settings.transcription,
        sample_mode=settings.features.use_sample_mode,
    )

    assert isinstance(analysis_adapter, OpenAIAnalysisAdapter)
    assert isinstance(transcription_service, OpenAITranscriptionService)


def test_app_service_fails_fast_for_missing_live_credentials() -> None:
    settings = build_settings(sample_mode=False)

    with pytest.raises(SettingsConfigurationError):
        PaydayAppService(settings)


def test_app_service_fails_fast_when_live_llm_key_is_missing() -> None:
    settings = build_settings(sample_mode=False, transcription_api_key="tx-key")

    with pytest.raises(SettingsConfigurationError, match="LLM_API_KEY is required"):
        PaydayAppService(settings)


def test_app_service_fails_fast_when_live_transcription_key_is_missing() -> None:
    settings = build_settings(sample_mode=False, llm_api_key="llm-key")

    with pytest.raises(SettingsConfigurationError, match="TRANSCRIPTION_API_KEY is required"):
        PaydayAppService(settings)
