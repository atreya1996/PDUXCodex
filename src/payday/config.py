from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_DATABASE_PATH = Path("data/payday.db")
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "anthropic"})
SUPPORTED_TRANSCRIPTION_PROVIDERS = frozenset({"openai"})
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


class SettingsConfigurationError(ValueError):
    """Raised when the runtime provider configuration is invalid."""


def resolve_runtime_commit_sha(default: str = "unknown") -> str:
    """Resolve the release/runtime commit SHA exposed in the dashboard."""

    raw_value = (
        os.getenv("PAYDAY_RELEASE_SHA")
        or os.getenv("GIT_COMMIT_SHA")
        or os.getenv("COMMIT_SHA")
        or ""
    ).strip()
    if not raw_value:
        return default
    if not COMMIT_SHA_PATTERN.match(raw_value):
        return default
    return raw_value[:12].lower()



def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def _normalized_provider(value: str | None, *, default: str) -> str:
    normalized = (value or default).strip().lower()
    return normalized or default


@dataclass(frozen=True)
class SupabaseSettings:
    url: str = ""
    anon_key: str = ""
    service_role_key: str = ""
    storage_bucket: str = "payday-assets"


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key: str = ""


@dataclass(frozen=True)
class TranscriptionSettings:
    provider: str = "openai"
    model: str = "gpt-4o-mini-transcribe"
    api_key: str = ""
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class DatabaseSettings:
    sqlite_path: str = str(DEFAULT_DATABASE_PATH)


@dataclass(frozen=True)
class FeatureFlags:
    use_sample_mode: bool = False
    enable_uploads: bool = True
    enable_dashboard: bool = True
    enable_analysis: bool = True


@dataclass(frozen=True)
class Settings:
    app_env: str
    database: DatabaseSettings
    supabase: SupabaseSettings
    llm: LLMSettings
    transcription: TranscriptionSettings
    features: FeatureFlags


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    sqlite_path = Path(os.getenv("PAYDAY_SQLITE_PATH", str(DEFAULT_DATABASE_PATH))).expanduser()
    return Settings(
        app_env=os.getenv("PAYDAY_APP_ENV", "development"),
        database=DatabaseSettings(sqlite_path=str(sqlite_path)),
        supabase=SupabaseSettings(
            url=os.getenv("SUPABASE_URL", ""),
            anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            storage_bucket=os.getenv("SUPABASE_STORAGE_BUCKET", "payday-assets"),
        ),
        llm=LLMSettings(
            provider=_normalized_provider(os.getenv("LLM_PROVIDER"), default="openai"),
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("LLM_API_KEY", ""),
        ),
        transcription=TranscriptionSettings(
            provider=_normalized_provider(os.getenv("TRANSCRIPTION_PROVIDER"), default="openai"),
            model=os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
            api_key=os.getenv("TRANSCRIPTION_API_KEY", ""),
            timeout_seconds=float(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "60")),
        ),
        features=FeatureFlags(
            use_sample_mode=_as_bool(os.getenv("PAYDAY_USE_SAMPLE_MODE"), False),
            enable_uploads=_as_bool(os.getenv("PAYDAY_ENABLE_UPLOADS"), True),
            enable_dashboard=_as_bool(os.getenv("PAYDAY_ENABLE_DASHBOARD"), True),
            enable_analysis=_as_bool(os.getenv("PAYDAY_ENABLE_ANALYSIS"), True),
        ),
    )



def validate_runtime_settings(settings: Settings) -> None:
    """Validate supported providers and required credentials for the current mode."""

    errors: list[str] = []

    if settings.llm.provider not in SUPPORTED_LLM_PROVIDERS:
        errors.append(
            "LLM_PROVIDER must be one of: "
            f"{', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}. Got '{settings.llm.provider}'."
        )

    if settings.transcription.provider not in SUPPORTED_TRANSCRIPTION_PROVIDERS:
        errors.append(
            "TRANSCRIPTION_PROVIDER must be one of: "
            f"{', '.join(sorted(SUPPORTED_TRANSCRIPTION_PROVIDERS))}. "
            f"Got '{settings.transcription.provider}'."
        )

    if settings.features.use_sample_mode:
        if errors:
            raise SettingsConfigurationError("Invalid provider configuration:\n- " + "\n- ".join(errors))
        return

    if not settings.llm.api_key.strip():
        errors.append(
            "LLM_API_KEY is required when PAYDAY_USE_SAMPLE_MODE=false so analysis can use "
            f"the configured '{settings.llm.provider}' provider."
        )

    if not settings.transcription.api_key.strip():
        errors.append(
            "TRANSCRIPTION_API_KEY is required when PAYDAY_USE_SAMPLE_MODE=false so transcription can use "
            f"the configured '{settings.transcription.provider}' provider."
        )

    if errors:
        raise SettingsConfigurationError("Invalid provider configuration:\n- " + "\n- ".join(errors))
