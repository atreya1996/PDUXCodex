from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


@dataclass(frozen=True)
class DatabaseSettings:
    path: str = str(DEFAULT_DATABASE_PATH)


@dataclass(frozen=True)
class FeatureFlags:
    use_sample_mode: bool = True
    enable_uploads: bool = True
    enable_dashboard: bool = True
    enable_analysis: bool = True


@dataclass(frozen=True)
class DatabaseSettings:
    sqlite_path: str = "data/payday.db"


@dataclass(frozen=True)
class Settings:
    app_env: str
    database: DatabaseSettings
    supabase: SupabaseSettings
    llm: LLMSettings
    transcription: TranscriptionSettings
    database: DatabaseSettings
    features: FeatureFlags


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    sqlite_path = Path(os.getenv("PAYDAY_SQLITE_PATH", "data/payday.db")).expanduser()
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
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("LLM_API_KEY", ""),
        ),
        transcription=TranscriptionSettings(
            provider=os.getenv("TRANSCRIPTION_PROVIDER", "openai"),
            model=os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
            api_key=os.getenv("TRANSCRIPTION_API_KEY", ""),
        ),
        database=DatabaseSettings(path=database_path),
        features=FeatureFlags(
            use_sample_mode=_as_bool(os.getenv("PAYDAY_USE_SAMPLE_MODE"), True),
            enable_uploads=_as_bool(os.getenv("PAYDAY_ENABLE_UPLOADS"), True),
            enable_dashboard=_as_bool(os.getenv("PAYDAY_ENABLE_DASHBOARD"), True),
            enable_analysis=_as_bool(os.getenv("PAYDAY_ENABLE_ANALYSIS"), True),
        ),
    )
