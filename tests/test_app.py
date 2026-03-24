from __future__ import annotations

from contextlib import nullcontext

from payday.config import (
    DatabaseSettings,
    FeatureFlags,
    LLMSettings,
    Settings,
    SupabaseSettings,
    TranscriptionSettings,
)
from payday.transcription import describe_transcription_file_size_limit


class _FakeSidebar:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.file_uploader_help: str | None = None
        self.captions: list[str] = []

    def header(self, _text: str) -> None:
        return None

    def caption(self, _text: str) -> None:
        if isinstance(_text, str):
            self.captions.append(_text)
        return None

    def info(self, text: str) -> None:
        self.info_messages.append(text)

    def file_uploader(self, _label: str, **kwargs: object) -> list[object]:
        self.file_uploader_help = kwargs.get("help") if isinstance(kwargs.get("help"), str) else None
        return []

    def write(self, _value: object) -> None:
        return None

    def button(self, _label: str, **_kwargs: object) -> bool:
        return False

    def success(self, _text: str) -> None:
        return None

    def warning(self, _text: str) -> None:
        return None

    def error(self, _text: str) -> None:
        return None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.sidebar = _FakeSidebar()
        self.session_state: dict[str, object] = {}

    def set_page_config(self, **_kwargs: object) -> None:
        return None

    def title(self, _text: str) -> None:
        return None

    def caption(self, _text: str) -> None:
        return None

    def warning(self, _text: str) -> None:
        return None

    def spinner(self, _text: str):
        return nullcontext()

    def rerun(self) -> None:
        return None


class _FakeAppService:
    def list_results(self) -> list[object]:
        return []

    def list_recent_interviews(self) -> list[object]:
        return []

    def get_status_overview(self) -> object:
        return {}

    def get_interview_detail(self, _file_id: str) -> object:
        return {}

    def runtime_summary(self) -> dict[str, object]:
        return {
            "sample_mode": True,
            "transcription": {"provider": "openai", "model": "gpt-4o-mini-transcribe", "required_key_present": False},
            "analysis": {"provider": "openai", "model": "gpt-4.1-mini", "required_key_present": False},
            "runtime_commit_sha": "abc123def456",
        }

    def runtime_diagnostics(self) -> dict[str, object]:
        return {"sample_mode": True, "git_branch": "main", "git_commit": "abc123d"}


class _FakeDashboardRenderer:
    def __init__(self) -> None:
        self.render_calls: list[dict[str, object]] = []

    def render(self, **kwargs: object) -> None:
        self.render_calls.append(kwargs)


def test_app_main_import_and_sidebar_guidance_stay_consistent(monkeypatch) -> None:
    from payday import app as payday_app

    main = payday_app.main
    assert callable(main)

    fake_st = _FakeStreamlit()
    fake_dashboard = _FakeDashboardRenderer()
    settings = Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=":memory:"),
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(provider="openai"),
        features=FeatureFlags(enable_analysis=True),
    )
    expected_guidance = describe_transcription_file_size_limit(settings.transcription)

    monkeypatch.setattr(payday_app, "st", fake_st)
    monkeypatch.setattr(payday_app, "build_app_service", lambda: (_FakeAppService(), settings))
    monkeypatch.setattr(payday_app, "DashboardRenderer", lambda: fake_dashboard)

    main()

    assert expected_guidance is not None
    assert fake_st.sidebar.info_messages == [expected_guidance]
    assert fake_st.sidebar.file_uploader_help is not None
    assert expected_guidance in fake_st.sidebar.file_uploader_help
    assert "Runtime commit SHA: `abc123def456`" in fake_st.sidebar.captions
    assert len(fake_dashboard.render_calls) == 1
