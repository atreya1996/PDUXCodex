from __future__ import annotations

import builtins
import sys
import types

import pytest

from payday.analysis import build_analysis_adapter


class FakeOpenAITimeoutError(Exception):
    pass


class StubTranscriptionResponse:
    def __init__(self, text: str, *, language: str | None = None, duration: float | None = None) -> None:
        self.text = text
        self.language = language
        self.duration = duration


class StubOpenAITranscriptions:
    def __init__(self, response: StubTranscriptionResponse | None = None, error: Exception | None = None) -> None:
        self.response = response or StubTranscriptionResponse("stub transcript")
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, file: object):
        recorded_file: object = file
        if hasattr(file, "read"):
            payload = file.read()
            file.seek(0)
            recorded_file = {
                "name": getattr(file, "name", ""),
                "payload": payload,
            }
        self.calls.append({"model": model, "file": recorded_file})
        if self.error is not None:
            raise self.error
        return self.response


class StubOpenAIClient:
    def __init__(self, transcriptions: StubOpenAITranscriptions) -> None:
        self.audio = types.SimpleNamespace(transcriptions=transcriptions)


def install_fake_openai_module(monkeypatch) -> None:
    module = types.ModuleType("openai")
    module.APITimeoutError = FakeOpenAITimeoutError
    module.APIConnectionError = RuntimeError
    module.APIStatusError = RuntimeError
    module.OpenAI = object
    monkeypatch.setitem(sys.modules, "openai", module)


builtins.build_analysis_adapter = build_analysis_adapter
builtins.install_fake_openai_module = install_fake_openai_module
builtins.StubOpenAITranscriptions = StubOpenAITranscriptions
builtins.StubOpenAIClient = StubOpenAIClient
builtins.StubTranscriptionResponse = StubTranscriptionResponse
builtins.FakeOpenAITimeoutError = FakeOpenAITimeoutError


@pytest.fixture(autouse=True)
def enable_sample_mode_env_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAYDAY_USE_SAMPLE_MODE", "true")
