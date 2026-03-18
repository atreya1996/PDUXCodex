from __future__ import annotations

import builtins
import sys
import types

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

    def create(self, *, model: str, file: tuple[str, bytes, str]):
        self.calls.append({"model": model, "file": file})
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
