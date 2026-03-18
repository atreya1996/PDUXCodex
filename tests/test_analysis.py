from __future__ import annotations

import json

import pytest

from payday.analysis import (
    AnalysisService,
    AnthropicAnalysisAdapter,
    ExternalProviderError,
    OpenAIAnalysisAdapter,
    build_analysis_adapter,
)
from payday.config import LLMSettings
from payday.models import Transcript


class SequenceAdapter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.provider_name = "test-provider"
        self.model_name = "test-model"
        self.calls = 0

    def generate_analysis(self, *, prompt, transcript, expected_schema, attempt):
        del prompt, transcript, expected_schema, attempt
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class SequenceTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def __call__(self, *, prompt: str, model: str, api_key: str) -> object:
        self.calls.append({"prompt": prompt, "model": model, "api_key": api_key})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class ErrorTransport:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def __call__(self, *, prompt: str, model: str, api_key: str) -> object:
        del prompt, model, api_key
        self.calls += 1
        return {"error": {"message": self.message}}


VALID_ANALYSIS_PAYLOAD = {
    "smartphone_usage": {
        "value": "has_smartphone",
        "status": "observed",
        "evidence_quotes": ["I use WhatsApp every day"],
        "notes": "Directly stated.",
    },
    "bank_account_status": {
        "value": "has_bank_account",
        "status": "observed",
        "evidence_quotes": ["My bank account is active"],
        "notes": "Directly stated.",
    },
    "income_range": {
        "value": "₹12,000",
        "status": "observed",
        "evidence_quotes": ["I earn ₹12,000 per month"],
        "notes": "Directly stated.",
    },
    "borrowing_history": {
        "value": "has_borrowed",
        "status": "observed",
        "evidence_quotes": ["I borrow from neighbors sometimes"],
        "notes": "Directly stated.",
    },
    "repayment_preference": {
        "value": "monthly",
        "status": "observed",
        "evidence_quotes": ["I repay monthly after salary"],
        "notes": "Directly stated.",
    },
    "loan_interest": {
        "value": "fearful_or_uncertain",
        "status": "observed",
        "evidence_quotes": ["I am worried about scams"],
        "notes": "Trust barrier present.",
    },
    "summary": {
        "value": "The participant uses WhatsApp, has a bank account, earns ₹12,000, and worries about scams.",
        "status": "observed",
        "evidence_quotes": ["I use WhatsApp every day", "I am worried about scams"],
        "notes": "Grounded in transcript.",
    },
    "key_quotes": [
        "I use WhatsApp every day",
        "I am worried about scams",
    ],
    "confidence_signals": {
        "observed_evidence": ["trust barrier mentioned"],
        "missing_or_unknown": [],
    },
}


def build_transcript() -> Transcript:
    return Transcript(
        text=(
            "I use WhatsApp every day. My bank account is active. I earn ₹12,000 per month. "
            "I borrow from neighbors sometimes. I repay monthly after salary. I am worried about scams."
        ),
        provider="test-transcription",
        model="test-model",
    )


def test_analysis_service_retries_invalid_json_and_returns_validated_output() -> None:
    adapter = SequenceAdapter(
        responses=[
            "not-json",
            json.dumps(VALID_ANALYSIS_PAYLOAD),
        ]
    )
    service = AnalysisService(adapter=adapter, settings=LLMSettings(), max_json_retries=2)

    result = service.analyze(build_transcript())

    assert adapter.calls == 2
    assert result.metrics["json_attempts"] == 2
    assert result.structured_output["smartphone_usage"]["value"] == "has_smartphone"
    assert result.structured_output["bank_account_status"]["value"] == "has_bank_account"
    assert result.evidence_quotes == ["I use WhatsApp every day", "I am worried about scams"]


def test_openai_analysis_adapter_retries_invalid_json_until_valid_payload() -> None:
    transport = SequenceTransport(
        responses=[
            {"output_text": "not-json"},
            {"output": [{"content": [{"text": json.dumps(VALID_ANALYSIS_PAYLOAD)}]}]},
        ]
    )
    adapter = OpenAIAnalysisAdapter(
        LLMSettings(provider="openai", model="gpt-test", api_key="test-key"),
        transport=transport,
    )
    service = AnalysisService(adapter=adapter, settings=LLMSettings(), max_json_retries=2)

    result = service.analyze(build_transcript())

    assert len(transport.calls) == 2
    assert transport.calls[0]["api_key"] == "test-key"
    assert "The previous response was invalid" in transport.calls[1]["prompt"]
    assert result.metrics["json_attempts"] == 2
    assert result.structured_output["summary"]["status"] == "observed"


def test_openai_analysis_adapter_surfaces_provider_errors() -> None:
    transport = ErrorTransport("provider unavailable")
    adapter = OpenAIAnalysisAdapter(
        LLMSettings(provider="openai", model="gpt-test", api_key="test-key"),
        transport=transport,
    )
    service = AnalysisService(adapter=adapter, settings=LLMSettings(), max_json_retries=2)

    with pytest.raises(ExternalProviderError, match="provider unavailable"):
        service.analyze(build_transcript())

    assert transport.calls == 1


def test_analysis_service_applies_defaults_and_tracks_missing_values() -> None:
    adapter = SequenceAdapter(
        responses=[
            json.dumps(
                {
                    "summary": {
                        "value": "The participant said they use a basic phone.",
                        "status": "observed",
                        "evidence_quotes": ["I use a basic phone"],
                        "notes": "Grounded in transcript.",
                    },
                    "key_quotes": ["I use a basic phone"],
                }
            )
        ]
    )
    service = AnalysisService(adapter=adapter, settings=LLMSettings())
    transcript = Transcript(
        text="I use a basic phone. I do not have a bank account.",
        provider="test-transcription",
        model="test-model",
    )

    result = service.analyze(transcript)

    assert result.structured_output["summary"]["evidence_quotes"] == ["I use a basic phone"]
    assert result.structured_output["smartphone_usage"]["value"] == "unknown"
    assert result.structured_output["smartphone_usage"]["status"] == "missing"
    assert result.structured_output["bank_account_status"]["status"] == "missing"
    assert "smartphone_usage: unknown" in result.structured_output["confidence_signals"]["missing_or_unknown"]


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_openai_analysis_adapter_returns_raw_json_text(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request_obj, timeout):
        captured["url"] = request_obj.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request_obj.header_items())
        captured["body"] = json.loads(request_obj.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"summary": {"value": "ok", "status": "observed", "evidence_quotes": [], "notes": ""}, "key_quotes": []}',
                            }
                        ]
                    }
                ]
            }
        )

    monkeypatch.setattr("payday.analysis.request.urlopen", fake_urlopen)

    adapter = OpenAIAnalysisAdapter(LLMSettings(provider="openai", model="gpt-test", api_key="secret"))
    transcript = Transcript(text="I use WhatsApp.", provider="test", model="test")

    raw = adapter.generate_analysis(
        prompt="Return JSON",
        transcript=transcript,
        expected_schema={},
        attempt=1,
    )

    assert raw.startswith('{"summary"')
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["timeout"] == 60
    assert captured["body"] == {
        "model": "gpt-test",
        "input": "Return JSON",
        "text": {"format": {"type": "json_object"}},
    }
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_build_analysis_adapter_uses_heuristic_in_sample_mode_and_provider_specific_live_modes() -> None:
    sample_adapter = build_analysis_adapter(
        LLMSettings(provider="openai", model="gpt-4.1-mini", api_key="secret"),
        sample_mode=True,
    )
    live_openai_adapter = build_analysis_adapter(
        LLMSettings(provider="openai", model="gpt-4.1-mini", api_key="secret"),
        sample_mode=False,
    )
    live_anthropic_adapter = build_analysis_adapter(
        LLMSettings(provider="anthropic", model="claude-test", api_key="secret"),
        sample_mode=False,
    )

    assert sample_adapter.provider_name == "heuristic"
    assert sample_adapter.model_name == "heuristic-json"
    assert live_openai_adapter.provider_name == "openai"
    assert live_openai_adapter.model_name == "gpt-4.1-mini"
    assert isinstance(live_anthropic_adapter, AnthropicAnalysisAdapter)
    assert live_anthropic_adapter.provider_name == "anthropic-heuristic"
    assert live_anthropic_adapter.model_name == "heuristic-json"
