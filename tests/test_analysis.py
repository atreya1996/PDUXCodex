from __future__ import annotations

import json
from pathlib import Path

import pytest

from payday.analysis import (
    AnalysisSchemaError,
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
        "english_translation": "I use WhatsApp every day",
    },
    "bank_account_status": {
        "value": "has_bank_account",
        "status": "observed",
        "evidence_quotes": ["My bank account is active"],
        "notes": "Directly stated.",
        "english_translation": "My bank account is active",
    },
    "per_household_earnings": {
        "value": "unknown",
        "status": "unknown",
        "evidence_quotes": [],
        "notes": "No per-household quote.",
        "evidence_type": "unknown",
        "english_translation": "",
    },
    "participant_personal_monthly_income": {
        "value": "₹12,000",
        "status": "observed",
        "evidence_quotes": ["I earn ₹12,000 per month"],
        "notes": "Directly stated.",
        "evidence_type": "direct",
        "english_translation": "I earn ₹12,000 per month",
    },
    "total_household_monthly_income": {
        "value": "unknown",
        "status": "unknown",
        "evidence_quotes": [],
        "notes": "No household total quote.",
        "evidence_type": "unknown",
        "english_translation": "",
    },
    "borrowing_history": {
        "value": "has_borrowed",
        "status": "observed",
        "evidence_quotes": ["I borrow from neighbors sometimes"],
        "notes": "Directly stated.",
        "english_translation": "I borrow from neighbors sometimes",
    },
    "repayment_preference": {
        "value": "monthly",
        "status": "observed",
        "evidence_quotes": ["I repay monthly after salary"],
        "notes": "Directly stated.",
        "english_translation": "I repay monthly after salary",
    },
    "loan_interest": {
        "value": "fearful_or_uncertain",
        "status": "observed",
        "evidence_quotes": ["I am worried about scams"],
        "notes": "Trust barrier present.",
        "english_translation": "I am worried about scams",
    },
    "summary": {
        "value": "The participant uses WhatsApp, has a bank account, earns ₹12,000, and worries about scams.",
        "status": "observed",
        "evidence_quotes": ["I use WhatsApp every day", "I am worried about scams"],
        "notes": "Grounded in transcript.",
        "english_translation": "The participant uses WhatsApp, has a bank account, earns ₹12,000, and worries about scams.",
    },
    "key_quotes": [
        "I use WhatsApp every day",
        "I am worried about scams",
    ],
    "key_quote_details": [
        {
            "original_text": "I use WhatsApp every day",
            "english_translation": "I use WhatsApp every day",
            "speaker_label": "participant",
            "turn_index": 0,
        },
        {
            "original_text": "I am worried about scams",
            "english_translation": "I am worried about scams",
            "speaker_label": "participant",
            "turn_index": 1,
        },
    ],
    "income_mentions": [
        {
            "meaning_label": "participant_personal_monthly_income",
            "amount": "₹12,000",
            "evidence_quote": "I earn ₹12,000 per month",
            "english_translation": "I earn ₹12,000 per month",
            "speaker_label": "unknown",
            "turn_index": None,
            "evidence_type": "direct",
        }
    ],
    "confidence_signals": {
        "observed_evidence": ["trust barrier mentioned"],
        "missing_or_unknown": [],
    },
    "segmented_dialogue": [
        {
            "turn_index": 0,
            "speaker_label": "participant",
            "utterance_text": "I use WhatsApp every day",
            "english_translation": "I use WhatsApp every day",
            "speaker_confidence": "medium",
            "speaker_uncertainty": "First-person phrasing suggests the participant is speaking.",
        },
        {
            "turn_index": 1,
            "speaker_label": "participant",
            "utterance_text": "I am worried about scams",
            "english_translation": "I am worried about scams",
            "speaker_confidence": "medium",
            "speaker_uncertainty": "First-person phrasing suggests the participant is speaking.",
        },
    ],
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
    assert result.structured_output["segmented_dialogue"][0]["speaker_label"] == "participant"
    assert result.structured_output["key_quote_details"][0]["english_translation"] == "I use WhatsApp every day"


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


INCOME_CASE_FIXTURES = json.loads(Path("tests/fixtures/income_analysis_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", INCOME_CASE_FIXTURES, ids=lambda case: case["name"])
def test_heuristic_analysis_separates_income_fields(case: dict[str, object]) -> None:
    service = AnalysisService()
    transcript = Transcript(text=str(case["transcript"]), provider="test-transcription", model="test-model")

    result = service.analyze(transcript)

    expected = case["expected"]
    assert isinstance(expected, dict)
    for field_name, field_expected in expected.items():
        field = result.structured_output[field_name]
        assert field["value"] == field_expected["value"]
        assert field["status"] == field_expected["status"]
        assert field["evidence_type"] == field_expected["evidence_type"]
        if field["status"] == "observed":
            assert field["evidence_quotes"]
    assert result.structured_output["income_mentions"]
    assert all("meaning_label" in item for item in result.structured_output["income_mentions"])


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
    assert result.structured_output["segmented_dialogue"] == []
    assert result.structured_output["key_quote_details"] == [{"original_text": "I use a basic phone", "english_translation": "I use a basic phone", "speaker_label": "unknown", "turn_index": None}]


def test_analysis_service_filters_malformed_quotes_and_marks_transcript_quality() -> None:
    adapter = SequenceAdapter(
        responses=[
            json.dumps(
                {
                    "smartphone_usage": {
                        "value": "has_smartphone",
                        "status": "observed",
                        "evidence_quotes": ["010101010101", "I use WhatsApp every day"],
                        "notes": "Direct evidence",
                    },
                    "summary": {
                        "value": "Participant uses WhatsApp.",
                        "status": "observed",
                        "evidence_quotes": ["%%%%%%%///////", "I use WhatsApp every day"],
                        "notes": "Grounded in transcript.",
                    },
                    "key_quotes": ["010101010101", "I use WhatsApp every day"],
                }
            )
        ]
    )
    service = AnalysisService(adapter=adapter, settings=LLMSettings())
    transcript = Transcript(
        text="I use WhatsApp every day.",
        provider="test-transcription",
        model="test-model",
    )

    result = service.analyze(transcript)

    assert result.structured_output["smartphone_usage"]["evidence_quotes"] == ["I use WhatsApp every day"]
    assert result.structured_output["key_quotes"] == ["I use WhatsApp every day"]
    assert result.structured_output["transcript_quality"]["status"] == "degraded"
    assert result.structured_output["transcript_quality"]["dropped_malformed_quote_count"] >= 1


def test_analysis_service_rejects_non_transcript_income_quote_in_strict_mode() -> None:
    service = AnalysisService(adapter=SequenceAdapter([json.dumps(VALID_ANALYSIS_PAYLOAD)]), settings=LLMSettings())
    payload = json.loads(json.dumps(VALID_ANALYSIS_PAYLOAD))
    payload["participant_personal_monthly_income"]["evidence_quotes"] = ["Monthly salary around 12k"]

    with pytest.raises(AnalysisSchemaError, match="must include at least one direct quote"):
        service._parse_and_validate(  # noqa: SLF001
            json.dumps(payload, ensure_ascii=False),
            build_transcript().text,
            metadata={"evidence_mode": "strict"},
        )


def test_analysis_service_manual_edit_mode_preserves_valid_income_quotes_not_in_transcript() -> None:
    service = AnalysisService(adapter=SequenceAdapter([json.dumps(VALID_ANALYSIS_PAYLOAD)]), settings=LLMSettings())
    payload = json.loads(json.dumps(VALID_ANALYSIS_PAYLOAD))
    payload["participant_personal_monthly_income"]["evidence_quotes"] = ["Monthly salary around 12k"]
    payload["key_quotes"] = ["Monthly salary around 12k"]

    normalized = service._parse_and_validate(  # noqa: SLF001
        json.dumps(payload, ensure_ascii=False),
        build_transcript().text,
        metadata={"evidence_mode": "manual_edit"},
    )

    assert normalized["participant_personal_monthly_income"]["evidence_quotes"] == ["Monthly salary around 12k"]
    assert normalized["key_quotes"] == ["Monthly salary around 12k"]


def test_analysis_prompt_includes_weak_metadata_hints_and_separates_dialogue_when_evident() -> None:
    transcript = Transcript(
        text="Interviewer: Do you use WhatsApp? Participant: Yes, I use it every day.",
        provider="test-transcription",
        model="test-model",
        metadata={"filename": "meena_interview.wav", "participant_name_hint": "Meena"},
    )
    service = AnalysisService(adapter=SequenceAdapter([json.dumps(VALID_ANALYSIS_PAYLOAD)]), settings=LLMSettings())

    prompt = service._build_prompt(transcript)  # noqa: SLF001

    assert "Use filename or interview metadata only as a weak hint for participant identity." in prompt
    assert '"participant_name_hint": "Meena"' in prompt
    assert "Separate interviewer questions from participant answers" in prompt
    assert "Never treat wages from one house/home as the participant's full monthly income" in prompt


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
