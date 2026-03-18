from __future__ import annotations

import json

from payday.analysis import AnalysisService
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


def test_analysis_service_retries_invalid_json_and_returns_validated_output() -> None:
    adapter = SequenceAdapter(
        responses=[
            "not-json",
            json.dumps(
                {
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
            ),
        ]
    )
    service = AnalysisService(adapter=adapter, settings=LLMSettings(), max_json_retries=2)
    transcript = Transcript(
        text=(
            "I use WhatsApp every day. My bank account is active. I earn ₹12,000 per month. "
            "I borrow from neighbors sometimes. I repay monthly after salary. I am worried about scams."
        ),
        provider="test-transcription",
        model="test-model",
    )

    result = service.analyze(transcript)

    assert adapter.calls == 2
    assert result.metrics["json_attempts"] == 2
    assert result.structured_output["smartphone_usage"]["value"] == "has_smartphone"
    assert result.structured_output["bank_account_status"]["value"] == "has_bank_account"
    assert result.evidence_quotes == ["I use WhatsApp every day", "I am worried about scams"]


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
