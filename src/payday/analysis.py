from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from payday.config import LLMSettings
from payday.context_loader import ContextMemory, build_analysis_prompt, load_context_memory
from payday.models import AnalysisResult, Transcript

DEFAULT_UNKNOWN_VALUE = "unknown"
FIELD_NAMES = (
    "smartphone_usage",
    "bank_account_status",
    "income_range",
    "borrowing_history",
    "repayment_preference",
    "loan_interest",
    "summary",
)
ALLOWED_FIELD_STATUSES = {"observed", "unknown", "missing"}


class AnalysisProviderAdapter(Protocol):
    """Provider-agnostic contract for transcript analysis backends."""

    provider_name: str
    model_name: str

    def generate_analysis(
        self,
        *,
        prompt: str,
        transcript: Transcript,
        expected_schema: dict[str, Any],
        attempt: int,
    ) -> str:
        """Return raw JSON text for a transcript analysis request."""


@dataclass(frozen=True, slots=True)
class AnalysisField:
    value: str = DEFAULT_UNKNOWN_VALUE
    status: str = "missing"
    evidence_quotes: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "evidence_quotes": list(self.evidence_quotes),
            "notes": self.notes,
        }


class AnalysisSchemaError(ValueError):
    """Raised when an analysis response is not valid JSON or violates the schema."""


class HeuristicAnalysisAdapter:
    """Local adapter that emits schema-compliant JSON for development and tests."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        resolved = settings or LLMSettings(provider="mock-llm", model="heuristic-json")
        self.provider_name = resolved.provider
        self.model_name = resolved.model

    def generate_analysis(
        self,
        *,
        prompt: str,
        transcript: Transcript,
        expected_schema: dict[str, Any],
        attempt: int,
    ) -> str:
        del prompt, expected_schema, attempt
        text = transcript.text.strip()
        lowered = text.lower()
        quotes = _extract_direct_quotes(text)

        payload = {
            "smartphone_usage": _field_payload(
                value=(
                    "no_smartphone"
                    if _contains_any(lowered, _NO_SMARTPHONE_PHRASES)
                    else "has_smartphone"
                    if _contains_any(lowered, ("smartphone", "whatsapp", "upi", "loan app", "online"))
                    else DEFAULT_UNKNOWN_VALUE
                ),
                status=(
                    "observed"
                    if _contains_any(lowered, _NO_SMARTPHONE_PHRASES)
                    or _contains_any(lowered, ("smartphone", "whatsapp", "upi", "loan app", "online"))
                    else "unknown"
                ),
                evidence_quotes=_matching_quotes(
                    quotes,
                    ("smartphone", "whatsapp", "upi", "loan app", "online", "basic phone"),
                ),
                notes="Observed from direct mention of device or channel usage." if _contains_any(
                    lowered,
                    ("smartphone", "whatsapp", "upi", "loan app", "online", "basic phone"),
                ) else "No direct smartphone evidence found in transcript.",
            ),
            "bank_account_status": _field_payload(
                value=(
                    "no_bank_account"
                    if _contains_any(lowered, _NO_BANK_ACCOUNT_PHRASES)
                    else "has_bank_account"
                    if _contains_any(lowered, ("bank account", "account", "passbook", "atm", "upi"))
                    else DEFAULT_UNKNOWN_VALUE
                ),
                status=(
                    "observed"
                    if _contains_any(lowered, _NO_BANK_ACCOUNT_PHRASES)
                    or _contains_any(lowered, ("bank account", "passbook", "atm", "upi"))
                    else "unknown"
                ),
                evidence_quotes=_matching_quotes(quotes, ("bank", "account", "passbook", "atm", "upi", "unbanked")),
                notes="Observed from a direct banking reference." if _contains_any(
                    lowered,
                    ("bank account", "passbook", "atm", "upi", "unbanked"),
                ) else "No direct banking evidence found in transcript.",
            ),
            "income_range": _field_payload(
                value=_extract_income_range(text),
                status="observed" if _extract_income_range(text) != DEFAULT_UNKNOWN_VALUE else "unknown",
                evidence_quotes=_matching_quotes(quotes, ("rupees", "salary", "income", "earn", "month")),
                notes="Income range captured only when directly stated." if _extract_income_range(
                    text
                ) != DEFAULT_UNKNOWN_VALUE else "Transcript does not state a clear income range.",
            ),
            "borrowing_history": _field_payload(
                value=(
                    "has_borrowed"
                    if _contains_any(lowered, ("borrow", "loan", "debt", "moneylender", "neighbors"))
                    else "no_borrowing_mentioned"
                ),
                status="observed" if _contains_any(lowered, ("borrow", "loan", "debt", "moneylender", "neighbors")) else "unknown",
                evidence_quotes=_matching_quotes(quotes, ("borrow", "loan", "debt", "moneylender", "neighbor")),
                notes="Observed from explicit borrowing references." if _contains_any(
                    lowered,
                    ("borrow", "loan", "debt", "moneylender", "neighbors"),
                ) else "Borrowing history is not directly described.",
            ),
            "repayment_preference": _field_payload(
                value=_extract_repayment_preference(lowered),
                status="observed" if _extract_repayment_preference(lowered) != DEFAULT_UNKNOWN_VALUE else "unknown",
                evidence_quotes=_matching_quotes(quotes, ("repay", "monthly", "weekly", "daily", "installment", "flexible")),
                notes="Preference only recorded when repayment timing or method is stated." if _extract_repayment_preference(
                    lowered
                ) != DEFAULT_UNKNOWN_VALUE else "No repayment preference is directly stated.",
            ),
            "loan_interest": _field_payload(
                value=_extract_loan_interest(lowered),
                status="observed" if _extract_loan_interest(lowered) != DEFAULT_UNKNOWN_VALUE else "unknown",
                evidence_quotes=_matching_quotes(quotes, ("interest", "loan", "afraid", "fear", "trust", "need money")),
                notes="Derived from direct statements about willingness or hesitation to borrow." if _extract_loan_interest(
                    lowered
                ) != DEFAULT_UNKNOWN_VALUE else "No direct statement about loan interest was found.",
            ),
            "summary": _field_payload(
                value=_build_summary(text),
                status="observed" if text else "unknown",
                evidence_quotes=quotes[:2],
                notes="Summary is grounded in the supplied transcript only.",
            ),
            "key_quotes": quotes[:5],
            "confidence_signals": {
                "observed_evidence": [],
                "missing_or_unknown": [],
            },
        }

        payload["confidence_signals"] = _build_confidence_signals(payload)
        return json.dumps(payload)


class AnalysisService:
    """Transforms transcripts into validated, evidence-aware structured JSON."""

    def __init__(
        self,
        adapter: AnalysisProviderAdapter | None = None,
        *,
        settings: LLMSettings | None = None,
        memory: ContextMemory | None = None,
        max_json_retries: int = 2,
    ) -> None:
        self.settings = settings or LLMSettings()
        self.adapter = adapter or HeuristicAnalysisAdapter(self.settings)
        self.memory = memory or load_context_memory()
        self.max_json_retries = max_json_retries

    def analyze(self, transcript: Transcript) -> AnalysisResult:
        prompt = self._build_prompt(transcript)
        last_error: AnalysisSchemaError | None = None
        raw_response = ""

        for attempt in range(1, self.max_json_retries + 2):
            raw_response = self.adapter.generate_analysis(
                prompt=prompt,
                transcript=transcript,
                expected_schema=self.expected_schema(),
                attempt=attempt,
            )
            try:
                structured_output = self._parse_and_validate(raw_response, transcript.text)
                evidence_quotes = structured_output["key_quotes"]
                return AnalysisResult(
                    summary=structured_output["summary"]["value"],
                    metrics={
                        "word_count": len(transcript.text.split()),
                        "character_count": len(transcript.text),
                        "transcript_word_count": len(transcript.text.split()),
                        "transcript_character_count": len(transcript.text),
                        "analysis_provider": self.adapter.provider_name,
                        "analysis_model": self.adapter.model_name,
                        "json_attempts": attempt,
                    },
                    structured_output=structured_output,
                    evidence_quotes=evidence_quotes,
                )
            except AnalysisSchemaError as exc:
                last_error = exc
                prompt = self._build_retry_prompt(transcript, previous_prompt=prompt, error=exc, raw_response=raw_response)

        raise AnalysisSchemaError(
            f"Analysis provider returned invalid JSON after {self.max_json_retries + 1} attempts: {last_error}"
        )

    def expected_schema(self) -> dict[str, Any]:
        return {
            "smartphone_usage": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "bank_account_status": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "income_range": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "borrowing_history": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "repayment_preference": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "loan_interest": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "summary": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
            },
            "key_quotes": ["string"],
            "confidence_signals": {
                "observed_evidence": ["string"],
                "missing_or_unknown": ["string"],
            },
        }

    def _build_prompt(self, transcript: Transcript) -> str:
        task = (
            "Analyze the interview transcript and return only valid JSON that matches the required schema.\n\n"
            "Rules:\n"
            "- Use only evidence from the transcript.\n"
            "- Preserve direct quotes exactly when evidence exists.\n"
            "- Do not infer unsupported facts.\n"
            "- Mark fields as unknown when the transcript does not provide evidence.\n"
            "- The no-smartphone and no-bank-account cases must be clearly surfaced for downstream Persona 3 handling.\n\n"
            f"Required schema:\n{json.dumps(self.expected_schema(), indent=2)}\n\n"
            f"Transcript:\n{transcript.text.strip() or '[empty transcript]'}"
        )
        return build_analysis_prompt(task=task, memory=self.memory)

    def _build_retry_prompt(
        self,
        transcript: Transcript,
        *,
        previous_prompt: str,
        error: AnalysisSchemaError,
        raw_response: str,
    ) -> str:
        return (
            f"{previous_prompt}\n\n"
            "The previous response was invalid. Return corrected JSON only.\n"
            f"Validation error: {error}\n"
            f"Previous response:\n{raw_response}"
        )

    def _parse_and_validate(self, raw_response: str, transcript_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise AnalysisSchemaError(f"Invalid JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise AnalysisSchemaError("Top-level JSON must be an object.")

        normalized: dict[str, Any] = {}
        for field_name in FIELD_NAMES:
            normalized[field_name] = self._normalize_field(
                field_name,
                payload.get(field_name),
                transcript_text,
            )

        normalized["key_quotes"] = self._normalize_quote_list(payload.get("key_quotes"), transcript_text)
        normalized["confidence_signals"] = self._normalize_confidence_signals(
            payload.get("confidence_signals"),
            normalized,
        )
        return normalized

    def _normalize_field(self, field_name: str, value: Any, transcript_text: str) -> dict[str, Any]:
        if value is None:
            return AnalysisField(
                notes=f"Fallback default applied because '{field_name}' was missing from the provider response.",
            ).as_dict()

        if isinstance(value, str):
            normalized_status = "unknown" if value.strip().lower() == DEFAULT_UNKNOWN_VALUE else "observed"
            return AnalysisField(value=value.strip() or DEFAULT_UNKNOWN_VALUE, status=normalized_status).as_dict()

        if not isinstance(value, dict):
            raise AnalysisSchemaError(f"Field '{field_name}' must be an object or string.")

        normalized_value = str(value.get("value", DEFAULT_UNKNOWN_VALUE)).strip() or DEFAULT_UNKNOWN_VALUE
        normalized_status = str(value.get("status", "unknown")).strip().lower() or "unknown"
        if normalized_status not in ALLOWED_FIELD_STATUSES:
            raise AnalysisSchemaError(
                f"Field '{field_name}' has invalid status '{normalized_status}'."
            )

        evidence_quotes = self._normalize_quote_list(value.get("evidence_quotes"), transcript_text)
        notes = str(value.get("notes", "")).strip()

        if normalized_value == DEFAULT_UNKNOWN_VALUE and normalized_status == "observed":
            normalized_status = "unknown"
        if normalized_status == "observed" and not evidence_quotes and field_name != "summary":
            notes = notes or "Marked observed without a direct quote; review provider output."

        return AnalysisField(
            value=normalized_value,
            status=normalized_status,
            evidence_quotes=tuple(evidence_quotes),
            notes=notes,
        ).as_dict()

    def _normalize_quote_list(self, value: Any, transcript_text: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise AnalysisSchemaError("Quote collections must be arrays of strings.")

        normalized_quotes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise AnalysisSchemaError("Quotes must be strings.")
            aligned = _align_quote_to_transcript(item, transcript_text)
            if aligned and aligned not in normalized_quotes:
                normalized_quotes.append(aligned)
        return normalized_quotes

    def _normalize_confidence_signals(
        self,
        value: Any,
        normalized_fields: dict[str, Any],
    ) -> dict[str, list[str]]:
        if value is not None and not isinstance(value, dict):
            raise AnalysisSchemaError("'confidence_signals' must be an object.")

        observed = []
        missing_or_unknown = []
        for field_name in FIELD_NAMES:
            field = normalized_fields[field_name]
            label = f"{field_name}: {field['value']}"
            if field["status"] == "observed":
                observed.append(label)
            else:
                missing_or_unknown.append(label)

        if isinstance(value, dict):
            observed.extend(_string_list(value.get("observed_evidence")))
            missing_or_unknown.extend(_string_list(value.get("missing_or_unknown")))

        return {
            "observed_evidence": _dedupe_preserve_order(observed),
            "missing_or_unknown": _dedupe_preserve_order(missing_or_unknown),
        }


_NO_SMARTPHONE_PHRASES = (
    "no smartphone",
    "do not have a smartphone",
    "don't have a smartphone",
    "without smartphone",
    "i use a basic phone",
)

_NO_BANK_ACCOUNT_PHRASES = (
    "no bank account",
    "do not have a bank account",
    "don't have a bank account",
    "without bank account",
    "i am unbanked",
)


def _field_payload(
    *,
    value: str,
    status: str,
    evidence_quotes: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "value": value,
        "status": status,
        "evidence_quotes": evidence_quotes,
        "notes": notes,
    }


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _extract_direct_quotes(text: str) -> list[str]:
    if not text:
        return []
    segments = [segment.strip() for segment in re.split(r"[.?!\n]+", text) if segment.strip()]
    return segments[:5]


def _matching_quotes(quotes: list[str], keywords: tuple[str, ...]) -> list[str]:
    return [quote for quote in quotes if any(keyword in quote.lower() for keyword in keywords)][:3]


def _extract_income_range(text: str) -> str:
    lowered = text.lower()
    if not any(term in lowered for term in ("salary", "income", "earn", "rupees", "month")):
        return DEFAULT_UNKNOWN_VALUE
    match = re.search(r"(?:₹|rs\.?|rupees?)\s*([\d,]+(?:\s*(?:to|-|–)\s*[\d,]+)?)", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return DEFAULT_UNKNOWN_VALUE


def _extract_repayment_preference(lowered: str) -> str:
    if "monthly" in lowered or "month end" in lowered:
        return "monthly"
    if "weekly" in lowered:
        return "weekly"
    if "daily" in lowered:
        return "daily"
    if "flexible" in lowered or "when i can" in lowered:
        return "flexible"
    return DEFAULT_UNKNOWN_VALUE


def _extract_loan_interest(lowered: str) -> str:
    if any(term in lowered for term in ("want a loan", "need a loan", "interested in loan", "need money")):
        return "interested"
    if any(term in lowered for term in ("not interested", "do not want loan", "don't want loan", "avoid loan")):
        return "not_interested"
    if any(term in lowered for term in ("afraid", "fear", "scam", "trust", "worried")):
        return "fearful_or_uncertain"
    return DEFAULT_UNKNOWN_VALUE


def _build_summary(text: str) -> str:
    words = text.split()
    if not words:
        return "No transcript content available."
    summary = " ".join(words[:30])
    if len(words) > 30:
        summary += "..."
    return summary


def _build_confidence_signals(payload: dict[str, Any]) -> dict[str, list[str]]:
    observed = []
    missing_or_unknown = []
    for field_name in FIELD_NAMES:
        field = payload[field_name]
        entry = f"{field_name}: {field['value']}"
        if field["status"] == "observed":
            observed.append(entry)
        else:
            missing_or_unknown.append(entry)
    return {
        "observed_evidence": observed,
        "missing_or_unknown": missing_or_unknown,
    }


def _align_quote_to_transcript(quote: str, transcript_text: str) -> str:
    candidate = quote.strip().strip('"')
    if not candidate:
        return ""
    if candidate in transcript_text:
        return candidate

    lowered_transcript = transcript_text.lower()
    lowered_candidate = candidate.lower()
    start = lowered_transcript.find(lowered_candidate)
    if start == -1:
        return ""
    end = start + len(candidate)
    return transcript_text[start:end].strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisSchemaError("Confidence signal collections must be arrays of strings.")
    if any(not isinstance(item, str) for item in value):
        raise AnalysisSchemaError("Confidence signal collections must contain strings only.")
    return [item.strip() for item in value if item.strip()]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
