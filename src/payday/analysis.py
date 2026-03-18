from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib import error, request

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
TOP_LEVEL_SCHEMA_KEYS = FIELD_NAMES + ("key_quotes", "confidence_signals")
ALLOWED_FIELD_STATUSES = {"observed", "unknown", "missing"}
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_REQUEST_TIMEOUT_SECONDS = 60


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


class ExternalProviderError(RuntimeError):
    """Raised when an external provider cannot complete an analysis request."""


class OpenAIAnalysisAdapter:
    """Concrete adapter for OpenAI-compatible structured-analysis calls."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.provider_name = settings.provider
        self.model_name = settings.model
        self._transport = transport or self._default_transport

    def generate_analysis(
        self,
        *,
        prompt: str,
        transcript: Transcript,
        expected_schema: dict[str, Any],
        attempt: int,
    ) -> str:
        del transcript, expected_schema, attempt
        api_key = self.settings.api_key.strip()
        if not api_key:
            raise ExternalProviderError(
                f"LLM_API_KEY is required when sample mode is disabled for provider '{self.settings.provider}'."
            )

        response_payload = self._transport(
            prompt=prompt,
            model=self.settings.model,
            api_key=api_key,
        )
        return self._extract_output_text(response_payload)

    def _default_transport(self, *, prompt: str, model: str, api_key: str) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": model,
                "input": prompt,
                "text": {"format": {"type": "json_object"}},
            }
        ).encode("utf-8")
        http_request = request.Request(
            OPENAI_RESPONSES_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=OPENAI_REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - depends on live provider
            body = exc.read().decode("utf-8", errors="ignore")
            raise ExternalProviderError(f"OpenAI analysis request failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:  # pragma: no cover - depends on live provider
            raise ExternalProviderError(f"OpenAI analysis request failed: {exc.reason}") from exc

    def _extract_output_text(self, response_payload: Any) -> str:
        if isinstance(response_payload, str):
            return response_payload
        if not isinstance(response_payload, dict):
            raise ExternalProviderError("OpenAI analysis provider returned an unsupported response shape.")

        provider_error = response_payload.get("error")
        if isinstance(provider_error, dict):
            message = provider_error.get("message") or provider_error.get("type") or "unknown provider error"
            raise ExternalProviderError(f"OpenAI analysis provider error: {message}")

        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = response_payload.get("output")
        if isinstance(output, list):
            collected: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        collected.append(text_value)
            if collected:
                return "\n".join(collected)

        raise ExternalProviderError("OpenAI analysis provider response did not include output text.")


class HeuristicAnalysisAdapter:
    """Local adapter that emits schema-compliant JSON for development and tests."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        resolved = settings or LLMSettings(provider="heuristic", model="heuristic-json")
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
        del prompt, attempt
        text = transcript.text.strip()
        sentences = _split_sentences(text)
        key_quotes = sentences[:2]
        payload: dict[str, Any] = {
            "smartphone_usage": self._smartphone_usage(text),
            "bank_account_status": self._bank_account_status(text),
            "income_range": self._income_range(text),
            "borrowing_history": self._borrowing_history(text),
            "repayment_preference": self._repayment_preference(text),
            "loan_interest": self._loan_interest(text),
            "summary": {
                "value": self._summary(text),
                "status": "observed" if text else "missing",
                "evidence_quotes": key_quotes,
                "notes": "Heuristic sample-mode summary grounded in transcript text.",
            },
            "key_quotes": key_quotes,
        }
        payload["confidence_signals"] = _build_confidence_signals(payload)
        extra_keys = set(payload) - set(expected_schema)
        if extra_keys:
            raise AnalysisSchemaError(
                f"Heuristic adapter produced unexpected keys: {sorted(extra_keys)}"
            )
        return json.dumps(payload, ensure_ascii=False)

    def _smartphone_usage(self, text: str) -> dict[str, Any]:
        evidence = _matching_phrases(
            text,
            ("whatsapp", "smartphone", "upi", "google pay", "phonepe", "paytm"),
        )
        negative = _matching_phrases(
            text,
            ("basic phone", "feature phone", "no smartphone", "without smartphone"),
        )
        if negative:
            return _field("no_smartphone", negative, "Direct evidence that the participant lacks a smartphone.")
        if evidence:
            return _field("has_smartphone", evidence, "Direct evidence of smartphone or digital-app usage.")
        return AnalysisField(notes="No direct transcript evidence about smartphone access.").as_dict()

    def _bank_account_status(self, text: str) -> dict[str, Any]:
        positive = _matching_phrases(
            text,
            ("bank account", "salary in bank", "account is active", "money goes to bank"),
        )
        negative = _matching_phrases(
            text,
            ("no bank account", "without bank account", "unbanked", "do not have a bank account"),
        )
        if negative:
            return _field("no_bank_account", negative, "Direct evidence that the participant lacks a bank account.")
        if positive:
            return _field("has_bank_account", positive, "Direct evidence of an active or available bank account.")
        return AnalysisField(notes="No direct transcript evidence about bank-account access.").as_dict()

    def _income_range(self, text: str) -> dict[str, Any]:
        rupee_match = re.search(r"₹\s?([\d,]+)", text)
        monthly_match = re.search(r"(\d{1,2}[,\d]{0,4})\s*(?:rupees|rs)\s*(?:per month|monthly)", text, re.IGNORECASE)
        if rupee_match:
            amount = rupee_match.group(1)
            return _field(f"₹{amount}", _evidence_for_match(text, rupee_match.group(0)), "Income amount quoted directly.")
        if monthly_match:
            amount = monthly_match.group(1)
            return _field(f"₹{amount}", _evidence_for_match(text, monthly_match.group(0)), "Income amount quoted directly.")
        return AnalysisField(notes="No direct transcript evidence about income.").as_dict()

    def _borrowing_history(self, text: str) -> dict[str, Any]:
        borrowed = _matching_phrases(
            text,
            (
                "borrow",
                "borrowed",
                "loan",
                "employer first",
                "neighbors",
                "moneylender",
                "family",
            ),
        )
        not_borrowing = _matching_phrases(text, ("do not borrow", "don't borrow", "avoid loans", "use my savings"))
        if not_borrowing:
            return _field("has_not_borrowed_recently", not_borrowing, "Direct evidence of non-borrowing or self-reliance.")
        if borrowed:
            return _field("has_borrowed", borrowed, "Direct evidence that the participant borrows or has borrowed.")
        return AnalysisField(notes="No direct transcript evidence about borrowing history.").as_dict()

    def _repayment_preference(self, text: str) -> dict[str, Any]:
        for value, phrases in {
            "daily": ("daily", "every day"),
            "weekly": ("weekly", "every week"),
            "monthly": ("monthly", "after salary", "every month"),
        }.items():
            evidence = _matching_phrases(text, phrases)
            if evidence:
                return _field(value, evidence, "Repayment timing stated directly in the transcript.")
        return AnalysisField(notes="No direct transcript evidence about repayment preference.").as_dict()

    def _loan_interest(self, text: str) -> dict[str, Any]:
        fearful = _matching_phrases(text, ("afraid", "fear", "scam", "worried", "don't trust", "do not trust"))
        interested = _matching_phrases(text, ("interested", "would take", "want loan", "need a loan"))
        not_interested = _matching_phrases(text, ("not interested", "don't want a loan", "do not want a loan"))
        if fearful:
            return _field(
                "fearful_or_uncertain",
                fearful,
                "Trust or safety concern is explicitly mentioned in the transcript.",
            )
        if interested:
            return _field("interested", interested, "Participant directly expresses interest in borrowing.")
        if not_interested:
            return _field("not_interested", not_interested, "Participant directly declines loan interest.")
        return AnalysisField(notes="No direct transcript evidence about loan interest.").as_dict()

    def _summary(self, text: str) -> str:
        if not text:
            return "No transcript content available."
        sentences = _split_sentences(text)
        if sentences:
            return " ".join(sentences[:2])
        words = text.split()
        return " ".join(words[:24]) + ("..." if len(words) > 24 else "")


class AnthropicAnalysisAdapter:
    """Placeholder adapter reserved for future live Anthropic support."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.provider_name = settings.provider
        self.model_name = settings.model

    def generate_analysis(
        self,
        *,
        prompt: str,
        transcript: Transcript,
        expected_schema: dict[str, Any],
        attempt: int,
    ) -> str:
        del prompt, transcript, expected_schema, attempt
        raise RuntimeError(
            "Anthropic analysis adapter is reserved for future live provider support and is not implemented yet."
        )


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
        self.adapter = adapter or HeuristicAnalysisAdapter()
        self.memory = memory or load_context_memory()
        self.max_json_retries = max_json_retries

    def analyze(self, transcript: Transcript) -> AnalysisResult:
        prompt = self._build_prompt(transcript)
        last_error: AnalysisSchemaError | None = None

        for attempt in range(1, self.max_json_retries + 2):
            raw_response = self.adapter.generate_analysis(
                prompt=prompt,
                transcript=transcript,
                expected_schema=self.expected_schema(),
                attempt=attempt,
            )
            try:
                structured_output = self._parse_and_validate(raw_response, transcript.text)
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
                    evidence_quotes=structured_output["key_quotes"],
                )
            except AnalysisSchemaError as exc:
                last_error = exc
                prompt = self._build_retry_prompt(
                    transcript,
                    previous_prompt=prompt,
                    error=exc,
                    raw_response=raw_response,
                )

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
        del transcript
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
        unexpected_keys = set(payload) - set(TOP_LEVEL_SCHEMA_KEYS)
        if unexpected_keys:
            raise AnalysisSchemaError(
                f"Top-level JSON contains unexpected keys: {sorted(unexpected_keys)}."
            )

        normalized: dict[str, Any] = {}
        for field_name in FIELD_NAMES:
            normalized[field_name] = self._normalize_field(field_name, payload.get(field_name), transcript_text)

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
            raise AnalysisSchemaError(f"Field '{field_name}' has invalid status '{normalized_status}'.")

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

        observed: list[str] = []
        missing_or_unknown: list[str] = []
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


def build_analysis_adapter(settings: LLMSettings, *, sample_mode: bool) -> AnalysisProviderAdapter:
    if sample_mode:
        return HeuristicAnalysisAdapter()

    provider = settings.provider.strip().lower()
    if provider == "openai":
        return OpenAIAnalysisAdapter(settings)
    if provider == "anthropic":
        return AnthropicAnalysisAdapter(settings)

    raise ValueError(f"Unsupported LLM provider '{settings.provider}' for live analysis.")


def _field(value: str, evidence_quotes: list[str], notes: str) -> dict[str, Any]:
    return AnalysisField(
        value=value,
        status="observed",
        evidence_quotes=tuple(evidence_quotes),
        notes=notes,
    ).as_dict()


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]


def _matching_phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for phrase in phrases:
        if phrase in lowered:
            aligned = _align_quote_to_transcript(phrase, text)
            matches.append(aligned or phrase)
    return _dedupe_preserve_order(matches)


def _evidence_for_match(text: str, matched_text: str) -> list[str]:
    aligned = _align_quote_to_transcript(matched_text, text)
    return [aligned] if aligned else [matched_text]


def _align_quote_to_transcript(candidate: str, transcript_text: str) -> str:
    candidate = candidate.strip()
    transcript_text = transcript_text.strip()
    if not candidate or not transcript_text:
        return ""
    if candidate in transcript_text:
        return candidate

    lowered_candidate = candidate.lower()
    lowered_transcript = transcript_text.lower()
    start = lowered_transcript.find(lowered_candidate)
    if start == -1:
        for sentence in _split_sentences(transcript_text):
            if lowered_candidate in sentence.lower():
                return sentence.strip()
        return candidate
    end = start + len(candidate)
    return transcript_text[start:end]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisSchemaError("Confidence signal collections must be arrays of strings.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise AnalysisSchemaError("Confidence signal entries must be strings.")
        normalized = item.strip()
        if normalized:
            items.append(normalized)
    return items


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _build_confidence_signals(payload: dict[str, Any]) -> dict[str, list[str]]:
    observed: list[str] = []
    missing_or_unknown: list[str] = []
    for field_name in FIELD_NAMES:
        field = payload[field_name]
        label = f"{field_name}: {field['value']}"
        if field["status"] == "observed":
            observed.append(label)
        else:
            missing_or_unknown.append(label)
    return {
        "observed_evidence": observed,
        "missing_or_unknown": missing_or_unknown,
    }


def _extract_openai_output_text(payload: dict[str, Any]) -> str:
    direct_output = payload.get("output_text")
    if isinstance(direct_output, str) and direct_output.strip():
        return direct_output

    collected: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content_item in item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    collected.append(text)
    return "\n".join(collected)
