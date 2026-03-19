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
SMARTPHONE_HAS_VALUE = "has_smartphone"
SMARTPHONE_NO_VALUE = "no_smartphone"
BANK_ACCOUNT_HAS_VALUE = "has_bank_account"
BANK_ACCOUNT_NO_VALUE = "no_bank_account"
FIELD_NAMES = (
    "smartphone_usage",
    "bank_account_status",
    "per_household_earnings",
    "participant_personal_monthly_income",
    "total_household_monthly_income",
    "borrowing_history",
    "repayment_preference",
    "loan_interest",
    "summary",
)
TOP_LEVEL_SCHEMA_KEYS = FIELD_NAMES + ("key_quotes", "confidence_signals", "segmented_dialogue")
ALLOWED_FIELD_STATUSES = {"observed", "unknown", "missing"}
ALLOWED_SPEAKER_CONFIDENCE = {"high", "medium", "low"}
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
    evidence_type: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "evidence_quotes": list(self.evidence_quotes),
            "notes": self.notes,
            "evidence_type": self.evidence_type,
        }


class AnalysisSchemaError(ValueError):
    """Raised when an analysis response is not valid JSON or violates the schema."""


class ExternalProviderError(RuntimeError):
    """Raised when an external provider cannot complete an analysis request."""


class OpenAIAnalysisAdapter:
    """Live adapter that sends prompts to the OpenAI Responses API."""

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

        response_payload = self._transport(prompt=prompt, model=self.settings.model, api_key=api_key)
        if isinstance(response_payload, dict):
            error_payload = response_payload.get("error")
            if isinstance(error_payload, dict):
                message = str(error_payload.get("message", "")).strip()
                if message:
                    raise ExternalProviderError(message)
        raw_text = self._extract_output_text(response_payload)
        if not raw_text.strip():
            raise ExternalProviderError("OpenAI analysis request succeeded but returned no text output.")
        return raw_text

    def _default_transport(self, *, prompt: str, model: str, api_key: str) -> dict[str, Any]:
        payload = json.dumps({"model": model, "input": prompt, "text": {"format": {"type": "json_object"}}}).encode("utf-8")
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
            detail = body.strip() or exc.reason
            raise ExternalProviderError(
                f"OpenAI analysis request failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:  # pragma: no cover - depends on live provider
            raise ExternalProviderError(f"Unable to reach OpenAI Responses API: {exc.reason}") from exc
        except TimeoutError as exc:  # pragma: no cover - depends on live provider
            raise ExternalProviderError("Timed out while waiting for the OpenAI Responses API.") from exc

    @staticmethod
    def _extract_output_text(response_payload: dict[str, Any]) -> str:
        provider_error = response_payload.get("error")
        if isinstance(provider_error, dict):
            message = provider_error.get("message") or provider_error.get("type") or "unknown provider error"
            raise ExternalProviderError(f"OpenAI analysis provider error: {message}")

        direct_output = response_payload.get("output_text")
        if isinstance(direct_output, str) and direct_output.strip():
            return direct_output.strip()

        output_items = response_payload.get("output")
        if not isinstance(output_items, list):
            return ""

        text_chunks: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {None, "output_text", "text"}:
                    text_value = content.get("text")
                    if isinstance(text_value, str):
                        text_chunks.append(text_value)
        return "".join(text_chunks).strip()


class AnthropicAnalysisAdapter:
    """Placeholder adapter reserved for future Anthropic live support."""

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
        raise ExternalProviderError(
            "Anthropic live analysis is not implemented yet. Use OpenAI or enable sample mode."
        )



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
            **self._income_fields(text),
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
            "segmented_dialogue": _heuristic_segmented_dialogue(text),
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
            return _field(
                SMARTPHONE_NO_VALUE,
                negative,
                "Direct evidence that the participant lacks a smartphone.",
            )
        if evidence:
            return _field(
                SMARTPHONE_HAS_VALUE,
                evidence,
                "Direct evidence of smartphone or digital-app usage.",
            )
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
            return _field(
                BANK_ACCOUNT_NO_VALUE,
                negative,
                "Direct evidence that the participant lacks a bank account.",
            )
        if positive:
            return _field(
                BANK_ACCOUNT_HAS_VALUE,
                positive,
                "Direct evidence of an active or available bank account.",
            )
        return AnalysisField(notes="No direct transcript evidence about bank-account access.").as_dict()

    def _income_fields(self, text: str) -> dict[str, dict[str, Any]]:
        transcript = text.strip()
        lowered = transcript.lower()
        sentences = _split_sentences(transcript)

        extracted: dict[str, dict[str, Any]] = {}
        for field_name in INCOME_FIELD_NAMES:
            extracted[field_name] = AnalysisField(notes="No direct transcript evidence about income.").as_dict()

        patterns = {
            "per_household_earnings": (
                r"(?:from|in)\s+(?:one\s+)?(?:house|home|ghar)\S*[^₹\d]{0,25}(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))",
                r"(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))[^.]{0,30}(?:for|from)\s+(?:one\s+)?(?:house|home|ghar)",
                r"(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))[^.]{0,25}(?:per|a)\s+(?:house|home)",
            ),
            "participant_personal_monthly_income": (
                r"(?:my|i)\s+(?:monthly\s+income|income|earn|get|receive)[^.]{0,40}?(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))[^.]{0,20}(?:per\s+month|monthly)",
                r"(?:in\s+total|altogether|total)\s+(?:i\s+)?(?:earn|get|receive)[^.]{0,30}?(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))",
                r"(?:i\s+work\s+in\s+\d+\s+houses[^.]{0,30})?(?:my\s+salary|salary)[^.]{0,25}?(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))[^.]{0,20}(?:per\s+month|monthly)",
            ),
            "total_household_monthly_income": (
                r"(?:our|total)\s+(?:household|family)\s+(?:monthly\s+)?income[^.]{0,30}?(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))",
                r"(?:together|altogether)[^.]{0,35}(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))[^.]{0,20}(?:per\s+month|monthly)",
                r"(?:my\s+husband\s+and\s+i|all\s+of\s+us)[^.]{0,35}(₹\s?[\d,]+|[\d,]+\s*(?:rupees|rs))[^.]{0,20}(?:per\s+month|monthly)",
            ),
        }

        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match is None:
                    continue
                sentence = _sentence_for_span(sentences, match.span()) or _align_quote_to_transcript(match.group(0), transcript)
                if not sentence:
                    continue
                amount = _normalize_currency_amount(match.group(1))
                evidence_type = "direct"
                notes = _income_notes(field_name, evidence_type)
                extracted[field_name] = _field(amount, [sentence], notes, evidence_type=evidence_type)
                break

        if extracted["participant_personal_monthly_income"]["status"] != "observed":
            inferred = self._infer_participant_monthly_income(
                transcript=transcript,
                lowered=lowered,
                sentences=sentences,
                per_household_field=extracted["per_household_earnings"],
            )
            if inferred is not None:
                extracted["participant_personal_monthly_income"] = inferred

        return extracted

    def _infer_participant_monthly_income(
        self,
        *,
        transcript: str,
        lowered: str,
        sentences: list[str],
        per_household_field: dict[str, Any],
    ) -> dict[str, Any] | None:
        if per_household_field.get("status") != "observed":
            return None
        if any(token in lowered for token in ("total monthly income", "household income", "family income", "in total i earn")):
            return None
        house_count_match = re.search(r"(\d+)\s+(?:houses|homes)", lowered)
        amount_text = str(per_household_field.get("value", "")).strip()
        if not house_count_match or not amount_text.startswith("₹"):
            return None
        amount = int(amount_text.replace("₹", "").replace(",", ""))
        house_count = int(house_count_match.group(1))
        inferred_total = f"₹{amount * house_count:,}"
        supporting_sentence = _sentence_containing(sentences, house_count_match.group(0)) or _align_quote_to_transcript(house_count_match.group(0), transcript)
        evidence_quotes = [quote for quote in [*per_household_field.get("evidence_quotes", []), supporting_sentence] if quote]
        if not evidence_quotes:
            return None
        return _field(
            inferred_total,
            _dedupe_preserve_order(evidence_quotes),
            _income_notes("participant_personal_monthly_income", "inferred_or_uncertain"),
            evidence_type="inferred_or_uncertain",
        )

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
            "per_household_earnings": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "evidence_type": "direct|inferred_or_uncertain|unknown",
            },
            "participant_personal_monthly_income": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "evidence_type": "direct|inferred_or_uncertain|unknown",
            },
            "total_household_monthly_income": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "evidence_type": "direct|inferred_or_uncertain|unknown",
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
            "segmented_dialogue": [
                {
                    "speaker_label": "participant|interviewer|unknown",
                    "utterance_text": "string",
                    "speaker_confidence": "high|medium|low",
                    "speaker_uncertainty": "string",
                }
            ],
        }

    def _build_prompt(self, transcript: Transcript) -> str:
        metadata_json = json.dumps(_prompt_metadata(transcript.metadata), ensure_ascii=False, indent=2)
        task = (
            "Analyze the interview transcript and return only valid JSON that matches the required schema.\n\n"
            "Rules:\n"
            "- Use only evidence from the transcript.\n"
            "- Preserve direct quotes exactly when evidence exists.\n"
            "- Do not infer unsupported facts.\n"
            "- Mark fields as unknown when the transcript does not provide evidence.\n"
            "- Populate segmented_dialogue with ordered dialogue turns when the transcript makes speaker changes evident.\n"
            "- Separate interviewer questions from participant answers when the transcript wording or punctuation makes that distinction evident.\n"
            "- Use filename or interview metadata only as a weak hint for participant identity.\n"
            "- Do not fabricate unsupported speaker names or role tags; use 'unknown' and explain uncertainty when speaker identity is unclear.\n"
            "- The no-smartphone and no-bank-account cases must be clearly surfaced for downstream Persona 3 handling.\n\n"
            f"Required schema:\n{json.dumps(self.expected_schema(), indent=2)}\n\n"
            f"Interview metadata (weak hints only):\n{metadata_json}\n\n"
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
        normalized["segmented_dialogue"] = self._normalize_segmented_dialogue(
            payload.get("segmented_dialogue"),
            transcript_text,
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
        evidence_type = str(value.get("evidence_type", "")).strip().lower() or (
            "direct" if normalized_status == "observed" and evidence_quotes else "unknown"
        )
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            raise AnalysisSchemaError(f"Field '{field_name}' has invalid evidence_type '{evidence_type}'.")

        if normalized_value == DEFAULT_UNKNOWN_VALUE and normalized_status == "observed":
            normalized_status = "unknown"
        if field_name in INCOME_FIELD_NAMES and normalized_status == "observed" and not evidence_quotes:
            raise AnalysisSchemaError(f"Income field '{field_name}' must include at least one direct quote in evidence_quotes.")
        if normalized_status != "observed":
            evidence_type = "unknown"
        elif normalized_status == "observed" and not evidence_quotes and field_name != "summary":
            notes = notes or "Marked observed without a direct quote; review provider output."

        return AnalysisField(
            value=normalized_value,
            status=normalized_status,
            evidence_quotes=tuple(evidence_quotes),
            notes=notes,
            evidence_type=evidence_type,
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

    def _normalize_segmented_dialogue(self, value: Any, transcript_text: str) -> list[dict[str, str]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise AnalysisSchemaError("'segmented_dialogue' must be an array of dialogue-turn objects.")

        normalized_turns: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                raise AnalysisSchemaError("Each segmented dialogue turn must be an object.")

            speaker_label = str(item.get("speaker_label", "unknown")).strip().lower() or "unknown"
            utterance_text = str(item.get("utterance_text", "")).strip()
            speaker_confidence = str(item.get("speaker_confidence", "low")).strip().lower() or "low"
            speaker_uncertainty = str(item.get("speaker_uncertainty", "")).strip()

            if speaker_confidence not in ALLOWED_SPEAKER_CONFIDENCE:
                raise AnalysisSchemaError(
                    f"Segmented dialogue speaker_confidence must be one of {sorted(ALLOWED_SPEAKER_CONFIDENCE)}."
                )

            aligned_text = _align_quote_to_transcript(utterance_text, transcript_text)
            if not aligned_text:
                continue

            normalized_turn = {
                "speaker_label": speaker_label,
                "utterance_text": aligned_text,
                "speaker_confidence": speaker_confidence,
                "speaker_uncertainty": speaker_uncertainty,
            }
            if normalized_turn not in normalized_turns:
                normalized_turns.append(normalized_turn)
        return normalized_turns


def build_analysis_adapter(settings: LLMSettings, *, sample_mode: bool) -> AnalysisProviderAdapter:
    if sample_mode:
        return HeuristicAnalysisAdapter()

    provider = settings.provider.strip().lower()
    if provider == "openai":
        return OpenAIAnalysisAdapter(settings)
    if provider == "anthropic":
        return AnthropicAnalysisAdapter(
            LLMSettings(provider=f"{settings.provider}-heuristic", model="heuristic-json")
        )

    return HeuristicAnalysisAdapter(
        LLMSettings(provider=f"{settings.provider}-heuristic", model="heuristic-json")
    )


def _field(
    value: str,
    evidence_quotes: list[str],
    notes: str,
    *,
    evidence_type: str = "direct",
) -> dict[str, Any]:
    return AnalysisField(
        value=value,
        status="observed",
        evidence_quotes=tuple(evidence_quotes),
        notes=notes,
        evidence_type=evidence_type,
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




def _sentence_for_span(sentences: list[str], span: tuple[int, int]) -> str:
    start, end = span
    cursor = 0
    for sentence in sentences:
        sentence_start = cursor
        sentence_end = cursor + len(sentence)
        if sentence_start <= start <= sentence_end or sentence_start <= end <= sentence_end:
            return sentence
        cursor = sentence_end + 1
    return ""


def _sentence_containing(sentences: list[str], needle: str) -> str:
    lowered_needle = needle.lower()
    for sentence in sentences:
        if lowered_needle in sentence.lower():
            return sentence
    return ""


def _normalize_currency_amount(raw_amount: str) -> str:
    digits = re.sub(r"[^\d]", "", raw_amount)
    if not digits:
        return DEFAULT_UNKNOWN_VALUE
    return f"₹{int(digits):,}"


def _income_notes(field_name: str, evidence_type: str) -> str:
    labels = {
        "per_household_earnings": "Per-household earnings",
        "participant_personal_monthly_income": "Participant personal monthly income",
        "total_household_monthly_income": "Total household monthly income",
    }
    base = f"{labels[field_name]} identified from transcript evidence."
    if evidence_type == "inferred_or_uncertain":
        return f"{base} Marked inferred_or_uncertain because the transcript supports an estimate rather than an explicit total."
    return f"{base} Quoted directly by the participant."


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


def _prompt_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    allowed_keys = (
        "filename",
        "source",
        "file_id",
        "language",
        "duration_seconds",
        "participant_name_hint",
        "interview_metadata",
    )
    return {
        key: metadata[key]
        for key in allowed_keys
        if key in metadata and metadata[key] not in (None, "", {}, [])
    }


def _heuristic_segmented_dialogue(text: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for sentence in _split_sentences(text):
        if sentence.endswith("?"):
            speaker_label = "interviewer"
            speaker_confidence = "high"
            speaker_uncertainty = ""
        elif re.search(r"\b(i|my|me|we|our|us)\b", sentence, re.IGNORECASE):
            speaker_label = "participant"
            speaker_confidence = "medium"
            speaker_uncertainty = (
                "Speaker inferred as participant because the utterance is framed in first person, but the transcript does not explicitly tag speakers."
            )
        else:
            speaker_label = "unknown"
            speaker_confidence = "low"
            speaker_uncertainty = "Transcript text did not clearly identify which speaker produced this utterance."
        turns.append(
            {
                "speaker_label": speaker_label,
                "utterance_text": sentence,
                "speaker_confidence": speaker_confidence,
                "speaker_uncertainty": speaker_uncertainty,
            }
        )
    return turns


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
            content_type = content_item.get("type")
            text = content_item.get("text")
            if isinstance(text, str) and text.strip() and content_type in {None, "output_text"}:
                collected.append(text)
    return "\n".join(collected)


__all__ = [
    "AnalysisProviderAdapter",
    "AnalysisSchemaError",
    "AnalysisService",
    "AnthropicAnalysisAdapter",
    "BANK_ACCOUNT_HAS_VALUE",
    "BANK_ACCOUNT_NO_VALUE",
    "ExternalProviderError",
    "HeuristicAnalysisAdapter",
    "OpenAIAnalysisAdapter",
    "SMARTPHONE_HAS_VALUE",
    "SMARTPHONE_NO_VALUE",
    "bank_account_user_from_analysis",
    "build_analysis_adapter",
    "get_analysis_evidence_quotes",
    "get_analysis_field",
    "get_analysis_value",
    "smartphone_user_from_analysis",
]
