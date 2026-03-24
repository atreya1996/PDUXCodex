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
INCOME_FIELD_NAMES = (
    "per_household_earnings",
    "participant_personal_monthly_income",
    "total_household_monthly_income",
)
TOP_LEVEL_SCHEMA_KEYS = FIELD_NAMES + (
    "key_quotes",
    "key_quote_details",
    "income_mentions",
    "confidence_signals",
    "segmented_dialogue",
    "transcript_quality",
)
ALLOWED_FIELD_STATUSES = {"observed", "unknown", "missing"}
ALLOWED_EVIDENCE_TYPES = {"direct", "inferred_or_uncertain", "unknown"}
ALLOWED_SPEAKER_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_TRANSCRIPT_QUALITY_STATUS = {"clean", "degraded"}
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_REQUEST_TIMEOUT_SECONDS = 60
NO_RELIABLE_QUOTE_PLACEHOLDER = "No reliable quote extracted"


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
    english_translation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "evidence_quotes": list(self.evidence_quotes),
            "notes": self.notes,
            "evidence_type": self.evidence_type,
            "english_translation": self.english_translation,
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
        segmented_dialogue = _heuristic_segmented_dialogue(text)
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
                "english_translation": self._summary(text),
            },
            "key_quotes": key_quotes,
            "key_quote_details": _build_key_quote_details(
                key_quotes,
                segmented_dialogue=segmented_dialogue,
            ),
            "segmented_dialogue": segmented_dialogue,
        }
        payload["income_mentions"] = _build_income_mentions(
            payload,
            segmented_dialogue=segmented_dialogue,
        )
        payload["confidence_signals"] = _build_confidence_signals(payload)
        payload["transcript_quality"] = {
            "status": "clean",
            "dropped_malformed_quote_count": 0,
            "notes": [],
        }
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

        return extracted

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
                "english_translation": "string",
            },
            "bank_account_status": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "english_translation": "string",
            },
            "per_household_earnings": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "evidence_type": "direct|inferred_or_uncertain|unknown",
                "english_translation": "string",
            },
            "participant_personal_monthly_income": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "evidence_type": "direct|inferred_or_uncertain|unknown",
                "english_translation": "string",
            },
            "total_household_monthly_income": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "evidence_type": "direct|inferred_or_uncertain|unknown",
                "english_translation": "string",
            },
            "borrowing_history": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "english_translation": "string",
            },
            "repayment_preference": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "english_translation": "string",
            },
            "loan_interest": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "english_translation": "string",
            },
            "summary": {
                "value": "string",
                "status": "observed|unknown|missing",
                "evidence_quotes": ["string"],
                "notes": "string",
                "english_translation": "string",
            },
            "key_quotes": ["string"],
            "key_quote_details": [
                {
                    "original_text": "string",
                    "english_translation": "string",
                    "speaker_label": "participant|interviewer|unknown",
                    "turn_index": "integer|null",
                }
            ],
            "income_mentions": [
                {
                    "meaning_label": "per_household_earnings|participant_personal_monthly_income|total_household_monthly_income|other_income_mention",
                    "amount": "string",
                    "evidence_quote": "string",
                    "english_translation": "string",
                    "speaker_label": "participant|interviewer|unknown",
                    "turn_index": "integer|null",
                    "evidence_type": "direct|inferred_or_uncertain|unknown",
                }
            ],
            "confidence_signals": {
                "observed_evidence": ["string"],
                "missing_or_unknown": ["string"],
            },
            "transcript_quality": {
                "status": "clean|degraded",
                "dropped_malformed_quote_count": "integer",
                "notes": ["string"],
            },
            "segmented_dialogue": [
                {
                    "turn_index": "integer",
                    "speaker_label": "participant|interviewer|unknown",
                    "utterance_text": "string",
                    "english_translation": "string",
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
            "- Never treat wages from one house/home as the participant's full monthly income unless the participant explicitly states that total.\n"
            "- If multiple income numbers appear, capture each income mention with the correct meaning label and keep participant personal income separate from total household income.\n"
            "- For each populated income field, include a direct evidence quote and preserve the original transcript wording.\n"
            "- Add summary-only English translations for key quotes/snippets while preserving the original transcript text in evidence quotes and dialogue turns.\n"
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
        dropped_malformed_quote_count = 0
        for field_name in FIELD_NAMES:
            field_payload, dropped_count = self._normalize_field(field_name, payload.get(field_name), transcript_text)
            normalized[field_name] = field_payload
            dropped_malformed_quote_count += dropped_count

        normalized["key_quotes"], dropped_key_quote_count = self._normalize_quote_list(
            payload.get("key_quotes"),
            transcript_text,
        )
        dropped_malformed_quote_count += dropped_key_quote_count
        normalized["key_quote_details"] = self._normalize_key_quote_details(
            payload.get("key_quote_details"),
            normalized["key_quotes"],
            transcript_text,
        )
        normalized["income_mentions"] = self._normalize_income_mentions(
            payload.get("income_mentions"),
            transcript_text,
        )
        normalized["confidence_signals"] = self._normalize_confidence_signals(
            payload.get("confidence_signals"),
            normalized,
        )
        normalized["segmented_dialogue"] = self._normalize_segmented_dialogue(
            payload.get("segmented_dialogue"),
            transcript_text,
        )
        normalized["transcript_quality"] = self._normalize_transcript_quality(
            payload.get("transcript_quality"),
            dropped_malformed_quote_count=dropped_malformed_quote_count,
        )
        return normalized

    def _normalize_field(self, field_name: str, value: Any, transcript_text: str) -> tuple[dict[str, Any], int]:
        if value is None:
            return AnalysisField(
                notes=f"Fallback default applied because '{field_name}' was missing from the provider response.",
            ).as_dict(), 0

        if isinstance(value, str):
            normalized_status = "unknown" if value.strip().lower() == DEFAULT_UNKNOWN_VALUE else "observed"
            return AnalysisField(value=value.strip() or DEFAULT_UNKNOWN_VALUE, status=normalized_status).as_dict(), 0

        if not isinstance(value, dict):
            raise AnalysisSchemaError(f"Field '{field_name}' must be an object or string.")

        normalized_value = str(value.get("value", DEFAULT_UNKNOWN_VALUE)).strip() or DEFAULT_UNKNOWN_VALUE
        normalized_status = str(value.get("status", "unknown")).strip().lower() or "unknown"
        if normalized_status not in ALLOWED_FIELD_STATUSES:
            raise AnalysisSchemaError(f"Field '{field_name}' has invalid status '{normalized_status}'.")

        evidence_quotes, dropped_malformed_quote_count = self._normalize_quote_list(
            value.get("evidence_quotes"),
            transcript_text,
        )
        notes = str(value.get("notes", "")).strip()
        evidence_type = str(value.get("evidence_type", "")).strip().lower() or (
            "direct" if normalized_status == "observed" and evidence_quotes else "unknown"
        )
        english_translation = str(value.get("english_translation", "")).strip()
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
            english_translation=english_translation,
        ).as_dict(), dropped_malformed_quote_count

    def _normalize_quote_list(self, value: Any, transcript_text: str) -> tuple[list[str], int]:
        if value is None:
            return [], 0
        if not isinstance(value, list):
            raise AnalysisSchemaError("Quote collections must be arrays of strings.")

        normalized_quotes: list[str] = []
        dropped_malformed_quote_count = 0
        for item in value:
            if not isinstance(item, str):
                raise AnalysisSchemaError("Quotes must be strings.")
            aligned = _align_quote_to_transcript(item, transcript_text)
            if not _is_reliable_quote_text(aligned):
                dropped_malformed_quote_count += 1
                continue
            if aligned and aligned not in normalized_quotes:
                normalized_quotes.append(aligned)
        return normalized_quotes, dropped_malformed_quote_count

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

    def _normalize_key_quote_details(
        self,
        value: Any,
        key_quotes: list[str],
        transcript_text: str,
    ) -> list[dict[str, Any]]:
        if value is None:
            return _build_key_quote_details(key_quotes)
        if not isinstance(value, list):
            raise AnalysisSchemaError("'key_quote_details' must be an array of quote-detail objects.")

        normalized_details: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise AnalysisSchemaError("Each key quote detail must be an object.")
            original_text = _align_quote_to_transcript(str(item.get("original_text", "")).strip(), transcript_text)
            if not _is_reliable_quote_text(original_text):
                continue
            speaker_label = str(item.get("speaker_label", "unknown")).strip().lower() or "unknown"
            turn_index = item.get("turn_index")
            if turn_index is not None and not isinstance(turn_index, int):
                raise AnalysisSchemaError("key_quote_details.turn_index must be an integer or null.")
            normalized_item = {
                "original_text": original_text,
                "english_translation": str(item.get("english_translation", "")).strip(),
                "speaker_label": speaker_label,
                "turn_index": turn_index,
            }
            if normalized_item not in normalized_details:
                normalized_details.append(normalized_item)

        if not normalized_details:
            return _build_key_quote_details(key_quotes)
        return normalized_details

    def _normalize_income_mentions(
        self,
        value: Any,
        transcript_text: str,
    ) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise AnalysisSchemaError("'income_mentions' must be an array of income-mention objects.")

        normalized_mentions: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise AnalysisSchemaError("Each income mention must be an object.")
            meaning_label = str(item.get("meaning_label", "")).strip()
            amount = str(item.get("amount", "")).strip() or DEFAULT_UNKNOWN_VALUE
            evidence_quote = _align_quote_to_transcript(str(item.get("evidence_quote", "")).strip(), transcript_text)
            english_translation = str(item.get("english_translation", "")).strip()
            speaker_label = str(item.get("speaker_label", "unknown")).strip().lower() or "unknown"
            turn_index = item.get("turn_index")
            evidence_type = str(item.get("evidence_type", "unknown")).strip().lower() or "unknown"

            if turn_index is not None and not isinstance(turn_index, int):
                raise AnalysisSchemaError("income_mentions.turn_index must be an integer or null.")
            if evidence_type not in ALLOWED_EVIDENCE_TYPES:
                raise AnalysisSchemaError(
                    f"Income mention '{meaning_label or 'unknown'}' has invalid evidence_type '{evidence_type}'."
                )
            if not meaning_label or not _is_reliable_quote_text(evidence_quote):
                continue

            normalized_item = {
                "meaning_label": meaning_label,
                "amount": amount,
                "evidence_quote": evidence_quote,
                "english_translation": english_translation,
                "speaker_label": speaker_label,
                "turn_index": turn_index,
                "evidence_type": evidence_type,
            }
            if normalized_item not in normalized_mentions:
                normalized_mentions.append(normalized_item)
        return normalized_mentions

    def _normalize_segmented_dialogue(self, value: Any, transcript_text: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise AnalysisSchemaError("'segmented_dialogue' must be an array of dialogue-turn objects.")

        normalized_turns: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise AnalysisSchemaError("Each segmented dialogue turn must be an object.")

            turn_index = item.get("turn_index", len(normalized_turns))
            speaker_label = str(item.get("speaker_label", "unknown")).strip().lower() or "unknown"
            utterance_text = str(item.get("utterance_text", "")).strip()
            english_translation = str(item.get("english_translation", "")).strip()
            speaker_confidence = str(item.get("speaker_confidence", "low")).strip().lower() or "low"
            speaker_uncertainty = str(item.get("speaker_uncertainty", "")).strip()

            if not isinstance(turn_index, int):
                raise AnalysisSchemaError("Segmented dialogue turn_index must be an integer.")
            if speaker_confidence not in ALLOWED_SPEAKER_CONFIDENCE:
                raise AnalysisSchemaError(
                    f"Segmented dialogue speaker_confidence must be one of {sorted(ALLOWED_SPEAKER_CONFIDENCE)}."
                )

            aligned_text = _align_quote_to_transcript(utterance_text, transcript_text)
            if not _is_reliable_quote_text(aligned_text):
                continue

            normalized_turn = {
                "turn_index": turn_index,
                "speaker_label": speaker_label,
                "utterance_text": aligned_text,
                "english_translation": english_translation,
                "speaker_confidence": speaker_confidence,
                "speaker_uncertainty": speaker_uncertainty,
            }
            if normalized_turn not in normalized_turns:
                normalized_turns.append(normalized_turn)
        return normalized_turns

    def _normalize_transcript_quality(
        self,
        value: Any,
        *,
        dropped_malformed_quote_count: int,
    ) -> dict[str, Any]:
        status = "degraded" if dropped_malformed_quote_count > 0 else "clean"
        notes: list[str] = []
        if dropped_malformed_quote_count > 0:
            notes.append("Malformed or non-linguistic quote candidates were removed before evidence storage.")

        if value is not None:
            if not isinstance(value, dict):
                raise AnalysisSchemaError("'transcript_quality' must be an object.")
            provided_status = str(value.get("status", "")).strip().lower()
            if provided_status and provided_status not in ALLOWED_TRANSCRIPT_QUALITY_STATUS:
                raise AnalysisSchemaError(
                    f"'transcript_quality.status' must be one of {sorted(ALLOWED_TRANSCRIPT_QUALITY_STATUS)}."
                )
            notes.extend(_string_list(value.get("notes")))

        return {
            "status": status,
            "dropped_malformed_quote_count": dropped_malformed_quote_count,
            "notes": _dedupe_preserve_order(notes),
        }


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
    english_translation: str = "",
) -> dict[str, Any]:
    translation = english_translation.strip() or (evidence_quotes[0].strip() if evidence_quotes else "")
    return AnalysisField(
        value=value,
        status="observed",
        evidence_quotes=tuple(evidence_quotes),
        notes=notes,
        evidence_type=evidence_type,
        english_translation=translation,
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


def _is_reliable_quote_text(value: str) -> bool:
    text = str(value).strip()
    if len(text) < 4:
        return False
    if text.lower() == NO_RELIABLE_QUOTE_PLACEHOLDER.lower():
        return False
    if not any(char.isalpha() for char in text):
        return False
    if re.fullmatch(r"[01\s|,.;:_\-]{4,}", text):
        return False
    compact = text.replace(" ", "")
    if len(compact) >= 24 and re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return False
    punctuation_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    if punctuation_count / max(len(text), 1) > 0.45:
        return False
    return True


def get_income_display_value(structured_output: dict[str, Any]) -> str | None:
    labels = {
        "participant_personal_monthly_income": "Participant monthly income",
        "total_household_monthly_income": "Household monthly income",
        "per_household_earnings": "Per-household earnings",
    }
    for field_name in (
        "participant_personal_monthly_income",
        "total_household_monthly_income",
        "per_household_earnings",
    ):
        value = get_analysis_value(structured_output, field_name).strip()
        if value and value != DEFAULT_UNKNOWN_VALUE:
            return f"{labels[field_name]}: {value}"
    return None




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


def _heuristic_segmented_dialogue(text: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for turn_index, sentence in enumerate(_split_sentences(text)):
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
                "turn_index": turn_index,
                "speaker_label": speaker_label,
                "utterance_text": sentence,
                "english_translation": sentence,
                "speaker_confidence": speaker_confidence,
                "speaker_uncertainty": speaker_uncertainty,
            }
        )
    return turns


def _build_key_quote_details(
    key_quotes: list[str],
    *,
    segmented_dialogue: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    turns = segmented_dialogue or []
    details: list[dict[str, Any]] = []
    for quote in key_quotes:
        matched_turn = next((turn for turn in turns if quote == turn.get("utterance_text")), None)
        details.append(
            {
                "original_text": quote,
                "english_translation": quote,
                "speaker_label": str(matched_turn.get("speaker_label", "unknown")) if matched_turn else "unknown",
                "turn_index": matched_turn.get("turn_index") if matched_turn else None,
            }
        )
    return details


def _build_income_mentions(
    payload: dict[str, Any],
    *,
    segmented_dialogue: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    turns = segmented_dialogue or []
    mentions: list[dict[str, Any]] = []
    for field_name in INCOME_FIELD_NAMES:
        field = payload.get(field_name)
        if not isinstance(field, dict) or field.get("status") != "observed":
            continue
        evidence_quotes = field.get("evidence_quotes")
        if not isinstance(evidence_quotes, list) or not evidence_quotes:
            continue
        evidence_quote = str(evidence_quotes[0]).strip()
        matched_turn = next((turn for turn in turns if evidence_quote == turn.get("utterance_text")), None)
        mentions.append(
            {
                "meaning_label": field_name,
                "amount": str(field.get("value", DEFAULT_UNKNOWN_VALUE)).strip() or DEFAULT_UNKNOWN_VALUE,
                "evidence_quote": evidence_quote,
                "english_translation": str(field.get("english_translation", "")).strip() or evidence_quote,
                "speaker_label": str(matched_turn.get("speaker_label", "unknown")) if matched_turn else "unknown",
                "turn_index": matched_turn.get("turn_index") if matched_turn else None,
                "evidence_type": str(field.get("evidence_type", "unknown")).strip() or "unknown",
            }
        )
    return mentions


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


def get_analysis_field(structured_output: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = structured_output.get(field_name)
    if isinstance(value, dict):
        return value
    legacy_field = _legacy_field_from_nested_sections(structured_output, field_name)
    if legacy_field is not None:
        return legacy_field
    if value is None:
        return AnalysisField().as_dict()
    normalized_status = "unknown" if str(value).strip().lower() == DEFAULT_UNKNOWN_VALUE else "observed"
    return AnalysisField(value=str(value).strip() or DEFAULT_UNKNOWN_VALUE, status=normalized_status).as_dict()


def get_analysis_value(structured_output: dict[str, Any], field_name: str, default: str = DEFAULT_UNKNOWN_VALUE) -> str:
    field = get_analysis_field(structured_output, field_name)
    value = field.get("value")
    if isinstance(value, str) and value.strip():
        return value
    return default


def get_analysis_evidence_quotes(structured_output: dict[str, Any], field_name: str) -> list[str]:
    field = get_analysis_field(structured_output, field_name)
    evidence_quotes = field.get("evidence_quotes")
    if not isinstance(evidence_quotes, list):
        return []
    return clean_evidence_quotes(evidence_quotes)


def smartphone_user_from_analysis(structured_output: dict[str, Any]) -> bool | None:
    value = get_analysis_value(structured_output, "smartphone_usage")
    if value == SMARTPHONE_HAS_VALUE:
        return True
    if value == SMARTPHONE_NO_VALUE:
        return False
    return None


def bank_account_user_from_analysis(structured_output: dict[str, Any]) -> bool | None:
    value = get_analysis_value(structured_output, "bank_account_status")
    if value == BANK_ACCOUNT_HAS_VALUE:
        return True
    if value == BANK_ACCOUNT_NO_VALUE:
        return False
    return None


def get_persona_signal_value(structured_output: dict[str, Any], signal_name: str) -> bool | None:
    signal = _nested_object(structured_output, "persona_signals", signal_name)
    value = signal.get("value")
    if isinstance(value, bool):
        return value
    return None


def get_persona_signal_evidence_quotes(structured_output: dict[str, Any], signal_name: str) -> list[str]:
    signal = _nested_object(structured_output, "persona_signals", signal_name)
    evidence = signal.get("evidence_quotes")
    if not isinstance(evidence, list):
        evidence = signal.get("evidence")
    if not isinstance(evidence, list):
        return []
    return clean_evidence_quotes(evidence)


def clean_evidence_quotes(quotes: list[Any] | tuple[Any, ...], *, limit: int | None = None) -> list[str]:
    cleaned: list[str] = []
    for item in quotes:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not _is_reliable_quote_text(normalized):
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def has_borrowing_evidence(structured_output: dict[str, Any]) -> bool:
    borrowing_history = get_analysis_value(structured_output, "borrowing_history")
    if borrowing_history == "has_borrowed":
        return True
    if get_persona_signal_evidence_quotes(structured_output, "cyclical_borrowing"):
        return True
    if get_persona_signal_evidence_quotes(structured_output, "digital_borrowing"):
        return True
    return False


def _legacy_field_from_nested_sections(structured_output: dict[str, Any], field_name: str) -> dict[str, Any] | None:
    if field_name == "smartphone_usage":
        legacy_value = _nested_boolean(structured_output, "participant_profile", "smartphone_user")
        if legacy_value is None:
            return None
        return _legacy_boolean_field(
            value=legacy_value,
            positive_value=SMARTPHONE_HAS_VALUE,
            negative_value=SMARTPHONE_NO_VALUE,
            evidence_quotes=_nested_evidence_quotes(structured_output, "participant_profile", "smartphone_user"),
        )

    if field_name == "bank_account_status":
        legacy_value = _nested_boolean(structured_output, "participant_profile", "has_bank_account")
        if legacy_value is None:
            return None
        return _legacy_boolean_field(
            value=legacy_value,
            positive_value=BANK_ACCOUNT_HAS_VALUE,
            negative_value=BANK_ACCOUNT_NO_VALUE,
            evidence_quotes=_nested_evidence_quotes(structured_output, "participant_profile", "has_bank_account"),
        )

    if field_name == "borrowing_history":
        if get_persona_signal_value(structured_output, "cyclical_borrowing") is True:
            return _legacy_boolean_field(
                value=True,
                positive_value="has_borrowed",
                negative_value="has_not_borrowed_recently",
                evidence_quotes=get_persona_signal_evidence_quotes(structured_output, "cyclical_borrowing"),
            )
        if get_persona_signal_value(structured_output, "digital_borrowing") is True:
            return _legacy_boolean_field(
                value=True,
                positive_value="has_borrowed",
                negative_value="has_not_borrowed_recently",
                evidence_quotes=get_persona_signal_evidence_quotes(structured_output, "digital_borrowing"),
            )
        if get_persona_signal_value(structured_output, "self_reliance_non_borrowing") is True:
            return _legacy_boolean_field(
                value=False,
                positive_value="has_borrowed",
                negative_value="has_not_borrowed_recently",
                evidence_quotes=get_persona_signal_evidence_quotes(structured_output, "self_reliance_non_borrowing"),
            )
        return None

    if field_name == "loan_interest":
        if get_persona_signal_value(structured_output, "trust_fear_barrier") is True:
            return _legacy_boolean_field(
                value=True,
                positive_value="fearful_or_uncertain",
                negative_value=DEFAULT_UNKNOWN_VALUE,
                evidence_quotes=get_persona_signal_evidence_quotes(structured_output, "trust_fear_barrier"),
            )
        if get_persona_signal_value(structured_output, "repayment_stress") is True:
            return _legacy_boolean_field(
                value=True,
                positive_value="fearful_or_uncertain",
                negative_value=DEFAULT_UNKNOWN_VALUE,
                evidence_quotes=get_persona_signal_evidence_quotes(structured_output, "repayment_stress"),
            )
        return None

    return None


def _legacy_boolean_field(
    *,
    value: bool,
    positive_value: str,
    negative_value: str,
    evidence_quotes: list[str],
) -> dict[str, Any]:
    return AnalysisField(
        value=positive_value if value else negative_value,
        status="observed",
        evidence_quotes=tuple(evidence_quotes),
        notes="Derived from legacy nested structured output.",
    ).as_dict()


def _nested_object(structured_output: dict[str, Any], section: str, field_name: str) -> dict[str, Any]:
    parent = structured_output.get(section)
    if not isinstance(parent, dict):
        return {}
    nested = parent.get(field_name)
    if not isinstance(nested, dict):
        return {}
    return nested


def _nested_boolean(structured_output: dict[str, Any], section: str, field_name: str) -> bool | None:
    value = _nested_object(structured_output, section, field_name).get("value")
    if isinstance(value, bool):
        return value
    return None


def _nested_evidence_quotes(structured_output: dict[str, Any], section: str, field_name: str) -> list[str]:
    nested = _nested_object(structured_output, section, field_name)
    evidence = nested.get("evidence_quotes")
    if not isinstance(evidence, list):
        evidence = nested.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [quote for quote in evidence if isinstance(quote, str) and quote.strip()]


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
    "get_income_display_value",
    "get_analysis_value",
    "smartphone_user_from_analysis",
]
