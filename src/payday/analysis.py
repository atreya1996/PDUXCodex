from __future__ import annotations

from collections.abc import Iterable

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
        words = text.split()
        summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
        evidence_quotes = self._extract_quotes(text)
        structured_output = {
            "transcript_summary": summary or "No transcript content available.",
            "evidence_quotes": evidence_quotes,
            "participant_profile": {
                "smartphone_user": self._detect_smartphone_user(text),
                "has_bank_account": self._detect_bank_account(text),
            },
            "persona_signals": {
                "employer_dependency": self._detect_employer_dependency(text),
                "digital_readiness": self._detect_digital_readiness(text),
                "digital_borrowing": self._detect_digital_borrowing(text),
                "trust_fear_barrier": self._detect_trust_fear_barrier(text),
                "self_reliance_non_borrowing": self._detect_self_reliance_non_borrowing(text),
                "cyclical_borrowing": self._detect_cyclical_borrowing(text),
                "repayment_stress": self._detect_repayment_stress(text),
            },
            "signals": self._extract_signals(text),
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

    def _build_evidence_field(self, field_name: str, value: bool | None, evidence: Iterable[str]) -> dict[str, object]:
        return {
            "value": value,
            "field": field_name,
            "evidence": [item for item in evidence if item],
        }

    def _detect_smartphone_user(self, text: str) -> dict[str, object]:
        negative_matches = self._matching_phrases(
            text,
            (
                "no smartphone",
                "do not have a smartphone",
                "don't have a smartphone",
                "without smartphone",
                "basic phone",
                "feature phone",
            ),
        )
        if negative_matches:
            return self._build_evidence_field("participant_profile.smartphone_user", False, negative_matches)

        positive_matches = self._matching_phrases(
            text,
            (
                "smartphone",
                "whatsapp",
                "upi",
                "google pay",
                "phonepe",
                "paytm",
                "online app",
            ),
        )
        if positive_matches:
            return self._build_evidence_field("participant_profile.smartphone_user", True, positive_matches)

        return self._build_evidence_field("participant_profile.smartphone_user", None, ())

    def _detect_bank_account(self, text: str) -> dict[str, object]:
        negative_matches = self._matching_phrases(
            text,
            (
                "no bank account",
                "do not have a bank account",
                "don't have a bank account",
                "without bank account",
                "unbanked",
            ),
        )
        if negative_matches:
            return self._build_evidence_field("participant_profile.has_bank_account", False, negative_matches)

        positive_matches = self._matching_phrases(
            text,
            (
                "bank account",
                "my account",
                "salary in bank",
                "money goes to bank",
                "account is active",
            ),
        )
        if positive_matches:
            return self._build_evidence_field("participant_profile.has_bank_account", True, positive_matches)

        return self._build_evidence_field("participant_profile.has_bank_account", None, ())

    def _detect_employer_dependency(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "employer told me",
                "employer helped",
                "madam helped",
                "sir helped",
                "madam asked me",
                "through my employer",
                "boss helped",
            ),
        )
        return self._build_evidence_field("persona_signals.employer_dependency", bool(matches), matches)

    def _detect_digital_readiness(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "whatsapp",
                "upi",
                "google pay",
                "phonepe",
                "paytm",
                "online",
                "loan app",
                "digital",
            ),
        )
        return self._build_evidence_field("persona_signals.digital_readiness", bool(matches), matches)

    def _detect_digital_borrowing(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "loan app",
                "borrow online",
                "digital loan",
                "took loan on app",
                "upi loan",
                "whatsapp loan",
            ),
        )
        return self._build_evidence_field("persona_signals.digital_borrowing", bool(matches), matches)

    def _detect_trust_fear_barrier(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "afraid",
                "fear",
                "scam",
                "do not trust",
                "don't trust",
                "worried",
                "nervous",
                "risk",
            ),
        )
        return self._build_evidence_field("persona_signals.trust_fear_barrier", bool(matches), matches)

    def _detect_self_reliance_non_borrowing(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "i avoid loans",
                "i do not borrow",
                "i don't borrow",
                "use my savings",
                "manage on my own",
                "self manage",
                "save first",
                "borrow only from myself",
            ),
        )
        return self._build_evidence_field("persona_signals.self_reliance_non_borrowing", bool(matches), matches)

    def _detect_cyclical_borrowing(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "every month",
                "monthly loan",
                "again and again",
                "borrow repeatedly",
                "loan cycle",
                "take another loan",
            ),
        )
        return self._build_evidence_field("persona_signals.cyclical_borrowing", bool(matches), matches)

    def _detect_repayment_stress(self, text: str) -> dict[str, object]:
        matches = self._matching_phrases(
            text,
            (
                "stress",
                "pressure",
                "repay",
                "repayment",
                "debt",
                "collection calls",
                "tension",
            ),
        )
        return self._build_evidence_field("persona_signals.repayment_stress", bool(matches), matches)

    def _matching_phrases(self, text: str, phrases: tuple[str, ...]) -> list[str]:
        lowered = text.lower()
        return [phrase for phrase in phrases if phrase in lowered]
