from __future__ import annotations

from collections.abc import Iterable

from payday.models import AnalysisResult, Transcript


class AnalysisService:
    """Transforms transcripts into structured, evidence-aware summaries."""

    def analyze(self, transcript: Transcript) -> AnalysisResult:
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
        metrics = {
            "word_count": len(words),
            "character_count": len(text),
            "provider": transcript.provider,
        }
        return AnalysisResult(
            summary=summary or "No transcript content available.",
            metrics=metrics,
            structured_output=structured_output,
            evidence_quotes=evidence_quotes,
        )

    def _extract_quotes(self, text: str) -> list[str]:
        if not text:
            return []
        segments = [segment.strip() for segment in text.replace("\n", " ").split(".")]
        quotes = [segment for segment in segments if segment][:3]
        return quotes

    def _extract_signals(self, text: str) -> dict[str, bool]:
        lowered = text.lower()
        return {
            "mentions_whatsapp": "whatsapp" in lowered,
            "mentions_trust": "trust" in lowered,
            "mentions_informal_lending": any(term in lowered for term in ("moneylender", "informal", "borrow from friends")),
            "mentions_employer_support": "employer" in lowered,
            "mentions_repayment_stress": any(term in lowered for term in ("stress", "pressure", "repay", "debt")),
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
