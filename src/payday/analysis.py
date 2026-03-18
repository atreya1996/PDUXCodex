from __future__ import annotations

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
