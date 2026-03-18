from __future__ import annotations

from payday.models import AnalysisResult, Transcript
from payday.personas import PersonaService


class AnalysisService:
    """Transforms transcripts into MVP-friendly summaries and metrics."""

    def __init__(self, persona_service: PersonaService) -> None:
        self.persona_service = persona_service

    def analyze(self, transcript: Transcript) -> AnalysisResult:
        words = transcript.text.split()
        summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
        persona_matches = self.persona_service.match_personas(transcript.text)
        metrics = {
            "word_count": len(words),
            "character_count": len(transcript.text),
            "provider": transcript.provider,
        }
        return AnalysisResult(
            summary=summary or "No transcript content available.",
            metrics=metrics,
            persona_matches=persona_matches,
        )
