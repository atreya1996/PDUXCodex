from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    description: str


DEFAULT_PERSONAS = [
    Persona(name="Saver", description="Optimizes for lower spend and stable budgeting."),
    Persona(name="Planner", description="Prefers predictable schedules and operational clarity."),
    Persona(name="Closer", description="Focused on urgency, conversions, and commitment signals."),
]


class PersonaService:
    def list_personas(self) -> list[Persona]:
        return DEFAULT_PERSONAS

    def match_personas(self, transcript_text: str) -> list[str]:
        lowered = transcript_text.lower()
        matches: list[str] = []
        if "budget" in lowered or "save" in lowered:
            matches.append("Saver")
        if "plan" in lowered or "schedule" in lowered:
            matches.append("Planner")
        if "close" in lowered or "buy" in lowered:
            matches.append("Closer")
        return matches or [persona.name for persona in DEFAULT_PERSONAS[:1]]
