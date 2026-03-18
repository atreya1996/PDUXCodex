from __future__ import annotations

from dataclasses import dataclass

from payday.models import AnalysisResult, PersonaClassification, Transcript


@dataclass(frozen=True, slots=True)
class Persona:
    id: str
    name: str
    description: str


PERSONAS = {
    "persona_1": Persona(
        id="persona_1",
        name="Employer-Dependent Digital Borrower",
        description="Borrows digitally with employer influence or support.",
    ),
    "persona_2": Persona(
        id="persona_2",
        name="Digitally Ready but Fearful",
        description="Has digital access but hesitates because of trust or risk concerns.",
    ),
    "persona_3": Persona(
        id="persona_3",
        name="Offline / Excluded",
        description="Excluded because they lack a smartphone or a bank account.",
    ),
    "persona_4": Persona(
        id="persona_4",
        name="Self-Reliant Non-Borrower",
        description="Avoids borrowing and relies on self-management or savings.",
    ),
    "persona_5": Persona(
        id="persona_5",
        name="High-Stress Cyclical Borrower",
        description="Faces repeated borrowing and repayment pressure.",
    ),
}


class PersonaService:
    def list_personas(self) -> list[Persona]:
        return list(PERSONAS.values())

    def classify(self, transcript: Transcript, analysis: AnalysisResult) -> PersonaClassification:
        lowered = transcript.text.lower()
        evidence = tuple(analysis.evidence_quotes[:2])

        if self._mentions_no_smartphone(lowered):
            return self._build_classification(
                "persona_3",
                "Transcript explicitly says the participant does not have a smartphone.",
                evidence,
                is_non_target=True,
            )
        if self._mentions_no_bank_account(lowered):
            return self._build_classification(
                "persona_3",
                "Transcript explicitly says the participant does not have a bank account.",
                evidence,
                is_non_target=True,
            )
        if any(term in lowered for term in ("every month", "monthly", "again and again", "cycle", "debt", "repay")):
            return self._build_classification(
                "persona_5",
                "Transcript mentions repeated borrowing or repayment stress.",
                evidence,
            )
        if ("employer" in lowered or "madam" in lowered or "sir" in lowered) and any(
            term in lowered for term in ("loan app", "borrow online", "digital loan", "upi", "whatsapp")
        ):
            return self._build_classification(
                "persona_1",
                "Transcript links borrowing decisions with employer support and digital channels.",
                evidence,
            )
        if any(term in lowered for term in ("afraid", "fear", "scam", "trust", "worried")) and any(
            term in lowered for term in ("smartphone", "bank account", "whatsapp", "upi", "online")
        ):
            return self._build_classification(
                "persona_2",
                "Transcript shows digital readiness alongside trust or risk concerns.",
                evidence,
            )
        return self._build_classification(
            "persona_4",
            "No stronger borrowing-risk signal was found, so this interview is treated as self-reliant by default.",
            evidence,
        )

    def _build_classification(
        self,
        persona_key: str,
        rationale: str,
        evidence_quotes: tuple[str, ...],
        *,
        is_non_target: bool = False,
    ) -> PersonaClassification:
        persona = PERSONAS[persona_key]
        return PersonaClassification(
            persona_id=persona.id,
            persona_name=persona.name,
            rationale=rationale,
            evidence_quotes=evidence_quotes,
            is_non_target=is_non_target,
        )

    def _mentions_no_smartphone(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "no smartphone",
                "do not have a smartphone",
                "don't have a smartphone",
                "without smartphone",
                "i use a basic phone",
            )
        )

    def _mentions_no_bank_account(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "no bank account",
                "do not have a bank account",
                "don't have a bank account",
                "without bank account",
                "i am unbanked",
            )
        )
