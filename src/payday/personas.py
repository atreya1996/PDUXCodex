from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from payday.models import AnalysisResult, PersonaClassification, Transcript


@dataclass(frozen=True, slots=True)
class Persona:
    id: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class PersonaDecision:
    persona_key: str
    rationale: str
    triggered_fields: tuple[str, ...]
    evidence_quotes: tuple[str, ...]
    is_non_target: bool = False


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
        del transcript
        structured_output = analysis.structured_output or {}

        decisions = (
            self._persona_three_no_smartphone(structured_output),
            self._persona_three_no_bank_account(structured_output),
            self._persona_five_high_stress_cyclical_borrower(structured_output),
            self._persona_one_employer_dependent_digital_borrower(structured_output),
            self._persona_two_digitally_ready_but_fearful(structured_output),
            self._persona_four_self_reliant_non_borrower(structured_output),
        )

        for decision in decisions:
            if decision is not None:
                return self._build_classification(decision)

        return self._build_classification(
            PersonaDecision(
                persona_key="persona_4",
                rationale=(
                    "No higher-priority persona rule matched, so the deterministic classifier "
                    "falls back to Persona 4."
                ),
                triggered_fields=(),
                evidence_quotes=(),
            )
        )

    def _persona_three_no_smartphone(self, structured_output: dict[str, Any]) -> PersonaDecision | None:
        smartphone = self._read_field(structured_output, "participant_profile.smartphone_user")
        if smartphone.get("value") is False:
            return PersonaDecision(
                persona_key="persona_3",
                rationale="Persona 3 override applied because smartphone_user is false.",
                triggered_fields=("participant_profile.smartphone_user",),
                evidence_quotes=self._evidence_tuple(smartphone),
                is_non_target=True,
            )
        return None

    def _persona_three_no_bank_account(self, structured_output: dict[str, Any]) -> PersonaDecision | None:
        bank_account = self._read_field(structured_output, "participant_profile.has_bank_account")
        if bank_account.get("value") is False:
            return PersonaDecision(
                persona_key="persona_3",
                rationale="Persona 3 override applied because has_bank_account is false.",
                triggered_fields=("participant_profile.has_bank_account",),
                evidence_quotes=self._evidence_tuple(bank_account),
                is_non_target=True,
            )
        return None

    def _persona_one_employer_dependent_digital_borrower(
        self,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        employer_dependency = self._read_field(structured_output, "persona_signals.employer_dependency")
        digital_borrowing = self._read_field(structured_output, "persona_signals.digital_borrowing")
        if employer_dependency.get("value") is True and digital_borrowing.get("value") is True:
            return PersonaDecision(
                persona_key="persona_1",
                rationale=(
                    "Persona 1 matched because the structured analysis shows both employer dependency "
                    "and digital borrowing evidence."
                ),
                triggered_fields=(
                    "persona_signals.employer_dependency",
                    "persona_signals.digital_borrowing",
                ),
                evidence_quotes=self._combine_evidence(employer_dependency, digital_borrowing),
            )
        return None

    def _persona_two_digitally_ready_but_fearful(
        self,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        digital_readiness = self._read_field(structured_output, "persona_signals.digital_readiness")
        trust_fear_barrier = self._read_field(structured_output, "persona_signals.trust_fear_barrier")
        if digital_readiness.get("value") is True and trust_fear_barrier.get("value") is True:
            return PersonaDecision(
                persona_key="persona_2",
                rationale=(
                    "Persona 2 matched because the structured analysis shows digital readiness alongside "
                    "trust or fear barriers."
                ),
                triggered_fields=(
                    "persona_signals.digital_readiness",
                    "persona_signals.trust_fear_barrier",
                ),
                evidence_quotes=self._combine_evidence(digital_readiness, trust_fear_barrier),
            )
        return None

    def _persona_four_self_reliant_non_borrower(
        self,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        self_reliance = self._read_field(structured_output, "persona_signals.self_reliance_non_borrowing")
        if self_reliance.get("value") is True:
            return PersonaDecision(
                persona_key="persona_4",
                rationale=(
                    "Persona 4 matched because the structured analysis shows self-reliance and explicit "
                    "non-borrowing behavior."
                ),
                triggered_fields=("persona_signals.self_reliance_non_borrowing",),
                evidence_quotes=self._evidence_tuple(self_reliance),
            )
        return None

    def _persona_five_high_stress_cyclical_borrower(
        self,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        cyclical_borrowing = self._read_field(structured_output, "persona_signals.cyclical_borrowing")
        repayment_stress = self._read_field(structured_output, "persona_signals.repayment_stress")
        if cyclical_borrowing.get("value") is True and repayment_stress.get("value") is True:
            return PersonaDecision(
                persona_key="persona_5",
                rationale=(
                    "Persona 5 matched because the structured analysis shows repeated borrowing together "
                    "with repayment stress."
                ),
                triggered_fields=(
                    "persona_signals.cyclical_borrowing",
                    "persona_signals.repayment_stress",
                ),
                evidence_quotes=self._combine_evidence(cyclical_borrowing, repayment_stress),
            )
        return None

    def _build_classification(self, decision: PersonaDecision) -> PersonaClassification:
        persona = PERSONAS[decision.persona_key]
        explanation_payload = {
            "persona_id": persona.id,
            "persona_name": persona.name,
            "decision_type": "override" if decision.is_non_target else "rule_match",
            "triggered_fields": list(decision.triggered_fields),
            "evidence": [
                {
                    "field": field_name,
                    "evidence": quote,
                }
                for field_name, quote in zip(decision.triggered_fields, decision.evidence_quotes, strict=False)
            ],
            "rationale": decision.rationale,
        }
        return PersonaClassification(
            persona_id=persona.id,
            persona_name=persona.name,
            rationale=decision.rationale,
            evidence_quotes=decision.evidence_quotes,
            explanation_payload=explanation_payload,
            is_non_target=decision.is_non_target,
        )

    def _read_field(self, data: dict[str, Any], dotted_path: str) -> dict[str, Any]:
        current: Any = data
        for key in dotted_path.split("."):
            if not isinstance(current, dict):
                return {"field": dotted_path, "value": None, "evidence": []}
            current = current.get(key)
        if isinstance(current, dict):
            return {
                "field": current.get("field", dotted_path),
                "value": current.get("value"),
                "evidence": current.get("evidence", []),
            }
        return {"field": dotted_path, "value": current, "evidence": []}

    def _evidence_tuple(self, field_payload: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(item) for item in field_payload.get("evidence", []) if item)[:2]

    def _combine_evidence(self, *field_payloads: dict[str, Any]) -> tuple[str, ...]:
        evidence: list[str] = []
        for payload in field_payloads:
            for item in payload.get("evidence", []):
                text = str(item)
                if text and text not in evidence:
                    evidence.append(text)
        return tuple(evidence[:2])
