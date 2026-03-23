from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from payday.analysis import (
    DEFAULT_UNKNOWN_VALUE,
    bank_account_user_from_analysis,
    get_analysis_evidence_quotes,
    get_persona_signal_evidence_quotes,
    get_persona_signal_value,
    get_analysis_value,
    smartphone_user_from_analysis,
)
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
        structured_output = analysis.structured_output or {}

        decisions = (
            self._persona_three_no_smartphone(structured_output),
            self._persona_three_no_bank_account(structured_output),
            self._persona_five_high_stress_cyclical_borrower(transcript, structured_output),
            self._persona_one_employer_dependent_digital_borrower(transcript, structured_output),
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
        smartphone_user = smartphone_user_from_analysis(structured_output)
        if smartphone_user is False:
            return PersonaDecision(
                persona_key="persona_3",
                rationale="Persona 3 override applied because smartphone_usage shows no smartphone.",
                triggered_fields=("smartphone_usage",),
                evidence_quotes=get_analysis_evidence_quotes(structured_output, "smartphone_usage")[:2],
                is_non_target=True,
            )
        return None

    def _persona_three_no_bank_account(self, structured_output: dict[str, Any]) -> PersonaDecision | None:
        has_bank_account = bank_account_user_from_analysis(structured_output)
        if has_bank_account is False:
            return PersonaDecision(
                persona_key="persona_3",
                rationale="Persona 3 override applied because bank_account_status shows no bank account.",
                triggered_fields=("bank_account_status",),
                evidence_quotes=get_analysis_evidence_quotes(structured_output, "bank_account_status")[:2],
                is_non_target=True,
            )
        return None

    def _persona_one_employer_dependent_digital_borrower(
        self,
        transcript: Transcript,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        if smartphone_user_from_analysis(structured_output) is not True:
            return None
        if bank_account_user_from_analysis(structured_output) is not True:
            return None

        borrowing_history = get_analysis_value(structured_output, "borrowing_history")
        employer_dependency = get_persona_signal_value(structured_output, "employer_dependency")
        digital_borrowing = get_persona_signal_value(structured_output, "digital_borrowing")
        lowered_transcript = transcript.text.lower()
        mentions_employer = any(token in lowered_transcript for token in ("employer", "madam", "sir", "boss"))
        if borrowing_history == "has_borrowed" and (mentions_employer or employer_dependency is True):
            borrowing_quotes = list(get_analysis_evidence_quotes(structured_output, "borrowing_history"))
            borrowing_quotes.extend(get_persona_signal_evidence_quotes(structured_output, "digital_borrowing"))
            borrowing_quotes.extend(get_persona_signal_evidence_quotes(structured_output, "employer_dependency"))
            employer_quotes = [
                sentence
                for sentence in transcript.text.split(".")
                if any(token in sentence.lower() for token in ("employer", "madam", "sir", "boss"))
            ]
            return PersonaDecision(
                persona_key="persona_1",
                rationale=(
                    "Persona 1 matched because the participant has digital access, has borrowed, "
                    "and the transcript ties borrowing context to the employer."
                ),
                triggered_fields=(
                    "borrowing_history",
                    "smartphone_usage",
                ),
                evidence_quotes=tuple((borrowing_quotes + [quote.strip() for quote in employer_quotes if quote.strip()])[:2]),
            )
        return None

    def _persona_two_digitally_ready_but_fearful(
        self,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        if smartphone_user_from_analysis(structured_output) is not True:
            return None
        if bank_account_user_from_analysis(structured_output) is not True:
            return None

        loan_interest = get_analysis_value(structured_output, "loan_interest")
        trust_fear_barrier = get_persona_signal_value(structured_output, "trust_fear_barrier")
        if loan_interest == "fearful_or_uncertain" or trust_fear_barrier is True:
            return PersonaDecision(
                persona_key="persona_2",
                rationale=(
                    "Persona 2 matched because the participant is digitally in-scope and explicitly "
                    "expresses trust or fear barriers about borrowing."
                ),
                triggered_fields=(
                    "smartphone_usage",
                    "loan_interest",
                ),
                evidence_quotes=tuple(
                    (
                        list(get_analysis_evidence_quotes(structured_output, "smartphone_usage"))
                        + list(get_analysis_evidence_quotes(structured_output, "loan_interest"))
                        + list(get_persona_signal_evidence_quotes(structured_output, "trust_fear_barrier"))
                    )[:2]
                ),
            )
        return None

    def _persona_four_self_reliant_non_borrower(
        self,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        borrowing_history = get_analysis_value(structured_output, "borrowing_history")
        self_reliance_non_borrowing = get_persona_signal_value(structured_output, "self_reliance_non_borrowing")
        if borrowing_history == "has_not_borrowed_recently" or self_reliance_non_borrowing is True:
            return PersonaDecision(
                persona_key="persona_4",
                rationale=(
                    "Persona 4 matched because the participant explicitly reports not borrowing recently."
                ),
                triggered_fields=("borrowing_history",),
                evidence_quotes=tuple(
                    (
                        list(get_analysis_evidence_quotes(structured_output, "borrowing_history"))
                        + list(get_persona_signal_evidence_quotes(structured_output, "self_reliance_non_borrowing"))
                    )[:2]
                ),
            )
        return None

    def _persona_five_high_stress_cyclical_borrower(
        self,
        transcript: Transcript,
        structured_output: dict[str, Any],
    ) -> PersonaDecision | None:
        if smartphone_user_from_analysis(structured_output) is not True:
            return None
        if bank_account_user_from_analysis(structured_output) is not True:
            return None

        borrowing_history = get_analysis_value(structured_output, "borrowing_history")
        repayment_preference = get_analysis_value(structured_output, "repayment_preference")
        loan_interest = get_analysis_value(structured_output, "loan_interest")
        cyclical_borrowing = get_persona_signal_value(structured_output, "cyclical_borrowing")
        repayment_stress = get_persona_signal_value(structured_output, "repayment_stress")
        lowered_transcript = transcript.text.lower()
        stress_markers = ("pressure", "stress", "tension", "cycle", "again", "every month")
        has_stress_evidence = any(marker in lowered_transcript for marker in stress_markers)

        if (
            (borrowing_history == "has_borrowed" or cyclical_borrowing is True)
            and (
                repayment_preference != DEFAULT_UNKNOWN_VALUE
                or cyclical_borrowing is True
                or repayment_stress is True
                or has_stress_evidence
            )
            and (loan_interest == "fearful_or_uncertain" or has_stress_evidence or repayment_stress is True)
        ):
            return PersonaDecision(
                persona_key="persona_5",
                rationale=(
                    "Persona 5 matched because the participant reports borrowing plus stress around repeated "
                    "repayment, including legacy cyclical-borrowing signals when present."
                ),
                triggered_fields=(
                    "borrowing_history",
                    "repayment_preference",
                    "loan_interest",
                ),
                evidence_quotes=tuple(
                    (
                        list(get_analysis_evidence_quotes(structured_output, "borrowing_history"))
                        + list(get_analysis_evidence_quotes(structured_output, "repayment_preference"))
                        + list(get_analysis_evidence_quotes(structured_output, "loan_interest"))
                        + list(get_persona_signal_evidence_quotes(structured_output, "cyclical_borrowing"))
                        + list(get_persona_signal_evidence_quotes(structured_output, "repayment_stress"))
                    )[:2]
                ),
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
