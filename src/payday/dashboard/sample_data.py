from __future__ import annotations

import math
import wave
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Callable

from payday.dashboard.views import DashboardInterview
from payday.models import ProcessingStatus


def build_sample_interviews(
    *,
    persona_lookup: dict[str, str],
    digital_access_label: Callable[[bool | None, bool | None], str],
) -> list[DashboardInterview]:
    now = datetime.now(timezone.utc)
    samples = [
        {
            "id": "sample-001",
            "filename": "meena_whatsapp_borrower.wav",
            "transcript": (
                "I use WhatsApp every day and my bank account is active. I earn ₹12,000 per month. "
                "When money is short I borrow from my employer first, but I am worried about scams from apps."
            ),
            "persona_id": "persona_2",
            "income_band": "₹10k–15k",
            "borrowing_source": "Employer",
            "borrowing_label": "Borrower",
            "loan_interest_label": "Fearful / uncertain",
            "interested_in_loan": False,
            "smartphone_user": True,
            "has_bank_account": True,
            "status": ProcessingStatus.COMPLETED.value,
        },
        {
            "id": "sample-002",
            "filename": "rekha_offline_non_target.wav",
            "transcript": (
                "I use a basic phone and I do not have a bank account. When there is an emergency, I ask my sister for help."
            ),
            "persona_id": "persona_3",
            "income_band": "Below ₹10k",
            "borrowing_source": "Family / friends",
            "borrowing_label": "Borrower",
            "loan_interest_label": "Unknown",
            "interested_in_loan": False,
            "smartphone_user": False,
            "has_bank_account": False,
            "status": ProcessingStatus.COMPLETED.value,
        },
        {
            "id": "sample-003",
            "filename": "saira_self_reliant.wav",
            "transcript": (
                "My salary goes to my bank account and I use WhatsApp. I save first and I do not borrow because I manage on my own."
            ),
            "persona_id": "persona_4",
            "income_band": "₹15k–20k",
            "borrowing_source": "No borrowing disclosed",
            "borrowing_label": "Non-borrower",
            "loan_interest_label": "Not interested",
            "interested_in_loan": False,
            "smartphone_user": True,
            "has_bank_account": True,
            "status": ProcessingStatus.COMPLETED.value,
        },
        {
            "id": "sample-004",
            "filename": "anita_cyclical_borrower.wav",
            "transcript": (
                "Every month I take another loan from a local moneylender and the repayment pressure gives me tension. "
                "I have a smartphone and a bank account, but debt follows me again and again."
            ),
            "persona_id": "persona_5",
            "income_band": "₹10k–15k",
            "borrowing_source": "Informal lender",
            "borrowing_label": "Borrower",
            "loan_interest_label": "Interested",
            "interested_in_loan": True,
            "smartphone_user": True,
            "has_bank_account": True,
            "status": ProcessingStatus.COMPLETED.value,
        },
        {
            "id": "sample-005",
            "filename": "lata_employer_digital.wav",
            "transcript": (
                "My employer helped me use a loan app on my smartphone. I have a bank account and I repay monthly after salary."
            ),
            "persona_id": "persona_1",
            "income_band": "₹15k–20k",
            "borrowing_source": "Digital / app",
            "borrowing_label": "Borrower",
            "loan_interest_label": "Interested",
            "interested_in_loan": True,
            "smartphone_user": True,
            "has_bank_account": True,
            "status": ProcessingStatus.COMPLETED.value,
        },
        {
            "id": "sample-006",
            "filename": "pooja_partial_access.wav",
            "transcript": (
                "I have a smartphone and WhatsApp, but I share my sister's account because I still do not have my own bank account."
            ),
            "persona_id": "persona_3",
            "income_band": "Below ₹10k",
            "borrowing_source": "Family / friends",
            "borrowing_label": "Borrower",
            "loan_interest_label": "Fearful / uncertain",
            "interested_in_loan": False,
            "smartphone_user": True,
            "has_bank_account": False,
            "status": ProcessingStatus.PROCESSING.value,
        },
    ]
    interviews: list[DashboardInterview] = []
    for index, sample in enumerate(samples):
        transcript = sample["transcript"]
        persona_id = sample["persona_id"]
        persona_name = persona_lookup[persona_id]
        digital_access = digital_access_label(sample["smartphone_user"], sample["has_bank_account"])
        extracted_json = {
            "smartphone_usage": {
                "value": "has_smartphone" if sample["smartphone_user"] else "no_smartphone",
                "status": "observed",
                "evidence_quotes": [transcript.split(". ")[0]],
                "notes": "Sample-mode structured output.",
            },
            "bank_account_status": {
                "value": "has_bank_account" if sample["has_bank_account"] else "no_bank_account",
                "status": "observed",
                "evidence_quotes": [transcript.split(". ")[0]],
                "notes": "Sample-mode structured output.",
            },
            "per_household_earnings": {
                "value": "unknown",
                "status": "unknown",
                "evidence_quotes": [],
                "notes": "Sample-mode structured output.",
                "evidence_type": "unknown",
            },
            "participant_personal_monthly_income": {
                "value": sample["income_band"],
                "status": "observed",
                "evidence_quotes": [transcript.split(". ")[0]],
                "notes": "Sample-mode structured output.",
                "evidence_type": "direct",
            },
            "total_household_monthly_income": {
                "value": "unknown",
                "status": "unknown",
                "evidence_quotes": [],
                "notes": "Sample-mode structured output.",
                "evidence_type": "unknown",
            },
            "borrowing_history": {
                "value": sample["borrowing_label"].lower().replace("-", "_"),
                "status": "observed",
                "evidence_quotes": [transcript],
                "notes": "Sample-mode structured output.",
            },
            "repayment_preference": {
                "value": "monthly" if "monthly" in transcript.lower() or "every month" in transcript.lower() else "unknown",
                "status": "observed" if "monthly" in transcript.lower() or "every month" in transcript.lower() else "unknown",
                "evidence_quotes": [transcript] if "month" in transcript.lower() else [],
                "notes": "Sample-mode structured output.",
            },
            "loan_interest": {
                "value": sample["loan_interest_label"].lower().replace(" / ", "_").replace(" ", "_"),
                "status": "observed" if sample["loan_interest_label"] != "Unknown" else "unknown",
                "evidence_quotes": [transcript],
                "notes": "Sample-mode structured output.",
            },
            "summary": {
                "value": transcript,
                "status": "observed",
                "evidence_quotes": [transcript],
                "notes": "Sample-mode structured output.",
            },
            "key_quotes": [transcript.split(". ")[0]],
            "confidence_signals": {
                "observed_evidence": ["sample interview"],
                "missing_or_unknown": [],
            },
        }
        interviews.append(
            DashboardInterview(
                id=sample["id"],
                filename=sample["filename"],
                created_at=(now - timedelta(days=index)).date().isoformat(),
                status=sample["status"],
                current_stage="storage" if sample["status"] == ProcessingStatus.COMPLETED.value else "analysis",
                last_error=None,
                summary=transcript,
                transcript=transcript,
                persona_id=persona_id,
                persona_name=persona_name,
                is_non_target=persona_id == "persona_3",
                income_band=sample["income_band"],
                borrowing_source=sample["borrowing_source"],
                borrowing_label=sample["borrowing_label"],
                is_borrower=sample["borrowing_label"] == "Borrower",
                loan_interest_label=sample["loan_interest_label"],
                interested_in_loan=sample["interested_in_loan"],
                smartphone_user=sample["smartphone_user"],
                has_bank_account=sample["has_bank_account"],
                digital_access=digital_access,
                extracted_json=extracted_json,
                evidence_quotes=(transcript.split(". ")[0],),
                audio_bytes=_sample_audio_bytes(frequency=220 + index * 40),
            )
        )
    return interviews


def _sample_audio_bytes(frequency: int) -> bytes:
    sample_rate = 22050
    duration_seconds = 1
    amplitude = 12000
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for step in range(sample_rate * duration_seconds):
                value = int(amplitude * math.sin(2 * math.pi * frequency * step / sample_rate))
                frames.extend(value.to_bytes(2, byteorder="little", signed=True))
            wav_file.writeframes(bytes(frames))
        return buffer.getvalue()
