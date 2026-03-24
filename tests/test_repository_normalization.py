from __future__ import annotations

from payday.models import ProcessingStatus
from payday.repository import PaydayRepository


def test_repository_normalizes_legacy_html_fragments_and_backfills_structured_fields() -> None:
    repository = PaydayRepository(database_path=":memory:")
    interview = repository.create_interview(
        audio_url="audio/interview-1.wav",
        transcript=(
            "<div class='interview-summary'><strong>Transcript preview:</strong> "
            "I borrow from my employer when needed.</div>"
        ),
        status=ProcessingStatus.COMPLETED.value,
        latest_stage="storage",
    )
    repository.upsert_structured_response(
        interview.id,
        smartphone_user=True,
        has_bank_account=True,
        per_household_earnings=None,
        participant_personal_monthly_income=None,
        total_household_monthly_income=None,
        income_range=None,
        borrowing_history=None,
        repayment_preference=None,
        loan_interest=None,
    )
    repository.upsert_insight(
        interview.id,
        summary=(
            "<div class='interview-summary'>Emergency borrowing from employer.</div>"
            "<div class='interview-meta'>"
            "<span class='meta-label'>Borrowing</span><span class='meta-value'>Borrower</span>"
            "<span class='meta-label'>Income</span><span class='meta-value'>₹10k–15k</span>"
            "</div>"
        ),
        key_quotes=["quoted"],
        persona="Digitally Ready but Fearful",
        confidence_score=0.9,
    )

    records = repository.list_recent_interviews(limit=10)

    assert len(records) == 1
    assert records[0].summary == "Emergency borrowing from employer."
    assert records[0].transcript == "I borrow from my employer when needed."
    assert records[0].data_malformed is False
    assert records[0].data_malformed_details == []

    detail = repository.get_interview_detail(interview.id)
    assert detail.structured_response is not None
    assert detail.structured_response.borrowing_history == "has_borrowed"
    assert detail.structured_response.income_range == "₹10k–15k"


def test_repository_uses_safe_fallback_for_unparseable_html_blob() -> None:
    repository = PaydayRepository(database_path=":memory:")
    interview = repository.create_interview(
        audio_url="audio/interview-2.wav",
        transcript="<div><b>broken html without known fragment</b></div>",
        status=ProcessingStatus.COMPLETED.value,
        latest_stage="storage",
    )
    repository.upsert_insight(
        interview.id,
        summary="<section><p>not a supported legacy format</p></section>",
        key_quotes=[],
        persona="Unknown",
        confidence_score=0.1,
    )

    record = repository.get_dashboard_interview_detail(interview.id)

    assert record.summary == "Analysis pending."
    assert record.transcript == "Transcript unavailable due to malformed stored data."
    assert record.data_malformed is True
    assert any("could not be normalized" in detail for detail in record.data_malformed_details)
