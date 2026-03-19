from __future__ import annotations

from payday.analysis import AnalysisResult
from payday.dashboard.views import DashboardRenderer
from payday.models import PipelineResult, ProcessingStatus, Transcript, UploadedAsset
from payday.personas import PersonaClassification
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview


def test_dashboard_renderer_prefers_durable_repository_values_for_status_rows() -> None:
    renderer = DashboardRenderer()
    cached_result = PipelineResult(
        file_id="interview-1",
        filename="cached-name.wav",
        status=ProcessingStatus.PROCESSING,
        transcript=Transcript(text="Cached transcript", provider="test", model="test"),
        analysis=AnalysisResult(
            summary="Cached summary",
            structured_output={
                "participant_profile": {
                    "smartphone_user": {"value": True},
                    "has_bank_account": {"value": True},
                },
                "income_range": {"value": "Unknown"},
                "borrowing_history": {"value": "unknown"},
                "loan_interest": {"value": "unknown"},
            },
            evidence_quotes=["Cached quote"],
        ),
        persona=PersonaClassification(
            persona_id="persona_4",
            persona_name="Self-Reliant Non-Borrower",
            rationale="cache only",
        ),
        asset=UploadedAsset(
            filename="cached-name.wav",
            content_type="audio/wav",
            size_bytes=4,
            raw_bytes=b"1234",
            file_id="interview-1",
        ),
    )
    repository_record = DashboardInterviewRecord(
        id="interview-1",
        audio_url="audio/interview-1/repo-name.wav",
        filename="repo-name.wav",
        transcript="Repository transcript",
        status=ProcessingStatus.COMPLETED.value,
        created_at="2026-03-18T10:00:00+00:00",
        smartphone_user=True,
        has_bank_account=True,
        income_range="₹10k–15k",
        borrowing_history="has_borrowed",
        repayment_preference="monthly",
        loan_interest="interested",
        summary="Repository summary",
        key_quotes=["Repository quote"],
        persona="Digitally Ready but Fearful",
        confidence_score=0.91,
    )

    interviews = renderer._build_dashboard_interviews([cached_result], [repository_record])

    assert len(interviews) == 1
    assert interviews[0].filename == "repo-name.wav"
    assert interviews[0].status == ProcessingStatus.COMPLETED.value
    assert interviews[0].summary == "Repository summary"
    assert interviews[0].transcript == "Repository transcript"
    assert interviews[0].evidence_quotes == ("Repository quote",)
    assert interviews[0].audio_bytes == b"1234"


def test_dashboard_renderer_status_counts_use_durable_overview_when_available() -> None:
    renderer = DashboardRenderer()
    overview = DashboardStatusOverview(
        total_interviews=4,
        status_counts={
            ProcessingStatus.PENDING.value: 1,
            ProcessingStatus.PROCESSING.value: 1,
            ProcessingStatus.COMPLETED.value: 1,
            ProcessingStatus.FAILED.value: 1,
        },
    )

    counts = renderer._status_counts([], overview)

    assert counts == {
        ProcessingStatus.PENDING.value: 1,
        ProcessingStatus.PROCESSING.value: 1,
        ProcessingStatus.COMPLETED.value: 1,
        ProcessingStatus.FAILED.value: 1,
    }
