from __future__ import annotations

from streamlit.testing.v1 import AppTest

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
        latest_stage="storage",
        last_error=None,
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
    assert interviews[0].current_stage == "storage"
    assert interviews[0].last_error is None
    assert interviews[0].summary == "Repository summary"
    assert interviews[0].transcript == "Repository transcript"
    assert interviews[0].evidence_quotes == ("Repository quote",)
    assert interviews[0].audio_bytes == b"1234"


def test_dashboard_renderer_returns_empty_interviews_without_explicit_sample_mode() -> None:
    renderer = DashboardRenderer()

    interviews = renderer._build_dashboard_interviews([], [])

    assert interviews == []


def test_dashboard_renderer_can_load_sample_interviews_only_when_explicitly_enabled() -> None:
    renderer = DashboardRenderer()

    interviews = renderer._build_dashboard_interviews([], [], sample_mode=True)

    assert interviews
    assert all(interview.id.startswith("sample-") for interview in interviews)


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


def test_dashboard_empty_repository_shows_empty_states_without_fabricated_personas_or_quotes() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.repository import DashboardStatusOverview

renderer = DashboardRenderer()
renderer.render(
    cached_results=[],
    recent_interviews=[],
    status_overview=DashboardStatusOverview(total_interviews=0, status_counts={}),
    interview_detail_loader=lambda interview_id: (_ for _ in ()).throw(KeyError(interview_id)),
    sample_mode=False,
)
'''

    app = AppTest.from_string(script)
    app.run()

    info_values = [element.value for element in app.info]
    markdown_values = [element.value for element in app.markdown]

    assert any("Upload interview audio from the sidebar to populate the dashboard overview." in value for value in info_values)
    assert any("Upload audio files or intentionally load sample fixtures to create interview cards." in value for value in info_values)
    assert any(
        "Upload and process at least one interview, or intentionally load sample fixtures, to inspect transcript details here." in value
        for value in info_values
    )
    assert not any("Employer-Dependent Digital Borrower" in value for value in markdown_values)
    assert not any("Digitally Ready but Fearful" in value for value in markdown_values)
    assert not any("Offline / Excluded" in value for value in markdown_values)
    assert not any("Self-Reliant Non-Borrower" in value for value in markdown_values)
    assert not any("High-Stress Cyclical Borrower" in value for value in markdown_values)
    assert not any("WhatsApp every day" in value for value in markdown_values)
    assert not any("Open" == element.label for element in app.button)


def test_dashboard_interview_selection_shows_inline_detail_and_syncs_detail_tab() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

records = [
    DashboardInterviewRecord(
        id="interview-1",
        audio_url="audio/interview-1/first.wav",
        filename="first.wav",
        transcript="First transcript with evidence.",
        status=ProcessingStatus.COMPLETED.value,
        latest_stage="analysis",
        last_error=None,
        created_at="2026-03-18T10:00:00+00:00",
        smartphone_user=True,
        has_bank_account=True,
        income_range="₹10k–15k",
        borrowing_history="has_borrowed",
        repayment_preference="monthly",
        loan_interest="interested",
        summary="First summary",
        key_quotes=["First quote"],
        persona="Digitally Ready but Fearful",
        confidence_score=0.9,
    ),
    DashboardInterviewRecord(
        id="interview-2",
        audio_url="audio/interview-2/second.wav",
        filename="second.wav",
        transcript="Second transcript with better evidence for the inline preview.",
        status=ProcessingStatus.COMPLETED.value,
        latest_stage="analysis",
        last_error=None,
        created_at="2026-03-18T11:00:00+00:00",
        smartphone_user=True,
        has_bank_account=True,
        income_range="₹15k–20k",
        borrowing_history="has_borrowed",
        repayment_preference="weekly",
        loan_interest="interested",
        summary="Second summary",
        key_quotes=["Second quote"],
        persona="High-Stress Cyclical Borrower",
        confidence_score=0.95,
    ),
]

renderer = DashboardRenderer()
renderer.render(
    cached_results=[],
    recent_interviews=records,
    status_overview=DashboardStatusOverview(
        total_interviews=2,
        status_counts={ProcessingStatus.COMPLETED.value: 2},
    ),
    interview_detail_loader=lambda interview_id: next(record for record in records if record.id == interview_id),
    sample_mode=False,
)
'''

    app = AppTest.from_string(script)
    app.run(timeout=10)

    select_button = next(button for button in app.button if button.key == "select_interview_interview-2")
    select_button.click()
    app.run(timeout=10)

    success_values = [element.value for element in app.success]
    info_values = [element.value for element in app.info]
    markdown_values = [element.value for element in app.markdown]

    assert any("Showing inline detail for second.wav." in value for value in success_values)
    assert any(
        "Interview selection synced from the Interviews tab: second.wav is now active in Interview Detail." in value
        for value in info_values
    )
    assert any("#### second.wav" in value for value in markdown_values)
