from __future__ import annotations

from streamlit.testing.v1 import AppTest

from payday.analysis import AnalysisResult
from payday.dashboard.views import DashboardRenderer
from payday.models import PipelineResult, ProcessingStatus, Transcript, UploadedAsset
from payday.personas import PersonaClassification
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview


OPEN_INTERVIEW_KEY_PREFIX = "open_interview_"
OPEN_INTERVIEW_LABEL_ALIASES = {"Open", "Opening interview…", "Opening…"}


def _find_open_interview_button(app: AppTest):
    for button in app.button:
        if getattr(button, "key", "").startswith(OPEN_INTERVIEW_KEY_PREFIX):
            return button
    for button in app.button:
        if button.label in OPEN_INTERVIEW_LABEL_ALIASES:
            return button
    raise AssertionError("No interview-open action was rendered.")


def _dashboard_interview(
    *,
    interview_id: str,
    participant_income_value: str = "Unknown",
    borrowing_source: str = "Unknown",
) -> object:
    return type(
        "DashboardInterviewStub",
        (),
        {
            "id": interview_id,
            "filename": f"{interview_id}.wav",
            "created_at": "2026-03-18",
            "status": ProcessingStatus.COMPLETED.value,
            "current_stage": "storage",
            "last_error": None,
            "summary": "Summary",
            "transcript": "Transcript",
            "persona_id": "persona_4",
            "persona_name": "Self-Reliant Non-Borrower",
            "is_non_target": False,
            "participant_income_value": participant_income_value,
            "income_band": participant_income_value,
            "borrowing_source": borrowing_source,
            "borrowing_label": "Borrower" if borrowing_source != "Unknown" else "Non-borrower",
            "is_borrower": borrowing_source != "Unknown",
            "loan_interest_label": "Unknown",
            "interested_in_loan": False,
            "smartphone_user": True,
            "has_bank_account": True,
            "digital_access": "Smartphone + bank account",
            "transcript_quality": "good",
            "extracted_json": {},
            "evidence_quotes": ("Quote",),
            "segmented_dialogue": (),
            "audio_bytes": None,
            "audio_format": "audio/wav",
        },
    )()


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
                "participant_personal_monthly_income": {"value": "Unknown"},
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
        per_household_earnings=None,
        participant_personal_monthly_income="₹10k–15k",
        total_household_monthly_income=None,
        income_range="₹10k–15k",
        borrowing_history="has_borrowed",
        repayment_preference="monthly",
        loan_interest="interested",
        summary="Repository summary",
        key_quotes=["Repository quote"],
        persona="Digitally Ready but Fearful",
        confidence_score=0.91,
        segmented_dialogue=[
            {
                "speaker_label": "participant",
                "utterance_text": "Repository transcript",
                "speaker_confidence": "medium",
                "speaker_uncertainty": "Transcript content suggests participant narration.",
            }
        ],
    )

    interviews = renderer._build_dashboard_interviews([cached_result], [repository_record])

    assert len(interviews) == 1
    assert interviews[0].filename == "repo-name.wav"
    assert interviews[0].status == ProcessingStatus.COMPLETED.value
    assert interviews[0].current_stage == "storage"
    assert interviews[0].last_error is None
    assert interviews[0].summary == "Repository summary"
    assert interviews[0].transcript == "Repository transcript"
    assert interviews[0].income_band == "Participant monthly income: ₹10k–15k"
    assert interviews[0].evidence_quotes == ("Repository quote",)
    assert interviews[0].segmented_dialogue[0]["speaker_label"] == "participant"
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


def test_dashboard_renderer_normalizes_income_buckets_from_comparable_participant_income_values() -> None:
    renderer = DashboardRenderer()
    interviews = [
        _dashboard_interview(interview_id="one", participant_income_value="₹12,000"),
        _dashboard_interview(interview_id="two", participant_income_value="₹10k–15k"),
        _dashboard_interview(interview_id="three", participant_income_value="Below ₹10k"),
        _dashboard_interview(interview_id="four", participant_income_value="Unknown"),
    ]

    income_values = renderer._normalized_participant_income_values(interviews)
    income_rows = renderer._income_bucket_rows(interviews)

    assert income_values == [12000, 12500, 9999]
    assert income_rows == [
        ("Below ₹10k", 1, "33%"),
        ("₹10k–15k", 2, "67%"),
    ]


def test_dashboard_renderer_uses_normalized_borrowing_sources_only_for_overview_table() -> None:
    renderer = DashboardRenderer()
    interviews = [
        _dashboard_interview(interview_id="one", borrowing_source="Employer"),
        _dashboard_interview(interview_id="two", borrowing_source="Employer"),
        _dashboard_interview(interview_id="three", borrowing_source="Family / friends"),
        _dashboard_interview(interview_id="four", borrowing_source="Unknown"),
    ]

    rows = renderer._normalized_borrowing_rows(interviews)

    assert rows == [
        ("Employer", 2, "67%"),
        ("Family / friends", 1, "33%"),
    ]


def test_dashboard_renderer_normalizes_chart_rows_to_flat_category_count_share_shape() -> None:
    renderer = DashboardRenderer()

    chart_rows = renderer._normalized_chart_rows(
        {
            "  Employer  ": 2,
            "unknown": 1,
            "": 3,
            None: 4,
            "Family / friends": "2",
            "Ignore me": 0,
        }
    )

    assert chart_rows == [
        {"category": "Employer", "count": 2, "share": 16.7},
        {"category": "Unknown", "count": 8, "share": 66.7},
        {"category": "Family / friends", "count": 2, "share": 16.7},
    ]
    assert renderer._chart_rows_are_flat(chart_rows) is True


def test_dashboard_renderer_normalizes_unknown_or_malformed_chart_categories() -> None:
    renderer = DashboardRenderer()

    assert renderer._normalize_chart_category(None) == "Unknown"
    assert renderer._normalize_chart_category("   ") == "Unknown"
    assert renderer._normalize_chart_category("NIL") == "Unknown"
    assert renderer._normalize_chart_category(" -- ") == "Unknown"
    assert renderer._normalize_chart_category("  Family/friends  ") == "Family / friends"


def test_dashboard_renderer_chart_rows_validation_rejects_nested_or_missing_shape() -> None:
    renderer = DashboardRenderer()

    assert renderer._chart_rows_are_flat([{"category": "Borrower", "count": {"nested": 1}}]) is False
    assert renderer._chart_rows_are_flat([{"category": "Borrower"}]) is False
    assert renderer._chart_rows_are_flat([{"count": 1, "share": 20.0}]) is False


def test_dashboard_renderer_uses_neutral_placeholder_when_quotes_are_malformed() -> None:
    renderer = DashboardRenderer()

    placeholder = renderer._clean_quote_snippet(("0101010101", "///////"))

    assert placeholder == "No reliable quote extracted"


def test_dashboard_renderer_shows_malformed_transcript_badge_for_malformed_errors() -> None:
    renderer = DashboardRenderer()
    interview = renderer._from_pipeline_result(
        PipelineResult(
            file_id="bad-1",
            filename="bad-1.mp4",
            status=ProcessingStatus.FAILED,
            last_error="Transcription failed: malformed transcript detected (excessive non-text ratio)",
            transcript=Transcript(text="broken", provider="test", model="test"),
        )
    )

    assert renderer._malformed_transcript_badge(interview) == "Malformed transcript"


def test_dashboard_renderer_excludes_malformed_from_cohort_base_by_default() -> None:
    renderer = DashboardRenderer()
    good = _dashboard_interview(interview_id="good-1")
    malformed = _dashboard_interview(interview_id="bad-1")
    setattr(malformed, "transcript_quality", "malformed")

    comparable, excluded_count = renderer._cohort_base([good, malformed])

    assert len(comparable) == 1
    assert comparable[0].id == "good-1"
    assert excluded_count == 1


def test_dashboard_renderer_prefers_income_table_for_sparse_or_non_numeric_inputs() -> None:
    renderer = DashboardRenderer()

    sparse_chart_rows = renderer._normalized_chart_rows({"₹10k–15k": 1})
    assert renderer._prefer_income_table_fallback(
        title="Income bands",
        values={"₹10k–15k": 1},
        chart_rows=sparse_chart_rows,
    )

    mixed_chart_rows = renderer._normalized_chart_rows({"₹10k–15k": "2", "Unknown": "n/a"})
    assert renderer._prefer_income_table_fallback(
        title="Income distribution",
        values={"₹10k–15k": "2", "Unknown": "n/a"},
        chart_rows=mixed_chart_rows,
    )

    borrowing_rows = renderer._normalized_chart_rows({"Borrower": 4, "Non-borrower": 2})
    assert not renderer._prefer_income_table_fallback(
        title="Borrowing behavior",
        values={"Borrower": 4, "Non-borrower": 2},
        chart_rows=borrowing_rows,
    )


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
    app.run(timeout=10)

    info_values = [element.value for element in app.info]
    markdown_values = [element.value for element in app.markdown]

    assert any("Upload interview audio from the sidebar to populate the dashboard overview." in value for value in info_values)
    assert any("Upload audio in the sidebar, then process at least one interview to review it here." in value for value in info_values)
    assert any(
        "To test with sample interviews, turn on sample mode in app settings." in value
        for value in info_values
    )
    assert not any("Employer-Dependent Digital Borrower" in value for value in markdown_values)
    assert not any("Digitally Ready but Fearful" in value for value in markdown_values)
    assert not any("Offline / Excluded" in value for value in markdown_values)
    assert not any("Self-Reliant Non-Borrower" in value for value in markdown_values)
    assert not any("High-Stress Cyclical Borrower" in value for value in markdown_values)
    assert not any("WhatsApp every day" in value for value in markdown_values)
    assert not any(getattr(element, "key", "").startswith(OPEN_INTERVIEW_KEY_PREFIX) for element in app.button)


def test_dashboard_renderer_boots_with_empty_cached_results_and_recent_interviews() -> None:
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
    app.run(timeout=10)

    assert not app.exception


def test_dashboard_interview_detail_exposes_explicit_save_actions() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

renderer = DashboardRenderer()
if not hasattr(renderer, "_render_status_error_indicator"):
    renderer._render_status_error_indicator = lambda *args, **kwargs: None
record = DashboardInterviewRecord(
    id="interview-1",
    audio_url="audio/interview-1/repo-name.wav",
    filename="repo-name.wav",
    transcript="Repository transcript",
    status=ProcessingStatus.COMPLETED.value,
    latest_stage="persona",
    last_error=None,
    created_at="2026-03-18T10:00:00+00:00",
    smartphone_user=True,
    has_bank_account=True,
    per_household_earnings=None,
    participant_personal_monthly_income="₹10k–15k",
    total_household_monthly_income=None,
    income_range="₹10k–15k",
    borrowing_history="has_borrowed",
    repayment_preference="monthly",
    loan_interest="interested",
    summary="Repository summary",
    key_quotes=["Repository quote"],
    persona="Digitally Ready but Fearful",
    confidence_score=0.91,
)

renderer.render(
    cached_results=[],
    recent_interviews=[record],
    status_overview=DashboardStatusOverview(total_interviews=1, status_counts={ProcessingStatus.COMPLETED.value: 1}),
    interview_detail_loader=lambda interview_id: record,
    save_interview_edits=lambda **kwargs: record,
    sample_mode=False,
)
'''

    app = AppTest.from_string(script)
    app.run(timeout=10)
    _find_open_interview_button(app).click()
    app.run(timeout=10)

    button_labels = [element.label for element in app.button]

    assert "Save transcript" in button_labels
    assert "Save structured JSON" in button_labels
    assert "Save all edits" in button_labels
    assert "Reprocess interview" in button_labels
    assert "Re-analyze selected interview" in button_labels


def test_dashboard_interviews_open_button_reveals_overlay_in_same_view() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

renderer = DashboardRenderer()
if not hasattr(renderer, "_render_status_error_indicator"):
    renderer._render_status_error_indicator = lambda *args, **kwargs: None
record = DashboardInterviewRecord(
    id="interview-1",
    audio_url="audio/interview-1/repo-name.wav",
    filename="repo-name.wav",
    transcript="Repository transcript",
    status=ProcessingStatus.COMPLETED.value,
    latest_stage="persona",
    last_error=None,
    created_at="2026-03-18T10:00:00+00:00",
    smartphone_user=True,
    has_bank_account=True,
    per_household_earnings=None,
    participant_personal_monthly_income="₹10k–15k",
    total_household_monthly_income=None,
    income_range="₹10k–15k",
    borrowing_history="has_borrowed",
    repayment_preference="monthly",
    loan_interest="interested",
    summary="Repository summary",
    key_quotes=["Repository quote"],
    persona="Digitally Ready but Fearful",
    confidence_score=0.91,
)

renderer.render(
    cached_results=[],
    recent_interviews=[record],
    status_overview=DashboardStatusOverview(total_interviews=1, status_counts={ProcessingStatus.COMPLETED.value: 1}),
    interview_detail_loader=lambda interview_id: record,
    save_interview_edits=lambda **kwargs: record,
    sample_mode=False,
)
'''

    app = AppTest.from_string(script)
    app.run(timeout=10)

    assert any(getattr(button, "key", "").startswith(OPEN_INTERVIEW_KEY_PREFIX) for button in app.button)
    assert not any(button.label == "Close overlay" for button in app.button)

    open_button = _find_open_interview_button(app)
    open_button.click()
    app.run()

    button_labels = [button.label for button in app.button]
    success_values = [element.value for element in app.success]
    markdown_values = [element.value for element in app.markdown]
    text_area_values = [element.value for element in app.text_area]

    assert "Close overlay" in button_labels
    assert not success_values
    assert any("Structured evidence" in value for value in markdown_values)
    assert "Repository transcript" in text_area_values


def test_dashboard_interviews_open_action_sets_overlay_state_after_click() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer, FILTER_SESSION_KEYS
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview
import streamlit as st

renderer = DashboardRenderer()
if not hasattr(renderer, "_render_status_error_indicator"):
    renderer._render_status_error_indicator = lambda *args, **kwargs: None
record = DashboardInterviewRecord(
    id="interview-1",
    audio_url="audio/interview-1/repo-name.wav",
    filename="repo-name.wav",
    transcript="Repository transcript",
    status=ProcessingStatus.COMPLETED.value,
    latest_stage="persona",
    last_error=None,
    created_at="2026-03-18T10:00:00+00:00",
    smartphone_user=True,
    has_bank_account=True,
    per_household_earnings=None,
    participant_personal_monthly_income="₹10k–15k",
    total_household_monthly_income=None,
    income_range="₹10k–15k",
    borrowing_history="has_borrowed",
    repayment_preference="monthly",
    loan_interest="interested",
    summary="Repository summary",
    key_quotes=["Repository quote"],
    persona="Digitally Ready but Fearful",
    confidence_score=0.91,
)

renderer.render(
    cached_results=[],
    recent_interviews=[record],
    status_overview=DashboardStatusOverview(total_interviews=1, status_counts={ProcessingStatus.COMPLETED.value: 1}),
    interview_detail_loader=lambda interview_id: record,
    save_interview_edits=lambda **kwargs: record,
    sample_mode=False,
)

if st.session_state.get(FILTER_SESSION_KEYS["overlay_open"]):
    st.caption("Overlay open state: true")
'''

    app = AppTest.from_string(script)
    app.run(timeout=10)
    _find_open_interview_button(app).click()
    app.run(timeout=10)

    caption_values = [element.value for element in app.caption]
    button_labels = [button.label for button in app.button]
    assert any("Overlay open state: true" in value for value in caption_values)
    assert "Close overlay" in button_labels


def test_dashboard_personas_tab_renders_without_sample_summary_name_error() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

renderer = DashboardRenderer()
record = DashboardInterviewRecord(
    id="interview-1",
    audio_url="audio/interview-1/repo-name.wav",
    filename="repo-name.wav",
    transcript="Repository transcript",
    status=ProcessingStatus.COMPLETED.value,
    latest_stage="persona",
    last_error=None,
    created_at="2026-03-18T10:00:00+00:00",
    smartphone_user=True,
    has_bank_account=True,
    per_household_earnings=None,
    participant_personal_monthly_income="₹10k–15k",
    total_household_monthly_income=None,
    income_range="₹10k–15k",
    borrowing_history="has_borrowed",
    repayment_preference="monthly",
    loan_interest="interested",
    summary="Repository summary",
    key_quotes=["Repository quote"],
    persona="Digitally Ready but Fearful",
    confidence_score=0.91,
)

renderer.render(
    cached_results=[],
    recent_interviews=[record],
    status_overview=DashboardStatusOverview(total_interviews=1, status_counts={ProcessingStatus.COMPLETED.value: 1}),
    interview_detail_loader=lambda interview_id: record,
    sample_mode=False,
)
'''
    app = AppTest.from_string(script)
    app.run(timeout=10)

    assert not app.exception


def test_dashboard_interview_card_hides_failed_transcript_preview_content() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

renderer = DashboardRenderer()
record = DashboardInterviewRecord(
    id="interview-1",
    audio_url="audio/interview-1/repo-name.wav",
    filename="repo-name.wav",
    transcript="<span class='meta-label'>garbled</span>",
    status=ProcessingStatus.FAILED.value,
    latest_stage="transcription",
    last_error="transcription failed",
    created_at="2026-03-18T10:00:00+00:00",
    smartphone_user=True,
    has_bank_account=True,
    per_household_earnings=None,
    participant_personal_monthly_income="₹10k–15k",
    total_household_monthly_income=None,
    income_range="₹10k–15k",
    borrowing_history="unknown",
    repayment_preference="unknown",
    loan_interest="unknown",
    summary="Analysis pending.",
    key_quotes=[],
    persona="Digitally Ready but Fearful",
    confidence_score=0.0,
    transcript_quality="failed",
)

renderer.render(
    cached_results=[],
    recent_interviews=[record],
    status_overview=DashboardStatusOverview(total_interviews=1, status_counts={ProcessingStatus.FAILED.value: 1}),
    interview_detail_loader=lambda interview_id: record,
    sample_mode=False,
)
'''
    app = AppTest.from_string(script)
    app.run(timeout=10)

    assert any(
        "Transcript unavailable due to failed/malformed transcription." in markdown.value
        for markdown in app.markdown
    )


def test_dashboard_interview_detail_requires_confirmation_before_delete() -> None:
    script = '''
from payday.dashboard.views import DashboardRenderer
from payday.models import ProcessingStatus
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

renderer = DashboardRenderer()
record = DashboardInterviewRecord(
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
    per_household_earnings=None,
    participant_personal_monthly_income="₹10k–15k",
    total_household_monthly_income=None,
    income_range="₹10k–15k",
    borrowing_history="has_borrowed",
    repayment_preference="monthly",
    loan_interest="interested",
    summary="Repository summary",
    key_quotes=["Repository quote"],
    persona="Digitally Ready but Fearful",
    confidence_score=0.91,
)

renderer.render(
    cached_results=[],
    recent_interviews=[record],
    status_overview=DashboardStatusOverview(total_interviews=1, status_counts={ProcessingStatus.COMPLETED.value: 1}),
    interview_detail_loader=lambda interview_id: record,
    delete_interview=lambda interview_id: True,
    sample_mode=False,
)
'''

    app = AppTest.from_string(script)
    app.run(timeout=10)
    _find_open_interview_button(app).click()
    app.run(timeout=10)

    assert any(button.label == "Delete interview" for button in app.button)
    assert not any(button.label == "Confirm delete" for button in app.button)

    delete_button = next(button for button in app.button if button.label == "Delete interview")
    delete_button.click()
    app.run()

    button_labels = [button.label for button in app.button]
    warning_values = [warning.value for warning in app.warning]

    assert "Confirm delete" in button_labels
    assert "Cancel delete" in button_labels
    assert any("related audio file" in value for value in warning_values)
