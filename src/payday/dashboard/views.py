from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Callable

import streamlit as st

from payday.analysis import (
    DEFAULT_UNKNOWN_VALUE,
    bank_account_user_from_analysis,
    get_analysis_value,
    smartphone_user_from_analysis,
)
from payday.models import PipelineResult, ProcessingStatus
from payday.personas import PERSONAS
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

PERSONA_LOOKUP = {persona.id: persona.name for persona in PERSONAS.values()}
STATUS_DISPLAY_ORDER = tuple(status.value for status in ProcessingStatus)
FILTER_SESSION_KEYS = {
    "income": "dashboard_filter_income",
    "borrowing": "dashboard_filter_borrowing",
    "persona": "dashboard_filter_persona",
    "digital_access": "dashboard_filter_digital_access",
    "search": "dashboard_search_query",
    "selected": "dashboard_selected_interview_id",
    "transcripts": "dashboard_transcript_edits",
    "json": "dashboard_json_edits",
    "detail_message": "dashboard_detail_message",
    "delete_confirm": "dashboard_delete_confirm",
    "selection_notice": "dashboard_selection_notice",
}


@dataclass(frozen=True, slots=True)
class DashboardInterview:
    id: str
    filename: str
    created_at: str
    status: str
    current_stage: str
    last_error: str | None
    summary: str
    transcript: str
    persona_id: str
    persona_name: str
    is_non_target: bool
    income_band: str
    borrowing_source: str
    borrowing_label: str
    is_borrower: bool
    loan_interest_label: str
    interested_in_loan: bool
    smartphone_user: bool | None
    has_bank_account: bool | None
    digital_access: str
    extracted_json: dict[str, Any]
    evidence_quotes: tuple[str, ...]
    audio_bytes: bytes | None = None
    audio_format: str = "audio/wav"


class DashboardRenderer:
    def render(
        self,
        *,
        cached_results: list[PipelineResult],
        recent_interviews: list[DashboardInterviewRecord],
        status_overview: DashboardStatusOverview,
        interview_detail_loader: Callable[[str], DashboardInterviewRecord],
        save_interview_edits: Callable[..., DashboardInterviewRecord] | None = None,
        delete_interview: Callable[[str], bool] | None = None,
        sample_mode: bool = False,
    ) -> None:
        self._inject_styles()
        interviews = self._build_dashboard_interviews(
            cached_results,
            recent_interviews,
            sample_mode=sample_mode,
        )
        self._initialize_session_state(interviews)
        filtered = self._apply_filters(interviews)

        tabs = st.tabs(["Overview", "Cohorts", "Personas", "Interviews", "Interview Detail"])
        with tabs[0]:
            self._render_overview(filtered, interviews, status_overview, sample_mode=sample_mode)
        with tabs[1]:
            self._render_cohorts(filtered)
        with tabs[2]:
            self._render_personas(filtered, sample_mode=sample_mode)
        with tabs[3]:
            self._render_interviews(filtered, sample_mode=sample_mode)
        with tabs[4]:
            self._render_interview_detail(
                filtered,
                interviews,
                interview_detail_loader,
                save_interview_edits=save_interview_edits,
                delete_interview=delete_interview,
                sample_mode=sample_mode,
            )

    def _initialize_session_state(self, interviews: list[DashboardInterview]) -> None:
        st.session_state.setdefault(FILTER_SESSION_KEYS["income"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["borrowing"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["persona"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["digital_access"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["search"], "")
        st.session_state.setdefault(FILTER_SESSION_KEYS["transcripts"], {})
        st.session_state.setdefault(FILTER_SESSION_KEYS["json"], {})
        st.session_state.setdefault(FILTER_SESSION_KEYS["detail_message"], None)
        st.session_state.setdefault(FILTER_SESSION_KEYS["delete_confirm"], None)
        st.session_state.setdefault(FILTER_SESSION_KEYS["selection_notice"], None)

        selected_key = FILTER_SESSION_KEYS["selected"]
        if interviews and not st.session_state.get(selected_key):
            st.session_state[selected_key] = interviews[0].id
        if interviews and st.session_state.get(selected_key) not in {item.id for item in interviews}:
            st.session_state[selected_key] = interviews[0].id

    def _apply_filters(self, interviews: list[DashboardInterview]) -> list[DashboardInterview]:
        search_query = st.session_state[FILTER_SESSION_KEYS["search"]].strip().lower()
        income_filters = set(st.session_state[FILTER_SESSION_KEYS["income"]])
        borrowing_filters = set(st.session_state[FILTER_SESSION_KEYS["borrowing"]])
        persona_filters = set(st.session_state[FILTER_SESSION_KEYS["persona"]])
        digital_filters = set(st.session_state[FILTER_SESSION_KEYS["digital_access"]])

        filtered: list[DashboardInterview] = []
        for interview in interviews:
            haystack = " ".join(
                [
                    interview.filename,
                    interview.id,
                    interview.summary,
                    interview.transcript,
                    interview.persona_name,
                    interview.borrowing_source,
                ]
            ).lower()
            if search_query and search_query not in haystack:
                continue
            if income_filters and interview.income_band not in income_filters:
                continue
            if borrowing_filters and interview.borrowing_label not in borrowing_filters:
                continue
            if persona_filters and interview.persona_name not in persona_filters:
                continue
            if digital_filters and interview.digital_access not in digital_filters:
                continue
            filtered.append(interview)
        return filtered

    def _render_overview(
        self,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
        *,
        sample_mode: bool,
    ) -> None:
        st.markdown("### Interview portfolio")
        self._render_status_panel(all_interviews, status_overview, sample_mode=sample_mode)
        self._render_filter_summary(filtered, all_interviews, status_overview)
        self._render_kpis(filtered, status_overview)

        if not filtered:
            if all_interviews:
                st.info("No interviews match the current filters.")
            else:
                self._render_empty_state(
                    "No interviews yet",
                    sample_mode=sample_mode,
                    context="Upload interview audio from the sidebar to populate the dashboard overview.",
                )
            return

        income_col, borrowing_col = st.columns(2, gap="large")
        with income_col:
            self._render_income_summary(filtered)
        with borrowing_col:
            self._render_borrowing_sources_summary(filtered)

        top_quotes = [quote for interview in filtered for quote in interview.evidence_quotes][:4]
        if top_quotes:
            st.markdown("### Evidence highlights")
            quote_columns = st.columns(min(2, len(top_quotes)), gap="large")
            for index, quote in enumerate(top_quotes):
                with quote_columns[index % len(quote_columns)]:
                    st.markdown(
                        f"<div class='pd-card quote-card'>“{quote}”</div>",
                        unsafe_allow_html=True,
                    )

    def _render_cohorts(self, filtered: list[DashboardInterview]) -> None:
        st.markdown("### Cohort slices")
        if not filtered:
            st.info("No cohorts to summarize for the current filter set.")
            return

        digital_col, borrowing_col = st.columns(2, gap="large")
        with digital_col:
            self._render_table_card(
                title="Digital access cohorts",
                rows=self._build_cohort_rows(filtered, "digital_access"),
                headers=("Cohort", "Count", "% of filtered"),
            )
        with borrowing_col:
            self._render_table_card(
                title="Borrowing behavior cohorts",
                rows=self._build_cohort_rows(filtered, "borrowing_label"),
                headers=("Cohort", "Count", "% of filtered"),
            )

        persona_col, status_col = st.columns(2, gap="large")
        with persona_col:
            self._render_table_card(
                title="Persona mix",
                rows=self._build_cohort_rows(filtered, "persona_name"),
                headers=("Persona", "Count", "% of filtered"),
            )
        with status_col:
            self._render_table_card(
                title="Processing status",
                rows=self._build_cohort_rows(filtered, "status"),
                headers=("Status", "Count", "% of filtered"),
            )

    def _render_personas(self, filtered: list[DashboardInterview], *, sample_mode: bool) -> None:
        st.markdown("### Persona review")
        if not filtered:
            self._render_empty_state(
                "No personas to review yet",
                sample_mode=sample_mode,
                context="Run at least one interview through transcription and analysis before persona cards can appear.",
            )
            return

        counts = self._count_by(filtered, lambda item: item.persona_name)
        columns = st.columns(2, gap="large")
        for index, (persona_name, count) in enumerate(counts.items()):
            matching = [item for item in filtered if item.persona_name == persona_name]
            sample_quote = matching[0].evidence_quotes[0] if matching and matching[0].evidence_quotes else "No quote captured"
            non_target_count = sum(1 for item in matching if item.is_non_target)
            with columns[index % len(columns)]:
                st.markdown(
                    """
                    <div class='pd-card persona-card'>
                        <div class='persona-label'>{persona_name}</div>
                        <div class='persona-count'>{count}</div>
                        <div class='persona-meta'>{non_target_count} marked non-target</div>
                        <div class='persona-quote'>“{sample_quote}”</div>
                    </div>
                    """.format(
                        persona_name=persona_name,
                        count=count,
                        non_target_count=non_target_count,
                        sample_quote=sample_quote,
                    ),
                    unsafe_allow_html=True,
                )

    def _render_interviews(self, filtered: list[DashboardInterview], *, sample_mode: bool) -> None:
        st.markdown("### Searchable interview list")
        st.caption("Click an interview card to open an inline preview immediately. The Interview Detail tab stays synced to the same selection.")
        if not filtered:
            self._render_empty_state(
                "No interview cards yet",
                sample_mode=sample_mode,
                context="Upload audio files or intentionally load sample fixtures to create interview cards.",
            )
            return

        selected_id = st.session_state.get(FILTER_SESSION_KEYS["selected"])
        for interview in filtered:
            if self._render_interview_row(interview, selected_id=selected_id):
                selected_id = interview.id

    def _render_interview_detail(
        self,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        interview_detail_loader: Callable[[str], DashboardInterviewRecord],
        *,
        save_interview_edits: Callable[..., DashboardInterviewRecord] | None,
        delete_interview: Callable[[str], bool] | None,
        sample_mode: bool,
    ) -> None:
        st.markdown("### Interview detail")
        if not all_interviews:
            self._render_empty_state(
                "No interview selected",
                sample_mode=sample_mode,
                context="Upload and process at least one interview, or intentionally load sample fixtures, to inspect transcript details here.",
            )
            return

        selected_id = st.session_state[FILTER_SESSION_KEYS["selected"]]
        selected = next((item for item in all_interviews if item.id == selected_id), None)
        if selected is None:
            st.info("Select an interview from the list to inspect it here.")
            return

        if filtered and selected.id not in {item.id for item in filtered}:
            st.warning("The selected interview is outside the current filter set, but remains available for review.")

        selected = self._merge_detail_record(selected, interview_detail_loader)

        transcript_edits: dict[str, str] = st.session_state[FILTER_SESSION_KEYS["transcripts"]]
        json_edits: dict[str, str] = st.session_state[FILTER_SESSION_KEYS["json"]]
        original_json = json.dumps(selected.extracted_json, indent=2, ensure_ascii=False)
        transcript_edits.setdefault(selected.id, selected.transcript)
        json_edits.setdefault(selected.id, original_json)

        detail_message = st.session_state.get(FILTER_SESSION_KEYS["detail_message"])
        if isinstance(detail_message, dict):
            kind = detail_message.get("kind")
            message = str(detail_message.get("message", "")).strip()
            if kind == "success" and message:
                st.success(message)
            elif kind == "error" and message:
                st.error(message)
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = None

        selection_notice = st.session_state.get(FILTER_SESSION_KEYS["selection_notice"])
        if selection_notice == selected.id:
            st.info(f"Interview selection synced from the Interviews tab: {selected.filename} is now active in Interview Detail.")

        header_col, badge_col = st.columns([3, 1])
        with header_col:
            st.markdown(f"#### {selected.filename}")
            st.caption(f"Interview ID: {selected.id} · Created: {selected.created_at} · Status: {selected.status}")
        with badge_col:
            badge_class = "badge-nontarget" if selected.is_non_target else "badge-target"
            st.markdown(
                f"<div class='persona-badge {badge_class}'>{selected.persona_name}</div>",
                unsafe_allow_html=True,
            )

        audio_col, meta_col = st.columns([2, 1], gap="large")
        with audio_col:
            st.markdown("##### Audio")
            if selected.audio_bytes:
                st.audio(selected.audio_bytes, format=selected.audio_format)
            else:
                st.info("Audio bytes are unavailable after restart, but the durable interview record remains available.")
                if selected.extracted_json.get("audio_url"):
                    st.caption(f"Stored audio path: {selected.extracted_json['audio_url']}")
        with meta_col:
            st.markdown("##### Extracted flags")
            flag_rows = [
                ("Digital access", selected.digital_access),
                ("Income band", selected.income_band),
                ("Borrowing", selected.borrowing_label),
                ("Borrowing source", selected.borrowing_source),
                ("Loan interest", selected.loan_interest_label),
            ]
            flag_markup = "".join(
                f"<div class='detail-flag-row'><span class='detail-flag-label'>{label}</span><span class='detail-flag-value'>{value}</span></div>"
                for label, value in flag_rows
            )
            st.markdown(
                f"<div class='pd-card detail-flags-card'>{flag_markup}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("##### Editable transcript")
        updated_transcript = st.text_area(
            "Transcript",
            key=f"detail_transcript_{selected.id}",
            value=transcript_edits[selected.id],
            height=220,
            label_visibility="collapsed",
        )
        transcript_edits[selected.id] = updated_transcript

        st.markdown("##### Extracted JSON")
        updated_json = st.text_area(
            "Extracted JSON",
            key=f"detail_json_{selected.id}",
            value=json_edits[selected.id],
            height=260,
            label_visibility="collapsed",
        )
        json_edits[selected.id] = updated_json

        transcript_changed = updated_transcript != selected.transcript
        structured_json_changed = updated_json != original_json

        transcript_col, json_col, save_all_col, delete_col = st.columns([1.1, 1.2, 1.1, 1], gap="medium")
        with transcript_col:
            transcript_save_disabled = save_interview_edits is None or not transcript_changed
            if st.button(
                "Save transcript",
                key=f"save_transcript_{selected.id}",
                type="primary",
                use_container_width=True,
                disabled=transcript_save_disabled,
            ):
                self._save_interview_detail_edits(
                    interview_id=selected.id,
                    transcript=updated_transcript,
                    extracted_json=original_json,
                    transcript_changed=True,
                    structured_json_changed=False,
                    save_interview_edits=save_interview_edits,
                    transcript_edits=transcript_edits,
                    json_edits=json_edits,
                    success_message="Transcript saved. Downstream analysis, persona derivation, and dashboard views were refreshed from SQLite.",
                )
            if save_interview_edits is None:
                st.caption("Transcript save is unavailable because no backend save handler was provided.")
            elif not transcript_changed:
                st.caption("No transcript edits to save.")
            else:
                st.caption("Saves only the transcript, then reruns analysis/persona and reloads dashboard summaries.")

        with json_col:
            json_save_disabled = save_interview_edits is None or not structured_json_changed
            if st.button(
                "Save structured JSON",
                key=f"save_json_{selected.id}",
                use_container_width=True,
                disabled=json_save_disabled,
            ):
                self._save_interview_detail_edits(
                    interview_id=selected.id,
                    transcript=selected.transcript or "",
                    extracted_json=updated_json,
                    transcript_changed=False,
                    structured_json_changed=True,
                    save_interview_edits=save_interview_edits,
                    transcript_edits=transcript_edits,
                    json_edits=json_edits,
                    success_message="Structured JSON saved. Persona outputs and dashboard summaries were refreshed from SQLite.",
                )
            if save_interview_edits is None:
                st.caption("Structured JSON save is unavailable because no backend save handler was provided.")
            elif not structured_json_changed:
                st.caption("No structured JSON edits to save.")
            else:
                st.caption("Validates the JSON, persists it to SQLite, and reruns persona derivation from the edited values.")

        with save_all_col:
            save_all_disabled = save_interview_edits is None or (not transcript_changed and not structured_json_changed)
            if st.button(
                "Save all edits",
                key=f"save_all_{selected.id}",
                use_container_width=True,
                disabled=save_all_disabled,
            ):
                self._save_interview_detail_edits(
                    interview_id=selected.id,
                    transcript=updated_transcript,
                    extracted_json=updated_json,
                    transcript_changed=transcript_changed,
                    structured_json_changed=structured_json_changed,
                    save_interview_edits=save_interview_edits,
                    transcript_edits=transcript_edits,
                    json_edits=json_edits,
                    success_message="Transcript and structured JSON saved. Analysis, persona outputs, and dashboard summaries were refreshed from SQLite.",
                )
            if save_interview_edits is None:
                st.caption("Save all is unavailable because no backend save handler was provided.")
            elif not transcript_changed and not structured_json_changed:
                st.caption("No unsaved interview detail edits detected.")
            elif transcript_changed and structured_json_changed:
                st.caption("Recommended when both transcript and JSON drafts changed, so both edits persist together.")
            else:
                st.caption("Saves whichever interview detail drafts are currently unsaved.")

        with delete_col:
            delete_disabled = delete_interview is None
            if st.button(
                "Delete interview",
                key=f"delete_interview_{selected.id}",
                use_container_width=True,
                disabled=delete_disabled,
            ):
                st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = selected.id
            if delete_disabled:
                st.caption("Delete is unavailable because no backend delete handler was provided.")

        if st.session_state.get(FILTER_SESSION_KEYS["delete_confirm"]) == selected.id and delete_interview is not None:
            st.warning(
                "Delete this interview from durable storage? This also removes linked structured responses and insights."
            )
            confirm_col, cancel_col = st.columns(2, gap="medium")
            with confirm_col:
                if st.button(
                    "Confirm delete",
                    key=f"confirm_delete_interview_{selected.id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    try:
                        deleted = delete_interview(selected.id)
                    except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                        st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                            "kind": "error",
                            "message": f"Delete failed: {exc}",
                        }
                    else:
                        if deleted:
                            transcript_edits.pop(selected.id, None)
                            json_edits.pop(selected.id, None)
                            st.session_state[FILTER_SESSION_KEYS["selected"]] = self._next_selected_id(
                                all_interviews,
                                deleted_id=selected.id,
                            )
                            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                                "kind": "success",
                                "message": "Interview deleted. Overview, cohort, and persona counts were refreshed from durable storage.",
                            }
                        else:
                            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                                "kind": "error",
                                "message": f"Interview {selected.id} could not be deleted because it no longer exists.",
                            }
                        st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                        st.rerun()
            with cancel_col:
                if st.button(
                    "Cancel delete",
                    key=f"cancel_delete_interview_{selected.id}",
                    use_container_width=True,
                ):
                    st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                    st.rerun()

    def _save_interview_detail_edits(
        self,
        *,
        interview_id: str,
        transcript: str,
        extracted_json: str,
        transcript_changed: bool,
        structured_json_changed: bool,
        save_interview_edits: Callable[..., DashboardInterviewRecord] | None,
        transcript_edits: dict[str, str],
        json_edits: dict[str, str],
        success_message: str,
    ) -> None:
        if save_interview_edits is None:
            return
        try:
            save_interview_edits(
                interview_id,
                transcript=transcript,
                extracted_json=extracted_json,
                transcript_changed=transcript_changed,
                structured_json_changed=structured_json_changed,
            )
        except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                "kind": "error",
                "message": f"Save failed: {exc}",
            }
        else:
            transcript_edits.pop(interview_id, None)
            json_edits.pop(interview_id, None)
            st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                "kind": "success",
                "message": success_message,
            }
            st.rerun()

    def _render_filter_summary(
        self,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
    ) -> None:
        income_options = sorted({item.income_band for item in all_interviews})
        borrowing_options = sorted({item.borrowing_label for item in all_interviews})
        persona_options = sorted({item.persona_name for item in all_interviews})
        digital_options = sorted({item.digital_access for item in all_interviews})

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        st.sidebar.multiselect(
            "Income",
            options=income_options,
            key=FILTER_SESSION_KEYS["income"],
            placeholder="All income bands",
        )
        st.sidebar.multiselect(
            "Borrowing",
            options=borrowing_options,
            key=FILTER_SESSION_KEYS["borrowing"],
            placeholder="All borrowing types",
        )
        st.sidebar.multiselect(
            "Persona",
            options=persona_options,
            key=FILTER_SESSION_KEYS["persona"],
            placeholder="All personas",
        )
        st.sidebar.multiselect(
            "Digital access",
            options=digital_options,
            key=FILTER_SESSION_KEYS["digital_access"],
            placeholder="All access levels",
        )
        st.sidebar.text_input(
            "Search interviews",
            key=FILTER_SESSION_KEYS["search"],
            placeholder="Search filename, quote, summary, transcript",
        )

        st.caption(
            f"Showing {len(filtered)} of {len(all_interviews)} interviews. Durable interviews in SQLite: {status_overview.total_interviews}."
        )

    def _render_kpis(self, interviews: list[DashboardInterview], status_overview: DashboardStatusOverview) -> None:
        completed_count = status_overview.status_counts.get(ProcessingStatus.COMPLETED.value, 0)
        metrics = [
            ("Filtered interviews", str(len(interviews))),
            ("Completed (durable)", str(completed_count)),
            ("Smartphone %", self._percent(sum(1 for item in interviews if item.smartphone_user is True), len(interviews))),
            ("Borrowers %", self._percent(sum(1 for item in interviews if item.is_borrower), len(interviews))),
            (
                "Loan-interest %",
                self._percent(sum(1 for item in interviews if item.interested_in_loan), len(interviews)),
            ),
        ]
        columns = st.columns(5, gap="large")
        for column, (label, value) in zip(columns, metrics, strict=False):
            with column:
                st.markdown(
                    f"<div class='pd-card kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>",
                    unsafe_allow_html=True,
                )

    def _render_status_panel(
        self,
        interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
        *,
        sample_mode: bool,
    ) -> None:
        st.markdown("#### Processing status")
        st.caption("Per-file pipeline states are reloaded from the durable SQLite store whenever the app reruns or you use Refresh status.")

        counts = self._status_counts(interviews, status_overview)
        count_columns = st.columns(len(STATUS_DISPLAY_ORDER), gap="medium")
        for column, status in zip(count_columns, STATUS_DISPLAY_ORDER, strict=False):
            with column:
                st.markdown(
                    (
                        "<div class='pd-card status-card'>"
                        f"<div class='status-label'>{status.title()}</div>"
                        f"<div class='status-value'>{counts.get(status, 0)}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        if not interviews:
            self._render_empty_state(
                "No interview records are available yet",
                sample_mode=sample_mode,
                context="Upload a batch to populate the durable status table.",
            )
            return

        status_rows = [
            {
                "Filename": interview.filename,
                "Status": interview.status.title(),
                "Stage": interview.current_stage.title(),
                "Last error": self._truncate_text(interview.last_error or "—", limit=120),
                "Created": interview.created_at,
                "Persona": interview.persona_name,
                "Summary": self._truncate_text(interview.summary, limit=100),
            }
            for interview in interviews
        ]
        st.dataframe(
            status_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Filename": st.column_config.TextColumn(width="medium"),
                "Status": st.column_config.TextColumn(width="small"),
                "Stage": st.column_config.TextColumn(width="small"),
                "Last error": st.column_config.TextColumn(width="large"),
                "Created": st.column_config.TextColumn(width="small"),
                "Persona": st.column_config.TextColumn(width="medium"),
                "Summary": st.column_config.TextColumn(width="large"),
            },
        )

    def _render_chart_card(
        self,
        *,
        title: str,
        subtitle: str,
        values: dict[str, int],
        color: str,
    ) -> None:
        st.markdown(f"<div class='chart-title'>{title}</div><div class='chart-subtitle'>{subtitle}</div>", unsafe_allow_html=True)
        chart_data = [{"category": key, "count": value} for key, value in values.items()]
        if not chart_data:
            st.info("No chart data available.")
            return
        st.vega_lite_chart(
            chart_data,
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 10, "cornerRadiusTopRight": 10, "color": color},
                "encoding": {
                    "x": {"field": "category", "type": "nominal", "sort": None, "axis": {"title": None, "labelAngle": 0}},
                    "y": {"field": "count", "type": "quantitative", "axis": {"title": None}},
                    "tooltip": [
                        {"field": "category", "type": "nominal", "title": "Category"},
                        {"field": "count", "type": "quantitative", "title": "Interviews"},
                    ],
                },
                "height": 320,
            },
            use_container_width=True,
        )

    def _render_table_card(self, *, title: str, rows: list[tuple[str, int, str]], headers: tuple[str, str, str]) -> None:
        st.markdown(f"<div class='table-title'>{title}</div>", unsafe_allow_html=True)
        table_rows = "".join(
            f"<tr><td>{label}</td><td>{count}</td><td>{share}</td></tr>" for label, count, share in rows
        )
        st.markdown(
            f"""
            <div class='pd-card table-card'>
                <table class='pd-table'>
                    <thead>
                        <tr><th>{headers[0]}</th><th>{headers[1]}</th><th>{headers[2]}</th></tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_interview_row(self, interview: DashboardInterview, *, selected_id: str | None) -> bool:
        button_label = (
            f"**{interview.filename}**  \n"
            f"{self._truncate_text(interview.summary, limit=140)}  \n"
            f"Borrowing: {interview.borrowing_source} · Income: {interview.income_band}"
        )
        row_selected = interview.id == selected_id

        col_main, col_meta = st.columns([3.2, 1.4], gap="medium")
        with col_main:
            clicked = st.button(
                button_label,
                key=f"select_interview_{interview.id}",
                use_container_width=True,
                type="primary" if row_selected else "secondary",
            )
        with col_meta:
            st.markdown(
                f"""
                <div class='pd-card interview-meta-card'>
                    <div class='interview-meta-primary'>{interview.persona_name}</div>
                    <div class='interview-meta-secondary'>{interview.digital_access}</div>
                    <div class='interview-meta-secondary'>Status: {interview.status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if clicked:
            st.session_state[FILTER_SESSION_KEYS["selected"]] = interview.id
            st.session_state[FILTER_SESSION_KEYS["selection_notice"]] = interview.id
            row_selected = True

        if row_selected:
            self._render_inline_interview_detail(interview)

        return clicked

    def _render_inline_interview_detail(self, interview: DashboardInterview) -> None:
        st.success(f"Showing inline detail for {interview.filename}. The Interview Detail tab was updated to this selection.")

        quote_markup = "".join(
            f"<li>{quote}</li>" for quote in interview.evidence_quotes[:3]
        ) or "<li>No direct quote captured yet.</li>"
        transcript_preview = self._truncate_text(interview.transcript, limit=320)
        st.markdown(
            f"""
            <div class='pd-card interview-detail-inline'>
                <div class='interview-detail-inline-header'>Selected interview snapshot</div>
                <div class='interview-meta'>Interview ID: {interview.id} · Persona: {interview.persona_name} · Status: {interview.status}</div>
                <div class='interview-summary'>{interview.summary}</div>
                <div class='interview-inline-grid'>
                    <div>
                        <div class='interview-inline-label'>Evidence quotes</div>
                        <ul class='interview-inline-list'>{quote_markup}</ul>
                    </div>
                    <div>
                        <div class='interview-inline-label'>Transcript preview</div>
                        <div class='interview-inline-preview'>{transcript_preview}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _next_selected_id(self, interviews: list[DashboardInterview], *, deleted_id: str) -> str | None:
        remaining_ids = [item.id for item in interviews if item.id != deleted_id]
        if not remaining_ids:
            return None
        return remaining_ids[0]

    def _build_cohort_rows(self, interviews: list[DashboardInterview], attribute: str) -> list[tuple[str, int, str]]:
        counts = self._count_by(interviews, lambda item: getattr(item, attribute))
        total = len(interviews)
        return [(label, count, self._percent(count, total)) for label, count in counts.items()]

    def _numeric_income_values(self, interviews: list[DashboardInterview]) -> list[int]:
        values: list[int] = []
        for interview in interviews:
            income_value = self._parse_direct_income_value(interview.income_band)
            if income_value is not None:
                values.append(income_value)
        return values

    def _parse_direct_income_value(self, income_band: str) -> int | None:
        normalized = income_band.strip()
        if not normalized:
            return None
        if any(separator in normalized for separator in ("-", "–", "to")):
            return None
        if any(term in normalized.lower() for term in ("below", "under", "less than", "unknown")):
            return None

        match = re.search(r"(\d[\d,]*)(?:\s*([kK]))?", normalized)
        if match is None:
            return None

        amount = int(match.group(1).replace(",", ""))
        if match.group(2):
            amount *= 1000
        return amount

    def _format_currency(self, amount: int) -> str:
        return f"₹{amount:,}"

    def _count_by(self, interviews: list[DashboardInterview], key_func: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for interview in interviews:
            key = key_func(interview)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _build_dashboard_interviews(
        self,
        results: list[PipelineResult],
        recent_interviews: list[DashboardInterviewRecord],
        *,
        sample_mode: bool = False,
    ) -> list[DashboardInterview]:
        mapped_results = {result.file_id: self._from_pipeline_result(result) for result in results}
        merged = [self._overlay_cached_result(record, mapped_results.get(record.id)) for record in recent_interviews]
        if merged:
            repository_ids = {record.id for record in recent_interviews}
            cache_only = [interview for file_id, interview in mapped_results.items() if file_id not in repository_ids]
            return [*merged, *cache_only]
        if mapped_results:
            return list(mapped_results.values())
        if sample_mode:
            from payday.dashboard.sample_data import build_sample_interviews

            return build_sample_interviews(
                persona_lookup=PERSONA_LOOKUP,
                digital_access_label=self._digital_access_label,
            )
        return []

    def _from_pipeline_result(self, result: PipelineResult) -> DashboardInterview:
        structured = result.analysis.structured_output if result.analysis is not None else {}
        transcript = result.transcript.text if result.transcript is not None else "Transcript pending."
        smartphone_user = smartphone_user_from_analysis(structured)
        has_bank_account = bank_account_user_from_analysis(structured)
        digital_access = self._digital_access_label(smartphone_user, has_bank_account)
        borrowing_value = get_analysis_value(structured, "borrowing_history")
        loan_interest_value = get_analysis_value(structured, "loan_interest")
        summary = result.analysis.summary if result.analysis is not None else "Analysis pending."
        evidence_quotes = tuple(result.analysis.evidence_quotes if result.analysis is not None else [])
        persona_id = result.persona.persona_id if result.persona is not None else "persona_4"
        persona_name = result.persona.persona_name if result.persona is not None else PERSONA_LOOKUP["persona_4"]
        return DashboardInterview(
            id=result.file_id,
            filename=result.filename,
            created_at=datetime.now(timezone.utc).date().isoformat(),
            status=result.status.value,
            current_stage=result.current_stage.value,
            last_error=result.last_error,
            summary=summary,
            transcript=transcript,
            persona_id=persona_id,
            persona_name=persona_name,
            is_non_target=bool(result.persona.is_non_target) if result.persona is not None else False,
            income_band=self._display_value(get_analysis_value(structured, "income_range")),
            borrowing_source=self._borrowing_source_label(transcript, borrowing_value),
            borrowing_label="Borrower" if self._is_borrower_value(borrowing_value, transcript) else "Non-borrower",
            is_borrower=self._is_borrower_value(borrowing_value, transcript),
            loan_interest_label=self._loan_interest_label(loan_interest_value),
            interested_in_loan=self._is_loan_interest_positive(loan_interest_value),
            smartphone_user=smartphone_user,
            has_bank_account=has_bank_account,
            digital_access=digital_access,
            extracted_json=structured or self._empty_extracted_json(summary="Analysis pending."),
            evidence_quotes=evidence_quotes,
            audio_bytes=result.asset.raw_bytes if result.asset is not None else None,
            audio_format=result.asset.content_type if result.asset is not None else "audio/wav",
        )

    def _from_repository_record(self, record: DashboardInterviewRecord) -> DashboardInterview:
        summary = record.summary or "Analysis pending."
        transcript = record.transcript or "Transcript pending."
        persona_name = record.persona or PERSONA_LOOKUP["persona_4"]
        persona_id = self._persona_id_from_name(persona_name)
        borrowing_value = record.borrowing_history or "unknown"
        loan_interest_value = record.loan_interest or "unknown"
        extracted_json = self._repository_analysis_json(record)
        return DashboardInterview(
            id=record.id,
            filename=record.filename,
            created_at=self._format_created_at(record.created_at),
            status=record.status,
            current_stage=record.latest_stage,
            last_error=record.last_error,
            summary=summary,
            transcript=transcript,
            persona_id=persona_id,
            persona_name=persona_name,
            is_non_target=record.smartphone_user is False or record.has_bank_account is False,
            income_band=record.income_range or "Unknown",
            borrowing_source=self._borrowing_source_label(transcript, borrowing_value),
            borrowing_label="Borrower" if self._is_borrower_value(borrowing_value, transcript) else "Non-borrower",
            is_borrower=self._is_borrower_value(borrowing_value, transcript),
            loan_interest_label=self._loan_interest_label(loan_interest_value),
            interested_in_loan=self._is_loan_interest_positive(loan_interest_value),
            smartphone_user=record.smartphone_user,
            has_bank_account=record.has_bank_account,
            digital_access=self._digital_access_label(record.smartphone_user, record.has_bank_account),
            extracted_json=extracted_json,
            evidence_quotes=tuple(record.key_quotes),
        )

    def _overlay_cached_result(
        self,
        record: DashboardInterviewRecord,
        cached_result: DashboardInterview | None,
    ) -> DashboardInterview:
        repository_interview = self._from_repository_record(record)
        if cached_result is None:
            return repository_interview
        persona_name = repository_interview.persona_name or cached_result.persona_name
        return DashboardInterview(
            id=repository_interview.id,
            filename=repository_interview.filename or cached_result.filename,
            created_at=repository_interview.created_at,
            status=repository_interview.status,
            current_stage=repository_interview.current_stage,
            last_error=repository_interview.last_error or cached_result.last_error,
            summary=repository_interview.summary or cached_result.summary,
            transcript=repository_interview.transcript or cached_result.transcript,
            persona_id=repository_interview.persona_id if repository_interview.persona_name else cached_result.persona_id,
            persona_name=persona_name,
            is_non_target=repository_interview.is_non_target,
            income_band=repository_interview.income_band if repository_interview.income_band != "Unknown" else cached_result.income_band,
            borrowing_source=repository_interview.borrowing_source
            if repository_interview.borrowing_source != "Unknown"
            else cached_result.borrowing_source,
            borrowing_label=repository_interview.borrowing_label,
            is_borrower=repository_interview.is_borrower,
            loan_interest_label=repository_interview.loan_interest_label,
            interested_in_loan=repository_interview.interested_in_loan,
            smartphone_user=repository_interview.smartphone_user,
            has_bank_account=repository_interview.has_bank_account,
            digital_access=repository_interview.digital_access,
            extracted_json={
                **cached_result.extracted_json,
                **repository_interview.extracted_json,
            },
            evidence_quotes=repository_interview.evidence_quotes or cached_result.evidence_quotes,
            audio_bytes=cached_result.audio_bytes,
            audio_format=cached_result.audio_format,
        )

    def _merge_detail_record(
        self,
        selected: DashboardInterview,
        interview_detail_loader: Callable[[str], DashboardInterviewRecord],
    ) -> DashboardInterview:
        try:
            detail = interview_detail_loader(selected.id)
        except KeyError:
            return selected
        return self._overlay_cached_result(detail, selected)

    def _persona_id_from_name(self, persona_name: str) -> str:
        normalized = persona_name.strip().lower()
        for persona_id, canonical_name in PERSONA_LOOKUP.items():
            if canonical_name.lower() == normalized:
                return persona_id
        return "persona_4"

    def _format_created_at(self, created_at: str) -> str:
        try:
            return datetime.fromisoformat(created_at).date().isoformat()
        except ValueError:
            return created_at

    def _repository_analysis_json(self, record: DashboardInterviewRecord) -> dict[str, Any]:
        return {
            "smartphone_usage": self._bool_field(
                record.smartphone_user,
                true_value="has_smartphone",
                false_value="no_smartphone",
            ),
            "bank_account_status": self._bool_field(
                record.has_bank_account,
                true_value="has_bank_account",
                false_value="no_bank_account",
            ),
            "income_range": self._value_field(record.income_range),
            "borrowing_history": self._value_field(record.borrowing_history),
            "repayment_preference": self._value_field(record.repayment_preference),
            "loan_interest": self._value_field(record.loan_interest),
            "summary": self._value_field(record.summary, evidence_quotes=record.key_quotes),
            "key_quotes": list(record.key_quotes),
            "confidence_signals": {
                "observed_evidence": [],
                "missing_or_unknown": [],
            },
        }

    def _bool_field(
        self,
        value: bool | None,
        *,
        true_value: str,
        false_value: str,
    ) -> dict[str, Any]:
        if value is True:
            return self._value_field(true_value)
        if value is False:
            return self._value_field(false_value)
        return self._value_field(None)

    def _value_field(self, value: str | None, *, evidence_quotes: list[str] | None = None) -> dict[str, Any]:
        normalized = (value or "").strip()
        if not normalized:
            normalized = DEFAULT_UNKNOWN_VALUE
        return {
            "value": normalized,
            "status": "unknown" if normalized == DEFAULT_UNKNOWN_VALUE else "observed",
            "evidence_quotes": list(evidence_quotes or []),
            "notes": "",
        }

    def _empty_extracted_json(self, *, summary: str) -> dict[str, Any]:
        return {
            "smartphone_usage": self._value_field(None),
            "bank_account_status": self._value_field(None),
            "income_range": self._value_field(None),
            "borrowing_history": self._value_field(None),
            "repayment_preference": self._value_field(None),
            "loan_interest": self._value_field(None),
            "summary": self._value_field(summary),
            "key_quotes": [],
            "confidence_signals": {"observed_evidence": [], "missing_or_unknown": []},
        }

    def _display_value(self, value: str) -> str:
        return "Unknown" if value == DEFAULT_UNKNOWN_VALUE else value

    def _digital_access_label(self, smartphone_user: bool | None, has_bank_account: bool | None) -> str:
        if smartphone_user is False or has_bank_account is False:
            return "Excluded / offline"
        if smartphone_user is True and has_bank_account is True:
            return "Smartphone + bank account"
        return "Unknown / partial"

    def _is_borrower_value(self, borrowing_value: str, transcript: str) -> bool:
        lowered = f"{borrowing_value} {transcript}".lower()
        return any(token in lowered for token in ("borrow", "loan", "moneylender", "neighbors", "employer")) and not any(
            token in lowered for token in ("do not borrow", "don't borrow", "avoid loans")
        )

    def _borrowing_source_label(self, transcript: str, borrowing_value: str) -> str:
        lowered = f"{borrowing_value} {transcript}".lower()
        source_map = {
            "Employer": ("employer", "madam", "sir", "boss"),
            "Family / friends": ("family", "friend", "neighbor", "neighbour"),
            "Informal lender": ("moneylender", "local lender", "chit fund"),
            "Digital / app": ("loan app", "online", "upi loan", "whatsapp loan"),
            "No borrowing disclosed": ("do not borrow", "don't borrow", "avoid loans", "use my savings"),
        }
        for label, phrases in source_map.items():
            if any(phrase in lowered for phrase in phrases):
                return label
        return "Unknown"

    def _loan_interest_label(self, loan_interest_value: str) -> str:
        mapping = {
            "fearful_or_uncertain": "Fearful / uncertain",
            "interested": "Interested",
            "not_interested": "Not interested",
            "unknown": "Unknown",
        }
        return mapping.get(loan_interest_value, loan_interest_value.replace("_", " ").title())

    def _is_loan_interest_positive(self, loan_interest_value: str) -> bool:
        return loan_interest_value == "interested"

    def _render_empty_state(self, title: str, *, sample_mode: bool, context: str) -> None:
        sample_guidance = (
            " Because sample mode is enabled, you can also intentionally load developer sample fixtures from the app layer."
            if sample_mode
            else " To inspect demo content, enable the developer-only sample-mode path instead of relying on implicit fallback data."
        )
        st.info(f"{title}. {context}{sample_guidance}")

    def _percent(self, numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "0%"
        return f"{round((numerator / denominator) * 100)}%"

    def _status_counts(
        self,
        interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
    ) -> dict[str, int]:
        counts = {status: 0 for status in STATUS_DISPLAY_ORDER}
        if status_overview.total_interviews > 0:
            for status in STATUS_DISPLAY_ORDER:
                counts[status] = status_overview.status_counts.get(status, 0)
            return counts
        for interview in interviews:
            counts[interview.status] = counts.get(interview.status, 0) + 1
        return counts

    def _truncate_text(self, value: str, *, limit: int) -> str:
        normalized = value.strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 1].rstrip()}…"

    def _inject_styles(self) -> None:
        st.markdown(
            """
            <style>
            ::selection {
                background: #1d4ed8;
                color: #f8fafc;
            }
            .stApp {
                background: #f6f8fc;
                color: #0f172a;
            }
            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }
            .pd-card {
                background: #ffffff;
                border-radius: 20px;
                padding: 1.15rem 1.25rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.16);
                margin-bottom: 1rem;
                color: #0f172a;
            }
            .pd-card div, .pd-card span, .pd-card p, .pd-card strong {
                color: inherit;
            }
            .kpi-card {
                min-height: 124px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .status-card {
                min-height: 112px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .kpi-label, .chart-subtitle, .table-title, .interview-meta, .persona-meta,
            .interview-meta-secondary, .detail-flag-label {
                color: #475569;
                font-size: 0.95rem;
                font-weight: 500;
            }
            .kpi-value, .persona-count, .status-value {
                font-size: 2rem;
                font-weight: 700;
                color: #0f172a;
            }
            .status-label, .detail-flag-value, .interview-meta-primary {
                color: #1e293b;
                font-size: 0.98rem;
                font-weight: 600;
            }
            .chart-title, .table-title {
                font-size: 1.15rem;
                font-weight: 650;
                color: #0f172a;
                margin-bottom: 0.2rem;
            }
            .persona-label, .interview-title, .interview-detail-inline-header {
                font-size: 1.1rem;
                font-weight: 650;
                color: #0f172a;
                margin-bottom: 0.4rem;
            }
            .interview-summary, .persona-quote, .quote-card {
                font-size: 1rem;
                line-height: 1.6;
                color: #1e293b;
            }
            .interview-detail-inline {
                margin: 0.5rem 0 1.25rem 0;
                border: 1px solid rgba(79, 124, 255, 0.18);
                background: linear-gradient(180deg, rgba(239, 246, 255, 0.95), rgba(255, 255, 255, 1));
            }
            .interview-inline-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 0.9rem;
            }
            .interview-inline-label {
                font-size: 0.95rem;
                font-weight: 700;
                color: #1d4ed8;
                margin-bottom: 0.45rem;
            }
            .interview-inline-list {
                margin: 0;
                padding-left: 1.2rem;
                color: #1e293b;
            }
            .interview-inline-preview {
                color: #1e293b;
                line-height: 1.6;
            }
            .pd-table {
                width: 100%;
                border-collapse: collapse;
            }
            .pd-table th, .pd-table td {
                text-align: left;
                padding: 0.75rem 0.25rem;
                border-bottom: 1px solid rgba(226, 232, 240, 0.9);
                font-size: 0.98rem;
            }
            .pd-table th {
                color: #334155;
                font-weight: 700;
            }
            .pd-table td {
                color: #0f172a;
                font-weight: 500;
            }
            .detail-flags-card {
                padding-top: 0.9rem;
                padding-bottom: 0.9rem;
            }
            .detail-flag-row {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: baseline;
                padding: 0.45rem 0;
                border-bottom: 1px solid rgba(226, 232, 240, 0.9);
            }
            .detail-flag-row:last-child {
                border-bottom: none;
            }
            .detail-flag-value {
                text-align: right;
            }
            .persona-badge {
                border-radius: 999px;
                padding: 0.9rem 1rem;
                text-align: center;
                font-size: 0.95rem;
                font-weight: 700;
                margin-top: 1.75rem;
            }
            .badge-target {
                background: rgba(20, 184, 166, 0.12);
                color: #115e59;
            }
            .badge-nontarget {
                background: rgba(244, 63, 94, 0.12);
                color: #9f1239;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: #ffffff;
                color: #0f172a;
                border-radius: 999px;
                padding: 0.6rem 1rem;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
                border: 1px solid rgba(100, 116, 139, 0.28);
                transition: color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
                font-weight: 600;
            }
            .stTabs [data-baseweb="tab"] p {
                color: inherit;
            }
            .stTabs [data-baseweb="tab"]:hover {
                color: #1d4ed8;
                border-color: rgba(29, 78, 216, 0.42);
                box-shadow: 0 10px 24px rgba(79, 124, 255, 0.14);
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background: #1d4ed8;
                color: #eff6ff;
                border-color: #1d4ed8;
                box-shadow: 0 12px 28px rgba(29, 78, 216, 0.26);
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] p {
                color: inherit;
            }
            .stTextInput label, .stMultiSelect label, .stTextArea label {
                font-size: 1rem !important;
                font-weight: 600 !important;
            }
            .stTextInput input, .stTextArea textarea, .stMultiSelect div[data-baseweb="select"] > div {
                color: #0f172a !important;
            }
            .stTextInput input::selection, .stTextArea textarea::selection {
                background: #1d4ed8;
                color: #f8fafc;
            }
            div[data-testid="stDataFrame"] * {
                color: #0f172a !important;
            }
            div[data-testid="stDataFrame"] [role="columnheader"] *,
            div[data-testid="stDataFrame"] thead * {
                color: #334155 !important;
                font-weight: 700 !important;
            }
            p, li, span, label {
                font-size: 1rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
