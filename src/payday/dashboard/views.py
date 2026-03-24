from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Callable

import streamlit as st

from payday.analysis import (
    bank_account_user_from_analysis,
    get_income_display_value,
    get_analysis_value,
    smartphone_user_from_analysis,
)
from payday.models import PipelineResult, ProcessingStatus
from payday.personas import PERSONAS
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

PERSONA_LOOKUP = {persona.id: persona.name for persona in PERSONAS.values()}
STATUS_DISPLAY_ORDER = tuple(status.value for status in ProcessingStatus)
NORMALIZED_INCOME_BUCKETS = (
    (0, 10000, "Below ₹10k"),
    (10000, 15000, "₹10k–15k"),
    (15000, 20000, "₹15k–20k"),
    (20000, None, "₹20k+"),
)
DESIGN_SYSTEM = {
    "background": "#0B0F19",
    "card": "#111827",
    "primary": "#4F46E5",
    "text_primary": "#F9FAFB",
    "text_secondary": "#9CA3AF",
    "border": "rgba(156, 163, 175, 0.18)",
    "success": "#14B8A6",
    "danger": "#F43F5E",
}
FILTER_SESSION_KEYS = {
    "income": "dashboard_filter_income",
    "borrowing": "dashboard_filter_borrowing",
    "persona": "dashboard_filter_persona",
    "digital_access": "dashboard_filter_digital_access",
    "search": "dashboard_search_query",
    "selected": "dashboard_selected_interview_id",
    "detail_open": "dashboard_detail_overlay_open",
    "transcripts": "dashboard_transcript_edits",
    "detail_message": "dashboard_detail_message",
    "delete_confirm": "dashboard_delete_confirm",
    "overlay_open": "dashboard_interview_overlay_open",
    "force_sqlite_reload": "dashboard_force_sqlite_reload",
    "toast_message": "dashboard_toast_message",
    "include_low_quality": "dashboard_include_low_quality_rows",
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
    participant_income_value: str
    income_band: str
    borrowing_source: str
    borrowing_label: str
    is_borrower: bool
    loan_interest_label: str
    interested_in_loan: bool
    smartphone_user: bool | None
    has_bank_account: bool | None
    digital_access: str
    transcript_quality: str
    extracted_json: dict[str, Any]
    evidence_quotes: tuple[str, ...]
    segmented_dialogue: tuple[dict[str, Any], ...] = ()
    analysis_version: str | None = None
    analyzed_at: str | None = None
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
        reprocess_interview: Callable[[str], DashboardInterviewRecord] | None = None,
        reanalyze_interviews: Callable[[list[str]], list[DashboardInterviewRecord]] | None = None,
        reanalyze_stale_interviews: Callable[[], dict[str, object]] | None = None,
        list_stale_interview_ids: Callable[[], list[str]] | None = None,
        reprocess_failed_or_malformed: Callable[[], dict[str, object]] | None = None,
        list_failed_or_malformed_ids: Callable[[], list[str]] | None = None,
        delete_stale_corrupted: Callable[[], dict[str, object]] | None = None,
        list_stale_corrupted_ids: Callable[[], list[str]] | None = None,
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
        self._render_pending_toast()
        filtered = self._apply_filters(interviews)
        self._render_sidebar_filters(filtered, interviews, status_overview)

        tabs = st.tabs(["Overview", "Cohorts", "Personas", "Interviews"])
        with tabs[0]:
            self._render_overview(filtered, interviews, status_overview, sample_mode=sample_mode)
        with tabs[1]:
            self._render_cohorts(filtered, all_interviews=interviews, status_overview=status_overview)
        with tabs[2]:
            self._render_personas(filtered, sample_mode=sample_mode)
        with tabs[3]:
            self._render_interviews(
                filtered,
                all_interviews=interviews,
                interview_detail_loader=interview_detail_loader,
                save_interview_edits=save_interview_edits,
                reprocess_interview=reprocess_interview,
                reanalyze_interviews=reanalyze_interviews,
                reanalyze_stale_interviews=reanalyze_stale_interviews,
                list_stale_interview_ids=list_stale_interview_ids,
                reprocess_failed_or_malformed=reprocess_failed_or_malformed,
                list_failed_or_malformed_ids=list_failed_or_malformed_ids,
                delete_stale_corrupted=delete_stale_corrupted,
                list_stale_corrupted_ids=list_stale_corrupted_ids,
                delete_interview=delete_interview,
                sample_mode=sample_mode,
            )

    def _initialize_session_state(self, interviews: list[DashboardInterview]) -> None:
        st.session_state.setdefault(FILTER_SESSION_KEYS["income"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["borrowing"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["persona"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["digital_access"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["search"], "")
        st.session_state.setdefault(FILTER_SESSION_KEYS["detail_open"], False)
        st.session_state.setdefault(FILTER_SESSION_KEYS["transcripts"], {})
        st.session_state.setdefault(FILTER_SESSION_KEYS["detail_message"], None)
        st.session_state.setdefault(FILTER_SESSION_KEYS["delete_confirm"], None)
        st.session_state.setdefault(FILTER_SESSION_KEYS["overlay_open"], False)
        st.session_state.setdefault(FILTER_SESSION_KEYS["toast_message"], None)
        st.session_state.setdefault(FILTER_SESSION_KEYS["include_low_quality"], False)

        selected_key = FILTER_SESSION_KEYS["selected"]
        if interviews and not st.session_state.get(selected_key):
            st.session_state[selected_key] = interviews[0].id
        if interviews and st.session_state.get(selected_key) not in {item.id for item in interviews}:
            st.session_state[selected_key] = interviews[0].id
        if not interviews:
            st.session_state[selected_key] = None
            st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = False

    def _render_pending_toast(self) -> None:
        toast = st.session_state.pop(FILTER_SESSION_KEYS["toast_message"], None)
        if not isinstance(toast, dict):
            return

        message = str(toast.get("message", "")).strip()
        if not message:
            return
        kind = str(toast.get("kind", "success")).lower()
        icon = "✅" if kind == "success" else "⚠️"
        st.toast(message, icon=icon)

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

    def _render_sidebar_filters(
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
        st.sidebar.markdown("### Filters")
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
        st.sidebar.checkbox(
            "Include failed/malformed in cohort metrics",
            key=FILTER_SESSION_KEYS["include_low_quality"],
            help="Off by default so borrower/persona cohorts use only good-quality transcripts.",
        )
        st.sidebar.caption(
            f"Showing {len(filtered)} of {len(all_interviews)} interviews. Durable interviews in SQLite: {status_overview.total_interviews}."
        )

    def _render_overview(
        self,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
        *,
        sample_mode: bool,
    ) -> None:
        self._render_page_intro(
            "Overview",
            "Prominent KPIs, normalized portfolio views, and evidence-first summaries for the current filter set.",
        )
        comparable, excluded_count = self._cohort_base(filtered)
        self._render_status_strip(all_interviews, status_overview)
        self._render_analysis_version_notice(filtered)
        self._render_kpis(comparable, status_overview)
        self._render_quality_denominator_note(excluded_count=excluded_count, total_count=len(filtered))

        if not comparable:
            if all_interviews:
                self._render_empty_state(
                    "No interviews match the current filters",
                    sample_mode=sample_mode,
                    context="Adjust sidebar filters to repopulate the normalized overview cards and tables.",
                )
            else:
                self._render_empty_state(
                    "No interviews yet",
                    sample_mode=sample_mode,
                    context="Upload interview audio from the sidebar to populate the dashboard overview.",
                )
            return

        chart_col, table_col = st.columns([1.35, 1], gap="large")
        with chart_col:
            self._render_chart_card(
                title="Income bands",
                subtitle="Counts are normalized to the current filtered cohort.",
                values=self._count_by(comparable, lambda item: item.income_band),
                color=DESIGN_SYSTEM["primary"],
            )
            self._render_chart_card(
                title="Borrowing behavior",
                subtitle="Borrower vs non-borrower split across the filtered interviews.",
                values=self._count_by(comparable, lambda item: item.borrowing_label),
                color="#22C55E",
            )
        with table_col:
            self._render_table_card(
                title="Persona mix",
                subtitle="Share of filtered interviews by assigned persona.",
                rows=self._build_cohort_rows(comparable, "persona_name"),
                headers=("Persona", "Count", "% of filtered"),
            )
            self._render_table_card(
                title="Digital access",
                subtitle="Targeting status based on smartphone and bank-account evidence.",
                rows=self._build_cohort_rows(comparable, "digital_access"),
                headers=("Cohort", "Count", "% of filtered"),
            )

        summary_col, quote_col = st.columns([1, 1], gap="large")
        with summary_col:
            self._render_summary_blocks(comparable)
        with quote_col:
            self._render_evidence_highlights(comparable)

    def _render_cohorts(
        self,
        filtered: list[DashboardInterview],
        *,
        all_interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
    ) -> None:
        self._render_page_intro(
            "Cohorts",
            "Filters stay in the sidebar; this tab groups the current cohort into clean comparison tables.",
        )
        comparable, excluded_count = self._cohort_base(filtered)
        self._render_analysis_version_notice(filtered)
        self._render_filter_snapshot(filtered, all_interviews, status_overview)
        self._render_quality_denominator_note(excluded_count=excluded_count, total_count=len(filtered))

        if not comparable:
            st.info("No cohorts to summarize for the current filter set.")
            return

        top_col, bottom_col = st.columns(2, gap="large")
        with top_col:
            self._render_table_card(
                title="Digital access cohorts",
                subtitle="Grouped by eligibility and access signals.",
                rows=self._build_cohort_rows(comparable, "digital_access"),
                headers=("Cohort", "Count", "% of filtered"),
            )
            self._render_table_card(
                title="Borrowing cohorts",
                subtitle="Grouped by borrowing behavior label.",
                rows=self._build_cohort_rows(comparable, "borrowing_label"),
                headers=("Cohort", "Count", "% of filtered"),
            )
        with bottom_col:
            self._render_table_card(
                title="Persona cohorts",
                subtitle="Grouped by persona output shown in the dashboard.",
                rows=self._build_cohort_rows(comparable, "persona_name"),
                headers=("Persona", "Count", "% of filtered"),
            )
            self._render_table_card(
                title="Processing cohorts",
                subtitle="Grouped by durable pipeline status.",
                rows=self._build_cohort_rows(comparable, "status"),
                headers=("Status", "Count", "% of filtered"),
            )

    def _render_personas(self, filtered: list[DashboardInterview], *, sample_mode: bool) -> None:
        self._render_page_intro(
            "Personas",
            "Card-based persona summaries with size, description, and supporting quote coverage.",
        )
        comparable, excluded_count = self._cohort_base(filtered)
        self._render_quality_denominator_note(excluded_count=excluded_count, total_count=len(filtered))
        if not comparable:
            self._render_empty_state(
                "No personas to review yet",
                sample_mode=sample_mode,
                context="Run at least one interview through transcription and analysis before persona cards can appear.",
            )
            return

        persona_counts = self._count_by(comparable, lambda item: item.persona_id)
        persona_keys = [persona_id for persona_id in PERSONAS if persona_id in persona_counts]
        if not persona_keys:
            st.info("No persona outputs are available for the current filters.")
            return

        columns = st.columns(2, gap="large")
        for index, persona_id in enumerate(persona_keys):
            persona = PERSONAS[persona_id]
            matches = [item for item in comparable if item.persona_id == persona_id]
            quote = matches[0].evidence_quotes[0] if matches and matches[0].evidence_quotes else "No direct quote captured yet."
            non_target_count = sum(1 for item in matches if item.is_non_target)
            with columns[index % len(columns)]:
                self._render_persona_card(
                    persona_name=persona.name,
                    persona_description=persona.description,
                    count=persona_counts[persona_id],
                    non_target_count=non_target_count,
                    sample_quote=quote,
                )

    def _render_interviews(
        self,
        filtered: list[DashboardInterview],
        *,
        all_interviews: list[DashboardInterview],
        interview_detail_loader: Callable[[str], DashboardInterviewRecord],
        save_interview_edits: Callable[..., DashboardInterviewRecord] | None,
        reprocess_interview: Callable[[str], DashboardInterviewRecord] | None,
        reanalyze_interviews: Callable[[list[str]], list[DashboardInterviewRecord]] | None,
        reanalyze_stale_interviews: Callable[[], dict[str, object]] | None,
        list_stale_interview_ids: Callable[[], list[str]] | None,
        reprocess_failed_or_malformed: Callable[[], dict[str, object]] | None,
        list_failed_or_malformed_ids: Callable[[], list[str]] | None,
        delete_stale_corrupted: Callable[[], dict[str, object]] | None,
        list_stale_corrupted_ids: Callable[[], list[str]] | None,
        delete_interview: Callable[[str], bool] | None,
        sample_mode: bool,
    ) -> None:
        self._render_page_intro(
            "Interviews",
            "Browse interview cards, then open a modal-style overlay for audio, transcript editing, formatted insights, and persona review.",
        )
        self._render_interviews_visual_regression_checklist()

        if not filtered:
            self._render_empty_state(
                "No interview cards yet",
                sample_mode=sample_mode,
                context="Upload audio files or intentionally load sample fixtures to create interview cards. Upload and process at least one interview, or intentionally load sample fixtures, to inspect transcript details here.",
            )
            return

        action_col1, action_col2, action_col3, action_col4 = st.columns(4, gap="small")
        stale_count = len(list_stale_interview_ids()) if list_stale_interview_ids is not None else None
        failed_malformed_count = len(list_failed_or_malformed_ids()) if list_failed_or_malformed_ids is not None else None
        stale_corrupted_count = len(list_stale_corrupted_ids()) if list_stale_corrupted_ids is not None else None

        with action_col1:
            filtered_label = f"Reanalyze filtered ({len(filtered)})"
            reanalyze_filtered_clicked = st.button(
                filtered_label,
                key="reanalyze_filtered_dashboard",
                use_container_width=True,
                disabled=reanalyze_interviews is None or not filtered,
            )
        with action_col2:
            stale_label = "Reanalyze all stale"
            if stale_count is not None:
                stale_label += f" ({stale_count})"
            reanalyze_stale_clicked = st.button(
                stale_label,
                key="reanalyze_stale_dashboard",
                use_container_width=True,
                disabled=reanalyze_stale_interviews is None or stale_count == 0,
            )
        with action_col3:
            failed_malformed_label = "Reprocess failed/malformed"
            if failed_malformed_count is not None:
                failed_malformed_label += f" ({failed_malformed_count})"
            reprocess_failed_clicked = st.button(
                failed_malformed_label,
                key="reprocess_failed_malformed_dashboard",
                use_container_width=True,
                disabled=reprocess_failed_or_malformed is None or failed_malformed_count == 0,
            )
        with action_col4:
            delete_corrupted_label = "Delete stale corrupted"
            if stale_corrupted_count is not None:
                delete_corrupted_label += f" ({stale_corrupted_count})"
            delete_stale_corrupted_clicked = st.button(
                delete_corrupted_label,
                key="delete_stale_corrupted_dashboard",
                use_container_width=True,
                disabled=delete_stale_corrupted is None or stale_corrupted_count == 0,
            )

        if reanalyze_filtered_clicked:
            try:
                reanalyze_interviews([item.id for item in filtered])
            except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "error",
                    "message": f"Filtered reanalysis failed: {exc}",
                }
            else:
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "success",
                    "message": "Reanalysis complete for filtered interviews. Structured responses, insights, and personas were refreshed in SQLite.",
                }
            st.rerun()

        if reanalyze_stale_clicked:
            try:
                stale_summary = reanalyze_stale_interviews()
            except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "error",
                    "message": f"Stale reanalysis failed: {exc}",
                }
            else:
                refreshed_count = len(stale_summary.get("reprocessed_ids", []))
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "success",
                    "message": f"Reanalysis complete for {refreshed_count} stale interviews. Structured responses, insights, and personas were refreshed in SQLite.",
                }
            st.rerun()

        if reprocess_failed_clicked:
            try:
                result = reprocess_failed_or_malformed()
            except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "error",
                    "message": f"Failed/malformed reprocess failed: {exc}",
                }
            else:
                total = int(result.get("failed_or_malformed_count", 0))
                refreshed = len(result.get("reprocessed_ids", []))
                failed = result.get("failed", {})
                kind = "success" if not failed else "error"
                message = (
                    f"Reprocessed {refreshed} of {total} failed/malformed interviews."
                    if total
                    else "No failed/malformed interviews were found."
                )
                if failed:
                    message += " Remaining rows require delete + fresh transcription."
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": kind,
                    "message": message,
                }
            st.rerun()

        if delete_stale_corrupted_clicked:
            try:
                result = delete_stale_corrupted()
            except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "error",
                    "message": f"Delete stale corrupted failed: {exc}",
                }
            else:
                deleted = len(result.get("deleted_ids", []))
                total = int(result.get("stale_corrupted_count", 0))
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "success",
                    "message": (
                        f"Deleted {deleted} stale corrupted interviews out of {total}. "
                        "Re-upload source audio to recompute from fresh transcription only."
                    ),
                }
                st.session_state[FILTER_SESSION_KEYS["force_sqlite_reload"]] = True
            st.rerun()

        if reanalyze_interviews is None or reanalyze_stale_interviews is None:
            st.caption("Reanalysis actions are unavailable because one or more backend handlers were not provided.")

        overlay_rendered = self._render_overlay_if_needed(
            filtered=filtered,
            all_interviews=all_interviews,
            interview_detail_loader=interview_detail_loader,
            save_interview_edits=save_interview_edits,
            reprocess_interview=reprocess_interview,
            delete_interview=delete_interview,
        )
        if not overlay_rendered and st.session_state.get(FILTER_SESSION_KEYS["overlay_open"]):
            self._render_inline_detail_fallback(all_interviews=all_interviews)

        self._render_interview_row_actions(filtered)
        self._render_overlay_debug_indicator()

        st.markdown("<div class='pd-grid-section'>", unsafe_allow_html=True)
        card_columns = st.columns(2, gap="large")
        for index, interview in enumerate(filtered):
            with card_columns[index % len(card_columns)]:
                self._render_interview_card(
                    interview,
                    delete_interview=delete_interview,
                    all_interviews=all_interviews,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    def _render_interviews_visual_regression_checklist(self) -> None:
        st.markdown(
            (
                "<div class='pd-card checklist-card'>"
                "<div class='section-title'>Interviews tab visual regression checklist</div>"
                "<ul class='checklist-list'>"
                "<li>Card background uses the design token surface with visible card borders.</li>"
                "<li>Interview title, summary text, and metadata labels/values maintain readable contrast.</li>"
                "<li>Primary and secondary buttons preserve contrast against the card background.</li>"
                "<li>Hover and focus states are visible for cards, tabs, and action buttons.</li>"
                "</ul>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    def _render_page_intro(self, title: str, subtitle: str) -> None:
        st.markdown(
            (
                "<div class='page-hero'>"
                f"<div class='page-title'>{html.escape(title)}</div>"
                f"<div class='page-subtitle'>{html.escape(subtitle)}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    def _render_status_strip(
        self,
        interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
    ) -> None:
        counts = self._status_counts(interviews, status_overview)
        columns = st.columns(len(STATUS_DISPLAY_ORDER), gap="medium")
        for column, status in zip(columns, STATUS_DISPLAY_ORDER, strict=False):
            with column:
                self._render_metric_card(status.title(), str(counts.get(status, 0)), card_class="status-card")

    def _render_filter_snapshot(
        self,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        status_overview: DashboardStatusOverview,
    ) -> None:
        values = [
            ("Filtered interviews", str(len(filtered))),
            ("Portfolio interviews", str(len(all_interviews))),
            ("Durable interviews", str(status_overview.total_interviews)),
            ("Search term", st.session_state[FILTER_SESSION_KEYS["search"]].strip() or "All"),
        ]
        columns = st.columns(len(values), gap="medium")
        for column, (label, value) in zip(columns, values, strict=False):
            with column:
                self._render_metric_card(label, value, card_class="compact-card")

    def _render_analysis_version_notice(self, interviews: list[DashboardInterview]) -> None:
        versions = sorted({item.analysis_version for item in interviews if item.analysis_version})
        if not versions:
            st.info("Analysis version metadata is not available yet for this filter set.")
            return
        if len(versions) == 1:
            st.caption(f"Analysis version: {versions[0]}")
            return
        st.warning(
            "Mixed analysis versions detected in this filter set. "
            "Cohort and overview aggregates may combine rows from different prompt/model revisions. "
            "Use Re-analyze actions to refresh records onto one version."
        )
        st.caption("Versions in view: " + ", ".join(versions))

    def _render_kpis(self, interviews: list[DashboardInterview], status_overview: DashboardStatusOverview) -> None:
        completed_count = status_overview.status_counts.get(ProcessingStatus.COMPLETED.value, 0)
        metrics = [
            ("Filtered interviews", str(len(interviews))),
            ("Completed (durable)", str(completed_count)),
            ("Smartphone %", self._percent(sum(1 for item in interviews if item.smartphone_user is True), len(interviews))),
            ("Borrowers %", self._percent(sum(1 for item in interviews if item.is_borrower), len(interviews))),
            ("Loan-interest %", self._percent(sum(1 for item in interviews if item.interested_in_loan), len(interviews))),
        ]
        columns = st.columns(len(metrics), gap="medium")
        for column, (label, value) in zip(columns, metrics, strict=False):
            with column:
                self._render_metric_card(label, value, card_class="kpi-card")

    def _cohort_base(self, interviews: list[DashboardInterview]) -> tuple[list[DashboardInterview], int]:
        if st.session_state.get(FILTER_SESSION_KEYS["include_low_quality"], False):
            return interviews, 0
        comparable = [item for item in interviews if self._is_transcript_quality_comparable(item)]
        return comparable, len(interviews) - len(comparable)

    def _is_transcript_quality_comparable(self, interview: DashboardInterview) -> bool:
        return interview.transcript_quality == "good"

    def _render_quality_denominator_note(self, *, excluded_count: int, total_count: int) -> None:
        if total_count == 0:
            return
        if st.session_state.get(FILTER_SESSION_KEYS["include_low_quality"], False):
            st.caption("Cohort denominator includes failed/malformed transcripts because the sidebar override is enabled.")
            return
        if excluded_count <= 0:
            st.caption("Cohort denominator includes all filtered interviews (no transcript-quality exclusions).")
            return
        st.warning(
            f"{excluded_count} interviews excluded due to transcription quality "
            f"(failed/malformed) from a filtered set of {total_count}."
        )

    def _render_metric_card(self, label: str, value: str, *, card_class: str = "") -> None:
        st.markdown(
            (
                f"<div class='pd-card metric-card {card_class}'>"
                f"<div class='metric-label'>{html.escape(label)}</div>"
                f"<div class='metric-value'>{html.escape(value)}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    def _render_chart_card(
        self,
        *,
        title: str,
        subtitle: str,
        values: dict[str, Any],
        color: str,
    ) -> None:
        chart_data = self._normalized_chart_rows(values)
        with st.container():
            st.markdown(
                (
                    "<div class='pd-card section-card'>"
                    f"<div class='section-title'>{html.escape(title)}</div>"
                    f"<div class='section-subtitle'>{html.escape(subtitle)}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        if not chart_data or not self._chart_rows_are_flat(chart_data):
            self._emit_chart_diagnostics(
                title=title,
                chart_rows=chart_data,
                note="Fallback triggered because normalized chart rows were empty or invalid.",
            )
            self._render_chart_fallback_table(
                title=title,
                subtitle=subtitle,
                chart_rows=chart_data,
                reason="No valid chart rows were available, so a normalized table is shown instead.",
            )
            return

        if self._prefer_income_table_fallback(title=title, values=values, chart_rows=chart_data):
            self._emit_chart_diagnostics(
                title=title,
                chart_rows=chart_data,
                note="Income distribution shown as table due to sparse/non-numeric chart inputs.",
            )
            self._render_chart_fallback_table(
                title=title,
                subtitle=subtitle,
                chart_rows=chart_data,
                reason="Income values were sparse or non-numeric, so a tabular distribution is shown for clarity.",
            )
            return

        self._emit_chart_diagnostics(
            title=title,
            chart_rows=chart_data,
            note="Rendering Vega chart with normalized rows.",
        )

        try:
            st.vega_lite_chart(
                chart_data,
                {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "background": "transparent",
                    "data": {"values": chart_data},
                    "mark": {"type": "bar", "cornerRadiusTopLeft": 10, "cornerRadiusTopRight": 10, "color": color},
                    "encoding": {
                        "x": {
                            "field": "category",
                            "type": "nominal",
                            "sort": "-y",
                            "axis": {
                                "title": None,
                                "labelAngle": 0,
                                "labelColor": DESIGN_SYSTEM["text_secondary"],
                                "domainColor": DESIGN_SYSTEM["border"],
                                "tickColor": DESIGN_SYSTEM["border"],
                            },
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "axis": {
                                "title": None,
                                "labelColor": DESIGN_SYSTEM["text_secondary"],
                                "gridColor": "rgba(156, 163, 175, 0.14)",
                                "domainColor": DESIGN_SYSTEM["border"],
                                "tickColor": DESIGN_SYSTEM["border"],
                            },
                        },
                        "tooltip": [
                            {"field": "category", "type": "nominal", "title": "Category"},
                            {"field": "count", "type": "quantitative", "title": "Interviews"},
                            {"field": "share", "type": "quantitative", "title": "% of filtered"},
                        ],
                    },
                    "view": {"stroke": None},
                    "height": 280,
                    "config": {
                        "style": {"cell": {"stroke": "transparent"}},
                        "legend": {"labelColor": DESIGN_SYSTEM["text_secondary"], "titleColor": DESIGN_SYSTEM["text_primary"]},
                    },
                },
                use_container_width=True,
            )
        except Exception as exc:
            self._emit_chart_diagnostics(
                title=title,
                chart_rows=chart_data,
                note=f"Vega rendering failed ({type(exc).__name__}); using fallback table.",
            )
            self._render_chart_fallback_table(
                title=title,
                subtitle=subtitle,
                chart_rows=chart_data,
                reason="Chart rendering failed at runtime, so a normalized table is shown instead.",
            )

    def _normalized_chart_rows(self, values: dict[str, Any]) -> list[dict[str, int | float | str]]:
        normalized_counts: dict[str, int] = {}
        for raw_category, raw_count in values.items():
            count = self._safe_chart_count(raw_count)
            if count is None or count <= 0:
                continue
            category = self._normalize_chart_category(raw_category)
            normalized_counts[category] = normalized_counts.get(category, 0) + count
        total = sum(normalized_counts.values())
        if total <= 0:
            return []
        return [
            {
                "category": category,
                "count": count,
                "share": round((count / total) * 100, 1),
            }
            for category, count in normalized_counts.items()
        ]

    def _normalize_chart_category(self, raw_category: Any) -> str:
        unknown_label = "Unknown"
        if raw_category is None:
            return unknown_label
        if isinstance(raw_category, bytes):
            label = raw_category.decode("utf-8", errors="ignore").strip()
        else:
            label = str(raw_category).strip()
        label = re.sub(r"\s+", " ", label)
        label = label.strip(" -_:/\\|.,;")
        if not label:
            return unknown_label
        lowered = label.casefold()
        if lowered in {
            "unknown",
            "unk",
            "n/a",
            "na",
            "none",
            "null",
            "nil",
            "nan",
            "nill",
            "unspecified",
            "not specified",
            "not available",
            "missing",
            "undefined",
            "?",
            "-",
            "--",
            "{}",
            "[]",
            "()",
        }:
            return unknown_label
        if re.fullmatch(r"[?_\-./\s]+", label):
            return unknown_label
        label = re.sub(r"\s*/\s*", " / ", label)
        label = re.sub(r"\s+", " ", label).strip()
        return label

    def _safe_chart_count(self, raw_count: Any) -> int | None:
        if isinstance(raw_count, bool):
            return None
        if isinstance(raw_count, (int, float)):
            return int(raw_count)
        if isinstance(raw_count, str):
            parsed = raw_count.strip().replace(",", "")
            if not parsed:
                return None
            try:
                return int(float(parsed))
            except ValueError:
                return None
        return None

    def _chart_rows_are_flat(self, rows: list[dict[str, Any]]) -> bool:
        if not isinstance(rows, list):
            return False
        for row in rows:
            if not isinstance(row, dict):
                return False
            if "category" not in row or "count" not in row:
                return False
            if not isinstance(row["category"], str) or not row["category"].strip():
                return False
            if not isinstance(row["count"], (int, float)):
                return False
            if "share" in row and not isinstance(row["share"], (int, float)):
                return False
            if any(isinstance(value, (list, dict, tuple, set)) for value in row.values()):
                return False
        return True

    def _render_chart_fallback_table(
        self,
        *,
        title: str,
        subtitle: str,
        chart_rows: list[dict[str, Any]],
        reason: str,
    ) -> None:
        st.caption(reason)
        table_rows: list[tuple[str, int, str]] = []
        for row in chart_rows:
            if not isinstance(row, dict):
                continue
            category = self._normalize_chart_category(row.get("category"))
            count = self._safe_chart_count(row.get("count")) or 0
            raw_share = row.get("share", 0.0)
            share_value = 0.0
            if isinstance(raw_share, (int, float)):
                share_value = float(raw_share)
            elif isinstance(raw_share, str):
                try:
                    share_value = float(raw_share.strip().replace("%", ""))
                except ValueError:
                    share_value = 0.0
            table_rows.append((category, count, f"{share_value:.1f}%"))
        if not table_rows:
            table_rows = [("Unknown", 0, "0.0%")]
        self._render_table_card(
            title=f"{title} (Table fallback)",
            subtitle=subtitle,
            rows=table_rows,
            headers=("Category", "Count", "% of filtered"),
        )

    def _prefer_income_table_fallback(
        self,
        *,
        title: str,
        values: dict[str, Any],
        chart_rows: list[dict[str, Any]],
    ) -> bool:
        if "income" not in title.casefold():
            return False
        has_non_numeric = any(self._safe_chart_count(raw_count) is None for raw_count in values.values())
        non_zero_rows = [row for row in chart_rows if self._safe_chart_count(row.get("count"))]
        total_count = sum(self._safe_chart_count(row.get("count")) or 0 for row in chart_rows)
        is_sparse = len(non_zero_rows) <= 1 or total_count < 4
        return has_non_numeric or is_sparse

    def _emit_chart_diagnostics(self, *, title: str, chart_rows: list[dict[str, Any]], note: str) -> None:
        if not self._is_dev_mode():
            return
        compact_rows = chart_rows[:3]
        st.caption(
            f"[dev] {title}: {note} First normalized rows: "
            + json.dumps(compact_rows, ensure_ascii=False, sort_keys=True)
        )

    def _is_dev_mode(self) -> bool:
        raw_value = st.session_state.get("payday_dev_mode")
        if raw_value is None:
            raw_value = os.getenv("PAYDAY_DEV_MODE")
        if raw_value is None:
            return False
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            return raw_value.strip().casefold() in {"1", "true", "yes", "on"}
        return bool(raw_value)

    def _render_table_card(
        self,
        *,
        title: str,
        subtitle: str,
        rows: list[tuple[str, int, str]],
        headers: tuple[str, str, str],
    ) -> None:
        table_rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{count}</td><td>{html.escape(share)}</td></tr>"
            for label, count, share in rows
        )
        st.markdown(
            f"""
            <div class='pd-card section-card table-wrapper'>
                <div class='section-title'>{html.escape(title)}</div>
                <div class='section-subtitle'>{html.escape(subtitle)}</div>
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

    def _render_summary_blocks(self, interviews: list[DashboardInterview]) -> None:
        incomes = self._numeric_income_values(interviews)
        blocks = [
            (
                "Portfolio summary",
                [
                    ("Median direct income", self._format_currency(int(median(incomes))) if incomes else "No direct income captured"),
                    ("Average direct income", self._format_currency(int(mean(incomes))) if incomes else "No direct income captured"),
                    ("Non-target interviews", str(sum(1 for item in interviews if item.is_non_target))),
                ],
            ),
            (
                "Review notes",
                [
                    ("Transcripts with dialogue turns", str(sum(1 for item in interviews if self._segmented_dialogue_for(item)))),
                    ("Quotes captured", str(sum(len(item.evidence_quotes) for item in interviews))),
                    ("Current filter size", str(len(interviews))),
                ],
            ),
        ]
        for title, items in blocks:
            rows = "".join(
                (
                    "<div class='summary-row'>"
                    f"<span class='summary-label'>{html.escape(label)}</span>"
                    f"<span class='summary-value'>{html.escape(value)}</span>"
                    "</div>"
                )
                for label, value in items
            )
            st.markdown(
                f"""
                <div class='pd-card section-card'>
                    <div class='section-title'>{html.escape(title)}</div>
                    <div class='summary-block'>{rows}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_evidence_highlights(self, interviews: list[DashboardInterview]) -> None:
        quotes = [quote for interview in interviews for quote in interview.evidence_quotes][:4]
        if not quotes:
            st.info("No direct quotes are available for the filtered interviews yet.")
            return

        st.markdown(
            "<div class='pd-card section-card'><div class='section-title'>Evidence highlights</div><div class='section-subtitle'>Direct quotes shown without exposing raw structured JSON in the main UI.</div></div>",
            unsafe_allow_html=True,
        )
        for quote in quotes:
            st.markdown(
                f"<div class='pd-card quote-card'>“{html.escape(quote)}”</div>",
                unsafe_allow_html=True,
            )

    def _render_persona_card(
        self,
        *,
        persona_name: str,
        persona_description: str,
        count: int,
        non_target_count: int,
        sample_quote: str,
    ) -> None:
        st.markdown(
            f"""
            <div class='pd-card persona-card'>
                <div class='persona-header'>
                    <div>
                        <div class='persona-name'>{html.escape(persona_name)}</div>
                        <div class='persona-description'>{html.escape(persona_description)}</div>
                    </div>
                    <div class='persona-size'>{count}</div>
                </div>
                <div class='persona-meta'>Non-target in cohort: {non_target_count}</div>
                <div class='persona-quote'>“{html.escape(sample_quote)}”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_interview_card(
        self,
        interview: DashboardInterview,
        *,
        delete_interview: Callable[[str], bool] | None,
        all_interviews: list[DashboardInterview],
    ) -> None:
        badge_class = "badge-nontarget" if interview.is_non_target else "badge-target"
        persona_label = html.escape(interview.persona_name)
        summary = html.escape(self._truncate_text(interview.summary, limit=160) or "No summary captured yet.")
        st.markdown(
            f"""
            <div class='pd-card interview-card'>
                <div class='interview-card-top'>
                    <div class='interview-title'>{html.escape(interview.filename)}</div>
                    <div class='persona-badge {badge_class}'>{persona_label}</div>
                </div>
                <div class='interview-tags'>
                    <span class='meta-label'>Transcript quality:</span>
                    <span class='meta-value'>{html.escape(interview.transcript_quality.title())}</span>
                    <span class='meta-label' style='margin-left:8px;'>Analysis version:</span>
                    <span class='meta-value'>{html.escape(interview.analysis_version or "Unknown")}</span>
                </div>
                <div class='interview-summary'>{summary}</div>
                <div class='interview-meta-grid'>
                    <div><span class='meta-label'>Borrowing</span><span class='meta-value'>{html.escape(interview.borrowing_label)}</span></div>
                    <div><span class='meta-label'>Income</span><span class='meta-value'>{html.escape(interview.income_band)}</span></div>
                    <div><span class='meta-label'>Access</span><span class='meta-value'>{html.escape(interview.digital_access)}</span></div>
                    <div><span class='meta-label'>Status</span><span class='meta-value'>{html.escape(interview.status.title())}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        open_col, delete_col = st.columns(2, gap="small")
        with open_col:
            if st.button("Open", key=f"open_interview_{interview.id}", use_container_width=True, type="primary"):
                st.session_state[FILTER_SESSION_KEYS["selected"]] = interview.id
                st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = True
                st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                st.rerun()
        with delete_col:
            delete_disabled = delete_interview is None
            if st.button(
                "Delete",
                key=f"delete_interview_card_{interview.id}",
                use_container_width=True,
                disabled=delete_disabled,
            ):
                st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = interview.id
                st.rerun()

        if delete_disabled:
            st.caption("Delete is unavailable because no backend delete handler was provided.")
        elif st.session_state.get(FILTER_SESSION_KEYS["delete_confirm"]) == interview.id:
            st.warning("Confirm permanent delete for this interview and linked records.")
            confirm_col, cancel_col = st.columns(2, gap="small")
            with confirm_col:
                if st.button(
                    "Confirm",
                    key=f"confirm_delete_card_{interview.id}",
                    use_container_width=True,
                ):
                    self._delete_interview_and_refresh(
                        interview=interview,
                        all_interviews=all_interviews,
                        delete_interview=delete_interview,
                    )
            with cancel_col:
                if st.button(
                    "Cancel",
                    key=f"cancel_delete_card_{interview.id}",
                    use_container_width=True,
                ):
                    st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                    st.rerun()

    def _render_overlay_if_needed(
        self,
        *,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        interview_detail_loader: Callable[[str], DashboardInterviewRecord],
        save_interview_edits: Callable[..., DashboardInterviewRecord] | None,
        reprocess_interview: Callable[[str], DashboardInterviewRecord] | None,
        delete_interview: Callable[[str], bool] | None,
    ) -> bool:
        if not st.session_state.get(FILTER_SESSION_KEYS["overlay_open"]):
            return False

        selected_id = st.session_state.get(FILTER_SESSION_KEYS["selected"])
        if not selected_id:
            st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = False
            return False

        selected = next((item for item in all_interviews if item.id == selected_id), None)
        if selected is None:
            st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = False
            return False

        selected = self._merge_detail_record(selected, interview_detail_loader)
        transcript_edits: dict[str, str] = st.session_state[FILTER_SESSION_KEYS["transcripts"]]
        transcript_edits.setdefault(selected.id, selected.transcript)
        current_json = json.dumps(selected.extracted_json, indent=2, ensure_ascii=False)

        message = st.session_state.get(FILTER_SESSION_KEYS["detail_message"])
        if isinstance(message, dict):
            if message.get("kind") == "success":
                st.success(str(message.get("message", "")))
            elif message.get("kind") == "error":
                st.error(str(message.get("message", "")))
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = None

        st.markdown("<div class='overlay-backdrop'></div>", unsafe_allow_html=True)
        st.markdown("<div class='pd-card overlay-card'>", unsafe_allow_html=True)

        header_col, close_col = st.columns([5, 1], gap="medium")
        with header_col:
            st.markdown(
                f"<div class='overlay-title'>{html.escape(selected.filename)}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Interview ID: {selected.id} · Created: {selected.created_at} · Status: {selected.status} · Stage: {selected.current_stage}"
            )
        with close_col:
            badge_class = "badge-nontarget" if selected.is_non_target else "badge-target"
            st.markdown(
                f"<div class='persona-badge {badge_class} overlay-badge'>{html.escape(selected.persona_name)}</div>",
                unsafe_allow_html=True,
            )
            if st.button("Close overlay", key=f"close_overlay_{selected.id}", use_container_width=True):
                st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = False
                st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                st.rerun()

        top_col, side_col = st.columns([1.3, 1], gap="large")
        with top_col:
            st.markdown("#### Audio")
            if selected.audio_bytes:
                st.audio(selected.audio_bytes, format=selected.audio_format)
            else:
                st.info("Audio bytes are unavailable after restart, but the durable interview record remains available.")
                audio_url = selected.extracted_json.get("audio_url")
                if audio_url:
                    st.caption(f"Stored audio path: {audio_url}")

            st.markdown("#### Editable transcript")
            updated_transcript = st.text_area(
                "Transcript",
                key=f"overlay_transcript_{selected.id}",
                value=transcript_edits[selected.id],
                height=260,
                label_visibility="collapsed",
            )
            transcript_edits[selected.id] = updated_transcript
        with side_col:
            self._render_formatted_insights(selected)

        dialogue_turns = self._segmented_dialogue_for(selected)
        if dialogue_turns:
            st.markdown("#### Transcript turns")
            self._render_segmented_dialogue(dialogue_turns)

        transcript_changed = updated_transcript != selected.transcript
        action_col, delete_col = st.columns([2, 1], gap="medium")
        with action_col:
            if st.button(
                "Re-analyze selected interview",
                key=f"reanalyze_selected_overlay_{selected.id}",
                use_container_width=False,
                disabled=reprocess_interview is None,
            ):
                try:
                    reprocess_interview(selected.id)
                except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                    st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                        "kind": "error",
                        "message": f"Re-analysis failed: {exc}",
                    }
                else:
                    transcript_edits.pop(selected.id, None)
                    st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                        "kind": "success",
                        "message": "Selected interview re-analyzed. Structured responses and insights were overwritten in SQLite.",
                    }
                st.rerun()
            if st.button(
                "Reprocess interview",
                key=f"reprocess_overlay_{selected.id}",
                use_container_width=False,
                disabled=reprocess_interview is None,
            ):
                try:
                    reprocess_interview(selected.id)
                except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
                    st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                        "kind": "error",
                        "message": f"Re-analysis failed: {exc}",
                    }
                else:
                    transcript_edits.pop(selected.id, None)
                    st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                        "kind": "success",
                        "message": "Interview reprocessed and refreshed in SQLite.",
                    }
                st.rerun()

            save_disabled = save_interview_edits is None or not transcript_changed
            transcript_col, json_col, save_all_col = st.columns(3, gap="small")
            with transcript_col:
                if st.button(
                    "Save transcript",
                    key=f"save_transcript_overlay_{selected.id}",
                    type="primary",
                    use_container_width=True,
                    disabled=save_disabled,
                ):
                    self._save_interview_detail_edits(
                        interview_id=selected.id,
                        transcript=updated_transcript,
                        extracted_json=current_json,
                        transcript_changed=True,
                        structured_json_changed=False,
                        save_interview_edits=save_interview_edits,
                        transcript_edits=transcript_edits,
                        success_message="Transcript saved. Downstream analysis, persona derivation, and dashboard views were refreshed from SQLite.",
                    )
            with json_col:
                st.button(
                    "Save structured JSON",
                    key=f"save_json_overlay_{selected.id}",
                    use_container_width=True,
                    disabled=True,
                )
            with save_all_col:
                st.button(
                    "Save all edits",
                    key=f"save_all_overlay_{selected.id}",
                    use_container_width=True,
                    disabled=True,
                )
            if save_interview_edits is None:
                st.caption("Transcript save is unavailable because no backend save handler was provided.")
            if reprocess_interview is None:
                st.caption("Re-analyze selected interview is unavailable because no backend handler was provided.")
            elif not transcript_changed:
                st.caption("No transcript edits to save.")
            else:
                st.caption("Saving reruns downstream analysis and persona derivation without exposing raw JSON in the UI.")
            st.caption("Structured JSON editing was intentionally removed from the main UI. The buttons remain visible for workflow continuity.")
        with delete_col:
            delete_disabled = delete_interview is None
            if st.button(
                "Delete interview",
                key=f"delete_interview_overlay_{selected.id}",
                use_container_width=True,
                disabled=delete_disabled,
            ):
                st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = selected.id
                st.rerun()
            if delete_disabled:
                st.caption("Delete is unavailable because no backend delete handler was provided.")

        if st.session_state.get(FILTER_SESSION_KEYS["delete_confirm"]) == selected.id and delete_interview is not None:
            st.warning(
                "Delete this interview from durable storage? This removes linked structured responses and insights, and deletes the stored audio asset when live storage is enabled."
            )
            confirm_col, cancel_col = st.columns(2, gap="medium")
            with confirm_col:
                if st.button(
                    "Confirm delete",
                    key=f"confirm_delete_interview_{selected.id}",
                    use_container_width=True,
                ):
                    self._delete_interview_and_refresh(
                        interview=selected,
                        all_interviews=all_interviews,
                        delete_interview=delete_interview,
                    )
            with cancel_col:
                if st.button(
                    "Cancel delete",
                    key=f"cancel_delete_interview_{selected.id}",
                    use_container_width=True,
                ):
                    st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return True

    def _render_inline_detail_fallback(self, *, all_interviews: list[DashboardInterview]) -> None:
        selected_id = st.session_state.get(FILTER_SESSION_KEYS["selected"])
        selected = next((item for item in all_interviews if item.id == selected_id), None)
        if selected is None:
            st.warning("Detail overlay could not be rendered. Select an interview again.")
            return
        st.warning("Overlay was requested but could not render in this session. Showing inline detail fallback.")
        with st.expander(f"Inline detail fallback · {selected.filename}", expanded=True):
            st.caption(f"Interview ID: {selected.id} · Status: {selected.status} · Stage: {selected.current_stage}")
            st.markdown("**Summary**")
            st.write(selected.summary or "No summary captured yet.")
            st.markdown("**Transcript (read-only)**")
            st.text_area(
                "Fallback transcript",
                value=selected.transcript,
                height=220,
                key=f"fallback_transcript_{selected.id}",
                disabled=True,
                label_visibility="collapsed",
            )
            self._render_formatted_insights(selected)
            if st.button("Close fallback detail", key=f"close_fallback_{selected.id}", use_container_width=False):
                st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = False
                st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
                st.rerun()

    def _render_interview_row_actions(self, interviews: list[DashboardInterview]) -> None:
        with st.expander("Quick row actions", expanded=False):
            for interview in interviews:
                row_cols = st.columns([2.8, 1.4, 1.2, 1], gap="small")
                row_cols[0].markdown(f"**{html.escape(interview.filename)}**")
                row_cols[1].caption(f"ID: {interview.id}")
                row_cols[2].caption(interview.persona_name)
                if row_cols[3].button("Open", key=f"open_row_{interview.id}", use_container_width=True):
                    self._open_interview_overlay(interview.id)

    def _render_overlay_debug_indicator(self) -> None:
        if not self._is_dev_mode():
            return
        selected_id = st.session_state.get(FILTER_SESSION_KEYS["selected"])
        overlay_open = bool(st.session_state.get(FILTER_SESSION_KEYS["overlay_open"]))
        st.caption(f"[dev] overlay_debug selected_id={selected_id} overlay_open={overlay_open}")

    def _render_pending_toast(self) -> None:
        toast_state = st.session_state.get(FILTER_SESSION_KEYS["toast_message"])
        if not isinstance(toast_state, dict):
            return
        message = str(toast_state.get("message", "")).strip()
        if not message:
            st.session_state[FILTER_SESSION_KEYS["toast_message"]] = None
            return
        icon = "✅" if toast_state.get("kind") == "success" else "ℹ️"
        st.toast(message, icon=icon)
        st.session_state[FILTER_SESSION_KEYS["toast_message"]] = None

    def _open_interview_overlay(self, interview_id: str) -> None:
        st.session_state[FILTER_SESSION_KEYS["selected"]] = interview_id
        st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = True
        st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
        toast_payload = {
            "kind": "success",
            "message": f"Opened interview {interview_id}.",
        }
        st.session_state[FILTER_SESSION_KEYS["toast_message"]] = toast_payload
        st.toast(toast_payload["message"], icon="✅")
        st.rerun()

    def _is_dev_mode(self) -> bool:
        dev_flags = (
            os.getenv("PAYDAY_DEV_MODE", ""),
            os.getenv("STREAMLIT_ENV", ""),
            os.getenv("ENV", ""),
        )
        return any(flag.strip().lower() in {"1", "true", "yes", "dev", "development"} for flag in dev_flags)

    def _render_formatted_insights(self, interview: DashboardInterview) -> None:
        sections = [
            (
                "Persona",
                [
                    ("Persona", interview.persona_name),
                    ("Digital access", interview.digital_access),
                    ("Borrowing", interview.borrowing_label),
                    ("Loan interest", interview.loan_interest_label),
                ],
            ),
            (
                "Structured evidence",
                [
                    ("Borrowing source", interview.borrowing_source),
                    ("Income band", interview.income_band),
                    ("Smartphone user", self._bool_label(interview.smartphone_user)),
                    ("Bank account", self._bool_label(interview.has_bank_account)),
                    ("Analysis version", interview.analysis_version or "Unknown"),
                    ("Analyzed at", interview.analyzed_at or "Unknown"),
                ],
            ),
        ]
        for title, items in sections:
            rows = "".join(
                (
                    "<div class='insight-row'>"
                    f"<span class='insight-label'>{html.escape(label)}</span>"
                    f"<span class='insight-value'>{html.escape(value)}</span>"
                    "</div>"
                )
                for label, value in items
            )
            st.markdown(
                f"""
                <div class='pd-card section-card insight-card'>
                    <div class='section-title'>{html.escape(title)}</div>
                    <div class='insight-list'>{rows}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        quotes = interview.evidence_quotes[:3]
        quote_markup = "".join(f"<li>{html.escape(quote)}</li>" for quote in quotes) or "<li>No direct quote captured yet.</li>"
        st.markdown(
            f"""
            <div class='pd-card section-card insight-card'>
                <div class='section-title'>Evidence quotes</div>
                <ul class='quote-list'>{quote_markup}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_segmented_dialogue(self, turns: tuple[dict[str, Any], ...]) -> None:
        for turn in turns:
            speaker_label = html.escape(str(turn.get("speaker_label", "unknown")).replace("_", " ").title())
            utterance_text = html.escape(str(turn.get("utterance_text", "")).strip())
            speaker_confidence = html.escape(str(turn.get("speaker_confidence", "low")).strip().title())
            speaker_uncertainty = html.escape(str(turn.get("speaker_uncertainty", "")).strip())
            if not utterance_text:
                continue
            st.markdown(
                f"""
                <div class='pd-card dialogue-turn'>
                    <div class='dialogue-speaker'>{speaker_label}</div>
                    <div class='dialogue-utterance'>{utterance_text}</div>
                    <div class='dialogue-meta'>Speaker confidence: {speaker_confidence}</div>
                    {f"<div class='dialogue-uncertainty'>{speaker_uncertainty}</div>" if speaker_uncertainty else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                "kind": "success",
                "message": success_message,
            }
            st.rerun()

    def _segmented_dialogue_for(self, interview: DashboardInterview) -> tuple[dict[str, Any], ...]:
        if interview.segmented_dialogue:
            return interview.segmented_dialogue
        fallback = interview.extracted_json.get("segmented_dialogue")
        if isinstance(fallback, list):
            return tuple(item for item in fallback if isinstance(item, dict))
        return ()

    def _next_selected_id(self, interviews: list[DashboardInterview], *, deleted_id: str) -> str | None:
        remaining_ids = [item.id for item in interviews if item.id != deleted_id]
        if not remaining_ids:
            return None
        return remaining_ids[0]

    def _delete_interview_and_refresh(
        self,
        *,
        interview: DashboardInterview,
        all_interviews: list[DashboardInterview],
        delete_interview: Callable[[str], bool] | None,
    ) -> None:
        if delete_interview is None:
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                "kind": "error",
                "message": "Delete is unavailable because no backend delete handler was provided.",
            }
            st.session_state[FILTER_SESSION_KEYS["toast_message"]] = {
                "kind": "error",
                "message": f"Delete unavailable for {interview.filename} ({interview.id}).",
            }
            st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
            st.rerun()

        transcript_edits: dict[str, str] = st.session_state[FILTER_SESSION_KEYS["transcripts"]]
        try:
            deleted = delete_interview(interview.id)
        except Exception as exc:  # pragma: no cover - exercised through Streamlit interaction
            st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                "kind": "error",
                "message": f"Delete failed for {interview.filename} ({interview.id}): {exc}",
            }
            st.session_state[FILTER_SESSION_KEYS["toast_message"]] = {
                "kind": "error",
                "message": f"Delete failed for {interview.filename} ({interview.id}): {exc}",
            }
        else:
            if deleted:
                transcript_edits.pop(interview.id, None)
                st.session_state[FILTER_SESSION_KEYS["selected"]] = self._next_selected_id(
                    all_interviews,
                    deleted_id=interview.id,
                )
                st.session_state[FILTER_SESSION_KEYS["overlay_open"]] = bool(
                    st.session_state[FILTER_SESSION_KEYS["selected"]]
                )
                st.session_state[FILTER_SESSION_KEYS["force_sqlite_reload"]] = True
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "success",
                    "message": f"Deleted {interview.filename} ({interview.id}). Interview lists, overview KPIs, and cohort/persona tables were reloaded from SQLite.",
                }
                st.session_state[FILTER_SESSION_KEYS["toast_message"]] = {
                    "kind": "success",
                    "message": f"Deleted {interview.filename} ({interview.id}) and refreshed dashboard data.",
                }
            else:
                st.session_state[FILTER_SESSION_KEYS["detail_message"]] = {
                    "kind": "error",
                    "message": f"Interview {interview.id} could not be deleted because it no longer exists.",
                }
                st.session_state[FILTER_SESSION_KEYS["toast_message"]] = {
                    "kind": "error",
                    "message": f"Could not delete {interview.filename} ({interview.id}) because it was already removed.",
                }

        st.session_state[FILTER_SESSION_KEYS["delete_confirm"]] = None
        st.rerun()

    def _build_cohort_rows(self, interviews: list[DashboardInterview], attribute: str) -> list[tuple[str, int, str]]:
        counts = self._count_by(interviews, lambda item: getattr(item, attribute))
        total = len(interviews)
        return [(label, count, self._percent(count, total)) for label, count in counts.items()]

    def _normalized_participant_income_values(self, interviews: list[DashboardInterview]) -> list[int]:
        values: list[int] = []
        for interview in interviews:
            income_value = self._parse_income_value(interview.participant_income_value)
            if income_value is not None:
                values.append(income_value)
        return values

    def _numeric_income_values(self, interviews: list[DashboardInterview]) -> list[int]:
        return self._normalized_participant_income_values(interviews)

    def _parse_income_value(self, income_value: str) -> int | None:
        normalized = income_value.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if "unknown" in lowered:
            return None
        if any(term in lowered for term in ("below", "under", "less than")):
            match = re.search(r"(\d[\d,]*)(?:\s*([kK]))?", normalized)
            if match is None:
                return None
            return self._amount_from_match(match) - 1
        range_match = re.search(
            r"(\d[\d,]*)(?:\s*([kK]))?\s*(?:-|–|to)\s*(\d[\d,]*)(?:\s*([kK]))?",
            normalized,
        )
        if range_match is not None:
            lower_amount = self._amount_from_match(range_match, group_index=1, suffix_index=2)
            upper_amount = self._amount_from_match(range_match, group_index=3, suffix_index=4)
            return round((lower_amount + upper_amount) / 2)

        match = re.search(r"(\d[\d,]*)(?:\s*([kK]))?", normalized)
        if match is None:
            return None

        return self._amount_from_match(match)

    def _amount_from_match(
        self,
        match: re.Match[str],
        *,
        group_index: int = 1,
        suffix_index: int = 2,
    ) -> int:
        amount = int(match.group(group_index).replace(",", ""))
        if match.group(suffix_index):
            amount *= 1000
        return amount

    def _normalized_income_bucket(self, income_value: str) -> str | None:
        parsed_value = self._parse_income_value(income_value)
        if parsed_value is None:
            return None
        for lower_bound, upper_bound, label in NORMALIZED_INCOME_BUCKETS:
            if parsed_value < lower_bound:
                continue
            if upper_bound is None or parsed_value < upper_bound:
                return label
        return None

    def _income_bucket_rows(self, interviews: list[DashboardInterview]) -> list[tuple[str, int, str]]:
        counts = {label: 0 for _, _, label in NORMALIZED_INCOME_BUCKETS}
        total_normalized = 0
        for interview in interviews:
            bucket = self._normalized_income_bucket(interview.participant_income_value)
            if bucket is None:
                continue
            counts[bucket] += 1
            total_normalized += 1
        return [
            (label, count, self._percent(count, total_normalized))
            for _, _, label in NORMALIZED_INCOME_BUCKETS
            if (count := counts[label]) > 0
        ]

    def _normalized_borrowing_rows(self, interviews: list[DashboardInterview]) -> list[tuple[str, int, str]]:
        comparable = [item for item in interviews if item.borrowing_source != "Unknown"]
        counts = self._count_by(comparable, lambda item: item.borrowing_source)
        total = len(comparable)
        return [(label, count, self._percent(count, total)) for label, count in counts.items()]

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
        segmented_dialogue = tuple(structured.get("segmented_dialogue", [])) if isinstance(structured.get("segmented_dialogue"), list) else ()
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
            participant_income_value=get_analysis_value(structured, "participant_personal_monthly_income"),
            income_band=self._preferred_income_band(structured),
            borrowing_source=self._borrowing_source_label(transcript, borrowing_value),
            borrowing_label="Borrower" if self._is_borrower_value(borrowing_value, transcript) else "Non-borrower",
            is_borrower=self._is_borrower_value(borrowing_value, transcript),
            loan_interest_label=self._loan_interest_label(loan_interest_value),
            interested_in_loan=self._is_loan_interest_positive(loan_interest_value),
            smartphone_user=smartphone_user,
            has_bank_account=has_bank_account,
            digital_access=digital_access,
            transcript_quality=self._infer_transcript_quality(
                status=result.status.value,
                transcript=transcript,
                last_error=result.last_error,
            ),
            extracted_json=structured or self._empty_extracted_json(summary="Analysis pending."),
            evidence_quotes=evidence_quotes,
            segmented_dialogue=segmented_dialogue,
            analysis_version=result.analysis.metrics.get("analysis_version") if result.analysis else None,
            analyzed_at=result.analysis.metrics.get("analyzed_at") if result.analysis else None,
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
        extracted_json = {
            "audio_url": record.audio_url,
            "runtime": {
                "status": record.status,
                "current_stage": record.latest_stage,
                "last_error": record.last_error,
            },
            "participant_profile": {
                "smartphone_user": {"value": record.smartphone_user},
                "has_bank_account": {"value": record.has_bank_account},
            },
            "per_household_earnings": {"value": record.per_household_earnings},
            "participant_personal_monthly_income": {"value": record.participant_personal_monthly_income},
            "total_household_monthly_income": {"value": record.total_household_monthly_income},
            "borrowing_history": {"value": record.borrowing_history},
            "repayment_preference": {"value": record.repayment_preference},
            "loan_interest": {"value": record.loan_interest},
            "segmented_dialogue": record.segmented_dialogue,
            "insight": {
                "summary": record.summary,
                "persona": record.persona,
                "confidence_score": record.confidence_score,
                "key_quotes": record.key_quotes,
                "analysis_version": record.analysis_version,
                "analyzed_at": record.analyzed_at,
            },
        }
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
            participant_income_value=record.participant_personal_monthly_income or "Unknown",
            income_band=self._record_income_band(record),
            borrowing_source=self._borrowing_source_label(transcript, borrowing_value),
            borrowing_label="Borrower" if self._is_borrower_value(borrowing_value, transcript) else "Non-borrower",
            is_borrower=self._is_borrower_value(borrowing_value, transcript),
            loan_interest_label=self._loan_interest_label(loan_interest_value),
            interested_in_loan=self._is_loan_interest_positive(loan_interest_value),
            smartphone_user=record.smartphone_user,
            has_bank_account=record.has_bank_account,
            digital_access=self._digital_access_label(record.smartphone_user, record.has_bank_account),
            transcript_quality=record.transcript_quality or self._infer_transcript_quality(
                status=record.status,
                transcript=transcript,
                last_error=record.last_error,
            ),
            extracted_json=extracted_json,
            evidence_quotes=tuple(record.key_quotes),
            segmented_dialogue=tuple(record.segmented_dialogue),
            analysis_version=record.analysis_version,
            analyzed_at=record.analyzed_at,
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
            participant_income_value=(
                repository_interview.participant_income_value
                if repository_interview.participant_income_value != "Unknown"
                else cached_result.participant_income_value
            ),
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
            transcript_quality=repository_interview.transcript_quality,
            extracted_json={
                **cached_result.extracted_json,
                **repository_interview.extracted_json,
            },
            evidence_quotes=repository_interview.evidence_quotes or cached_result.evidence_quotes,
            segmented_dialogue=repository_interview.segmented_dialogue or cached_result.segmented_dialogue,
            analysis_version=repository_interview.analysis_version or cached_result.analysis_version,
            analyzed_at=repository_interview.analyzed_at or cached_result.analyzed_at,
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

    def _nested_value(self, structured: dict[str, Any], field_name: str) -> str:
        value = structured.get(field_name)
        if isinstance(value, dict):
            return str(value.get("value", "unknown"))
        participant_profile = structured.get("participant_profile", {})
        persona_signals = structured.get("persona_signals", {})
        if isinstance(participant_profile, dict) and field_name in participant_profile:
            nested = participant_profile[field_name]
            if isinstance(nested, dict):
                return str(nested.get("value", "unknown"))
        if isinstance(persona_signals, dict) and field_name in persona_signals:
            nested = persona_signals[field_name]
            if isinstance(nested, dict):
                return str(nested.get("value", "unknown"))
        return "unknown"

    def _preferred_income_band(self, structured: dict[str, Any]) -> str:
        corrected_value = get_income_display_value(structured)
        if corrected_value is not None:
            return corrected_value
        legacy_value = self._nested_value(structured, "income_range")
        if legacy_value != "unknown":
            return legacy_value
        return "Unknown"

    def _record_income_band(self, record: DashboardInterviewRecord) -> str:
        for label, value in (
            ("Participant monthly income", record.participant_personal_monthly_income),
            ("Household monthly income", record.total_household_monthly_income),
            ("Per-household earnings", record.per_household_earnings),
        ):
            if value:
                return f"{label}: {value}"
        if record.income_range:
            return record.income_range
        return "Unknown"

    def _digital_access_label(self, smartphone_user: bool | None, has_bank_account: bool | None) -> str:
        if smartphone_user is False or has_bank_account is False:
            return "Excluded / offline"
        if smartphone_user is True and has_bank_account is True:
            return "Smartphone + bank account"
        return "Unknown / partial"

    def _infer_transcript_quality(self, *, status: str, transcript: str, last_error: str | None) -> str:
        normalized_transcript = (transcript or "").strip()
        normalized_error = (last_error or "").strip().lower()
        if status == ProcessingStatus.FAILED.value:
            return "failed"
        if not normalized_transcript or normalized_transcript == "Transcript pending.":
            return "malformed"
        if len(normalized_transcript.split()) < 5:
            return "malformed"
        if any(token in normalized_error for token in ("transcription", "malformed", "invalid", "schema", "decode")):
            return "malformed"
        return "good"

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

    def _bool_label(self, value: bool | None) -> str:
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        return "Unknown"

    def _empty_extracted_json(self, *, summary: str) -> dict[str, Any]:
        return {
            "runtime": {
                "status": "pending",
                "current_stage": "analysis",
                "last_error": None,
            },
            "insight": {
                "summary": summary,
                "persona": PERSONA_LOOKUP["persona_4"],
                "confidence_score": None,
                "key_quotes": [],
            },
        }

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

    def _evidence_quotes_for(
        self,
        interviews: list[DashboardInterview],
        *,
        predicate: Callable[[DashboardInterview], bool],
        limit: int = 3,
    ) -> list[str]:
        quotes: list[str] = []
        for interview in interviews:
            if not predicate(interview):
                continue
            for quote in interview.evidence_quotes:
                if quote not in quotes:
                    quotes.append(quote)
                if len(quotes) >= limit:
                    return quotes
        return quotes

    def _render_evidence_quotes(self, title: str, quotes: list[str]) -> None:
        if not quotes:
            return
        st.markdown(f"##### {title}")
        for quote in quotes:
            st.markdown(
                f"<div class='pd-card quote-card'>“{quote}”</div>",
                unsafe_allow_html=True,
            )

    def _inject_styles(self) -> None:
        st.markdown(
            f"""
            <style>
            :root {{
                --pd-bg: {DESIGN_SYSTEM['background']};
                --pd-card: {DESIGN_SYSTEM['card']};
                --pd-primary: {DESIGN_SYSTEM['primary']};
                --pd-text: {DESIGN_SYSTEM['text_primary']};
                --pd-muted: {DESIGN_SYSTEM['text_secondary']};
                --pd-border: {DESIGN_SYSTEM['border']};
                --pd-success: {DESIGN_SYSTEM['success']};
                --pd-danger: {DESIGN_SYSTEM['danger']};
            }}
            ::selection {{
                background: var(--pd-primary);
                color: var(--pd-text);
            }}
            ::-moz-selection {{
                background: var(--pd-primary);
                color: var(--pd-text);
            }}
            .stApp, .stApp header {{
                background: var(--pd-bg);
                color: var(--pd-text);
            }}
            .block-container {{
                padding-top: 24px;
                padding-bottom: 48px;
            }}
            section[data-testid="stSidebar"] {{
                background: rgba(17, 24, 39, 0.92);
                border-right: 1px solid var(--pd-border);
            }}
            section[data-testid="stSidebar"] * {{
                color: var(--pd-text);
            }}
            .pd-card {{
                background: var(--pd-card);
                border: 1px solid var(--pd-border);
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 16px;
                color: var(--pd-text);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.28);
            }}
            .page-hero {{
                margin-bottom: 16px;
            }}
            .page-title {{
                color: var(--pd-text);
                font-size: 28px;
                font-weight: 700;
                line-height: 1.2;
                margin-bottom: 8px;
            }}
            .page-subtitle, .section-subtitle, .metric-label, .summary-label, .meta-label, .persona-description,
            .persona-meta, .dialogue-meta, .dialogue-uncertainty, .stCaption, .stMarkdown p, .stMarkdown li {{
                color: var(--pd-muted);
            }}
            .metric-card {{
                min-height: 112px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 8px;
            }}
            .kpi-card {{
                min-height: 132px;
            }}
            .compact-card {{
                min-height: 96px;
            }}
            .metric-value, .persona-size {{
                color: var(--pd-text);
                font-size: 32px;
                font-weight: 700;
                line-height: 1;
            }}
            .section-title, .persona-name, .interview-title, .overlay-title {{
                color: var(--pd-text);
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 8px;
            }}
            .table-wrapper {{
                overflow: hidden;
            }}
            .pd-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .pd-table th, .pd-table td {{
                text-align: left;
                padding: 12px 8px;
                border-bottom: 1px solid rgba(156, 163, 175, 0.12);
            }}
            .pd-table th {{
                color: var(--pd-muted);
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .pd-table td {{
                color: var(--pd-text);
                font-size: 14px;
            }}
            .summary-row, .insight-row {{
                display: flex;
                justify-content: space-between;
                gap: 16px;
                padding: 12px 0;
                border-bottom: 1px solid rgba(156, 163, 175, 0.12);
            }}
            .summary-row:last-child, .insight-row:last-child {{
                border-bottom: none;
            }}
            .summary-value, .insight-value, .meta-value {{
                color: var(--pd-text);
                font-weight: 600;
                text-align: right;
            }}
            .quote-card, .dialogue-turn {{
                line-height: 1.6;
            }}
            .persona-card {{
                min-height: 216px;
            }}
            .persona-header, .interview-card-top {{
                display: flex;
                justify-content: space-between;
                gap: 16px;
                align-items: flex-start;
                margin-bottom: 12px;
            }}
            .persona-quote, .interview-summary {{
                color: var(--pd-text);
                line-height: 1.6;
                margin-top: 12px;
            }}
            .interview-title {{
                color: var(--pd-text);
            }}
            .interview-summary {{
                color: var(--pd-text);
            }}
            .interview-card {{
                min-height: 250px;
            }}
            .interview-meta-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
                margin-top: 16px;
            }}
            .meta-label {{
                display: block;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 4px;
                color: var(--pd-muted);
            }}
            .meta-value {{
                color: var(--pd-text);
            }}
            .persona-badge {{
                border-radius: 999px;
                padding: 8px 12px;
                text-align: center;
                font-size: 12px;
                font-weight: 700;
                border: 1px solid transparent;
                white-space: nowrap;
            }}
            .overlay-badge {{
                margin-bottom: 8px;
            }}
            .badge-target {{
                background: rgba(20, 184, 166, 0.14);
                color: #99F6E4;
                border-color: rgba(20, 184, 166, 0.24);
            }}
            .badge-nontarget {{
                background: rgba(244, 63, 94, 0.14);
                color: #FDA4AF;
                border-color: rgba(244, 63, 94, 0.24);
            }}
            .dialogue-speaker {{
                color: #A5B4FC;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
            }}
            .dialogue-utterance {{
                color: var(--pd-text);
                line-height: 1.7;
                margin-bottom: 8px;
            }}
            .quote-list {{
                margin: 0;
                padding-left: 20px;
                color: var(--pd-text);
            }}
            .overlay-backdrop {{
                background: rgba(11, 15, 25, 0.72);
                border-radius: 20px;
                min-height: 8px;
                margin-bottom: -8px;
                position: relative !important;
                z-index: 2147483000 !important;
                visibility: visible !important;
                opacity: 1 !important;
            }}
            .overlay-card {{
                border: 1px solid rgba(79, 70, 229, 0.38);
                box-shadow: 0 28px 72px rgba(0, 0, 0, 0.42);
                margin-bottom: 24px;
                position: relative !important;
                z-index: 2147483001 !important;
                visibility: visible !important;
                opacity: 1 !important;
                isolation: isolate;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                margin-bottom: 16px;
            }}
            .stTabs [data-baseweb="tab"] {{
                background: rgba(17, 24, 39, 0.92);
                border: 1px solid var(--pd-border);
                border-radius: 999px;
                color: var(--pd-muted);
                padding: 8px 16px;
            }}
            .stTabs [data-baseweb="tab"] p {{
                color: inherit;
                font-weight: 600;
            }}
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background: var(--pd-primary);
                border-color: var(--pd-primary);
                color: var(--pd-text);
            }}
            .stButton > button, .stDownloadButton > button {{
                min-height: 40px;
                border-radius: 12px;
                padding: 8px 16px;
                border: 1px solid var(--pd-border);
            }}
            .stButton > button:hover, .stDownloadButton > button:hover {{
                border-color: rgba(79, 70, 229, 0.72);
            }}
            .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
                outline: 2px solid var(--pd-primary);
                outline-offset: 2px;
            }}
            .stTextInput label, .stMultiSelect label, .stTextArea label, .stSelectbox label {{
                font-size: 14px !important;
                font-weight: 600 !important;
                color: var(--pd-text) !important;
            }}
            .stTextInput input, .stTextArea textarea, .stMultiSelect div[data-baseweb="select"] > div {{
                background: rgba(15, 23, 42, 0.72) !important;
                color: var(--pd-text) !important;
                border-radius: 12px !important;
                border: 1px solid var(--pd-border) !important;
            }}
            .stAlert {{
                background: rgba(17, 24, 39, 0.92);
                border: 1px solid var(--pd-border);
                color: var(--pd-text);
            }}
            [data-testid="stVegaLiteChart"] > div {{
                background: var(--pd-card);
                border: 1px solid var(--pd-border);
                border-radius: 16px;
                padding: 16px;
            }}
            .checklist-card {{
                margin-bottom: 16px;
                padding: 16px;
            }}
            .checklist-list {{
                margin: 8px 0 0;
                padding-left: 24px;
                color: var(--pd-text);
                display: grid;
                gap: 8px;
            }}
            @media (max-width: 900px) {{
                .interview-meta-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
