from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Callable

import streamlit as st

from payday.models import PipelineResult, ProcessingStatus
from payday.personas import PERSONAS
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview

PERSONA_LOOKUP = {persona.id: persona.name for persona in PERSONAS.values()}
FILTER_SESSION_KEYS = {
    "income": "dashboard_filter_income",
    "borrowing": "dashboard_filter_borrowing",
    "persona": "dashboard_filter_persona",
    "digital_access": "dashboard_filter_digital_access",
    "search": "dashboard_search_query",
    "selected": "dashboard_selected_interview_id",
    "transcripts": "dashboard_transcript_edits",
    "json": "dashboard_json_edits",
}


@dataclass(frozen=True, slots=True)
class DashboardInterview:
    id: str
    filename: str
    created_at: str
    status: str
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
    ) -> None:
        self._inject_styles()
        interviews = self._build_dashboard_interviews(cached_results, recent_interviews)
        self._initialize_session_state(interviews)
        filtered = self._apply_filters(interviews)

        tabs = st.tabs(["Overview", "Cohorts", "Personas", "Interviews", "Interview Detail"])
        with tabs[0]:
            self._render_overview(filtered, interviews, status_overview)
        with tabs[1]:
            self._render_cohorts(filtered)
        with tabs[2]:
            self._render_personas(filtered)
        with tabs[3]:
            self._render_interviews(filtered)
        with tabs[4]:
            self._render_interview_detail(filtered, interviews, interview_detail_loader)

    def _initialize_session_state(self, interviews: list[DashboardInterview]) -> None:
        st.session_state.setdefault(FILTER_SESSION_KEYS["income"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["borrowing"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["persona"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["digital_access"], [])
        st.session_state.setdefault(FILTER_SESSION_KEYS["search"], "")
        st.session_state.setdefault(FILTER_SESSION_KEYS["transcripts"], {})
        st.session_state.setdefault(FILTER_SESSION_KEYS["json"], {})

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
    ) -> None:
        st.markdown("### Interview portfolio")
        self._render_filter_summary(filtered, all_interviews, status_overview)
        self._render_kpis(filtered, status_overview)

        if not filtered:
            st.info("No interviews match the current filters.")
            return

        left_col, right_col = st.columns(2, gap="large")
        with left_col:
            self._render_chart_card(
                title="Income distribution",
                subtitle="Current filtered set",
                values=self._count_by(filtered, lambda item: item.income_band),
                color="#4f7cff",
            )
        with right_col:
            self._render_chart_card(
                title="Borrowing sources",
                subtitle="Where participants say they borrow from",
                values=self._count_by(filtered, lambda item: item.borrowing_source),
                color="#14b8a6",
            )

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

    def _render_personas(self, filtered: list[DashboardInterview]) -> None:
        st.markdown("### Persona review")
        if not filtered:
            st.info("No persona cards to show for the current filter set.")
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

    def _render_interviews(self, filtered: list[DashboardInterview]) -> None:
        st.markdown("### Searchable interview list")
        if not filtered:
            st.info("No interviews match the current search and filters.")
            return

        for interview in filtered:
            self._render_interview_row(interview)

    def _render_interview_detail(
        self,
        filtered: list[DashboardInterview],
        all_interviews: list[DashboardInterview],
        interview_detail_loader: Callable[[str], DashboardInterviewRecord],
    ) -> None:
        st.markdown("### Interview detail")
        if not all_interviews:
            st.info("Upload or process interviews to inspect details here.")
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
        transcript_edits.setdefault(selected.id, selected.transcript)
        json_edits.setdefault(selected.id, json.dumps(selected.extracted_json, indent=2, ensure_ascii=False))

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
            st.write(
                {
                    "digital_access": selected.digital_access,
                    "income_band": selected.income_band,
                    "borrowing": selected.borrowing_label,
                    "borrowing_source": selected.borrowing_source,
                    "loan_interest": selected.loan_interest_label,
                }
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
            {"values": chart_data},
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

    def _render_interview_row(self, interview: DashboardInterview) -> None:
        col_main, col_meta, col_action = st.columns([3.2, 1.4, 0.9], gap="medium")
        with col_main:
            st.markdown(
                f"""
                <div class='pd-card interview-row'>
                    <div class='interview-title'>{interview.filename}</div>
                    <div class='interview-summary'>{interview.summary}</div>
                    <div class='interview-meta'>Borrowing: {interview.borrowing_source} · Income: {interview.income_band}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_meta:
            st.markdown(
                f"""
                <div class='pd-card interview-meta-card'>
                    <div><strong>{interview.persona_name}</strong></div>
                    <div>{interview.digital_access}</div>
                    <div>Status: {interview.status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_action:
            if st.button("Open", key=f"open_interview_{interview.id}", use_container_width=True):
                st.session_state[FILTER_SESSION_KEYS["selected"]] = interview.id

    def _build_cohort_rows(self, interviews: list[DashboardInterview], attribute: str) -> list[tuple[str, int, str]]:
        counts = self._count_by(interviews, lambda item: getattr(item, attribute))
        total = len(interviews)
        return [(label, count, self._percent(count, total)) for label, count in counts.items()]

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
    ) -> list[DashboardInterview]:
        mapped_results = {result.file_id: self._from_pipeline_result(result) for result in results}
        merged = [self._overlay_cached_result(record, mapped_results.get(record.id)) for record in recent_interviews]
        if merged:
            return merged
        if mapped_results:
            return list(mapped_results.values())
        return self._sample_interviews()

    def _from_pipeline_result(self, result: PipelineResult) -> DashboardInterview:
        structured = result.analysis.structured_output if result.analysis is not None else {}
        transcript = result.transcript.text if result.transcript is not None else "Transcript pending."
        smartphone_user = self._smartphone_user(structured)
        has_bank_account = self._bank_account_user(structured)
        digital_access = self._digital_access_label(smartphone_user, has_bank_account)
        borrowing_value = self._nested_value(structured, "borrowing_history")
        loan_interest_value = self._nested_value(structured, "loan_interest")
        summary = result.analysis.summary if result.analysis is not None else "Analysis pending."
        evidence_quotes = tuple(result.analysis.evidence_quotes if result.analysis is not None else [])
        persona_id = result.persona.persona_id if result.persona is not None else "persona_4"
        persona_name = result.persona.persona_name if result.persona is not None else PERSONA_LOOKUP["persona_4"]
        return DashboardInterview(
            id=result.file_id,
            filename=result.filename,
            created_at=datetime.now(timezone.utc).date().isoformat(),
            status=result.status.value,
            summary=summary,
            transcript=transcript,
            persona_id=persona_id,
            persona_name=persona_name,
            is_non_target=bool(result.persona.is_non_target) if result.persona is not None else False,
            income_band=self._nested_value(structured, "income_range") or "Unknown",
            borrowing_source=self._borrowing_source_label(transcript, borrowing_value),
            borrowing_label="Borrower" if self._is_borrower_value(borrowing_value, transcript) else "Non-borrower",
            is_borrower=self._is_borrower_value(borrowing_value, transcript),
            loan_interest_label=self._loan_interest_label(loan_interest_value),
            interested_in_loan=self._is_loan_interest_positive(loan_interest_value),
            smartphone_user=smartphone_user,
            has_bank_account=has_bank_account,
            digital_access=digital_access,
            extracted_json=structured or {"status": result.status.value},
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
        extracted_json = {
            "audio_url": record.audio_url,
            "participant_profile": {
                "smartphone_user": {"value": record.smartphone_user},
                "has_bank_account": {"value": record.has_bank_account},
            },
            "income_range": {"value": record.income_range},
            "borrowing_history": {"value": record.borrowing_history},
            "repayment_preference": {"value": record.repayment_preference},
            "loan_interest": {"value": record.loan_interest},
            "insight": {
                "summary": record.summary,
                "persona": record.persona,
                "confidence_score": record.confidence_score,
                "key_quotes": record.key_quotes,
            },
        }
        return DashboardInterview(
            id=record.id,
            filename=record.filename,
            created_at=self._format_created_at(record.created_at),
            status=record.status,
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
        return DashboardInterview(
            id=repository_interview.id,
            filename=cached_result.filename or repository_interview.filename,
            created_at=repository_interview.created_at,
            status=cached_result.status,
            summary=cached_result.summary,
            transcript=cached_result.transcript,
            persona_id=cached_result.persona_id,
            persona_name=cached_result.persona_name,
            is_non_target=cached_result.is_non_target,
            income_band=cached_result.income_band,
            borrowing_source=cached_result.borrowing_source,
            borrowing_label=cached_result.borrowing_label,
            is_borrower=cached_result.is_borrower,
            loan_interest_label=cached_result.loan_interest_label,
            interested_in_loan=cached_result.interested_in_loan,
            smartphone_user=cached_result.smartphone_user,
            has_bank_account=cached_result.has_bank_account,
            digital_access=cached_result.digital_access,
            extracted_json={
                **repository_interview.extracted_json,
                **cached_result.extracted_json,
                "audio_url": repository_interview.extracted_json.get("audio_url"),
            },
            evidence_quotes=cached_result.evidence_quotes or repository_interview.evidence_quotes,
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

    def _smartphone_user(self, structured: dict[str, Any]) -> bool | None:
        if "smartphone_usage" in structured:
            value = self._nested_value(structured, "smartphone_usage")
            if value == "has_smartphone":
                return True
            if value == "no_smartphone":
                return False
        participant_profile = structured.get("participant_profile", {})
        if isinstance(participant_profile, dict):
            smartphone = participant_profile.get("smartphone_user")
            if isinstance(smartphone, dict):
                return smartphone.get("value")
        return None

    def _bank_account_user(self, structured: dict[str, Any]) -> bool | None:
        if "bank_account_status" in structured:
            value = self._nested_value(structured, "bank_account_status")
            if value == "has_bank_account":
                return True
            if value == "no_bank_account":
                return False
        participant_profile = structured.get("participant_profile", {})
        if isinstance(participant_profile, dict):
            bank_account = participant_profile.get("has_bank_account")
            if isinstance(bank_account, dict):
                return bank_account.get("value")
        return None

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

    def _sample_interviews(self) -> list[DashboardInterview]:
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
            persona_name = PERSONA_LOOKUP[persona_id]
            digital_access = self._digital_access_label(sample["smartphone_user"], sample["has_bank_account"])
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
                "income_range": {
                    "value": sample["income_band"],
                    "status": "observed",
                    "evidence_quotes": [transcript.split(". ")[0]],
                    "notes": "Sample-mode structured output.",
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
                    audio_bytes=self._sample_audio_bytes(frequency=220 + index * 40),
                )
            )
        return interviews

    def _sample_audio_bytes(self, frequency: int) -> bytes:
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

    def _percent(self, numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "0%"
        return f"{round((numerator / denominator) * 100)}%"

    def _inject_styles(self) -> None:
        st.markdown(
            """
            <style>
            .stApp {
                background: #f6f8fc;
            }
            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }
            .pd-card {
                background: white;
                border-radius: 20px;
                padding: 1.15rem 1.25rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.16);
                margin-bottom: 1rem;
            }
            .kpi-card {
                min-height: 124px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .kpi-label, .chart-subtitle, .table-title, .interview-meta, .persona-meta {
                color: #64748b;
                font-size: 0.95rem;
            }
            .kpi-value, .persona-count {
                font-size: 2rem;
                font-weight: 700;
                color: #0f172a;
            }
            .chart-title, .table-title {
                font-size: 1.15rem;
                font-weight: 650;
                color: #0f172a;
                margin-bottom: 0.2rem;
            }
            .persona-label, .interview-title {
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
                color: #475569;
                font-weight: 600;
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
                color: #0f766e;
            }
            .badge-nontarget {
                background: rgba(244, 63, 94, 0.12);
                color: #be123c;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: white;
                color: #334155;
                border-radius: 999px;
                padding: 0.6rem 1rem;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
                border: 1px solid rgba(148, 163, 184, 0.24);
                transition: color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            }
            .stTabs [data-baseweb="tab"] p {
                color: inherit;
            }
            .stTabs [data-baseweb="tab"]:hover {
                color: #1d4ed8;
                border-color: rgba(79, 124, 255, 0.32);
                box-shadow: 0 10px 24px rgba(79, 124, 255, 0.1);
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                color: #1d4ed8;
                border-color: rgba(79, 124, 255, 0.38);
                box-shadow: 0 12px 28px rgba(79, 124, 255, 0.14);
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] p {
                color: inherit;
            }
            .stTextInput label, .stMultiSelect label, .stTextArea label {
                font-size: 1rem !important;
                font-weight: 600 !important;
            }
            p, li, span, label {
                font-size: 1rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
