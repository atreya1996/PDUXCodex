from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from payday.config import Settings, get_settings
from payday.dashboard.views import DashboardRenderer
from payday.models import BatchUploadItem, ProcessingStatus
from payday.service import PaydayAppService
from payday.transcription import describe_transcription_file_size_limit
from payday.upload import SUPPORTED_UPLOAD_EXTENSIONS

REFRESH_STATUS_FLAG = "dashboard_status_reloaded"


@st.cache_resource
def build_app_service() -> tuple[PaydayAppService, Settings]:
    load_dotenv()
    get_settings.cache_clear()
    settings = get_settings()
    return PaydayAppService(settings), settings


def load_dashboard_state(app_service: PaydayAppService) -> dict[str, object]:
    return {
        "cached_results": app_service.list_results(),
        "recent_interviews": app_service.list_recent_interviews(),
        "status_overview": app_service.get_status_overview(),
    }


def main() -> None:
    st.set_page_config(page_title="PayDay Interview Review", layout="wide")

    app_service, settings = build_app_service()
    dashboard = DashboardRenderer()
    supported_formats_label = ", ".join(extension.upper() for extension in SUPPORTED_UPLOAD_EXTENSIONS)
    upload_limit_guidance = describe_transcription_file_size_limit(settings.transcription)
    upload_help_text = (
        "Choose 1 to 10 interview recordings in "
        f"{supported_formats_label}. Start with one file for a live smoke test, then try a full batch."
    )
    if upload_limit_guidance:
        upload_help_text = f"{upload_help_text} {upload_limit_guidance}"

    st.title("PayDay interview review")
    st.caption(
        "Batch uploads, evidence-grounded analysis, persona review, and interview QA in one workspace."
    )

    st.sidebar.header("Upload interviews")
    st.sidebar.caption(
        "Start with one small recording to validate live processing, then scale up to a full batch. "
        f"Supported formats: {supported_formats_label}. Filters stay in session state for instant iteration."
    )
    if upload_limit_guidance:
        st.sidebar.info(upload_limit_guidance)
    uploaded_files = st.sidebar.file_uploader(
        "Audio batch",
        type=list(SUPPORTED_UPLOAD_EXTENSIONS),
        accept_multiple_files=True,
        help=upload_help_text,
    )

    upload_count = len(uploaded_files) if uploaded_files else 0
    runtime_summary = app_service.runtime_summary()
    st.sidebar.write(
        {
            "files_selected": upload_count,
            "analysis_enabled": settings.features.enable_analysis,
            "sqlite_path": settings.database.sqlite_path,
            **runtime_summary,
        }
    )

    st.sidebar.caption("Reload the latest durable interview rows from SQLite after a rerun or batch completion.")
    if st.sidebar.button("Refresh status", use_container_width=True):
        st.session_state[REFRESH_STATUS_FLAG] = True
        st.rerun()
    if st.session_state.pop(REFRESH_STATUS_FLAG, False):
        st.sidebar.success("Reloaded durable interview statuses from SQLite.")

    process_disabled = not settings.features.enable_analysis or upload_count == 0
    if st.sidebar.button("Process batch", type="primary", use_container_width=True, disabled=process_disabled):
        if upload_count < 1 or upload_count > 10:
            st.sidebar.warning("Please upload between 1 and 10 audio files.")
        else:
            items = [
                BatchUploadItem(
                    filename=file.name,
                    content_type=file.type or "application/octet-stream",
                    data=file.getvalue(),
                )
                for file in uploaded_files
            ]
            with st.spinner("Processing uploaded interviews..."):
                batch_result = app_service.process_batch_uploads(items)

            summary_message = (
                f"Processed batch {batch_result.batch_id[:8]} with {batch_result.completed_count} completed and "
                f"{batch_result.failed_count} failed."
            )
            if batch_result.failed_count == 0:
                st.sidebar.success(summary_message)
            elif batch_result.completed_count == 0:
                st.sidebar.error(summary_message)
            else:
                st.sidebar.warning(summary_message)

            failed_results = [
                result for result in batch_result.results if result.status is ProcessingStatus.FAILED
            ]
            for failed_result in failed_results:
                error_message = (
                    failed_result.errors[0]
                    if failed_result.errors
                    else "Processing failed before transcription began."
                )
                st.sidebar.error(f"{failed_result.filename}: {error_message}")

    if not settings.features.enable_analysis:
        st.warning(
            "Analysis is disabled by feature flag. The dashboard continues to show durable SQLite-backed interviews plus any current-session cache."
        )

    dashboard_state = load_dashboard_state(app_service)
    dashboard.render(
        cached_results=dashboard_state["cached_results"],
        recent_interviews=dashboard_state["recent_interviews"],
        status_overview=dashboard_state["status_overview"],
        interview_detail_loader=app_service.get_interview_detail,
        sample_mode=settings.features.use_sample_mode,
    )


if __name__ == "__main__":
    main()
