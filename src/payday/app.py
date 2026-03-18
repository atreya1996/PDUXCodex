from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from payday.config import Settings, get_settings
from payday.dashboard.views import DashboardRenderer
from payday.models import BatchUploadItem
from payday.service import PaydayAppService


@st.cache_resource
def build_app_service() -> tuple[PaydayAppService, Settings]:
    load_dotenv()
    get_settings.cache_clear()
    settings = get_settings()
    return PaydayAppService(settings), settings


def main() -> None:
    st.set_page_config(page_title="PayDay Interview Review", layout="wide")

    app_service, settings = build_app_service()
    dashboard = DashboardRenderer()

    st.title("PayDay interview review")
    st.caption(
        "Batch uploads, evidence-grounded analysis, persona review, and interview QA in one workspace."
    )

    st.sidebar.header("Upload interviews")
    st.sidebar.caption("Upload 5–10 audio files at a time for fast review. Filters stay in session state for instant iteration.")
    uploaded_files = st.sidebar.file_uploader(
        "Audio batch",
        type=["mp3", "wav", "m4a", "aac", "ogg"],
        accept_multiple_files=True,
        help="Choose between 5 and 10 interview recordings.",
    )

    upload_count = len(uploaded_files) if uploaded_files else 0
    st.sidebar.write(
        {
            "files_selected": upload_count,
            "sample_mode": settings.features.use_sample_mode,
            "analysis_enabled": settings.features.enable_analysis,
            "sqlite_path": settings.database.sqlite_path,
            "llm_provider": settings.llm.provider,
            "transcription_provider": settings.transcription.provider,
        }
    )

    process_disabled = not settings.features.enable_analysis or upload_count == 0
    if st.sidebar.button("Process batch", type="primary", use_container_width=True, disabled=process_disabled):
        if upload_count < 5 or upload_count > 10:
            st.sidebar.warning("Please upload between 5 and 10 audio files.")
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
            st.sidebar.success(
                f"Processed batch {batch_result.batch_id[:8]} with {batch_result.completed_count} completed and {batch_result.failed_count} failed."
            )

    if not settings.features.enable_analysis:
        st.warning(
            "Analysis is disabled by feature flag. The dashboard continues to show durable SQLite-backed interviews plus any current-session cache."
        )

    dashboard.render(
        cached_results=app_service.list_results(),
        recent_interviews=app_service.list_recent_interviews(),
        status_overview=app_service.get_status_overview(),
        interview_detail_loader=app_service.get_interview_detail,
    )


if __name__ == "__main__":
    main()
