from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from payday.config import Settings, get_settings
from payday.dashboard.views import DashboardRenderer
from payday.service import PaydayAppService


@st.cache_resource
def build_app_service() -> tuple[PaydayAppService, Settings]:
    load_dotenv()
    get_settings.cache_clear()
    settings = get_settings()
    return PaydayAppService(settings), settings


def main() -> None:
    app_service, settings = build_app_service()
    dashboard = DashboardRenderer()

    st.set_page_config(page_title="Payday MVP", layout="wide")
    st.title("Payday MVP")
    st.caption(
        f"Environment: {settings.app_env} · Sample mode: {'on' if settings.features.use_sample_mode else 'off'}"
    )

    st.sidebar.header("Providers")
    st.sidebar.write(
        {
            "supabase_configured": bool(settings.supabase.url),
            "llm_provider": settings.llm.provider,
            "transcription_provider": settings.transcription.provider,
            "analysis_enabled": settings.features.enable_analysis,
        }
    )

    uploaded_file = None
    if settings.features.enable_uploads:
        uploaded_file = st.file_uploader(
            "Upload a call recording or document",
            type=["mp3", "wav", "m4a", "mp4", "txt"],
        )
    else:
        st.info("Uploads are currently disabled by feature flag.")

    if st.button("Run pipeline", type="primary", disabled=not settings.features.enable_analysis):
        if uploaded_file is None:
            st.warning("Upload a file to run the pipeline.")
        else:
            result = app_service.process_upload(
                filename=uploaded_file.name,
                content_type=uploaded_file.type or "application/octet-stream",
                data=uploaded_file.getvalue(),
            )
            st.success("Pipeline completed.")
            st.subheader("Transcript")
            st.write(result.transcript.text)
            st.subheader("Analysis")
            st.write(result.analysis.summary)
            st.write(result.analysis.metrics)
            st.write({"personas": result.analysis.persona_matches, "persisted": result.persisted})

    if settings.features.enable_dashboard:
        dashboard.render(app_service.list_results())
