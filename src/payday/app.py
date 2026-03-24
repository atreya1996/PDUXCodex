from __future__ import annotations

import os
import subprocess

import streamlit as st
from dotenv import load_dotenv

from payday.config import Settings, get_settings
from payday.dashboard.views import DashboardRenderer
from payday.models import BatchUploadItem, ProcessingStatus
from payday.service import PaydayAppService
from payday.transcription import describe_transcription_file_size_limit
from payday.upload import SUPPORTED_UPLOAD_EXTENSIONS

REFRESH_STATUS_FLAG = "dashboard_status_reloaded"
FORCE_SQLITE_RELOAD_FLAG = "dashboard_force_sqlite_reload"
REPROCESS_STALE_FLAG = "dashboard_reprocess_stale_result"
DEFAULT_ENV_MAX_BATCH_BYTES = 45_000_000
DEFAULT_ENV_MAX_FILE_BYTES = 20_000_000
DEFAULT_CHUNK_SIZE = 3
MIN_CHUNK_SIZE = 1
MAX_CHUNK_SIZE = 3
NEAR_LIMIT_WARNING_RATIO = 0.85


def _get_runtime_limit_bytes(env_var: str, default_value: int) -> int:
    raw_value = os.getenv(env_var, "").strip()
    if not raw_value:
        return default_value
    try:
        parsed_mb = float(raw_value)
    except ValueError:
        return default_value
    if parsed_mb <= 0:
        return default_value
    return int(parsed_mb * 1_000_000)


def _get_chunk_size() -> int:
    raw_value = os.getenv("PAYDAY_UPLOAD_BATCH_CHUNK_SIZE", "").strip()
    if not raw_value:
        return DEFAULT_CHUNK_SIZE
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_CHUNK_SIZE
    return min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, parsed))


def _format_megabytes(value_bytes: int) -> str:
    return f"{value_bytes / 1_000_000:.1f} MB"


def _is_payload_too_large_error(message: str) -> bool:
    normalized = message.lower()
    return "413" in normalized or "payload too large" in normalized or "request entity too large" in normalized


def _chunk_upload_items(items: list[BatchUploadItem], chunk_size: int) -> list[list[BatchUploadItem]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


@st.cache_resource
def build_app_service() -> tuple[PaydayAppService, Settings]:
    load_dotenv()
    get_settings.cache_clear()
    settings = get_settings()
    return PaydayAppService(settings), settings


def load_dashboard_state(app_service: PaydayAppService, *, force_sqlite_reload: bool = False) -> dict[str, object]:
    return {
        "cached_results": [] if force_sqlite_reload else app_service.list_results(),
        "recent_interviews": app_service.list_recent_interviews(),
        "status_overview": app_service.get_status_overview(),
    }


def get_runtime_git_banner() -> str:
    try:
        short_sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        if short_sha and branch:
            return f"Runtime commit: `{short_sha}` on `{branch}`"
    except (OSError, subprocess.CalledProcessError):
        pass
    return "Runtime commit: unavailable"


def main() -> None:
    st.set_page_config(page_title="PayDay Interview Review", layout="wide")

    app_service, settings = build_app_service()
    dashboard = DashboardRenderer()
    supported_formats_label = ", ".join(extension.upper() for extension in SUPPORTED_UPLOAD_EXTENSIONS)
    upload_limit_guidance = describe_transcription_file_size_limit(settings.transcription)
    environment_file_limit_bytes = _get_runtime_limit_bytes(
        "PAYDAY_ENV_MAX_UPLOAD_FILE_MB", DEFAULT_ENV_MAX_FILE_BYTES
    )
    environment_batch_limit_bytes = _get_runtime_limit_bytes(
        "PAYDAY_ENV_MAX_UPLOAD_BATCH_MB", DEFAULT_ENV_MAX_BATCH_BYTES
    )
    upload_chunk_size = _get_chunk_size()
    upload_help_text = (
        "Choose 1 to 10 interview recordings in "
        f"{supported_formats_label}. Start with one file for a live smoke test, then try a full batch."
    )
    if upload_limit_guidance:
        upload_help_text = f"{upload_help_text} {upload_limit_guidance}"
    upload_help_text = (
        f"{upload_help_text} Runtime guardrails: max {_format_megabytes(environment_file_limit_bytes)} per file "
        f"and {_format_megabytes(environment_batch_limit_bytes)} per batch."
    )

    st.title("PayDay interview review")
    st.caption(
        "Batch uploads, evidence-grounded analysis, persona review, and interview QA in one workspace."
    )

    st.sidebar.header("Upload interviews")
    st.sidebar.caption(get_runtime_git_banner())
    st.sidebar.caption(
        "Start with one small recording to validate live processing, then scale up to a full batch. "
        f"Supported formats: {supported_formats_label}. Filters stay in session state for instant iteration."
    )
    if upload_limit_guidance:
        st.sidebar.info(upload_limit_guidance)
    st.sidebar.info(
        "Deployment/runtime upload guardrails: "
        f"{_format_megabytes(environment_file_limit_bytes)} per file, "
        f"{_format_megabytes(environment_batch_limit_bytes)} per batch."
    )
    uploaded_files = st.sidebar.file_uploader(
        "Audio batch",
        type=list(SUPPORTED_UPLOAD_EXTENSIONS),
        accept_multiple_files=True,
        help=upload_help_text,
    )

    upload_count = len(uploaded_files) if uploaded_files else 0
    file_sizes = [len(file.getvalue()) for file in uploaded_files] if uploaded_files else []
    total_selected_size = sum(file_sizes)
    largest_selected_file = max(file_sizes) if file_sizes else 0
    near_environment_limit = (
        environment_batch_limit_bytes > 0
        and total_selected_size >= int(environment_batch_limit_bytes * NEAR_LIMIT_WARNING_RATIO)
    )

    if near_environment_limit and total_selected_size <= environment_batch_limit_bytes:
        st.sidebar.warning(
            "Selected uploads are near the deployment request limit "
            f"({_format_megabytes(total_selected_size)} of {_format_megabytes(environment_batch_limit_bytes)}). "
            "If processing fails with HTTP 413, retry with fewer files per batch."
        )
    runtime_summary = app_service.runtime_summary()
    st.sidebar.write(
        {
            "files_selected": upload_count,
            "batch_size_mb": round(total_selected_size / 1_000_000, 2),
            "largest_file_mb": round(largest_selected_file / 1_000_000, 2),
            "analysis_enabled": settings.features.enable_analysis,
            **runtime_summary,
        }
    )
    st.sidebar.caption(f"Runtime commit SHA: `{runtime_summary.get('runtime_commit_sha', 'unknown')}`")

    st.sidebar.caption(
        "Reload durable interview status and runtime release metadata (including commit SHA) after reruns or deployments."
    )
    if st.sidebar.button("Refresh status + release metadata", use_container_width=True):
        build_app_service.clear()
        st.session_state[REFRESH_STATUS_FLAG] = True
        st.session_state[FORCE_SQLITE_RELOAD_FLAG] = True
        st.rerun()
    if st.session_state.pop(REFRESH_STATUS_FLAG, False):
        st.sidebar.success("Reloaded durable interview status and release metadata.")

    st.sidebar.caption("Recompute interviews saved under older analysis/persona schema versions.")
    if st.sidebar.button("Reprocess all stale interviews", use_container_width=True):
        st.session_state[REPROCESS_STALE_FLAG] = app_service.reprocess_stale_interviews()
        st.rerun()
    stale_reprocess_result = st.session_state.pop(REPROCESS_STALE_FLAG, None)
    if isinstance(stale_reprocess_result, dict):
        stale_count = int(stale_reprocess_result.get("stale_count", 0))
        refreshed_count = len(stale_reprocess_result.get("reprocessed_ids", []))
        failed = stale_reprocess_result.get("failed", {})
        if stale_count == 0:
            st.sidebar.info("No stale interviews were found.")
        elif failed:
            st.sidebar.warning(
                f"Reprocessed {refreshed_count} of {stale_count} stale interviews; {len(failed)} failed."
            )
            for interview_id, error in list(failed.items())[:5]:
                st.sidebar.error(f"{interview_id}: {error}")
        else:
            st.sidebar.success(f"Reprocessed all {stale_count} stale interviews.")

    process_disabled = not settings.features.enable_analysis or upload_count == 0
    if st.sidebar.button("Process batch", type="primary", use_container_width=True, disabled=process_disabled):
        if upload_count < 1 or upload_count > 10:
            st.sidebar.warning("Please upload between 1 and 10 audio files.")
        elif largest_selected_file > environment_file_limit_bytes:
            st.sidebar.error(
                "One or more files exceed the deployment/runtime per-file limit "
                f"({_format_megabytes(environment_file_limit_bytes)}). "
                "Compress/split oversized recordings before retrying."
            )
        elif total_selected_size > environment_batch_limit_bytes:
            st.sidebar.error(
                "Combined selection exceeds the deployment/runtime batch limit "
                f"({_format_megabytes(total_selected_size)} > {_format_megabytes(environment_batch_limit_bytes)}). "
                "Remove files or run multiple smaller batches."
            )
        else:
            items = [
                BatchUploadItem(
                    filename=file.name,
                    content_type=file.type or "application/octet-stream",
                    data=file.getvalue(),
                )
                for file in uploaded_files
            ]
            chunks = _chunk_upload_items(items, upload_chunk_size)
            chunk_results = []
            with st.spinner("Processing uploaded interviews in smaller chunks..."):
                for chunk_index, chunk in enumerate(chunks, start=1):
                    st.sidebar.caption(f"Processing chunk {chunk_index}/{len(chunks)} ({len(chunk)} files).")
                    chunk_results.append(app_service.process_batch_uploads(chunk))

            completed_count = sum(result.completed_count for result in chunk_results)
            failed_count = sum(result.failed_count for result in chunk_results)
            batch_ids = ", ".join(result.batch_id[:8] for result in chunk_results)
            summary_message = (
                f"Processed {len(items)} files across {len(chunks)} chunk(s) [{batch_ids}] with "
                f"{completed_count} completed and {failed_count} failed."
            )
            if failed_count == 0:
                st.sidebar.success(summary_message)
            elif completed_count == 0:
                st.sidebar.error(summary_message)
            else:
                st.sidebar.warning(summary_message)

            failed_results = [
                result
                for chunk_result in chunk_results
                for result in chunk_result.results
                if result.status is ProcessingStatus.FAILED
            ]
            for failed_result in failed_results:
                error_message = (
                    failed_result.errors[0]
                    if failed_result.errors
                    else "Processing failed before transcription began."
                )
                st.sidebar.error(f"{failed_result.filename}: {error_message}")
                if _is_payload_too_large_error(error_message):
                    st.sidebar.warning(
                        "HTTP 413 guidance: retry with fewer files (2-3 per run), compress audio, "
                        "or lower duration/bitrate before re-uploading."
                    )

    if not settings.features.enable_analysis:
        st.warning(
            "Analysis is disabled by feature flag. The dashboard continues to show durable SQLite-backed interviews plus any current-session cache."
        )

    force_sqlite_reload = bool(st.session_state.pop(FORCE_SQLITE_RELOAD_FLAG, False))
    dashboard_state = load_dashboard_state(app_service, force_sqlite_reload=force_sqlite_reload)
    dashboard.render(
        cached_results=dashboard_state["cached_results"],
        recent_interviews=dashboard_state["recent_interviews"],
        status_overview=dashboard_state["status_overview"],
        interview_detail_loader=app_service.get_interview_detail,
        save_interview_edits=getattr(app_service, "save_interview_edits", None),
        reprocess_interview=getattr(app_service, "reprocess_interview", None),
        reanalyze_interviews=getattr(app_service, "reanalyze_interviews", None),
        delete_interview=getattr(app_service, "delete_interview", lambda _interview_id: False),
        sample_mode=settings.features.use_sample_mode,
    )


if __name__ == "__main__":
    main()
