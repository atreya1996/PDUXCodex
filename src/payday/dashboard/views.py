from __future__ import annotations

import streamlit as st

from payday.models import PipelineResult


class DashboardRenderer:
    def render(self, results: list[PipelineResult]) -> None:
        st.subheader("Recent pipeline runs")
        if not results:
            st.info("No analyses have been processed yet.")
            return

        for result in reversed(results):
            with st.container(border=True):
                st.markdown(f"**File:** {result.asset.filename}")
                st.markdown(f"**Summary:** {result.analysis.summary}")
                st.write("Metrics", result.analysis.metrics)
                st.write("Personas", result.analysis.persona_matches)
                st.caption(f"Persisted: {'Yes' if result.persisted else 'No'}")
