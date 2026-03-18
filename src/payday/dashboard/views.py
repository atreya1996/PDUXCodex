from __future__ import annotations

import streamlit as st

from payday.models import PipelineResult, ProcessingStatus


class DashboardRenderer:
    def render(self, results: list[PipelineResult]) -> None:
        st.subheader("Recent pipeline runs")
        if not results:
            st.info("No analyses have been processed yet.")
            return

        for result in reversed(results):
            with st.container(border=True):
                st.markdown(f"**File:** {result.filename}")
                st.markdown(f"**Status:** {result.status.value}")
                st.markdown(f"**Current stage:** {result.current_stage.value}")

                if result.analysis is not None:
                    st.markdown(f"**Summary:** {result.analysis.summary}")
                    st.write("Metrics", result.analysis.metrics)
                    st.write("Evidence", result.analysis.evidence_quotes)
                else:
                    st.info("Analysis not available yet for this file.")

                if result.persona is not None:
                    st.write(
                        "Persona",
                        {
                            "persona_id": result.persona.persona_id,
                            "persona_name": result.persona.persona_name,
                            "is_non_target": result.persona.is_non_target,
                        },
                    )

                if result.status is ProcessingStatus.FAILED:
                    st.error(" ; ".join(result.errors) or "The pipeline failed for this file.")

                st.caption(f"Persisted: {'Yes' if result.persisted else 'No'}")
