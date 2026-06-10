from __future__ import annotations


def render_run_status(status: str | None, returncode: int | None = None) -> None:
    """Render workflow completion status."""
    import streamlit as st

    if status is None:
        st.info("No workflow has been run in this session.")
    elif status == "running":
        st.warning("Workflow is running…")
    elif status == "success":
        st.success("Workflow completed successfully.")
    elif status == "cancelled":
        st.warning(f"Workflow was cancelled. Return code: {returncode}")
    else:
        st.error(f"Workflow failed. Return code: {returncode}. Review the console log tail for details.")
