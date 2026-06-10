from __future__ import annotations

from pathlib import Path


def render_workflow_tabs(result_dir: str | Path) -> None:
    """Render workflow-specific tabs for a selected result directory."""
    import streamlit as st

    from dashboard.workflow_panels import analysis, kinopt, networkmodel, protwise, tfopt

    root = Path(result_dir)
    tabs = st.tabs(["KinOpt", "TFOpt", "ProtWise", "Networkmodel", "Advanced analysis"])
    with tabs[0]:
        kinopt.render(root)
    with tabs[1]:
        tfopt.render(root)
    with tabs[2]:
        protwise.render(root)
    with tabs[3]:
        networkmodel.render(root)
    with tabs[4]:
        analysis.render(root)
