from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.registry import WorkflowDescriptor, launchable_workflows, pixi_environments


def render_workflow_selector(repo_root: Path) -> tuple[WorkflowDescriptor, str, str]:
    """Render workflow, Pixi environment, and run-name controls."""
    workflows = launchable_workflows()
    workflow = st.selectbox("Workflow", workflows, format_func=lambda item: f"{item.label} ({item.key})")
    st.caption(workflow.description)
    if workflow.required_inputs:
        st.write("Known inputs:", ", ".join(workflow.required_inputs))

    envs = pixi_environments(repo_root / "pixi.toml")
    env = st.selectbox("Pixi environment", envs, index=0)
    run_name = st.text_input("Run name", value=f"{workflow.key}-run")
    return workflow, env, run_name
