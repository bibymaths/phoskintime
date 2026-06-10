from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.registry import WorkflowDescriptor, launchable_workflows, pixi_environments


def render_workflow_selector(repo_root: Path) -> tuple[WorkflowDescriptor, str, str, dict[str, Any]]:
    """Render structured workflow/env/argument controls for safe command building."""
    workflows = launchable_workflows()
    workflow = st.selectbox("Workflow", workflows, format_func=lambda item: f"{item.label} ({item.key})")
    st.caption(workflow.description)
    if workflow.required_inputs:
        st.write("Known inputs:", ", ".join(workflow.required_inputs))

    envs = pixi_environments(repo_root / "pixi.toml")
    env = st.selectbox("Pixi environment", envs, index=0)
    run_name = st.text_input("Run name", value=f"{workflow.key}-run")

    values: dict[str, Any] = {}
    if workflow.accepted_arguments:
        with st.expander("Structured CLI arguments", expanded=False):
            for spec in workflow.accepted_arguments:
                current = spec.default
                help_text = spec.description or None
                if spec.kind == "bool":
                    values[spec.name] = st.checkbox(spec.flag, value=bool(current), help=help_text)
                elif spec.kind in {"int", "float"}:
                    values[spec.name] = st.text_input(spec.flag, value="" if current is None else str(current), help=help_text)
                else:
                    values[spec.name] = st.text_input(spec.flag, value="" if current is None else str(current), help=help_text)
    return workflow, env, run_name, values
