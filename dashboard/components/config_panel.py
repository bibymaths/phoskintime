from __future__ import annotations

from pathlib import Path
from typing import Any

from dashboard.config_utils import parse_config_file
from dashboard.registry import WorkflowDescriptor


def render_config_panel(workflow: WorkflowDescriptor, available_files: list[Path]) -> tuple[
    dict[str, Any], dict[str, str]]:
    import streamlit as st

    """Render structured parameter and workflow-specific input assignment controls."""
    st.subheader("Configure workflow")
    arguments: dict[str, Any] = {}
    assignments: dict[str, str] = {}

    if workflow.accepted_arguments:
        with st.expander("Parameters", expanded=True):
            for spec in workflow.accepted_arguments:
                if any(input_spec.argument_name == spec.name for input_spec in workflow.input_assignments):
                    continue
                help_text = spec.description or None
                default = "" if spec.default is None else str(spec.default)
                if spec.kind == "bool":
                    arguments[spec.name] = st.checkbox(spec.flag, value=bool(spec.default), help=help_text)
                else:
                    arguments[spec.name] = st.text_input(spec.flag, value=default, help=help_text,
                                                         key=f"arg-{workflow.key}-{spec.name}")

    if workflow.input_assignments:
        with st.expander("Input assignments", expanded=True):
            labels = ["Do not pass"] + [str(path) for path in available_files]
            for spec in workflow.input_assignments:
                selected = st.selectbox(spec.label, labels, help=spec.description or None,
                                        key=f"input-{workflow.key}-{spec.role}")
                custom = st.text_input(f"Existing path for {spec.label}", value="",
                                       key=f"input-path-{workflow.key}-{spec.role}")
                if custom.strip():
                    assignments[spec.role] = custom.strip()
                elif selected != "Do not pass":
                    assignments[spec.role] = selected

    config_paths = [Path(path) for path in assignments.values() if
                    Path(path).suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".txt"}]
    if config_paths:
        with st.expander("Config preview", expanded=False):
            selected_config = st.selectbox("Config file", config_paths, format_func=lambda path: path.name)
            try:
                parsed = parse_config_file(selected_config)
                if isinstance(parsed, (dict, list)):
                    st.json(parsed)
                else:
                    st.code(str(parsed))
            except Exception as exc:
                st.error(f"Could not parse config: {exc}")
    return arguments, assignments
