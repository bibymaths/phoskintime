from __future__ import annotations

from dashboard.config_utils import validate_input_assignments
from dashboard.file_utils import detect_duplicate_filenames, validate_existing_file
from dashboard.registry import WorkflowDescriptor


def validate_dashboard_setup(workflow: WorkflowDescriptor, available_paths, assignments: dict[str, str]) -> list[str]:
    """Validate uploaded/selected files and workflow input assignments."""
    problems: list[str] = []
    duplicates = detect_duplicate_filenames(path.name for path in available_paths)
    if duplicates:
        problems.append("Duplicate filenames: " + ", ".join(duplicates))
    for path in available_paths:
        problems.extend(f"{path.name}: {problem}" for problem in validate_existing_file(path))
    problems.extend(validate_input_assignments(workflow, assignments))
    return problems


def render_validation_panel(problems: list[str]) -> bool:
    """Render validation status and return whether running should be enabled."""
    import streamlit as st

    st.subheader("Validation")
    if problems:
        st.error("Resolve validation issues before running.")
        for problem in problems:
            st.write(f"- {problem}")
        return False
    st.success("No validation issues detected for the current dashboard selections.")
    return True
