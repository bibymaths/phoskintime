from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.command_builder import build_workflow_command, sanitize_run_name
from dashboard.components.command_preview import render_command_preview
from dashboard.components.config_panel import render_config_panel
from dashboard.components.console_panel import render_cancellation_note, render_console
from dashboard.components.input_preview import render_input_preview
from dashboard.components.preset_panel import render_preset_panel
from dashboard.components.result_browser import render_result_browser
from dashboard.components.run_status import render_run_status
from dashboard.components.upload_panel import render_upload_panel
from dashboard.components.validation_panel import render_validation_panel, validate_dashboard_setup
from dashboard.components.workflow_selector import render_workflow_selector
from dashboard.components.workflow_tabs import render_workflow_tabs
from dashboard.config_utils import DashboardSelection
from dashboard.result_parser import discover_result_directory
from dashboard.runner import log_tail, run_built_command

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_result_dirs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    candidates = [base]
    candidates.extend(path for path in sorted(base.iterdir()) if path.is_dir())
    return candidates


def _render_browser_panel(default_directory: Path | None = None) -> None:
    with st.sidebar:
        st.header("Result directory")
        default_base = default_directory.parent if default_directory else Path("results")
        base = Path(st.text_input("Base results folder", value=str(default_base), key="browser-base")).expanduser()
        candidates = _candidate_result_dirs(base)
        if default_directory and default_directory.is_dir() and default_directory not in candidates:
            candidates.insert(0, default_directory)
        if candidates:
            index = candidates.index(default_directory) if default_directory in candidates else 0
            choice = st.selectbox("Select folder", candidates, index=index, format_func=lambda path: str(path), key="browser-choice")
            directory_text = st.text_input("Selected result directory", value=str(choice), key="browser-directory")
        else:
            st.info("No selectable folders found under the base path. Enter a result directory manually.")
            directory_text = st.text_input("Selected result directory", value=str(base), key="browser-directory-manual")

    try:
        inventory = discover_result_directory(directory_text)
    except (FileNotFoundError, NotADirectoryError) as exc:
        st.error(str(exc))
        return

    if not inventory.has_content:
        st.warning("This directory exists, but no standard PhosKinTime result files were discovered.")
    render_result_browser(inventory)
    render_workflow_tabs(inventory.root)


def _render_launcher_panel() -> None:
    st.header("Workflow launcher")
    st.write("Construct, preview, and run registered PhosKinTime workflows using existing CLI modules.")
    render_cancellation_note()

    workflow, env, run_name = render_workflow_selector(REPO_ROOT)
    safe_run_name = sanitize_run_name(run_name)
    uploaded_paths = render_upload_panel(REPO_ROOT, safe_run_name)
    retained_paths = [Path(path) for path in st.session_state.get("uploaded_paths", []) if Path(path).exists()]
    combined_paths = sorted({*retained_paths, *uploaded_paths}, key=lambda path: path.name.lower())
    if uploaded_paths:
        st.session_state["uploaded_paths"] = [str(path) for path in combined_paths]
    render_input_preview(combined_paths)
    argument_values, input_assignments = render_config_panel(workflow, combined_paths)
    validation_problems = validate_dashboard_setup(workflow, combined_paths, input_assignments)
    can_run = render_validation_panel(validation_problems)

    try:
        built = build_workflow_command(
            workflow.key,
            repo_root=REPO_ROOT,
            pixi_environment=env,
            run_name=safe_run_name,
            argument_values=argument_values,
            input_assignments=input_assignments,
        )
    except (KeyError, ValueError) as exc:
        st.error(str(exc))
        return

    render_command_preview(built)
    selection = DashboardSelection(
        workflow_key=workflow.key,
        run_name=safe_run_name,
        pixi_environment=env,
        arguments=argument_values,
        input_assignments=input_assignments,
    )
    render_preset_panel(selection, REPO_ROOT)
    render_run_status(st.session_state.get("launcher_status"), st.session_state.get("launcher_returncode"))

    if st.button("Run workflow", type="primary", disabled=not can_run):
        console_lines: list[str] = []
        console_placeholder = st.empty()
        status_placeholder = st.empty()
        st.session_state["launcher_status"] = "running"
        st.session_state["launcher_returncode"] = None
        with status_placeholder.container():
            render_run_status("running")
        final_event = None
        try:
            for event in run_built_command(built, repo_root=REPO_ROOT):
                final_event = event
                if event.line:
                    console_lines.append(event.line)
                    with console_placeholder.container():
                        render_console(console_lines)
        except FileNotFoundError as exc:
            st.session_state["launcher_status"] = "failure"
            st.session_state["launcher_returncode"] = 127
            st.error(f"Could not start workflow command: {exc}")
            return

        if final_event is not None:
            st.session_state["launcher_status"] = final_event.status
            st.session_state["launcher_returncode"] = final_event.returncode
            st.session_state["last_run_dir"] = str(built.outdir)
            with status_placeholder.container():
                render_run_status(final_event.status, final_event.returncode)
            if final_event.status == "failure":
                st.subheader("Log tail")
                st.code(log_tail(built.outdir), language="text")
            elif final_event.status == "success":
                st.success("Run completed. The result directory is shown below.")
                try:
                    inventory = discover_result_directory(built.outdir)
                    render_result_browser(inventory)
                    render_workflow_tabs(inventory.root)
                except (FileNotFoundError, NotADirectoryError) as exc:
                    st.warning(f"Run finished, but the result directory could not be opened: {exc}")


def main() -> None:
    st.set_page_config(page_title="PhosKinTime Dashboard", layout="wide")
    st.title("PhosKinTime Dashboard")
    st.write("Browse existing result directories or launch registered CLI workflows without reimplementing scientific logic.")

    launcher_tab, browser_tab = st.tabs(["Run workflow", "Browse results"])
    with launcher_tab:
        _render_launcher_panel()
    with browser_tab:
        last_run_dir = st.session_state.get("last_run_dir")
        _render_browser_panel(Path(last_run_dir) if last_run_dir else None)


if __name__ == "__main__":
    main()
