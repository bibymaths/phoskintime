from __future__ import annotations

import streamlit as st

from dashboard.components.download_panel import render_download_panel
from dashboard.components.log_viewer import render_logs, render_text_file
from dashboard.components.metadata_viewer import render_metadata
from dashboard.components.plot_viewer import render_plots
from dashboard.components.report_viewer import render_reports
from dashboard.components.table_viewer import render_tables
from dashboard.registry import infer_workflow
from dashboard.result_parser import ResultInventory


def _summary_metric(label: str, value: int) -> None:
    st.metric(label, value)


def render_result_browser(inventory: ResultInventory) -> None:
    """Render all discovered result-directory content."""
    workflow = infer_workflow(inventory)
    st.header("Result browser")
    st.caption(f"Directory: `{inventory.root}`")
    st.info(f"Detected workflow: **{workflow.label}** — {workflow.description}")

    columns = st.columns(5)
    with columns[0]:
        _summary_metric("Tables", len(inventory.tables))
    with columns[1]:
        _summary_metric("Plots", len(inventory.plots))
    with columns[2]:
        _summary_metric("Logs", len(inventory.logs))
    with columns[3]:
        _summary_metric("Reports", len(inventory.reports))
    with columns[4]:
        _summary_metric("Artifacts", len(inventory.artifacts))

    if inventory.missing_expected:
        with st.expander("Missing standard contract files/folders", expanded=False):
            st.write("The browser can still show recognised legacy outputs, but these standard items were not found:")
            st.code("\n".join(inventory.missing_expected))

    tabs = st.tabs(["Metadata", "Tables", "Plots", "Logs", "Reports", "Artifacts", "Download"])
    with tabs[0]:
        render_metadata(inventory.metadata)
        render_text_file("Command", inventory.command, language="bash")
        render_text_file("Resolved config", inventory.config, language="yaml")
    with tabs[1]:
        render_tables(inventory.tables)
    with tabs[2]:
        render_plots(inventory.plots)
    with tabs[3]:
        render_logs(inventory.logs)
    with tabs[4]:
        render_reports(inventory.reports)
    with tabs[5]:
        _render_downloadable_list("Artifacts", inventory.artifacts)
    with tabs[6]:
        render_download_panel(inventory)


def _render_downloadable_list(label: str, files) -> None:
    st.subheader(label)
    if not files:
        st.info(f"No {label.lower()} were found in the selected result directory.")
        return
    selected = st.selectbox(label, files, format_func=lambda item: item.relative_path)
    st.write(f"Selected: `{selected.relative_path}`")
    with selected.path.open("rb") as handle:
        st.download_button("Download", data=handle.read(), file_name=selected.name, key=f"download-{label}-{selected.relative_path}")
