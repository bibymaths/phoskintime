from __future__ import annotations

from hashlib import md5
from pathlib import Path

from dashboard.components.download_panel import render_download_panel
from dashboard.components.log_viewer import render_logs, render_text_file
from dashboard.components.metadata_viewer import render_metadata
from dashboard.components.plot_viewer import render_plots
from dashboard.components.report_viewer import render_reports
from dashboard.components.table_viewer import render_tables
from dashboard.registry import infer_workflow
from dashboard.result_parser import ResultInventory, discover_result_directory


def _path_key(path: Path) -> str:
    return md5(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def _safe_key_part(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _widget_key(prefix: str, *parts: object) -> str:
    return "::".join([prefix, *(_safe_key_part(part) for part in parts if part is not None)])


def _summary_metric(label: str, value: int) -> None:
    import streamlit as st

    st.metric(label, value)


def render_result_browser(inventory: ResultInventory, key_prefix: str = "results-browser") -> None:
    """Render all discovered result-directory content."""
    import streamlit as st

    root_key = _widget_key(key_prefix, _path_key(inventory.root))

    workflow = infer_workflow(inventory)
    st.header("Result browser")
    st.caption(f"Directory: `{inventory.root}`")
    st.info(f"Detected workflow: **{workflow.label}** — {workflow.description}")

    columns = st.columns(6)
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
    with columns[5]:
        _summary_metric("Child runs", len(inventory.child_runs))

    if inventory.missing_expected:
        with st.expander(
                "Missing standard contract files/folders",
                expanded=False,
        ):
            st.write(
                "The browser can still show recognised legacy outputs, "
                "but these standard items were not found:"
            )
            st.code("\n".join(inventory.missing_expected))

    tabs = st.tabs(
        [
            "Metadata",
            "Tables",
            "Plots",
            "Logs",
            "Reports",
            "Artifacts",
            "Child runs",
            "Download",
        ]
    )

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
        _render_downloadable_list(
            "Artifacts",
            inventory.artifacts,
            key_prefix=_widget_key(root_key, "artifacts"),
        )

    with tabs[6]:
        _render_child_runs(
            inventory,
            key_prefix=_widget_key(root_key, "child-runs"),
        )

    with tabs[7]:
        render_download_panel(
            inventory,
            key_prefix=_widget_key(root_key, "download"),
        )


def _render_downloadable_list(label: str, files, key_prefix: str) -> None:
    import streamlit as st

    st.subheader(label)
    if not files:
        st.info(f"No {label.lower()} were found in the selected result directory.")
        return

    selected = st.selectbox(
        label,
        files,
        format_func=lambda item: item.relative_path,
        key=_widget_key(key_prefix, "select"),
    )

    st.write(f"Selected: `{selected.relative_path}`")

    with selected.path.open("rb") as handle:
        st.download_button(
            "Download",
            data=handle.read(),
            file_name=selected.name,
            key=_widget_key(key_prefix, "download", selected.relative_path),
        )


def _render_child_runs(inventory: ResultInventory, key_prefix: str) -> None:
    import streamlit as st

    st.subheader("Child workflow result folders")
    if not inventory.child_runs:
        st.info("No nested workflow result folders were found in the selected directory.")
        return

    selected = st.selectbox(
        "Select child workflow result",
        inventory.child_runs,
        format_func=lambda path: (
            path.relative_to(inventory.root).as_posix()
            if path.is_relative_to(inventory.root)
            else str(path)
        ),
        key=_widget_key(key_prefix, "select", _path_key(inventory.root)),
    )

    st.caption(f"Opening `{selected}`")

    try:
        child_inventory = discover_result_directory(selected)
    except (FileNotFoundError, NotADirectoryError) as exc:
        st.warning(f"Child result directory could not be opened: {exc}")
        return

    render_result_browser(
        child_inventory,
        key_prefix=_widget_key(key_prefix, "child", _path_key(selected)),
    )
