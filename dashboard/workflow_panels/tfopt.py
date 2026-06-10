from __future__ import annotations

from pathlib import Path

from dashboard.workflow_panels.common import WorkflowPanelData, _first_existing, _inventory

TFOPT_SHEETS = ("Alpha Values", "Beta Values", "Observed", "Estimated", "Residuals", "Optimization Results")


def discover_tfopt_panel(root: str | Path) -> WorkflowPanelData:
    """Discover TFOpt result files and generated displays."""
    root = Path(root).resolve()
    inventory = _inventory(root)
    result = _first_existing(root, ("tfopt_results.xlsx",))
    tables = {"tfopt_results": result} if result else {}
    messages = [] if result else ["No tfopt_results.xlsx file found in the result directory or tables/."]
    return WorkflowPanelData(root=root, primary_result=result, tables=tables, plots=inventory.plots, reports=inventory.reports, artifacts=inventory.artifacts, messages=messages)


def render(root: str | Path) -> None:
    """Render TFOpt-specific outputs without recomputing analyses."""
    import pandas as pd
    import streamlit as st

    data = discover_tfopt_panel(root)
    st.subheader("TFOpt panel")
    for message in data.messages:
        st.info(message)
    if data.primary_result:
        st.caption(f"Workbook: `{data.primary_result}`")
        for sheet in TFOPT_SHEETS:
            try:
                df = pd.read_excel(data.primary_result, sheet_name=sheet)
            except Exception:
                continue
            with st.expander(sheet, expanded=sheet in {"Alpha Values", "Beta Values"}):
                st.dataframe(df, use_container_width=True)
    if data.plots:
        st.write("Existing plots and latent/dominance/knockout exports")
        for item in data.plots[:12]:
            if item.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                st.image(str(item.path), caption=item.relative_path, use_container_width=True)
            else:
                st.write(item.relative_path)
