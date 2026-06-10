from __future__ import annotations

from pathlib import Path

from dashboard.workflow_panels.common import WorkflowPanelData, _inventory

PROTWISE_KEYWORDS = ("fit", "residual", "sensitivity", "prediction", "model_error", "regularization")


def discover_protwise_panel(root: str | Path) -> WorkflowPanelData:
    """Discover ProtWise tables, plots, sensitivity outputs, and reports."""
    root = Path(root).resolve()
    inventory = _inventory(root)
    tables = {
        item.relative_path: item.path
        for item in inventory.tables
        if any(key in item.name.lower() for key in PROTWISE_KEYWORDS) or item.suffix in {".xlsx", ".csv"}
    }
    plots = [item for item in inventory.plots if any(key in item.name.lower() for key in PROTWISE_KEYWORDS)] or inventory.plots
    messages = [] if (tables or plots or inventory.reports) else ["No ProtWise-specific outputs were discovered; run the model first or select another result directory."]
    return WorkflowPanelData(root=root, tables=tables, plots=plots, reports=inventory.reports, artifacts=inventory.artifacts, messages=messages)


def render(root: str | Path) -> None:
    import pandas as pd
    import streamlit as st

    data = discover_protwise_panel(root)
    st.subheader("ProtWise panel")
    for message in data.messages:
        st.info(message)
    for label, path in data.tables.items():
        with st.expander(label):
            try:
                if path.suffix == ".csv":
                    st.dataframe(pd.read_csv(path), use_container_width=True)
                elif path.suffix in {".xlsx", ".xls"}:
                    st.dataframe(pd.read_excel(path), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not read {label}: {exc}")
    for item in data.plots[:12]:
        if item.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            st.image(str(item.path), caption=item.relative_path, use_container_width=True)
        else:
            st.write(item.relative_path)
