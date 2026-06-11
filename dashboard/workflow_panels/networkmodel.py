from __future__ import annotations

from pathlib import Path

from dashboard.workflow_panels.common import WorkflowPanelData, _first_existing, _inventory

NETWORK_TABLES = (
    "scalar_objective.csv",
    "convergence_history.csv",
    "pred_prot_picked.csv",
    "pred_rna_picked.csv",
    "pred_phospho_picked.csv",
    "picked_objectives.json",
)


def discover_networkmodel_panel(root: str | Path) -> WorkflowPanelData:
    """Discover networkmodel bundle, scalar objective, predictions, and inference outputs."""
    root = Path(root).resolve()
    inventory = _inventory(root)
    bundle = _first_existing(root, ("dashboard_bundle.pkl", "networkmodel_dashboard_bundle.pkl"))
    tables = {}
    for name in NETWORK_TABLES:
        found = _first_existing(root, (name,))
        if found:
            tables[name] = found
    for item in inventory.tables:
        if item.relative_path.startswith(("optimization/", "profiles/", "posterior/")):
            tables[item.relative_path] = item.path
    messages = [] if (bundle or tables or inventory.plots) else ["No networkmodel bundle, scalar objective, predictions, or inference outputs were found."]
    return WorkflowPanelData(root=root, primary_result=bundle, tables=tables, plots=inventory.plots, reports=inventory.reports, artifacts=inventory.artifacts, messages=messages)


def render(root: str | Path) -> None:
    import pandas as pd
    import streamlit as st

    data = discover_networkmodel_panel(root)
    st.subheader("Networkmodel panel")
    for message in data.messages:
        st.info(message)
    if data.primary_result:
        st.success(f"Dashboard bundle found: {data.primary_result}")
        try:
            from networkmodel.dashboard_bundle import load_dashboard_bundle
            bundle = load_dashboard_bundle(data.root)
            st.json({"bundle_keys": sorted(bundle.keys())})
        except Exception as exc:
            st.warning(f"Bundle exists but could not be loaded here: {exc}")
    for label, path in data.tables.items():
        with st.expander(label, expanded=label == "scalar_objective.csv"):
            try:
                if path.suffix == ".csv":
                    st.dataframe(pd.read_csv(path), use_container_width=True)
                elif path.suffix == ".json":
                    st.json(__import__("json").loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                st.warning(f"Could not read {label}: {exc}")
    if data.plots:
        st.write("Networkmodel plots")
        for item in data.plots[:12]:
            if item.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                st.image(str(item.path), caption=item.relative_path, use_container_width=True)
            else:
                st.write(item.relative_path)
