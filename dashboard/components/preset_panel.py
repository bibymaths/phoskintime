from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.config_utils import DashboardSelection, save_preset


def render_preset_panel(selection: DashboardSelection, repo_root: Path) -> None:
    """Allow saving the resolved dashboard selections as a preset file."""
    st.subheader("Preset")
    preset_name = st.text_input("Preset filename", value=f"{selection.run_name}.json")
    if st.button("Save preset"):
        target = repo_root / "dashboard_uploads" / selection.run_name / preset_name
        try:
            path = save_preset(selection, target)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(f"Saved preset to {path}")
    st.download_button(
        "Download preset JSON",
        data=__import__("json").dumps(selection.to_dict(), indent=2, sort_keys=True),
        file_name=f"{selection.run_name}.json",
        mime="application/json",
    )
