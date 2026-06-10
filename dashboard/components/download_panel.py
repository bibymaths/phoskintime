from __future__ import annotations

import streamlit as st

from dashboard.file_utils import create_result_zip
from dashboard.result_parser import ResultInventory


def render_download_panel(inventory: ResultInventory) -> None:
    st.subheader("Download")
    st.caption("Create a ZIP archive of the selected result directory. The archive is generated in memory and not written to disk.")
    if st.button("Prepare ZIP archive"):
        zip_bytes = create_result_zip(inventory.root)
        st.download_button(
            "Download result ZIP",
            data=zip_bytes,
            file_name=f"{inventory.root.name or 'phoskintime_results'}.zip",
            mime="application/zip",
        )
