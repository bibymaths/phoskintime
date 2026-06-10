from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.file_utils import preview_table, read_text_preview, validate_existing_file


def render_input_preview(paths: list[Path]) -> None:
    """Preview uploaded/selected files without modifying them."""
    st.subheader("Input preview")
    if not paths:
        st.info("Upload or select files to preview them here.")
        return
    selected = st.selectbox("Preview file", paths, format_func=lambda path: path.name)
    problems = validate_existing_file(selected)
    if problems:
        st.error("; ".join(problems))
        return
    if selected.suffix.lower() in {".csv", ".tsv", ".xlsx"}:
        try:
            st.dataframe(preview_table(selected), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not preview table: {exc}")
    else:
        text, truncated = read_text_preview(selected)
        st.code(text)
        if truncated:
            st.caption("Preview truncated for display.")
