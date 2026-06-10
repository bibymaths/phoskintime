from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.file_utils import DisplayFile, read_text_preview


def render_text_file(label: str, path: Path | None, language: str | None = None) -> None:
    st.subheader(label)
    if path is None:
        st.info(f"No {label} file was found in this result directory.")
        return
    text, truncated = read_text_preview(path)
    st.code(text, language=language)
    if truncated:
        st.caption("Preview truncated for display; download the file to inspect all content.")


def render_logs(logs: list[DisplayFile]) -> None:
    st.subheader("Logs")
    if not logs:
        st.info("No log files were found under logs/ or console.log.")
        return
    selected = st.selectbox("Log file", logs, format_func=lambda item: item.relative_path)
    text, truncated = read_text_preview(selected.path)
    st.text_area(selected.relative_path, text, height=420)
    if truncated:
        st.caption("Preview truncated for display; download the file to inspect all content.")
