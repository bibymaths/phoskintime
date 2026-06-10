from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dashboard.file_utils import read_text_preview


def render_metadata(path: Path | None) -> None:
    """Render metadata.json with a clear missing-state message."""
    st.subheader("Metadata")
    if path is None:
        st.info("No metadata.json file was found in this result directory.")
        return
    try:
        st.json(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        text, truncated = read_text_preview(path)
        st.warning("metadata.json is not valid JSON; showing raw content instead.")
        st.code(text, language="json")
        if truncated:
            st.caption("Preview truncated for display.")
