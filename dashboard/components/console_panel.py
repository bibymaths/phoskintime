from __future__ import annotations

import streamlit as st


def render_console(lines: list[str], height: int = 360) -> None:
    """Render streamed console output."""
    text = "".join(lines)
    st.text_area("Console", text, height=height)


def render_cancellation_note() -> None:
    """Explain the current cancellation boundary without exposing unsafe controls."""
    st.caption(
        "Cancellation is not exposed in this dashboard phase: Streamlit reruns make reliable foreground "
        "process termination fragile without a background job supervisor. The runner supports a cancellation "
        "callback for future supervised execution."
    )
