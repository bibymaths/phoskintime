from __future__ import annotations


def render_console(lines: list[str], height: int = 360) -> None:
    """Render streamed console output."""
    import streamlit as st

    text = "".join(lines)
    st.text_area("Console output", text, height=height, help="Live stdout/stderr from the workflow process.")


def render_cancellation_note() -> None:
    """Explain the current cancellation boundary without exposing unsafe controls."""
    import streamlit as st

    st.caption(
        "Cancellation is not exposed in this dashboard phase: Streamlit reruns make reliable foreground "
        "process termination fragile without a background job supervisor. The runner supports a cancellation "
        "callback for future supervised execution."
    )
