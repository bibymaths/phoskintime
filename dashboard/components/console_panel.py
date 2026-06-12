from __future__ import annotations


def render_console(lines: list[str], height: int = 360) -> None:
    """Render streamed ANSI-colored console output in Streamlit."""
    import streamlit as st
    from ansi2html import Ansi2HTMLConverter

    raw_text = "".join(lines)

    converter = Ansi2HTMLConverter(
        inline=True,
        escaped=True,
        scheme="ansi2html",
    )
    html = converter.convert(raw_text, full=False)

    st.markdown(
        f"""
        <div style="
            height: {height}px;
            overflow-y: auto;
            background-color: #0e1117;
            color: #f5f5f5;
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid #30363d;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.85rem;
            line-height: 1.35;
            white-space: pre-wrap;
        ">{html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_cancellation_note() -> None:
    """Explain the current cancellation boundary without exposing unsafe controls."""
    import streamlit as st

    st.caption(
        "Cancellation is not exposed in this dashboard phase: Streamlit reruns make reliable foreground "
        "process termination fragile without a background job supervisor. The runner supports a cancellation "
        "callback for future supervised execution."
    )
