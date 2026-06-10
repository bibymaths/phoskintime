from __future__ import annotations

from dashboard.command_builder import BuiltCommand


def render_command_preview(built: BuiltCommand) -> None:
    """Show an exact command preview before execution."""
    import streamlit as st

    st.subheader("Command preview")
    st.code(built.preview, language="bash")
    with st.expander("Argument list", expanded=False):
        st.json(built.command)
    st.caption(f"Output directory: `{built.outdir}`")
