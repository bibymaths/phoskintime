from __future__ import annotations

from dashboard.file_utils import DisplayFile, human_size, read_text_preview


def render_plots(plots: list[DisplayFile]) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.subheader("Plots")
    if not plots:
        st.info("No PNG, JPG, JPEG, SVG, or HTML plots were found in plots/ or recognised legacy plot folders.")
        return
    selected = st.selectbox("Plot", plots, format_func=lambda item: f"{item.relative_path} ({human_size(item.size_bytes)})")
    if selected.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        st.image(str(selected.path), caption=selected.relative_path, use_container_width=True)
    elif selected.suffix in {".html", ".htm"}:
        html, truncated = read_text_preview(selected.path, max_bytes=2_000_000)
        components.html(html, height=700, scrolling=True)
        if truncated:
            st.warning("HTML preview was truncated because the file is large. Download the file for the full plot.")
    else:
        st.info("This plot type can be downloaded but is not previewed inline.")
