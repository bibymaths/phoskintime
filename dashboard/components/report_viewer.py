from __future__ import annotations

import base64

from dashboard.file_utils import DisplayFile, human_size, read_text_preview


def render_reports(reports: list[DisplayFile]) -> None:
    """Render report files where Streamlit can preview them."""
    import streamlit as st
    import streamlit.components.v1 as components

    st.subheader("Reports")
    if not reports:
        st.info("No HTML, Markdown, or PDF reports were found in reports/.")
        return
    selected = st.selectbox("Report", reports,
                            format_func=lambda item: f"{item.relative_path} ({human_size(item.size_bytes)})")
    if selected.suffix in {".html", ".htm"}:
        html, truncated = read_text_preview(selected.path, max_bytes=2_000_000)
        components.html(html, height=700, scrolling=True)
        if truncated:
            st.warning(
                "HTML report preview was truncated because the file is large. Download the file for the full report.")
    elif selected.suffix == ".md":
        markdown, truncated = read_text_preview(selected.path, max_bytes=1_000_000)
        st.markdown(markdown)
        if truncated:
            st.warning(
                "Markdown report preview was truncated because the file is large. Download the file for the full report.")
    elif selected.suffix == ".pdf":
        pdf_bytes = selected.path.read_bytes()
        encoded = base64.b64encode(pdf_bytes).decode("ascii")
        components.html(
            f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="700"></iframe>',
            height=720,
            scrolling=True,
        )
    with selected.path.open("rb") as handle:
        st.download_button("Download report", data=handle.read(), file_name=selected.name)
