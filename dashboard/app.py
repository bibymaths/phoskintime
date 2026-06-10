from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.components.result_browser import render_result_browser
from dashboard.result_parser import discover_result_directory


def _candidate_result_dirs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    candidates = [base]
    candidates.extend(path for path in sorted(base.iterdir()) if path.is_dir())
    return candidates


def main() -> None:
    st.set_page_config(page_title="PhosKinTime Result Browser", layout="wide")
    st.title("PhosKinTime Result Browser")
    st.write("Browse existing PhosKinTime result directories without launching workflows or uploading files.")

    with st.sidebar:
        st.header("Result directory")
        base = Path(st.text_input("Base results folder", value="results")).expanduser()
        candidates = _candidate_result_dirs(base)
        if candidates:
            choice = st.selectbox("Select folder", candidates, format_func=lambda path: str(path))
            directory_text = st.text_input("Selected result directory", value=str(choice))
        else:
            st.info("No selectable folders found under the base path. Enter a result directory manually.")
            directory_text = st.text_input("Selected result directory", value=str(base))

    try:
        inventory = discover_result_directory(directory_text)
    except (FileNotFoundError, NotADirectoryError) as exc:
        st.error(str(exc))
        return

    if not inventory.has_content:
        st.warning("This directory exists, but no standard PhosKinTime result files were discovered.")
    render_result_browser(inventory)


if __name__ == "__main__":
    main()
