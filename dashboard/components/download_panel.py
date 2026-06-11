from __future__ import annotations

from hashlib import md5
from pathlib import Path

from dashboard.file_utils import create_result_zip
from dashboard.result_parser import ResultInventory


def _path_key(path: Path) -> str:
    return md5(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def _safe_key_part(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _widget_key(prefix: str, *parts: object) -> str:
    return "::".join([prefix, *(_safe_key_part(part) for part in parts if part is not None)])


def render_download_panel(inventory: ResultInventory, key_prefix: str = "download-panel") -> None:
    import streamlit as st

    root = Path(inventory.root)
    root_key = _widget_key(key_prefix, _path_key(root))
    zip_state_key = _widget_key(root_key, "zip-bytes")

    st.subheader("Download")
    st.caption(
        "Create a ZIP archive of the selected result directory. "
        "The archive is generated in memory and not written to disk."
    )

    if st.button(
        "Prepare ZIP archive",
        key=_widget_key(root_key, "prepare-zip"),
        help="Package all files in this result directory for download.",
    ):
        try:
            st.session_state[zip_state_key] = create_result_zip(root)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            st.error(f"Could not create result ZIP: {exc}")
            return

    if zip_state_key in st.session_state:
        st.download_button(
            "Download result ZIP",
            data=st.session_state[zip_state_key],
            file_name=f"{root.name or 'phoskintime_results'}.zip",
            mime="application/zip",
            key=_widget_key(root_key, "download-zip"),
        )