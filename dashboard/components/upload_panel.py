from __future__ import annotations

from pathlib import Path

from dashboard.file_utils import (
    UPLOAD_EXTENSIONS,
    create_upload_dir,
    detect_duplicate_filenames,
    save_uploaded_file,
    validate_upload_filename,
)


def render_upload_panel(repo_root: Path, run_name: str) -> list[Path]:
    """Render upload controls and save files under dashboard_uploads/<run_id>/ on demand."""
    import streamlit as st

    st.subheader("Upload inputs")
    uploaded = st.file_uploader(
        "Upload input/config files",
        type=sorted(ext.lstrip(".") for ext in UPLOAD_EXTENSIONS),
        accept_multiple_files=True,
        help="Files are copied to dashboard_uploads/<run name>/ and are not written into data/ automatically.",
    )
    saved: list[Path] = []
    if not uploaded:
        return saved

    duplicates = detect_duplicate_filenames(file.name for file in uploaded)
    if duplicates:
        st.error("Duplicate filenames after sanitization: " + ", ".join(duplicates))
        return saved

    upload_dir = create_upload_dir(repo_root, run_name)
    for file in uploaded:
        problems = validate_upload_filename(file.name)
        if problems:
            st.error("; ".join(problems))
            continue
        try:
            path = save_uploaded_file(file, upload_dir)
        except ValueError as exc:
            st.error(str(exc))
            continue
        saved.append(path)
    if saved:
        st.success(f"Saved {len(saved)} file(s) to {upload_dir}")
    return saved
