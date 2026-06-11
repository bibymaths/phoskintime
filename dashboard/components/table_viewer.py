from __future__ import annotations

from typing import Any

from dashboard.file_utils import DisplayFile, human_size


def _read_table(path: str, suffix: str, sheet_name: str | None = None) -> Any:
    import pandas as pd

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError(f"Unsupported table suffix: {suffix}")


def _excel_sheets(path: str) -> list[str]:
    import pandas as pd

    return pd.ExcelFile(path).sheet_names


def render_tables(tables: list[DisplayFile]) -> None:
    import streamlit as st

    st.subheader("Tables")
    if not tables:
        st.info("No CSV, TSV, or Excel tables were found in tables/ or recognised legacy locations.")
        return
    selected = st.selectbox("Table", tables, format_func=lambda item: f"{item.relative_path} ({human_size(item.size_bytes)})")
    sheet_name = None
    if selected.suffix in {".xlsx", ".xls"}:
        try:
            sheets = _excel_sheets(str(selected.path))
        except Exception as exc:
            st.error(f"Could not inspect Excel workbook {selected.relative_path}: {exc}")
            return
        sheet_name = st.selectbox("Sheet", sheets)
    try:
        st.dataframe(_read_table(str(selected.path), selected.suffix, sheet_name), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not load table {selected.relative_path}. Check that it is a readable CSV, TSV, or Excel file: {exc}")
