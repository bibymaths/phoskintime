from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.file_utils import DisplayFile, human_size


@st.cache_data(show_spinner=False)
def _read_table(path: str, suffix: str, sheet_name: str | None = None) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError(f"Unsupported table suffix: {suffix}")


@st.cache_data(show_spinner=False)
def _excel_sheets(path: str) -> list[str]:
    return pd.ExcelFile(path).sheet_names


def render_tables(tables: list[DisplayFile]) -> None:
    st.subheader("Tables")
    if not tables:
        st.info("No CSV, TSV, or Excel tables were found in tables/ or recognised legacy locations.")
        return
    selected = st.selectbox("Table", tables, format_func=lambda item: f"{item.relative_path} ({human_size(item.size_bytes)})")
    sheet_name = None
    if selected.suffix in {".xlsx", ".xls"}:
        sheets = _excel_sheets(str(selected.path))
        sheet_name = st.selectbox("Sheet", sheets)
    try:
        st.dataframe(_read_table(str(selected.path), selected.suffix, sheet_name), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not load table {selected.relative_path}: {exc}")
