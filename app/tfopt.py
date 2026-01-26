#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard for TFopt network readout + visualization + network rendering (gravis).

This app implements the same logic as:
  - tfopt_network_readout.py
  - tfopt_network_viz.py

Key adaptations
---------------
1) Single source of truth: tfopt_results.xlsx (no input2; no intermediate saved files required).
2) All analysis tables are computed in-memory from XLSX and shown as:
     - tables (st.dataframe) + CSV download buttons
     - Plotly visualizations (in-app equivalents of the matplotlib viz script)
3) Network rendering:
     - TF -> mRNA network rendered via gravis
     - Knockout preview (remove a TF) rendered via gravis
     - EGFR control logic (TF -> EGFR) rendered via gravis with edge width ~ |KO effect|
4) All options previously hard-coded are configurable
   through sliders/selectboxes with defaults matching your scripts.

Expected XLSX layout (minimal)
------------------------------
Sheets:
  - Observed:   columns [mRNA, x1..x9]
  - Estimated:  columns [mRNA, x1..x9]
  - Alpha Values: TF -> mRNA weights with columns that can be mapped to (TF, mRNA, Value)
  - Beta Values: TF betas with columns that can be mapped to (TF, PSite, Value)

Optional:
  - Edges-like sheet: (Source, Target) OR (TF, mRNA) OR synonyms.
    If missing, selecting "Alpha Values" as edges is valid: edges become TF->mRNA pairs from alpha.

input1.csv (required by your current logic)
------------------------------------------
Columns:
  - GeneID (or TF/Gene/Source)
  - PSite
  - x1..x14

Time grids
----------
mRNA grid x1..x9:  [4, 8, 15, 30, 60, 120, 240, 480, 960] min
TF grid  x1..x14:  [0, 0.5, 0.75, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480, 960] min
Interpolation onto mRNA grid: selectable (default: cubic)

Run
---
  streamlit run app/tfopt.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import networkx as nx
import gravis as gv

from scipy.integrate import trapezoid
from scipy.interpolate import interp1d


# =========================
# App configuration
# =========================

st.set_page_config(page_title="TFopt Network Readout Dashboard", layout="wide")
EPS = 1e-12


# =========================
# Defaults (match scripts)
# =========================

MRNA_TIME_COLS_DEFAULT = tuple(f"x{i}" for i in range(1, 10))
MRNA_TIME_POINTS_DEFAULT = tuple(float(x) for x in [4, 8, 15, 30, 60, 120, 240, 480, 960])

TF_TIME_COLS_DEFAULT = tuple(f"x{i}" for i in range(1, 15))
TF_TIME_POINTS_DEFAULT = tuple(
    float(x)
    for x in [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)


# =========================
# Helper functions
# =========================

def _standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Have: {list(df.columns)}")


def _find_first_sheet(sheet_names: list[str], candidates: list[str]) -> str:
    sheets = set(sheet_names)
    for c in candidates:
        if c in sheets:
            return c
    raise ValueError(f"Could not find any sheet in {candidates}. Available: {sheet_names}")


def auc_abs(y: np.ndarray, t: np.ndarray) -> float:
    return float(trapezoid(np.abs(y), t))


def peak_abs(y: np.ndarray) -> float:
    return float(np.max(np.abs(y))) if len(y) else 0.0


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def compute_time_windows_quantiles(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.quantile(t, 1 / 3)
    q2 = np.quantile(t, 2 / 3)
    early = np.where(t <= q1)[0]
    mid = np.where((t > q1) & (t <= q2))[0]
    late = np.where(t > q2)[0]
    return early, mid, late


@dataclass(frozen=True)
class TimeConfig:
    mrna_time_cols: tuple[str, ...]
    mrna_time_points: tuple[float, ...]
    tf_time_cols: tuple[str, ...]
    tf_time_points: tuple[float, ...]


@dataclass(frozen=True)
class AppParams:
    beta_bound: float
    bound_atol: float
    interp_kind: str
    restrict_alpha_to_edges: bool
    edges_sheet_name: str
    top_tf_bar: int
    top_targets: int
    max_point_labels: int
    outlier_band_quantile: float
    top_ko_per_target: int
    egfr_topk: int
    min_alpha_filter: float
    ko_metric: str  # "delta_auc_abs" or "delta_peak_abs"


# =========================
# XLSX readers (SAFE for caching)
# =========================

@st.cache_data(show_spinner=False)
def get_sheet_names(xlsx_bytes: bytes) -> list[str]:
    xl = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    return list(xl.sheet_names)


@st.cache_data(show_spinner=False)
def read_excel_sheet(xlsx_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=sheet_name)


# =========================
# Loaders from XLSX
# =========================

def load_observed_estimated(xlsx_bytes: bytes, sheet_names: list[str], mrna_time_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_name = _find_first_sheet(sheet_names, ["Observed", "observed", "OBSERVED"])
    est_name = _find_first_sheet(sheet_names, ["Estimated", "estimated", "ESTIMATED"])

    obs = _standardize_colnames(read_excel_sheet(xlsx_bytes, obs_name))
    est = _standardize_colnames(read_excel_sheet(xlsx_bytes, est_name))

    _require_cols(obs, ["mRNA"] + mrna_time_cols, obs_name)
    _require_cols(est, ["mRNA"] + mrna_time_cols, est_name)

    obs = obs.set_index("mRNA")[mrna_time_cols].astype(float)
    est = est.set_index("mRNA")[mrna_time_cols].astype(float)
    return obs, est


def load_alpha(xlsx_bytes: bytes, sheet_names: list[str]) -> pd.DataFrame:
    alpha_name = _find_first_sheet(sheet_names, ["Alpha Values", "Alpha", "alpha", "ALPHA"])
    a = _standardize_colnames(read_excel_sheet(xlsx_bytes, alpha_name))

    colmap = {}
    for c in a.columns:
        lc = str(c).strip().lower()
        if lc in {"mrna", "target", "gene", "geneid"}:
            colmap[c] = "mRNA"
        elif lc in {"tf", "source", "regulator"}:
            colmap[c] = "TF"
        elif lc in {"value", "alpha", "a"}:
            colmap[c] = "Value"
    a = a.rename(columns=colmap)

    _require_cols(a, ["mRNA", "TF", "Value"], alpha_name)
    a["mRNA"] = a["mRNA"].astype(str)
    a["TF"] = a["TF"].astype(str)
    a["Value"] = pd.to_numeric(a["Value"], errors="coerce").fillna(0.0).astype(float)
    return a[["mRNA", "TF", "Value"]]


def load_beta(xlsx_bytes: bytes, sheet_names: list[str]) -> pd.DataFrame:
    beta_name = _find_first_sheet(sheet_names, ["Beta Values", "Beta", "beta", "BETA"])
    b = _standardize_colnames(read_excel_sheet(xlsx_bytes, beta_name))

    colmap = {}
    for c in b.columns:
        lc = str(c).strip().lower()
        if lc in {"tf", "gene", "geneid", "source"}:
            colmap[c] = "TF"
        elif lc in {"psite", "psites", "site"}:
            colmap[c] = "PSite"
        elif lc in {"value", "beta", "b"}:
            colmap[c] = "Value"
    b = b.rename(columns=colmap)

    _require_cols(b, ["TF", "PSite", "Value"], beta_name)
    b["TF"] = b["TF"].astype(str)
    b["PSite"] = b["PSite"].astype(object)
    b["Value"] = pd.to_numeric(b["Value"], errors="coerce").fillna(0.0).astype(float)
    return b[["TF", "PSite", "Value"]]


def load_network_edges_from_sheet(xlsx_bytes: bytes, sheet_names: list[str], sheet_name: str) -> pd.DataFrame:
    """
    Edges can be any of the following column conventions:
      - Source, Target
      - TF, mRNA   (Alpha Values sheet works here)
      - Regulator, Target / Regulator, mRNA / TF, Target, etc.

    Always returns DataFrame with columns ["Source","Target"] (TF->mRNA).
    """
    if sheet_name not in sheet_names:
        raise ValueError(f"Edges sheet '{sheet_name}' not found. Available: {sheet_names}")

    edges = _standardize_colnames(read_excel_sheet(xlsx_bytes, sheet_name))

    # normalize column names to Source/Target
    cols_lc = {c: str(c).strip().lower() for c in edges.columns}

    src_candidates = {"source", "tf", "regulator", "parent", "from"}
    tgt_candidates = {"target", "mrna", "gene", "child", "to"}

    src_col = None
    tgt_col = None

    for c, lc in cols_lc.items():
        if lc in src_candidates and src_col is None:
            src_col = c
        if lc in tgt_candidates and tgt_col is None:
            tgt_col = c

    # If we still didn't find, try common exact pairs
    if src_col is None and "TF" in edges.columns:
        src_col = "TF"
    if tgt_col is None and "mRNA" in edges.columns:
        tgt_col = "mRNA"

    if src_col is None or tgt_col is None:
        raise ValueError(
            f"{sheet_name} must contain an edge list. "
            f"Need Source/Target (or TF/mRNA). Have: {list(edges.columns)}"
        )

    out = edges.rename(columns={src_col: "Source", tgt_col: "Target"}).copy()
    out["Source"] = out["Source"].astype(str)
    out["Target"] = out["Target"].astype(str)
    return out[["Source", "Target"]].dropna().drop_duplicates()


def restrict_alpha_to_edges(alpha: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    allowed = set(zip(edges["Target"], edges["Source"]))  # (mRNA, TF)
    mask = alpha.apply(lambda r: (r["mRNA"], r["TF"]) in allowed, axis=1)
    return alpha[mask].copy()


# =========================
# input1.csv parsing
# =========================

def parse_input1_csv(input1_bytes: bytes, tf_time_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_input1 = _standardize_colnames(pd.read_csv(io.BytesIO(input1_bytes)))

    tf_col = next((c for c in df_input1.columns if c.lower() in {"tf", "gene", "geneid", "source"}), None)
    if tf_col is None:
        raise ValueError(f"input1.csv needs a TF/Gene column (TF/Gene/GeneID/Source). Have: {list(df_input1.columns)}")

    psite_col = next((c for c in df_input1.columns if c.lower() == "psite"), None)
    if psite_col is None:
        raise ValueError(f"input1.csv needs a PSite column. Have: {list(df_input1.columns)}")

    _require_cols(df_input1, [tf_col, psite_col] + tf_time_cols, "input1.csv")

    df_input1 = df_input1.rename(columns={tf_col: "TF", psite_col: "PSite"}).copy()
    df_input1["TF"] = df_input1["TF"].astype(str)
    df_input1["PSite"] = df_input1["PSite"].astype(object)

    psite_str = df_input1["PSite"].astype(str).str.strip()
    is_prot = df_input1["PSite"].isna() | (psite_str == "") | (psite_str.str.lower().isin({"nan", "none"}))

    prot_df = df_input1[is_prot].drop_duplicates(subset=["TF"], keep="last").copy()
    tf_prot_14 = prot_df.set_index("TF")[tf_time_cols].astype(float)

    ps_df = df_input1[~is_prot].copy()
    ps_df["PSite"] = ps_df["PSite"].astype(str)
    tf_ps_14 = ps_df[["TF", "PSite"] + tf_time_cols].copy()
    for c in tf_time_cols:
        tf_ps_14[c] = pd.to_numeric(tf_ps_14[c], errors="coerce").fillna(0.0).astype(float)

    return tf_prot_14, tf_ps_14

# =========================
# input3.csv parsing
# =========================
def parse_input3_csv(input3_bytes: bytes, mrna_time_cols: list[str]) -> pd.DataFrame:
    """
    input3.csv fallback protein-level TF time series already on mRNA grid (x1..x9).
    Columns: GeneID, x1..x9
    Returns: DataFrame indexed by TF with columns mrna_time_cols (float).
    """
    df = _standardize_colnames(pd.read_csv(io.BytesIO(input3_bytes)))

    tf_col = next((c for c in df.columns if c.lower() in {"geneid", "tf", "gene", "source"}), None)
    if tf_col is None:
        raise ValueError(f"input3.csv needs GeneID/TF/Gene column. Have: {list(df.columns)}")

    _require_cols(df, [tf_col] + mrna_time_cols, "input3.csv")

    df = df.rename(columns={tf_col: "TF"}).copy()
    df["TF"] = df["TF"].astype(str)

    tf_prot_9 = df.drop_duplicates(subset=["TF"], keep="last").set_index("TF")[mrna_time_cols].copy()
    for c in mrna_time_cols:
        tf_prot_9[c] = pd.to_numeric(tf_prot_9[c], errors="coerce").fillna(0.0).astype(float)

    return tf_prot_9


# =========================
# Core computation (in-memory)
# =========================

def interpolate_tf_to_mrna_grid(y14: np.ndarray, t_tf: np.ndarray, t_mrna: np.ndarray, kind: str) -> np.ndarray:
    f = interp1d(
        t_tf,
        y14,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return f(t_mrna)


def build_tf_latent_activity(
    tf_prot_14: pd.DataFrame,
    tf_ps_14: pd.DataFrame,
    beta: pd.DataFrame,
    tcfg: TimeConfig,
    interp_kind: str,
    tf_prot_9_fallback: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    t_tf = np.array(tcfg.tf_time_points, dtype=float)
    t_mrna = np.array(tcfg.mrna_time_points, dtype=float)
    mrna_cols = list(tcfg.mrna_time_cols)
    tf_cols = list(tcfg.tf_time_cols)

    tf_prot_9: Dict[str, np.ndarray] = {}
    for tf in tf_prot_14.index.astype(str):
        y14 = tf_prot_14.loc[tf, tf_cols].to_numpy(dtype=float)
        tf_prot_9[tf] = interpolate_tf_to_mrna_grid(y14, t_tf, t_mrna, kind=interp_kind)

    if tf_prot_9_fallback is not None and not tf_prot_9_fallback.empty:
        for tf in tf_prot_9_fallback.index.astype(str):
            if tf not in tf_prot_9:  # only fill missing TFs
                tf_prot_9[tf] = tf_prot_9_fallback.loc[tf, mrna_cols].to_numpy(dtype=float)

    ps_map: Dict[Tuple[str, str], np.ndarray] = {}
    for r in tf_ps_14.itertuples(index=False):
        tf = str(getattr(r, "TF"))
        psite = str(getattr(r, "PSite"))
        y14 = np.array([getattr(r, c) for c in tf_cols], dtype=float)
        y9 = interpolate_tf_to_mrna_grid(y14, t_tf, t_mrna, kind=interp_kind)
        ps_map[(tf, psite)] = y9

    b2 = beta.copy()
    ps_str = b2["PSite"].astype(str).str.strip()
    has_ps = ~(b2["PSite"].isna() | (ps_str == "") | (ps_str.str.lower().isin({"nan", "none"})))
    tf_psite_stats = (
        b2.assign(has_psite=has_ps)
        .groupby("TF", as_index=False)
        .agg(
            n_beta=("Value", "size"),
            n_psites=("has_psite", "sum"),
            has_any_psite=("has_psite", "max"),
        )
    )

    latent: Dict[str, np.ndarray] = {}

    for tf, btf in beta.groupby("TF", sort=False):
        ps_str = btf["PSite"].astype(str).str.strip()
        is_b0 = btf["PSite"].isna() | (ps_str == "") | (ps_str.str.lower().isin({"nan", "none"}))
        beta0 = float(btf.loc[is_b0, "Value"].iloc[0]) if is_b0.any() else 0.0

        prot = tf_prot_9.get(tf, np.zeros(len(mrna_cols), dtype=float))
        y = beta0 * prot

        for r in btf.loc[~is_b0].itertuples(index=False):
            key = (tf, str(r.PSite))
            if key in ps_map:
                y = y + float(r.Value) * ps_map[key]
        latent[tf] = y

    return latent, tf_psite_stats

def compute_tf_activity_scalars(latent: dict[str, np.ndarray], tcfg: TimeConfig) -> tuple[dict[str, float], dict[str, int]]:
    """
    Returns:
      tf_auc_abs[TF]  = AUC(|A_TF(t)|)
      tf_polarity[TF] = sign(∫ A_TF(t) dt) in {-1,0,+1}
    """
    tf_auc_abs = {}
    tf_polarity = {}

    for tf, y in latent.items():
        y = np.asarray(y, dtype=float)
        tf_auc_abs[tf] = float(trapezoid(np.abs(y), tcfg.mrna_time_points))

        signed_area = float(trapezoid(y, tcfg.mrna_time_points))
        if signed_area > 0:
            tf_polarity[tf] = 1
        elif signed_area < 0:
            tf_polarity[tf] = -1
        else:
            tf_polarity[tf] = 0

    return tf_auc_abs, tf_polarity

def compute_predictions(alpha: pd.DataFrame, latent: dict[str, np.ndarray], nT: int) -> dict[str, np.ndarray]:
    pred: Dict[str, np.ndarray] = {}
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        y = np.zeros(nT, dtype=float)
        for r in grp.itertuples(index=False):
            tf = r.TF
            if tf in latent:
                y += float(r.Value) * latent[tf]
        pred[mrna] = y
    return pred


def compute_tf_load_table(
    alpha: pd.DataFrame,
    beta: pd.DataFrame,
    latent: dict[str, np.ndarray],
    tcfg: TimeConfig,
    beta_bound: float,
    bound_atol: float,
) -> pd.DataFrame:
    t_mrna = np.array(tcfg.mrna_time_points, dtype=float)
    latent_auc = {tf: auc_abs(y, t_mrna) for tf, y in latent.items()}
    latent_peak = {tf: peak_abs(y) for tf, y in latent.items()}

    btf = beta.copy()
    btf["at_bound"] = np.isclose(np.abs(btf["Value"].to_numpy(dtype=float)), beta_bound, atol=bound_atol)

    targets_per_tf = alpha.groupby("TF")["mRNA"].nunique().to_dict()

    rows = []
    for tf in sorted(set(alpha["TF"].astype(str))):
        a_tf = alpha[alpha["TF"] == tf]
        total_load = float(np.sum(np.abs(a_tf["Value"].to_numpy(dtype=float))) * latent_auc.get(tf, 0.0))

        b_sub = btf[btf["TF"] == tf]
        frac_bound = float(b_sub["at_bound"].mean()) if len(b_sub) else 0.0
        n_bound = int(b_sub["at_bound"].sum()) if len(b_sub) else 0

        rows.append(
            {
                "TF": tf,
                "n_targets": int(targets_per_tf.get(tf, 0)),
                "total_load_auc_abs": total_load,
                "frac_beta_at_bound": frac_bound,
                "n_beta_at_bound": n_bound,
                "latent_auc_abs": float(latent_auc.get(tf, 0.0)),
                "latent_peak_abs": float(latent_peak.get(tf, 0.0)),
            }
        )

    return pd.DataFrame(rows).sort_values("total_load_auc_abs", ascending=False)


def compute_target_dominance_table(
    alpha: pd.DataFrame,
    latent: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    obs: pd.DataFrame,
    est: pd.DataFrame,
    tcfg: TimeConfig,
) -> pd.DataFrame:
    t_mrna = np.array(tcfg.mrna_time_points, dtype=float)
    mrna_cols = list(tcfg.mrna_time_cols)
    early_idx, mid_idx, late_idx = compute_time_windows_quantiles(t_mrna)

    rows = []
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        contrib = []
        for r in grp.itertuples(index=False):
            tf = r.TF
            if tf in latent:
                contrib.append((tf, float(r.Value) * latent[tf]))

        if not contrib:
            continue

        aucs = [(tf, auc_abs(s, t_mrna)) for tf, s in contrib]
        aucs.sort(key=lambda x: x[1], reverse=True)
        dom_tf, dom_val = aucs[0]
        total = sum(v for _, v in aucs) + EPS
        dom_share = dom_val / total

        def win_dom(idxs):
            tt = t_mrna[idxs]
            w = [(tf, float(trapezoid(np.abs(s[idxs]), tt))) for tf, s in contrib]
            w.sort(key=lambda x: x[1], reverse=True)
            tf0, v0 = w[0]
            tot = sum(v for _, v in w) + EPS
            return tf0, v0 / tot

        dE, sE = win_dom(early_idx)
        dM, sM = win_dom(mid_idx)
        dL, sL = win_dom(late_idx)

        obs_y = obs.loc[mrna].to_numpy(dtype=float) if mrna in obs.index else np.zeros(len(mrna_cols))
        est_y = est.loc[mrna].to_numpy(dtype=float) if mrna in est.index else np.zeros(len(mrna_cols))
        pred_y = pred.get(mrna, np.zeros(len(mrna_cols)))

        rows.append(
            {
                "mRNA": mrna,
                "n_TFs": int(len(grp)),
                "dominant_overall": dom_tf,
                "dominant_overall_share": float(dom_share),
                "dominant_early": dE,
                "dominant_early_share": float(sE),
                "dominant_mid": dM,
                "dominant_mid_share": float(sM),
                "dominant_late": dL,
                "dominant_late_share": float(sL),
                "recon_auc_abs": auc_abs(pred_y, t_mrna),
                "recon_peak_abs": peak_abs(pred_y),
                "est_auc_abs": auc_abs(est_y, t_mrna),
                "est_peak_abs": peak_abs(est_y),
                "obs_auc_abs": auc_abs(obs_y, t_mrna),
                "obs_peak_abs": peak_abs(obs_y),
            }
        )

    return pd.DataFrame(rows).sort_values("dominant_overall_share", ascending=False)


def compute_knockout_table(
    alpha: pd.DataFrame,
    latent: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    tcfg: TimeConfig,
) -> pd.DataFrame:
    t_mrna = np.array(tcfg.mrna_time_points, dtype=float)
    nT = len(tcfg.mrna_time_cols)

    rows = []
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        base = pred.get(mrna, np.zeros(nT))
        base_auc = auc_abs(base, t_mrna)
        base_peak = peak_abs(base)

        for r in grp.itertuples(index=False):
            tf = r.TF
            if tf not in latent:
                continue
            ko_series = base - float(r.Value) * latent[tf]
            ko_auc = auc_abs(ko_series, t_mrna)
            ko_peak = peak_abs(ko_series)

            rows.append(
                {
                    "mRNA": mrna,
                    "KnockedTF": tf,
                    "alpha": float(r.Value),
                    "baseline_auc_abs": float(base_auc),
                    "baseline_peak_abs": float(base_peak),
                    "ko_auc_abs": float(ko_auc),
                    "ko_peak_abs": float(ko_peak),
                    "delta_auc_abs": float(base_auc - ko_auc),
                    "delta_peak_abs": float(base_peak - ko_peak),
                }
            )

    ko = pd.DataFrame(rows)
    if ko.empty:
        return ko

    ko["abs_delta_auc"] = ko["delta_auc_abs"].abs()
    ko["ko_rank_target"] = ko.groupby("mRNA")["abs_delta_auc"].rank(ascending=False, method="dense")
    ko = ko.drop(columns=["abs_delta_auc"]).sort_values(["mRNA", "ko_rank_target"])
    return ko


@st.cache_data(show_spinner=False)
def compute_all(xlsx_bytes: bytes, input1_bytes: bytes, params: AppParams, tcfg: TimeConfig,
                input3_bytes: Optional[bytes] = None) -> dict:
    sheet_names = get_sheet_names(xlsx_bytes)

    obs, est = load_observed_estimated(xlsx_bytes, sheet_names, list(tcfg.mrna_time_cols))
    alpha = load_alpha(xlsx_bytes, sheet_names)
    beta = load_beta(xlsx_bytes, sheet_names)

    edges = None
    if params.restrict_alpha_to_edges:
        # This now supports choosing "Alpha Values" as an edges sheet (TF/mRNA columns).
        edges = load_network_edges_from_sheet(xlsx_bytes, sheet_names, params.edges_sheet_name)
        alpha = restrict_alpha_to_edges(alpha, edges)

    if params.min_alpha_filter > 0:
        alpha = alpha[np.abs(alpha["Value"].astype(float)) >= params.min_alpha_filter].copy()

    tf_prot_14, tf_ps_14 = parse_input1_csv(input1_bytes, list(tcfg.tf_time_cols))

    tf_prot_9_fallback = None
    if input3_bytes is not None:
        tf_prot_9_fallback = parse_input3_csv(input3_bytes, list(tcfg.mrna_time_cols))

    # Simple check: any TF appearing in beta is expected to have a protein series either in input1 (14->9) or input3 (9)
    beta_tfs = set(beta["TF"].astype(str))
    prot_have = set(tf_prot_14.index.astype(str))
    prot_have |= (set(tf_prot_9_fallback.index.astype(str)) if tf_prot_9_fallback is not None else set())
    missing_prot = sorted([tf for tf in beta_tfs if tf not in prot_have])

    if missing_prot:
        st.error(
            f"Missing protein-level TF time series for {len(missing_prot)} TFs used in Beta Values. "
            f"Not found in input1.csv (PSite empty rows) nor input3.csv. Example: {missing_prot[:15]}"
        )
        st.stop()

    latent, tf_psite_stats = build_tf_latent_activity(
        tf_prot_14=tf_prot_14,
        tf_ps_14=tf_ps_14,
        beta=beta,
        tcfg=tcfg,
        interp_kind=params.interp_kind,
        tf_prot_9_fallback=tf_prot_9_fallback,   # <-- add
    )

    pred = compute_predictions(alpha, latent, nT=len(tcfg.mrna_time_cols))

    tf_load = compute_tf_load_table(alpha, beta, latent, tcfg, params.beta_bound, params.bound_atol)
    dom = compute_target_dominance_table(alpha, latent, pred, obs, est, tcfg)
    ko = compute_knockout_table(alpha, latent, pred, tcfg)

    target_cache: Dict[str, dict] = {}
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        tfs = grp["TF"].astype(str).tolist()
        a = grp["Value"].astype(float).to_numpy()
        C = np.zeros((len(tfs), len(tcfg.mrna_time_cols)), dtype=float)
        for i, tf in enumerate(tfs):
            C[i, :] = a[i] * latent.get(tf, np.zeros(len(tcfg.mrna_time_cols)))
        target_cache[mrna] = {
            "TFs": tfs,
            "alpha": a,
            "C": C,
            "pred": pred.get(mrna, np.zeros(len(tcfg.mrna_time_cols))),
            "obs": obs.loc[mrna].to_numpy(dtype=float) if mrna in obs.index else None,
            "est": est.loc[mrna].to_numpy(dtype=float) if mrna in est.index else None,
        }

    return {
        "meta": {"sheet_names": sheet_names},
        "obs": obs,
        "est": est,
        "alpha": alpha,
        "beta": beta,
        "edges": edges,
        "tf_prot_14": tf_prot_14,
        "tf_ps_14": tf_ps_14,
        "latent": latent,
        "pred": pred,
        "tf_psite_stats": tf_psite_stats,
        "tf_load": tf_load,
        "dom": dom,
        "ko": ko,
        "target_cache": target_cache,
    }


# =========================
# Plotly visualization builders
# =========================

def plot_tf_total_load_plotly(tf_load: pd.DataFrame, top_n: int) -> go.Figure:
    d = tf_load.sort_values("total_load_auc_abs", ascending=False).head(min(top_n, len(tf_load))).copy()
    d = d.sort_values("total_load_auc_abs", ascending=True)
    fig = px.bar(
        d, x="total_load_auc_abs", y="TF", orientation="h",
        title="TF total control load (top)",
        labels={"total_load_auc_abs": "Total control load"},
    )
    fig.update_layout(height=max(450, 18 * len(d) + 200), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_tf_bound_pressure_vs_load_plotly(tf_load: pd.DataFrame) -> go.Figure:
    d = tf_load.dropna(subset=["total_load_auc_abs", "frac_beta_at_bound"]).copy()
    d["score_label"] = (d["total_load_auc_abs"].rank(pct=True) + d["frac_beta_at_bound"].rank(pct=True))
    d["label"] = np.where(d["score_label"] >= d["score_label"].quantile(0.95), d["TF"].astype(str), "")
    fig = px.scatter(
        d,
        x="total_load_auc_abs", y="frac_beta_at_bound",
        text="label",
        hover_data=["TF", "n_targets", "latent_auc_abs", "latent_peak_abs", "n_beta_at_bound"],
        title="TF bound pressure vs control load",
        labels={"total_load_auc_abs": "Total control load", "frac_beta_at_bound": "Fraction β at bound"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_target_dominant_tf_share_plotly(dom: pd.DataFrame, top_targets: int) -> go.Figure:
    d = dom.sort_values("dominant_overall_share", ascending=False).head(min(top_targets, len(dom))).copy()
    d["Label"] = d["mRNA"].astype(str) + " (" + d["dominant_overall"].astype(str) + ")"
    d = d.sort_values("dominant_overall_share", ascending=True)
    fig = px.bar(
        d,
        x="dominant_overall_share", y="Label", orientation="h",
        title="Top targets by dominant TF share",
        labels={"dominant_overall_share": "Dominant TF share"},
        hover_data=["mRNA", "n_TFs", "dominant_early", "dominant_mid", "dominant_late"],
    )
    fig.update_layout(height=max(450, 18 * len(d) + 200), margin=dict(l=10, r=10, t=50, b=10), xaxis=dict(range=[0, 1.0]))
    return fig


def plot_obs_vs_recon_plotly(dom: pd.DataFrame, band_quantile: float, max_labels: int) -> go.Figure:
    d = dom.dropna(subset=["obs_auc_abs", "recon_auc_abs"]).copy()
    if d.empty:
        return go.Figure().update_layout(title="Observed vs reconstructed magnitude (no data)")

    d["obs"] = d["obs_auc_abs"].astype(float)
    d["recon"] = d["recon_auc_abs"].astype(float)
    d = d[(d["obs"].abs() > 1e-12) | (d["recon"].abs() > 1e-12)].copy()

    d_nz = d[d["obs"].abs() > 1e-12].copy()
    if d_nz.empty:
        d_nz = d.copy()

    d_nz["perp_dist"] = np.abs(d_nz["recon"] - d_nz["obs"]) / np.sqrt(2)
    thr = float(np.quantile(d_nz["perp_dist"].to_numpy(), band_quantile))
    offset = thr * np.sqrt(2)

    in_band = d_nz[d_nz["perp_dist"] <= thr].copy()
    lab = in_band.sort_values("perp_dist", ascending=True).head(max_labels)

    mx = float(np.nanmax(np.r_[d["obs"].to_numpy(), d["recon"].to_numpy(), 1e-9]))
    xline = np.array([0.0, mx])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["obs"], y=d["recon"], mode="markers",
        marker=dict(size=7),
        hovertext=d["mRNA"].astype(str),
        hoverinfo="text+x+y",
        name="mRNAs"
    ))
    fig.add_trace(go.Scatter(x=xline, y=xline, mode="lines", name="y=x", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=xline, y=xline + offset, mode="lines", name="band +", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=xline, y=xline - offset, mode="lines", name="band -", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(
        x=lab["obs"], y=lab["recon"],
        mode="text",
        text=lab["mRNA"].astype(str),
        textposition="top center",
        name="Labels"
    ))
    fig.update_layout(
        title="Observed vs reconstructed magnitude (AUC|·|)",
        xaxis_title="Observed magnitude",
        yaxis_title="Reconstructed magnitude",
        xaxis=dict(range=[0, mx]),
        yaxis=dict(range=[0, mx], scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_ko_distribution_plotly(ko: pd.DataFrame, metric: str) -> go.Figure:
    if ko.empty:
        return go.Figure().update_layout(title="KO distribution (no data)")
    fig = px.histogram(
        ko,
        x=metric,
        nbins=45,
        histnorm="probability density",
        title=f"Distribution of knockout effects ({metric})",
        labels={metric: "KO effect"},
    )
    fig.add_vline(x=0.0, line_width=2)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_ko_top_edges_plotly(ko: pd.DataFrame, metric: str, direction: str, top_n: int = 20) -> go.Figure:
    if ko.empty:
        return go.Figure().update_layout(title="KO edges (no data)")
    d = ko.copy()
    d["Edge"] = d["mRNA"].astype(str) + " \u2190 " + d["KnockedTF"].astype(str)
    if direction == "positive":
        dd = d.sort_values(metric, ascending=False).head(top_n).copy()
        title = f"Top positive KO effects ({metric})"
    else:
        dd = d.sort_values(metric, ascending=True).head(top_n).copy()
        title = f"Top negative KO effects ({metric})"
    dd = dd.sort_values(metric, ascending=True)

    fig = px.bar(
        dd, x=metric, y="Edge", orientation="h",
        title=title,
        hover_data=["alpha", "baseline_auc_abs", "baseline_peak_abs", "ko_auc_abs", "ko_peak_abs"],
    )
    fig.update_layout(height=max(450, 18 * len(dd) + 200), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_ko_heatmap_plotly(ko: pd.DataFrame, top_ko_per_target: int, metric: str) -> go.Figure:
    if ko.empty:
        return go.Figure().update_layout(title="KO heatmap (no data)")

    d = ko.copy()
    d["metric"] = d[metric].astype(float)

    base = d.groupby("mRNA", as_index=False)["baseline_auc_abs"].max()
    base = base.sort_values("baseline_auc_abs", ascending=False).head(min(18, base.shape[0]))
    targets = base["mRNA"].astype(str).tolist()

    rows = []
    all_tfs = set()
    for t in targets:
        dt = d[d["mRNA"].astype(str) == t].copy()
        dt["absd"] = dt["metric"].abs()
        dt = dt.sort_values("absd", ascending=False).head(top_ko_per_target)
        rows.append(dt)
        all_tfs.update(dt["KnockedTF"].astype(str).tolist())

    if not rows:
        return go.Figure().update_layout(title="KO heatmap (no data after filtering)")

    sub = pd.concat(rows, ignore_index=True)

    # drop TFs with near-zero effect across all targets
    eps = 0.0
    tf_max_effect = (
        sub.groupby("KnockedTF")["metric"]
        .apply(lambda x: np.max(np.abs(x)))
    )

    keep_tfs = tf_max_effect[tf_max_effect > eps].index
    sub = sub[sub["KnockedTF"].isin(keep_tfs)].copy()

    # TF ordering score - sum of absolute effect across all targets
    tf_score = (
        sub.assign(absd=sub["metric"].abs())
        .groupby("KnockedTF")["absd"]
        .sum()
        .sort_values(ascending=False)
    )
    tfs = tf_score.index.astype(str).tolist()

    # M = np.zeros((len(targets), len(tfs)), dtype=float)
    # for i, t in enumerate(targets):
    #     dt = sub[sub["mRNA"].astype(str) == t]
    #     for _, r in dt.iterrows():
    #         j = tfs.index(str(r["KnockedTF"]))
    #         M[i, j] = float(r["metric"])

    M = np.zeros((len(tfs), len(targets)), dtype=float)

    tf_to_i = {tf: i for i, tf in enumerate(tfs)}
    target_to_j = {t: j for j, t in enumerate(targets)}

    for _, r in sub.iterrows():
        tf = str(r["KnockedTF"])
        t = str(r["mRNA"])
        if tf in tf_to_i and t in target_to_j:
            i = tf_to_i[tf]
            j = target_to_j[t]
            M[i, j] = float(r["metric"])

    fig = go.Figure(
        data=go.Heatmap(
            z=M,
            x=targets,
            y=tfs,
            colorbar=dict(title=metric),
        )
    )
    fig.update_layout(
        title=f"KO landscape heatmap (top targets; top-{top_ko_per_target} TFs/target)",
        height=min(750, 30 * len(targets) + 260),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_latent_tf_timeseries_plotly(latent: dict[str, np.ndarray], tcfg: TimeConfig, tf: str) -> go.Figure:
    y = latent.get(tf)
    if y is None:
        return go.Figure().update_layout(title=f"Latent TF activity: {tf} (not available)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(tcfg.mrna_time_points), y=y, mode="lines+markers", name="A_TF(t)"))
    fig.update_layout(
        title=f"Latent TF activity A_TF(t): {tf}",
        xaxis_title="Time (min)",
        yaxis_title="Activity (arb.)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_target_reconstruction_timeseries_plotly(tcfg: TimeConfig, mrna: str, cache: dict, top_k: int = 8) -> go.Figure:
    pred = cache["pred"]
    obs = cache["obs"]
    est = cache["est"]
    C = cache["C"]
    tfs = cache["TFs"]

    t_mrna = np.array(tcfg.mrna_time_points, dtype=float)

    fig = go.Figure()
    if obs is not None:
        fig.add_trace(go.Scatter(x=t_mrna, y=obs, mode="lines+markers", name="Observed"))
    if est is not None:
        fig.add_trace(go.Scatter(x=t_mrna, y=est, mode="lines+markers", name="Estimated"))
    fig.add_trace(go.Scatter(x=t_mrna, y=pred, mode="lines+markers", name="Reconstructed (sum)"))

    if len(tfs) > 0:
        contrib_auc = np.array([auc_abs(C[i, :], t_mrna) for i in range(len(tfs))], dtype=float)
        order = np.argsort(-contrib_auc)[: min(top_k, len(tfs))]
        for i in order:
            fig.add_trace(go.Scatter(
                x=t_mrna, y=C[i, :],
                mode="lines",
                name=f"{tfs[i]} contrib",
                opacity=0.75
            ))

    fig.update_layout(
        title=f"mRNA reconstruction & TF contributions: {mrna}",
        xaxis_title="Time (min)",
        yaxis_title="Signal (arb.)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# =========================
# Network rendering (gravis)
# =========================

def build_tf_mrna_network(alpha: pd.DataFrame, tf_auc_abs: dict[str, float], tf_polarity: dict[str, int]) -> nx.DiGraph:
    """
    TF -> mRNA with effective sign and strength:
      strength = |alpha| * AUC(|A_TF|)
      sign     = sign(alpha) * polarity(TF)
    """
    G = nx.DiGraph()

    for _, r in alpha.iterrows():
        tf = str(r["TF"])
        mrna = str(r["mRNA"])
        a = float(r["Value"])

        # TF activity scalars (from beta-driven latent)
        auc_tf = float(tf_auc_abs.get(tf, 0.0))
        pol_tf = int(tf_polarity.get(tf, 0))

        # effective edge effect
        strength = abs(a) * auc_tf

        if a > 0:
            sign_a = 1
        elif a < 0:
            sign_a = -1
        else:
            sign_a = 0

        eff_sign = sign_a * pol_tf  # -1 suppressing, +1 activating, 0 unknown/neutral

        if eff_sign > 0:
            color = "#2ca02c"  # activating
            dashed = False
        elif eff_sign < 0:
            color = "#d62728"  # suppressing
            dashed = True
        else:
            color = "#7f7f7f"  # neutral/unknown
            dashed = False

        G.add_node(tf, kind="tf")
        G.add_node(mrna, kind="mrna")
        G.add_edge(
            tf, mrna,
            weight=strength,      # now uses beta through latent
            alpha=a,              # keep raw alpha for labels/hover
            tf_auc_abs=auc_tf,
            tf_polarity=pol_tf,
            eff_sign=int(eff_sign),
            color=color,
            dashed=dashed,
        )

    return G

def render_gravis(G: nx.DiGraph, node_size: int, edge_scale: float, edge_label: bool) -> str:
    """
    Render via gravis d3 and return HTML for Streamlit embedding.
    Uses:
      - width  from abs(weight)
      - color  from edge['color'] (set in build_tf_mrna_network)
      - dashed from edge['dashed'] (best-effort; depends on gravis version)
    """
    H = G.copy()

    ws = [abs(float(d.get("weight", 1.0))) for _, _, d in H.edges(data=True)]
    wmax = max(ws) if ws else 1.0

    for u, v, d in H.edges(data=True):
        w = abs(float(d.get("weight", 1.0)))
        d["w"] = 1.0 + 6.0 * (w / (wmax + EPS)) * edge_scale

        # label showing signed alpha
        if edge_label:
            d["label"] = f"{float(d.get('alpha', 0.0)):.3g}"

        # ensure color exists even if upstream didn’t set it
        if "color" not in d:
            a = float(d.get("alpha", 0.0))
            d["color"] = "#2ca02c" if a > 0 else ("#d62728" if a < 0 else "#7f7f7f")
            d["dashed"] = bool(a < 0)

    for n in H.nodes():
        deg = H.degree(n)
        H.nodes[n]["size"] = float(node_size + 2.0 * np.log1p(deg))

    custom = {
        "node": {
            "label": {"enable": True},
            "size": {"attribute": "size"},
        },
        "edge": {
            "size": {"attribute": "w"},
            "color": {"attribute": "color"},     # key change: color by sign
            "label": {"enable": bool(edge_label)},
            "arrow": {"enable": True},
            # dashed support is version-dependent; keep as best-effort
            "dashes": {"attribute": "dashed"},
        },
    }

    fig = gv.d3(H, use_node_size_normalization=False, use_edge_size_normalization=False)

    # inject options across gravis versions
    try:
        if hasattr(fig, "custom_options") and isinstance(fig.custom_options, dict):
            fig.custom_options.update(custom)
        elif hasattr(fig, "options") and isinstance(fig.options, dict):
            fig.options.update(custom)
        else:
            setattr(fig, "custom_options", custom)
    except Exception:
        pass

    return fig.to_html()



def build_egfr_control_graph_from_ko(ko: pd.DataFrame, target: str, metric: str, topk: int) -> nx.DiGraph:
    if ko.empty:
        return nx.DiGraph()

    d = ko[ko["mRNA"].astype(str) == str(target)].copy()
    if d.empty:
        return nx.DiGraph()

    d["m"] = d[metric].astype(float)
    d["abs_m"] = d["m"].abs()
    d = d.sort_values("abs_m", ascending=False).head(topk)

    G = nx.DiGraph()
    tgt = f"tgt::{target}"
    G.add_node(tgt, kind="mrna", label=target)

    for _, r in d.iterrows():
        tf = str(r["KnockedTF"])
        tf_id = f"tf::{tf}"
        G.add_node(tf_id, kind="tf", label=tf)
        sign = 1 if float(r["m"]) > 0 else -1
        G.add_edge(tf_id, tgt, weight=float(r["abs_m"]), sign=sign, alpha=float(r.get("alpha", np.nan)))
    return G


# =========================
# UI helpers
# =========================

def df_block(df: pd.DataFrame, fname: str):
    c1, c2 = st.columns([0.82, 0.18])
    with c1:
        st.dataframe(df, use_container_width=True, height=420)
    with c2:
        st.download_button("Download CSV", data=to_csv_bytes(df), file_name=f"{fname}.csv", mime="text/csv")


# =========================
# Sidebar + App
# =========================

st.title("TFopt Network Readout Dashboard")

# Time config
tcfg = TimeConfig(
    mrna_time_cols=MRNA_TIME_COLS_DEFAULT,
    mrna_time_points=MRNA_TIME_POINTS_DEFAULT,
    tf_time_cols=TF_TIME_COLS_DEFAULT,
    tf_time_points=TF_TIME_POINTS_DEFAULT,
)

with st.sidebar:
    st.header("Inputs")
    uploaded_xlsx = st.file_uploader("Upload tfopt_results.xlsx", type=["xlsx"])
    st.caption("Required: Observed, Estimated, Alpha Values, Beta Values.")
    uploaded_input1 = st.file_uploader("Upload input1.csv (TF time series)", type=["csv"])
    st.caption("Required: TF/GeneID, PSite, x1..x14")
    uploaded_input3 = st.file_uploader("Upload input3.csv (TF protein x1..x9 fallback)", type=["csv"])
    st.caption("Optional fallback: GeneID, x1..x9 (mRNA grid)")

    if uploaded_xlsx is None:
        st.stop()

    file_bytes = uploaded_xlsx.getvalue()
    sheet_names = get_sheet_names(file_bytes)

    st.divider()
    st.header("Sheet mapping")

    # Default edges sheet: prefer "Edges" if present; otherwise "Alpha Values" is valid (TF/mRNA columns).
    default_edges = "Edges" if "Edges" in sheet_names else ("Alpha Values" if "Alpha Values" in sheet_names else sheet_names[0])

    restrict_alpha = st.checkbox("Restrict α to explicit TF→mRNA edges", value=True)
    edges_sheet = st.selectbox(
        "Edges sheet (Source/Target OR TF/mRNA)",
        sheet_names,
        index=sheet_names.index(default_edges) if default_edges in sheet_names else 0,
        disabled=not restrict_alpha,
    )

    st.divider()
    st.header("Core options")

    beta_bound = st.number_input("β bound", min_value=0.0, value=4.0, step=0.5)
    bound_atol = st.number_input("β bound atol", min_value=0.0, value=1e-6, step=1e-6, format="%.1e")

    interp_kind = st.selectbox("Interpolation kind (TF→mRNA grid)", ["linear", "quadratic", "cubic"], index=2)

    min_alpha_filter = st.number_input("Min |α| edge filter", min_value=0.0, value=0.0, step=0.01)

    ko_metric = st.selectbox("KO metric for dashboards", ["delta_auc_abs", "delta_peak_abs"], index=0)

    st.divider()
    st.header("Visualization options")

    top_tf_bar = st.slider("Top TFs in load bar", min_value=5, max_value=80, value=25, step=1)
    top_targets = st.slider("Top targets in dominance bar", min_value=5, max_value=80, value=25, step=1)

    outlier_band_quantile = st.slider("Obs vs recon band quantile", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    max_point_labels = st.slider("Obs vs recon max labels", min_value=0, max_value=30, value=5, step=1)

    top_ko_per_target = st.slider("KO heatmap: top TFs per target", min_value=1, max_value=25, value=8, step=1)

    egfr_topk = st.slider("EGFR control: top TFs", min_value=3, max_value=60, value=18, step=1)

    st.divider()
    st.header("Network rendering")

    node_size = st.slider("Node base size", min_value=6, max_value=30, value=12, step=1)
    edge_scale = st.slider("Edge width scale", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    edge_label = st.checkbox("Show edge labels (slow)", value=False)

params = AppParams(
    beta_bound=float(beta_bound),
    bound_atol=float(bound_atol),
    interp_kind=str(interp_kind),
    restrict_alpha_to_edges=bool(restrict_alpha),
    edges_sheet_name=str(edges_sheet),
    top_tf_bar=int(top_tf_bar),
    top_targets=int(top_targets),
    max_point_labels=int(max_point_labels),
    outlier_band_quantile=float(outlier_band_quantile),
    top_ko_per_target=int(top_ko_per_target),
    egfr_topk=int(egfr_topk),
    min_alpha_filter=float(min_alpha_filter),
    ko_metric=str(ko_metric),
)

with st.spinner("Computing TFopt readouts from XLSX..."):
    if uploaded_input1 is None:
        st.error("input1.csv is required for TF time series (original TFopt readout logic).")
        st.stop()

    out = compute_all(
        xlsx_bytes=file_bytes,
        input1_bytes=uploaded_input1.getvalue(),
        input3_bytes=(uploaded_input3.getvalue() if uploaded_input3 is not None else None),
        params=params,
        tcfg=tcfg,
    )

obs = out["obs"]
est = out["est"]
alpha = out["alpha"]
beta = out["beta"]
latent = out["latent"]

tf_psite_stats = out["tf_psite_stats"]
tf_load = out["tf_load"]
dom = out["dom"]
ko = out["ko"]
target_cache = out["target_cache"]


# =========================
# Tabs
# =========================

tab_overview, tab_tables, tab_plots, tab_drilldown, tab_network = st.tabs(
    ["Overview", "Tables", "Plots (Plotly)", "Drilldown", "Networks (gravis)"]
)

with tab_overview:
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mRNAs (Observed)", obs.shape[0])
    c2.metric("α edges", alpha.shape[0])
    c3.metric("TFs (β)", beta["TF"].nunique() if not beta.empty else 0)
    c4.metric("KO rows", ko.shape[0] if not ko.empty else 0)

    st.write(
        "This dashboard reproduces the TFopt mechanistic audit: "
        "latent TF activities from β mixing of protein + phosphosites (interpolated to the mRNA grid), "
        "α-weighted reconstruction of mRNA trajectories, and in-silico TF knockouts."
    )

    st.subheader("Quick top lists")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top TFs by total control load")
        st.dataframe(tf_load.head(10), use_container_width=True)
    with c2:
        st.write("Top targets by dominant TF share")
        st.dataframe(dom.head(10), use_container_width=True)

with tab_tables:
    st.subheader("TF phosphosite stats")
    df_block(tf_psite_stats, "tfopt_tf_psite_stats")

    st.subheader("TF load")
    df_block(tf_load, "tfopt_tf_load")

    st.subheader("Target dominant TFs")
    df_block(dom, "tfopt_target_dominant_tfs")

    st.subheader("Knockout effects")
    df_block(ko, "tfopt_knockout_effects")

with tab_plots:
    st.subheader("TF-level plots")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_tf_total_load_plotly(tf_load, top_n=params.top_tf_bar), use_container_width=True)
    with c2:
        st.plotly_chart(plot_tf_bound_pressure_vs_load_plotly(tf_load), use_container_width=True)

    st.subheader("Target-level plots")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_target_dominant_tf_share_plotly(dom, top_targets=params.top_targets), use_container_width=True)
    with c2:
        st.plotly_chart(
            plot_obs_vs_recon_plotly(dom, band_quantile=params.outlier_band_quantile, max_labels=params.max_point_labels),
            use_container_width=True,
        )

    st.subheader("Knockout landscape")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_ko_distribution_plotly(ko, metric=params.ko_metric), use_container_width=True)
    with c2:
        st.plotly_chart(plot_ko_heatmap_plotly(ko, top_ko_per_target=params.top_ko_per_target, metric=params.ko_metric), use_container_width=True)

    st.plotly_chart(plot_ko_top_edges_plotly(ko, metric=params.ko_metric, direction="positive", top_n=20), use_container_width=True)
    st.plotly_chart(plot_ko_top_edges_plotly(ko, metric=params.ko_metric, direction="negative", top_n=20), use_container_width=True)

with tab_drilldown:
    st.subheader("Latent TF activity drilldown")
    tfs = sorted(list(latent.keys()))
    if not tfs:
        st.warning("No latent TF activities computed. Check input1.csv coverage and Beta Values.")
    else:
        c1, c2 = st.columns([0.58, 0.42])
        with c1:
            sel_tf = st.selectbox("Select TF", tfs, index=0)
            st.plotly_chart(plot_latent_tf_timeseries_plotly(latent, tcfg, sel_tf), use_container_width=True)
        with c2:
            st.write("β rows for selected TF")
            bsub = beta[beta["TF"].astype(str) == sel_tf].copy()
            bsub["PSite_str"] = bsub["PSite"].astype(str)
            st.dataframe(bsub, use_container_width=True, height=420)

    st.divider()
    st.subheader("mRNA reconstruction drilldown")
    mrnas = sorted(list(target_cache.keys()))
    if not mrnas:
        st.warning("No mRNAs in target cache. Check Alpha Values and edge restrictions.")
    else:
        sel_mrna = st.selectbox("Select mRNA", mrnas, index=0)
        top_k = st.slider("Show top-k TF contributions", min_value=1, max_value=25, value=8, step=1)
        st.plotly_chart(plot_target_reconstruction_timeseries_plotly(tcfg, sel_mrna, target_cache[sel_mrna], top_k=top_k), use_container_width=True)

        st.write("Knockout effects for selected mRNA")
        ko_sub = ko[ko["mRNA"].astype(str) == sel_mrna].sort_values(params.ko_metric, ascending=False)
        st.dataframe(ko_sub, use_container_width=True, height=380)

with tab_network:
    st.subheader("TF → mRNA network (from α) via gravis")
    st.caption(
        "Edge width uses an effective strength that includes β via latent TF activity "
        "(|α| × AUC(|A_TF|)). Edge sign is computed as sign(α) × sign(∫A_TF dt). "
        "Green = activating, red = suppressing, gray = neutral/unknown."
    )

    # ---------- filters ----------
    all_tf = sorted(alpha["TF"].astype(str).unique().tolist()) if not alpha.empty else []
    all_mrna = sorted(alpha["mRNA"].astype(str).unique().tolist()) if not alpha.empty else []

    c1, c2, c3 = st.columns(3)
    with c1:
        tf_filter = st.multiselect("Filter TFs (optional)", all_tf, default=[])
    with c2:
        mrna_filter = st.multiselect("Filter mRNAs (optional)", all_mrna, default=[])
    with c3:
        ko_preview = st.selectbox("KO preview (remove TF)", ["(none)"] + all_tf, index=0)

    alpha_net = alpha.copy()
    if tf_filter:
        alpha_net = alpha_net[alpha_net["TF"].astype(str).isin([str(x) for x in tf_filter])].copy()
    if mrna_filter:
        alpha_net = alpha_net[alpha_net["mRNA"].astype(str).isin([str(x) for x in mrna_filter])].copy()
    if ko_preview != "(none)":
        alpha_net = alpha_net[alpha_net["TF"].astype(str) != str(ko_preview)].copy()

    # ---------- build effective-sign network using beta-driven latent ----------
    tf_auc_abs, tf_polarity = compute_tf_activity_scalars(latent, tcfg)

    G = build_tf_mrna_network(alpha_net, tf_auc_abs=tf_auc_abs, tf_polarity=tf_polarity)
    html = render_gravis(G, node_size=node_size, edge_scale=edge_scale, edge_label=edge_label)
    st.components.v1.html(html, height=720, scrolling=True)

    # ---------- split into activating vs suppressing using effective sign ----------
    st.divider()
    st.subheader("Separate activating vs suppressing edges")
    st.caption(
        "This split uses the effective sign (sign(α) × sign(∫A_TF dt)), not raw α sign. "
        "If TF polarity is ~0 or missing, edges become neutral and will not appear in either split."
    )

    # Reuse the graph to filter by effective sign (avoids recomputing)
    eff_edges = []
    for u, v, d in G.edges(data=True):
        eff_edges.append(
            {
                "TF": u,
                "mRNA": v,
                "Value": float(d.get("alpha", 0.0)),          # raw alpha kept for reference
                "eff_sign": int(d.get("eff_sign", 0)),
                "eff_weight": float(d.get("weight", 0.0)),   # |α| * AUC(|A_TF|)
            }
        )
    eff_df = pd.DataFrame(eff_edges)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Activating edges only")
        if eff_df.empty:
            st.info("No edges available after filtering.")
        else:
            act_pairs = set(
                (r["TF"], r["mRNA"]) for _, r in eff_df[eff_df["eff_sign"] > 0].iterrows()
            )
            if not act_pairs:
                st.info("No activating edges under the effective-sign definition.")
            else:
                alpha_act = alpha_net[
                    alpha_net.apply(lambda r: (str(r["TF"]), str(r["mRNA"])) in act_pairs, axis=1)
                ].copy()

                G_act = build_tf_mrna_network(alpha_act, tf_auc_abs=tf_auc_abs, tf_polarity=tf_polarity)
                html_act = render_gravis(G_act, node_size=node_size, edge_scale=edge_scale, edge_label=edge_label)
                st.components.v1.html(html_act, height=720, scrolling=True)

    with colB:
        st.subheader("Suppressing edges only")
        if eff_df.empty:
            st.info("No edges available after filtering.")
        else:
            sup_pairs = set(
                (r["TF"], r["mRNA"]) for _, r in eff_df[eff_df["eff_sign"] < 0].iterrows()
            )
            if not sup_pairs:
                st.info("No suppressing edges under the effective-sign definition.")
            else:
                alpha_sup = alpha_net[
                    alpha_net.apply(lambda r: (str(r["TF"]), str(r["mRNA"])) in sup_pairs, axis=1)
                ].copy()

                G_sup = build_tf_mrna_network(alpha_sup, tf_auc_abs=tf_auc_abs, tf_polarity=tf_polarity)
                html_sup = render_gravis(G_sup, node_size=node_size, edge_scale=edge_scale, edge_label=edge_label)
                st.components.v1.html(html_sup, height=720, scrolling=True)

    # ---------- KO-based per-target control graph (already target-specific) ----------
    st.divider()
    st.subheader("Target control logic (TF → target) via gravis")
    st.caption(f"Edges are built from KO effects. Edge width ∝ |{params.ko_metric}|. Sign stored as attribute.")

    targets = sorted(ko["mRNA"].astype(str).unique().tolist()) if not ko.empty else ["EGFR"]
    target = st.selectbox(
        "Target for control logic",
        targets,
        index=targets.index("EGFR") if "EGFR" in targets else 0,
    )

    G2 = build_egfr_control_graph_from_ko(ko, target=target, metric=params.ko_metric, topk=params.egfr_topk)
    if G2.number_of_nodes() == 0:
        st.info("No control graph available (no KO data for this target).")
    else:
        html2 = render_gravis(G2, node_size=node_size, edge_scale=edge_scale, edge_label=False)
        st.components.v1.html(html2, height=720, scrolling=True)
