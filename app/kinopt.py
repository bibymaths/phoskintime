#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard for Kinopt network readouts and visualization.

What this app does
------------------
1) Loads a Kinopt XLSX (Observed / Estimated / Alpha Values / Beta Values).
2) Recomputes the model-consistent signal flow:

   A_j(t)        = sum_k beta_{k,j} * P^k_j(t)
   C_{i<-j}(t)   = alpha_{i,j} * A_j(t)
   P_hat_i(t)    = sum_j C_{i<-j}(t)

3) Produces the same analysis tables as your scripts, but in-memory:
   - latent kinase activity summary
   - target dominant kinases summary
   - kinase load summary
   - knockout effects table

4) Produces the same visualization families, but rendered via Plotly in-app:
   - Kinase control load bar chart
   - Breadth vs load scatter (+ labeling threshold)
   - Bound pressure vs load scatter
   - Target dominance distribution
   - Most fragile targets bar chart
   - Observed vs reconstructed magnitude with 95% band
   - Knockout effect distribution
   - Top activating / suppressing knockout edges

5) Network rendering:
   - Full Alpha network (targets <- kinases) rendered via gravis
   - Optional subgraph filtering (min alpha, by Gene, by Kinase)
   - Knockout preview (remove a kinase) and render the modified network via gravis
   - EGFR control logic DAG (EGFR → site → kinase), rendered via gravis (directed), edge width ~ |ΔAUC|, dashed for inhibitory

Design notes
------------
- Everything is computed from the XLSX; no intermediate figure files needed.
- Computation is cached. Changing parameters invalidates cache.
- All options that were constants in the scripts are exposed as widgets.

Run
---
  streamlit run app.py

Dependencies
------------
  pip install streamlit pandas numpy scipy plotly networkx gravis openpyxl pydot

If Graphviz is installed, EGFR DAG layout improves (DOT). Without it, a layered fallback is used.
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
from scipy.integrate import trapezoid

import gravis as gv


# =========================
# App config
# =========================

st.set_page_config(page_title="Kinopt Network Readout Dashboard", layout="wide")


# =========================
# Utilities
# =========================

EPS = 1e-12


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"Could not find any of {candidates} in columns={list(df.columns)}")


def _auc(t: np.ndarray, y: np.ndarray) -> float:
    return float(trapezoid(y, t))


def _windows(nT: int, n_early: int, n_late: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Early/mid/late windows by indices.
    """
    n_early = max(1, min(n_early, nT))
    n_late = max(1, min(n_late, nT))
    early = np.arange(0, min(n_early, nT))
    late = np.arange(max(0, nT - n_late), nT)
    mid = np.setdiff1d(np.arange(nT), np.union1d(early, late))
    if len(mid) == 0:
        mid = np.arange(nT)
    return early, mid, late


def _as_key(g: str, p: str) -> tuple[str, str]:
    return (str(g), str(p))


def _to_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _norm_id(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s

@dataclass(frozen=True)
class TimeConfig:
    time_cols: List[str]
    time_points_min: np.ndarray  # raw minutes
    t_norm: np.ndarray           # normalized time for AUC/load


@dataclass(frozen=True)
class AppParams:
    kinase_site_source: str           # "Observed" or "Estimated"
    beta_bound: float
    bound_tol: float
    ko_renormalize_alpha: bool
    n_early: int
    n_late: int
    kinase_label_load_threshold: float
    outlier_band_quantile: float
    outlier_max_labels: int
    egfr_topk_per_site: int
    min_alpha_filter: float


# =========================
# Data loading
# =========================

@st.cache_data(show_spinner=False)
def load_xlsx(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """
    Load required sheets from an uploaded XLSX.
    Returns a dict of sheet_name -> DataFrame.
    """
    bio = io.BytesIO(file_bytes)
    obs = pd.read_excel(bio, sheet_name="Observed")
    bio.seek(0)
    est = pd.read_excel(bio, sheet_name="Estimated")
    bio.seek(0)
    alpha = pd.read_excel(bio, sheet_name="Alpha Values")
    bio.seek(0)
    beta = pd.read_excel(bio, sheet_name="Beta Values")
    return {"Observed": obs, "Estimated": est, "Alpha Values": alpha, "Beta Values": beta}

@st.cache_data(show_spinner=False)
def load_input1_csv(input1_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(input1_bytes))

def build_ts_map_from_df(df: pd.DataFrame, gene_col: str, psite_col: str, time_cols: list[str]) -> Dict[Tuple[str, str], np.ndarray]:
    m: Dict[Tuple[str, str], np.ndarray] = {}
    sub = df[[gene_col, psite_col] + time_cols].copy()
    for _, r in sub.iterrows():
        key = _as_key(r[gene_col], r[psite_col])
        m[key] = r[time_cols].to_numpy(dtype=np.float64)
    return m


def build_time_config(
    time_mode: str,
    df_obs: pd.DataFrame,
    df_est: pd.DataFrame,
) -> TimeConfig:
    """
    Time handling.
    - "Kinopt default": use x1..x14 and the fixed 14 timepoints (minutes)
    - "Auto-detect": try to parse column headers as floats; fallback to numeric columns.
      (Not recommended unless your sheets truly use time headers.)
    """
    if time_mode == "Kinopt default":
        time_cols = [f"x{i}" for i in range(1, 15)]
        time_points_min = np.array(
            [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
            dtype=np.float64
        )
        missing_obs = [c for c in time_cols if c not in df_obs.columns]
        missing_est = [c for c in time_cols if c not in df_est.columns]
        if missing_obs:
            raise ValueError(f"Missing time columns in Observed sheet: {missing_obs}")
        if missing_est:
            raise ValueError(f"Missing time columns in Estimated sheet: {missing_est}")
        if len(time_cols) != len(time_points_min):
            raise ValueError("Internal error: time columns and time points length mismatch.")
        t_norm = time_points_min / (time_points_min.max() + EPS)
        return TimeConfig(time_cols=time_cols, time_points_min=time_points_min, t_norm=t_norm)

    # Auto-detect mode
    # Identify id columns first, then treat anything parseable as float as time.
    gene_col = _find_col(df_obs, ["Gene", "GeneID", "Protein", "mRNA"])
    psite_col = _find_col(df_obs, ["Psite", "PSite", "Position", "Site"])
    exclude = {gene_col, psite_col}

    time_cols = []
    for c in df_obs.columns:
        if c in exclude:
            continue
        s = str(c).strip()
        try:
            float(s)
            time_cols.append(c)
        except Exception:
            pass

    if not time_cols:
        # fallback numeric
        time_cols = [c for c in df_obs.select_dtypes(include="number").columns if c not in exclude]

    if not time_cols:
        raise ValueError("Auto-detect failed: no time columns detected.")

    # Sort by float(column name) if possible, else keep as-is
    def _key(c):
        try:
            return float(str(c).strip())
        except Exception:
            return 1e18

    time_cols = sorted(time_cols, key=_key)
    time_points_min = np.array([_key(c) for c in time_cols], dtype=np.float64)
    if np.any(~np.isfinite(time_points_min)):
        raise ValueError("Auto-detect failed: non-finite time column headers.")

    t_norm = time_points_min / (np.max(time_points_min) + EPS)
    return TimeConfig(time_cols=time_cols, time_points_min=time_points_min, t_norm=t_norm)


def build_ts_map(df: pd.DataFrame, gene_col: str, psite_col: str, time_cols: list[str]) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build a map (Gene, Psite) -> time series vector.
    """
    m: Dict[Tuple[str, str], np.ndarray] = {}
    for _, r in df[[gene_col, psite_col] + time_cols].iterrows():
        key = _as_key(r[gene_col], r[psite_col])
        y = r[time_cols].to_numpy(dtype=np.float64)
        m[key] = y
    return m


def parse_edges(alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse alpha edges: one row per (Gene, Psite, Kinase) with Alpha.
    """
    gene_col_a = _find_col(alpha_df, ["Gene", "GeneID", "Protein"])
    psite_col_a = _find_col(alpha_df, ["Psite", "PSite", "Position", "Site"])
    kinase_col_a = _find_col(alpha_df, ["Kinase"])
    alpha_col_a = _find_col(alpha_df, ["Alpha", "alpha"])

    edges = alpha_df[[gene_col_a, psite_col_a, kinase_col_a, alpha_col_a]].copy()
    edges.rename(columns={gene_col_a: "Gene", psite_col_a: "Psite", kinase_col_a: "Kinase", alpha_col_a: "Alpha"}, inplace=True)
    edges["Gene"] = edges["Gene"].astype(str)
    edges["Psite"] = edges["Psite"].astype(str)
    edges["Kinase"] = edges["Kinase"].astype(str)
    edges["Alpha"] = pd.to_numeric(edges["Alpha"], errors="coerce").fillna(0.0).astype(np.float64)
    return edges


def parse_betas(beta_df: pd.DataFrame, beta_bound: float, bound_tol: float) -> pd.DataFrame:
    """
    Parse beta table: one row per (Kinase, KinaseSite) with Beta.
    """
    kinase_col_b = _find_col(beta_df, ["Kinase"])
    psite_col_b = _find_col(beta_df, ["Psite", "PSite", "Position", "Site"])
    beta_col_b = _find_col(beta_df, ["Beta", "beta"])

    b = beta_df[[kinase_col_b, psite_col_b, beta_col_b]].copy()
    b.rename(columns={kinase_col_b: "Kinase", psite_col_b: "KinaseSite", beta_col_b: "Beta"}, inplace=True)
    b["Kinase"] = b["Kinase"].astype(str)
    b["KinaseSite"] = b["KinaseSite"].astype(str)
    b["Beta"] = pd.to_numeric(b["Beta"], errors="coerce").fillna(0.0).astype(np.float64)
    b["beta_at_bound"] = (b["Beta"].abs() >= (beta_bound - bound_tol))
    return b


# =========================
# Core computation
# =========================

def compute_all(file_bytes: bytes, input1_bytes: Optional[bytes], params: AppParams, time_mode: str) -> dict:
    """
    Compute all analysis tables + key time series products from the XLSX.
    Cached by (file_bytes, params, time_mode).
    """
    sheets = load_xlsx(file_bytes)
    obs = sheets["Observed"]
    est = sheets["Estimated"]
    alpha_df = sheets["Alpha Values"]
    beta_df = sheets["Beta Values"]

    # Identify id columns in time series
    gene_col_obs = _find_col(obs, ["Gene", "GeneID", "Protein", "mRNA"])
    psite_col_obs = _find_col(obs, ["Psite", "PSite", "Position", "Site"])
    gene_col_est = _find_col(est, ["Gene", "GeneID", "Protein", "mRNA"])
    psite_col_est = _find_col(est, ["Psite", "PSite", "Position", "Site"])

    # Time config
    tcfg = build_time_config(time_mode, obs, est)
    t = tcfg.t_norm
    nT = len(tcfg.time_cols)
    early_idx, mid_idx, late_idx = _windows(nT, params.n_early, params.n_late)

    # Build maps
    obs_map = build_ts_map(obs, gene_col_obs, psite_col_obs, tcfg.time_cols)
    est_map = build_ts_map(est, gene_col_est, psite_col_est, tcfg.time_cols)

    input1_map = {}
    if input1_bytes:
        input1_df = load_input1_csv(input1_bytes)

        gene_col_in = _find_col(input1_df, ["GeneID", "Gene", "Protein", "Kinase"])
        psite_col_in = _find_col(input1_df, ["Psite", "PSite", "Position", "Site"])

        # ensure time columns exist in input1 too
        missing_in = [c for c in tcfg.time_cols if c not in input1_df.columns]
        if missing_in:
            raise ValueError(f"Missing time columns in input1.csv: {missing_in}")

        input1_map = build_ts_map_from_df(input1_df, gene_col_in, psite_col_in, tcfg.time_cols)

    base_map = obs_map if params.kinase_site_source.lower() == "observed" else est_map

    def get_kinase_site_ts(kinase: str, ksite: str) -> Optional[np.ndarray]:
        key = _as_key(kinase, ksite)
        y = base_map.get(key)
        if y is not None:
            return y
        return input1_map.get(key)  # fallback
    # Alpha edges and Beta sites
    edges = parse_edges(alpha_df)
    if params.min_alpha_filter > 0.0:
        edges = edges[edges["Alpha"].abs() >= params.min_alpha_filter].copy()

    b = parse_betas(beta_df, params.beta_bound, params.bound_tol)

    # ------------------------
    # 1) Latent kinase activity A_j(t)
    # ------------------------
    latent: Dict[str, np.ndarray] = {}
    latent_meta_rows = []
    latent_site_detail: Dict[str, pd.DataFrame] = {}

    for kin, grp in b.groupby("Kinase", sort=False):
        A = np.zeros((nT,), dtype=np.float64)
        site_rows = []
        for _, rr in grp.iterrows():
            ksite = rr["KinaseSite"]
            beta_kj = float(rr["Beta"])
            y_site = get_kinase_site_ts(kin, ksite)


            if y_site is None:
                site_rows.append(
                    {
                        "Kinase": kin,
                        "KinaseSite": ksite,
                        "Beta": beta_kj,
                        "beta_at_bound": bool(rr["beta_at_bound"]),
                        "has_timeseries": False,
                        "site_auc_abs": np.nan,
                    }
                )
                continue

            A += beta_kj * y_site
            site_rows.append(
                {
                    "Kinase": kin,
                    "KinaseSite": ksite,
                    "Beta": beta_kj,
                    "beta_at_bound": bool(rr["beta_at_bound"]),
                    "has_timeseries": True,
                    "site_auc_abs": _auc(t, np.abs(y_site)),
                }
            )

        latent[kin] = A
        df_site = pd.DataFrame(site_rows)

        # Site dominance proxy: |beta| * AUC(|site|)
        top_site, top_share = None, np.nan
        if df_site["has_timeseries"].any():
            df_site["importance"] = df_site["Beta"].abs() * df_site["site_auc_abs"].fillna(0.0)
            df_ok = df_site[df_site["has_timeseries"]].sort_values("importance", ascending=False)
            top_site = str(df_ok.iloc[0]["KinaseSite"])
            top_import = float(df_ok.iloc[0]["importance"])
            tot_import = float(df_ok["importance"].sum() + EPS)
            top_share = float(top_import / tot_import)

        latent_meta_rows.append(
            {
                "Kinase": kin,
                "n_beta_sites": int(len(grp)),
                "n_beta_at_bound": int(grp["beta_at_bound"].sum()),
                "frac_beta_at_bound": float(grp["beta_at_bound"].mean()) if len(grp) else 0.0,
                "top_kinase_site": top_site,
                "top_kinase_site_share": top_share,
                "latent_auc_abs": _auc(t, np.abs(A)),
                "latent_peak_abs": float(np.max(np.abs(A))),
            }
        )

        latent_site_detail[kin] = df_site.sort_values(["has_timeseries", "importance" if "importance" in df_site.columns else "Beta"],
                                                     ascending=[False, False])

    latent_summary = pd.DataFrame(latent_meta_rows).sort_values(
        ["latent_auc_abs", "frac_beta_at_bound"], ascending=[False, False]
    )

    # ------------------------
    # 2) Target decomposition summary
    # ------------------------
    target_rows = []
    # Optional: store per-target timeseries decomposition in a light form (for interactive drilldown)
    target_decomp_cache: Dict[Tuple[str, str], dict] = {}

    for (gene, psite), grp in edges.groupby(["Gene", "Psite"], sort=False):
        kin_list = grp["Kinase"].tolist()
        a = grp["Alpha"].to_numpy(dtype=np.float64)

        C = np.zeros((len(kin_list), nT), dtype=np.float64)
        missing_latent = 0

        for i, kin in enumerate(kin_list):
            Aj = latent.get(kin)
            if Aj is None:
                Aj = np.zeros((nT,), dtype=np.float64)
                missing_latent += 1
            C[i, :] = a[i] * Aj

        P_hat = C.sum(axis=0)

        absC = np.abs(C)
        abs_total = absC.sum(axis=0) + EPS
        frac = absC / abs_total

        def dominant(idx: np.ndarray) -> tuple[str, float]:
            w = frac[:, idx].sum(axis=1)
            j = int(np.argmax(w))
            return kin_list[j], float(w[j] / (w.sum() + EPS))

        dom_overall_idx = int(np.argmax(absC.sum(axis=1))) if len(kin_list) else 0
        dom_overall = kin_list[dom_overall_idx] if len(kin_list) else None
        dom_overall_share = float(absC.sum(axis=1)[dom_overall_idx] / (absC.sum() + EPS)) if len(kin_list) else np.nan

        dom_early, dom_early_share = dominant(early_idx) if len(kin_list) else (None, np.nan)
        dom_mid, dom_mid_share = dominant(mid_idx) if len(kin_list) else (None, np.nan)
        dom_late, dom_late_share = dominant(late_idx) if len(kin_list) else (None, np.nan)

        obs_y = obs_map.get(_as_key(gene, psite))
        obs_auc_abs = _auc(t, np.abs(obs_y)) if obs_y is not None else np.nan
        obs_peak_abs = float(np.max(np.abs(obs_y))) if obs_y is not None else np.nan

        target_rows.append(
            {
                "Gene": gene,
                "Psite": psite,
                "n_kinases": int(len(kin_list)),
                "n_missing_kinase_latent": int(missing_latent),
                "dominant_overall": dom_overall,
                "dominant_overall_share": dom_overall_share,
                "dominant_early": dom_early,
                "dominant_early_share": dom_early_share,
                "dominant_mid": dom_mid,
                "dominant_mid_share": dom_mid_share,
                "dominant_late": dom_late,
                "dominant_late_share": dom_late_share,
                "pred_auc_abs": _auc(t, np.abs(P_hat)),
                "pred_peak_abs": float(np.max(np.abs(P_hat))),
                "obs_auc_abs": obs_auc_abs,
                "obs_peak_abs": obs_peak_abs,
            }
        )

        # store for drilldown
        target_decomp_cache[(gene, psite)] = {
            "kinases": kin_list,
            "alpha": a,
            "C": C,  # contribution matrix
            "P_hat": P_hat,
            "P_obs": obs_y,
        }

    target_dom = pd.DataFrame(target_rows).sort_values(
        ["dominant_overall_share", "n_kinases"], ascending=[False, True]
    )

    # ------------------------
    # 3) Kinase load across network
    # ------------------------
    edge_load_rows = []
    for _, r in edges.iterrows():
        gene, psite, kin, aij = r["Gene"], r["Psite"], r["Kinase"], float(r["Alpha"])
        Aj = latent.get(kin)
        if Aj is None:
            continue
        edge_load_rows.append(
            {
                "Kinase": kin,
                "Gene": gene,
                "Psite": psite,
                "Alpha": aij,
                "edge_load_auc_abs": _auc(t, np.abs(aij * Aj)),
            }
        )

    edge_load = pd.DataFrame(edge_load_rows)
    kinase_load = (
        edge_load.groupby("Kinase")
        .agg(
            n_targets=("edge_load_auc_abs", "size"),
            total_load_auc_abs=("edge_load_auc_abs", "sum"),
        )
        .reset_index()
        .merge(
            latent_summary[["Kinase", "frac_beta_at_bound", "n_beta_at_bound", "latent_auc_abs", "latent_peak_abs"]],
            on="Kinase",
            how="left",
        )
        .sort_values(["total_load_auc_abs", "n_targets"], ascending=[False, False])
    )

    # ------------------------
    # 4) Knockout effects per target
    # ------------------------
    ko_rows = []
    for (gene, psite), grp in edges.groupby(["Gene", "Psite"], sort=False):
        kin_list = grp["Kinase"].tolist()
        a = grp["Alpha"].to_numpy(dtype=np.float64)

        # baseline
        baseline = np.zeros((nT,), dtype=np.float64)
        for i, kin in enumerate(kin_list):
            Aj = latent.get(kin)
            if Aj is None:
                Aj = np.zeros((nT,), dtype=np.float64)
            baseline += a[i] * Aj

        base_auc = _auc(t, np.abs(baseline))
        base_peak_abs = float(np.max(np.abs(baseline)))

        for i, kin in enumerate(kin_list):
            a_ko = a.copy()
            a_ko[i] = 0.0
            if params.ko_renormalize_alpha:
                s = a_ko.sum()
                if s > EPS:
                    a_ko = a_ko / s

            y = np.zeros((nT,), dtype=np.float64)
            for ii, kin2 in enumerate(kin_list):
                Aj = latent.get(kin2)
                if Aj is None:
                    Aj = np.zeros((nT,), dtype=np.float64)
                y += a_ko[ii] * Aj

            ko_auc = _auc(t, np.abs(y))
            ko_peak_abs = float(np.max(np.abs(y)))

            ko_rows.append(
                {
                    "Gene": gene,
                    "Psite": psite,
                    "KnockedKinase": kin,
                    "renormalize_alpha": bool(params.ko_renormalize_alpha),
                    "delta_auc_abs": base_auc - ko_auc,
                    "delta_peak_abs": base_peak_abs - ko_peak_abs,
                    "baseline_auc_abs": base_auc,
                    "baseline_peak_abs": base_peak_abs,
                }
            )

    ko = pd.DataFrame(ko_rows)
    if not ko.empty:
        ko["ko_rank_site"] = ko.groupby(["Gene", "Psite"])["delta_auc_abs"].rank(ascending=False, method="min")
        ko = ko.sort_values(["Gene", "Psite", "ko_rank_site"])

    return {
        "sheets": sheets,
        "time": tcfg,
        "obs_map": obs_map,
        "est_map": est_map,
        "edges": edges,
        "betas": b,
        "latent": latent,
        "latent_summary": latent_summary,
        "latent_site_detail": latent_site_detail,
        "target_dom": target_dom,
        "target_decomp_cache": target_decomp_cache,
        "kinase_load": kinase_load,
        "edge_load": edge_load,
        "ko": ko,
        "params": params,
    }


# =========================
# Plotly viz functions (in-app analogs of your figure script)
# =========================

def fig_bar_kinase_control_load(kinase_load: pd.DataFrame, top_n: int = 25) -> go.Figure:
    d = kinase_load.sort_values("total_load_auc_abs", ascending=False).head(min(top_n, len(kinase_load))).copy()
    d = d.sort_values("total_load_auc_abs", ascending=True)
    fig = px.bar(
        d,
        x="total_load_auc_abs",
        y="Kinase",
        orientation="h",
        labels={"total_load_auc_abs": "Total routed activity"},
        title="Kinase control load (top)"
    )
    fig.update_layout(height=max(450, 18 * len(d) + 200), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_scatter_breadth_vs_load(kinase_load: pd.DataFrame, label_threshold: float) -> go.Figure:
    d = kinase_load.copy()
    d["label"] = np.where(d["total_load_auc_abs"].astype(float) > label_threshold, d["Kinase"].astype(str), "")
    fig = px.scatter(
        d,
        x="n_targets",
        y="total_load_auc_abs",
        hover_data=["Kinase", "frac_beta_at_bound", "latent_auc_abs", "latent_peak_abs"],
        text="label",
        title="Kinase breadth vs control load",
        labels={"n_targets": "Number of targets", "total_load_auc_abs": "Control load"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_scatter_boundpressure_vs_load(kinase_load: pd.DataFrame) -> go.Figure:
    d = kinase_load.copy()
    fig = px.scatter(
        d,
        x="frac_beta_at_bound",
        y="total_load_auc_abs",
        hover_data=["Kinase", "n_targets", "latent_auc_abs", "latent_peak_abs"],
        title="Constraint pressure vs control load",
        labels={"frac_beta_at_bound": "Fraction of β at bounds", "total_load_auc_abs": "Control load"},
    )
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_target_dominance_distribution(target_dom: pd.DataFrame) -> go.Figure:
    d = target_dom.dropna(subset=["dominant_overall_share"]).copy()
    fig = px.histogram(
        d,
        x="dominant_overall_share",
        nbins=35,
        histnorm="probability density",
        title="Target control dominance distribution",
        labels={"dominant_overall_share": "Fraction of signal explained by top kinase"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_fragile_targets_bar(target_dom: pd.DataFrame, top_n: int = 25) -> go.Figure:
    d = target_dom.sort_values("dominant_overall_share", ascending=False).head(min(top_n, len(target_dom))).copy()
    d["Target"] = d["Gene"].astype(str) + " " + d["Psite"].astype(str)
    d = d.sort_values("dominant_overall_share", ascending=True)
    fig = px.bar(
        d,
        x="dominant_overall_share",
        y="Target",
        orientation="h",
        hover_data=["dominant_overall", "n_kinases", "pred_auc_abs", "obs_auc_abs"],
        title="Most fragile targets",
        labels={"dominant_overall_share": "Dominance"},
    )
    fig.update_layout(height=max(450, 18 * len(d) + 200), margin=dict(l=10, r=10, t=50, b=10), xaxis=dict(range=[0, 1.05]))
    return fig


def fig_obs_vs_reconstructed(target_dom: pd.DataFrame, outlier_quantile: float, max_labels: int) -> go.Figure:
    d = target_dom.dropna(subset=["pred_auc_abs", "obs_auc_abs"]).copy()
    if d.empty:
        return go.Figure().update_layout(title="Observed vs reconstructed target magnitude (no data)")

    d["obs"] = d["obs_auc_abs"].astype(float)
    d["pred"] = d["pred_auc_abs"].astype(float)
    d["label"] = d["Gene"].astype(str) + " " + d["Psite"].astype(str)
    d["perp_dist"] = np.abs(d["pred"] - d["obs"]) / np.sqrt(2)

    thr = float(np.quantile(d["perp_dist"].to_numpy(), outlier_quantile))
    offset = thr * np.sqrt(2)

    # Label points inside the band (mirroring your script’s current behavior)
    in_band = d[d["perp_dist"] <= thr].copy()
    label_df = in_band.sort_values("perp_dist", ascending=True).head(max_labels)

    mx = float(np.nanmax(np.r_[d["obs"].to_numpy(), d["pred"].to_numpy(), 1e-9]))
    xline = np.array([0.0, mx])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d["obs"], y=d["pred"],
        mode="markers",
        marker=dict(size=7),
        hovertext=d["label"],
        hoverinfo="text+x+y",
        name="Targets"
    ))

    # identity + band
    fig.add_trace(go.Scatter(x=xline, y=xline, mode="lines", name="y=x", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=xline, y=xline + offset, mode="lines", name=f"y=x+{offset:.3g}", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=xline, y=xline - offset, mode="lines", name=f"y=x-{offset:.3g}", line=dict(width=1, dash="dot")))

    # labels
    fig.add_trace(go.Scatter(
        x=label_df["obs"], y=label_df["pred"],
        mode="text",
        text=label_df["label"],
        textposition="top center",
        name="Labels"
    ))

    fig.update_layout(
        title="Observed vs reconstructed target magnitude",
        xaxis_title="Observed signal magnitude",
        yaxis_title="Reconstructed signal magnitude",
        xaxis=dict(range=[0, mx]),
        yaxis=dict(range=[0, mx], scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def fig_knockout_distribution(ko: pd.DataFrame) -> go.Figure:
    d = ko.dropna(subset=["delta_auc_abs"]).copy()
    fig = px.histogram(
        d,
        x="delta_auc_abs",
        nbins=45,
        histnorm="probability density",
        title="Distribution of kinase knockout effects (ΔAUC)",
        labels={"delta_auc_abs": "Change in reconstructed signal"},
    )
    fig.add_vline(x=0.0, line_width=2, line_dash="solid")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_knockout_top_edges(ko: pd.DataFrame, direction: str, top_n: int = 20) -> go.Figure:
    d = ko.copy()
    if d.empty:
        return go.Figure().update_layout(title="No knockout data available.")
    d["Edge"] = d["Gene"].astype(str) + " " + d["Psite"].astype(str) + " \u2190 " + d["KnockedKinase"].astype(str)

    if direction == "activating":
        dd = d.sort_values("delta_auc_abs", ascending=False).head(top_n).copy()
        title = "Strongest activating edges (top ΔAUC)"
    else:
        dd = d.sort_values("delta_auc_abs", ascending=True).head(top_n).copy()
        title = "Strongest suppressing edges (most negative ΔAUC)"

    dd = dd.sort_values("delta_auc_abs", ascending=True)
    fig = px.bar(
        dd,
        x="delta_auc_abs",
        y="Edge",
        orientation="h",
        title=title,
        labels={"delta_auc_abs": "Change in reconstructed signal"},
        hover_data=["baseline_auc_abs", "baseline_peak_abs", "delta_peak_abs"]
    )
    fig.update_layout(height=max(450, 18 * len(dd) + 200), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_timeseries_kinase(latent: Dict[str, np.ndarray], tcfg: TimeConfig, kinase: str) -> go.Figure:
    y = latent.get(kinase)
    if y is None:
        return go.Figure().update_layout(title=f"Latent kinase activity: {kinase} (not available)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tcfg.time_points_min, y=y, mode="lines+markers", name="A_j(t)"))
    fig.update_layout(
        title=f"Latent kinase activity A_j(t): {kinase}",
        xaxis_title="Time (min)",
        yaxis_title="Activity (arb.)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_timeseries_target_decomp(tcfg: TimeConfig, gene: str, psite: str, decomp: dict, show_top_k: int = 8) -> go.Figure:
    P_hat = decomp["P_hat"]
    P_obs = decomp["P_obs"]
    kinases = decomp["kinases"]
    C = decomp["C"]  # [nKin, nT]

    fig = go.Figure()
    if P_obs is not None:
        fig.add_trace(go.Scatter(x=tcfg.time_points_min, y=P_obs, mode="lines+markers", name="Observed"))
    fig.add_trace(go.Scatter(x=tcfg.time_points_min, y=P_hat, mode="lines+markers", name="Predicted (sum)"))

    # Add top-k contributions by AUC(|C|)
    if len(kinases) > 0:
        contrib_auc = np.array([_auc(tcfg.t_norm, np.abs(C[i, :])) for i in range(len(kinases))], dtype=float)
        order = np.argsort(-contrib_auc)[: min(show_top_k, len(kinases))]
        for i in order:
            fig.add_trace(go.Scatter(
                x=tcfg.time_points_min, y=C[i, :],
                mode="lines",
                name=f"C: {kinases[i]}",
                opacity=0.75
            ))

    fig.update_layout(
        title=f"Target reconstruction and contributions: {gene} {psite}",
        xaxis_title="Time (min)",
        yaxis_title="Signal (arb.)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# =========================
# Network / gravis rendering
# =========================

def build_alpha_network(
    edges: pd.DataFrame,
    ko: Optional[pd.DataFrame] = None,
    target_dom: Optional[pd.DataFrame] = None,
    edge_mode: str = "alpha",   # "alpha" or "ko"
) -> nx.DiGraph:
    """
    Directed graph: Kinase -> Target(Gene|Psite)

    edge_mode:
      - "alpha": width ~ |alpha|
      - "ko":    width ~ |delta_auc_abs|, color by sign (green/red)
    """
    # KO lookup: (Gene,Psite,Kinase) -> delta
    ko_map = {}
    if ko is not None and not ko.empty:
        for r in ko.itertuples(index=False):
            ko_map[(str(r.Gene), str(r.Psite), str(r.KnockedKinase))] = float(r.delta_auc_abs)

    # Dominant kinase lookup per target
    dom_map = {}
    if target_dom is not None and not target_dom.empty:
        for r in target_dom.itertuples(index=False):
            dom_map[(str(r.Gene), str(r.Psite))] = str(r.dominant_overall)

    G = nx.DiGraph()

    # Precompute which kinases are dominant in the currently shown subgraph
    dominant_kinases = set()
    for _, r in edges[["Gene", "Psite"]].drop_duplicates().iterrows():
        k = dom_map.get((str(r["Gene"]), str(r["Psite"])))
        if k:
            dominant_kinases.add(k)

    for _, r in edges.iterrows():
        gene = str(r["Gene"])
        psite = str(r["Psite"])
        kin = str(r["Kinase"])
        a = float(r["Alpha"])
        target = f"{gene}|{psite}"

        # Nodes
        if kin not in G:
            G.add_node(kin, kind="kinase", label=kin,
                       color=("black" if kin in dominant_kinases else "gray"))
        if target not in G:
            G.add_node(target, kind="target", label=target, gene=gene, psite=psite, color="gray")

        # Edge attrs
        delta = ko_map.get((gene, psite, kin), 0.0)
        sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        edge_color = "green" if sign > 0 else ("red" if sign < 0 else "gray")

        # Choose width driver
        if edge_mode == "ko":
            w_base = abs(delta)
        else:
            w_base = abs(a)

        G.add_edge(
            kin, target,
            alpha=a,
            delta_auc=delta,
            sign=sign,
            color=edge_color,
            weight=w_base,  # this is what we'll map to width
            label=f"α={a:.3g}  ΔAUC={delta:.3g}"
        )

    return G


def render_gravis(
    G: nx.Graph,
    node_size: int = 12,
    edge_size_scale: float = 1.0,
    edge_label: bool = False,
    directed: bool = True,
) -> str:
    """
    Render a NetworkX graph via gravis and return HTML to embed in Streamlit.
    """
    # Edge widths
    # gravis will respect edge properties if provided via custom_options; keep it simple and consistent.
    # We set a computed "w" attribute used by gravis mapping.
    H = G.copy()
    ws = []
    for u, v, d in H.edges(data=True):
        w = abs(_safe_float(d.get("weight", d.get("alpha", 1.0)), 1.0))
        ws.append(w)
    wmax = max(ws) if ws else 1.0
    for u, v, d in H.edges(data=True):
        w = abs(_safe_float(d.get("weight", d.get("alpha", 1.0)), 1.0))
        d["w"] = 1.0 + 6.0 * (w / (wmax + EPS)) * edge_size_scale

    # Node sizing by degree (mild)
    for n in H.nodes():
        deg = H.degree(n)
        H.nodes[n]["size"] = node_size + 2.0 * np.log1p(deg)

    # Style mapping
    custom = {
        "node": {
            "label": {"enable": True},
            "size": {"attribute": "size"},
            "color": {"attribute": "color"},  # <-- add this
        },
        "edge": {
            "size": {"attribute": "w"},
            "label": {"enable": bool(edge_label), "attribute": "label"},  # <-- label from attr
            "color": {"attribute": "color"},  # <-- add this
        }
    }
    if directed:
        custom["edge"]["arrow"] = {"enable": True}

    fig = gv.d3(
        H,
        use_node_size_normalization=False,
        use_edge_size_normalization=False,
    )

    # --- compatibility: gravis versions differ in how custom D3 options are injected ---
    # Try common patterns; if none exist, silently skip (network still renders).
    try:
        # Some versions expose a dict you can update
        if hasattr(fig, "custom_options") and isinstance(fig.custom_options, dict):
            fig.custom_options.update(custom)
        # Some versions store options under .options
        elif hasattr(fig, "options") and isinstance(fig.options, dict):
            fig.options.update(custom)
        # Some versions allow setting arbitrary attributes
        else:
            setattr(fig, "custom_options", custom)
    except Exception:
        pass

    return fig.to_html()


def build_egfr_control_dag(ko: pd.DataFrame, gene_name: str, topk_per_site: int) -> nx.DiGraph:
    """
    Construct EGFR-like DAG: Gene -> site -> kinase, using |delta_auc_abs| as edge weight.
    Sign: delta_auc_abs > 0 => activating (solid), else inhibitory (dashed).
    """
    eg = ko[ko["Gene"].astype(str) == str(gene_name)].copy()
    if eg.empty:
        return nx.DiGraph()

    eg["abs_delta"] = eg["delta_auc_abs"].abs()
    eg = eg[eg["abs_delta"] > 1e-10]
    if eg.empty:
        return nx.DiGraph()

    eg = (
        eg.sort_values(["Psite", "abs_delta"], ascending=[True, False])
        .groupby("Psite", as_index=False)
        .head(topk_per_site)
    )

    G = nx.DiGraph()
    root = str(gene_name)
    G.add_node(root, kind="protein")

    sites = sorted(eg["Psite"].astype(str).unique())
    kinases = sorted(eg["KnockedKinase"].astype(str).unique())

    for s in sites:
        sid = f"site::{s}"
        G.add_node(sid, kind="psite", label=s)
        G.add_edge(root, sid, weight=0.5, sign=1)

    for k in kinases:
        kid = f"kin::{k}"
        G.add_node(kid, kind="kinase", label=k)

    for _, r in eg.iterrows():
        sid = f"site::{str(r['Psite'])}"
        kid = f"kin::{str(r['KnockedKinase'])}"
        w = float(abs(r["delta_auc_abs"]))
        sign = 1 if float(r["delta_auc_abs"]) > 0 else -1
        # encode sign for potential styling
        G.add_edge(sid, kid, weight=w, sign=sign)

    return G


# =========================
# UI helpers
# =========================

def section_header(title: str):
    st.markdown(f"### {title}")


def dataframe_block(df: pd.DataFrame, name: str):
    c1, c2 = st.columns([1, 0.2])
    with c1:
        st.dataframe(df, use_container_width=True, height=420)
    with c2:
        st.download_button(
            label="Download CSV",
            data=_to_bytes_csv(df),
            file_name=f"{name}.csv",
            mime="text/csv",
        )


# =========================
# Sidebar: inputs & parameters
# =========================

st.title("Kinopt Network Readout Dashboard")

with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Kinopt XLSX", type=["xlsx"])
    st.caption("Required sheets: Observed, Estimated, Alpha Values, Beta Values")

    uploaded_input1 = st.file_uploader("input1.csv (kinase/protein time series)", type=["csv"])

    st.divider()
    st.header("Computation options")

    time_mode = st.selectbox("Time columns mode", ["Kinopt default", "Auto-detect"], index=0)

    kinase_site_source = st.selectbox("Kinase site source (P^k_j(t))", ["Observed", "Estimated"], index=0)
    beta_bound = st.number_input("β bound (for 'at bound' flag)", min_value=0.0, value=4.0, step=0.5)
    bound_tol = st.number_input("Bound tolerance", min_value=0.0, value=1e-9, step=1e-9, format="%.2e")

    ko_renorm = st.checkbox("Knockout: renormalize remaining α to sum=1", value=True)

    st.subheader("Dominance windows")
    n_early = st.slider("Early window (#timepoints)", min_value=1, max_value=8, value=4, step=1)
    n_late = st.slider("Late window (#timepoints)", min_value=1, max_value=8, value=4, step=1)

    st.subheader("Visualization thresholds")
    kinase_label_load_threshold = st.number_input("Label kinases if load >", min_value=0.0, value=1.0, step=0.1)
    outlier_band_quantile = st.slider("Obs vs recon: band quantile", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    outlier_max_labels = st.slider("Obs vs recon: max labels", min_value=0, max_value=30, value=5, step=1)

    st.subheader("Network filters")
    min_alpha_filter = st.number_input("Min |α| edge filter", min_value=0.0, value=0.0, step=0.01)

    st.subheader("EGFR-like DAG")
    egfr_topk_per_site = st.slider("Top-k kinases per site", min_value=3, max_value=50, value=17, step=1)

    st.divider()
    st.header("Rendering")
    node_size = st.slider("Network node base size", min_value=6, max_value=30, value=12, step=1)
    edge_size_scale = st.slider("Network edge width scale", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    edge_label = st.checkbox("Show edge labels (slow for large graphs)", value=False)


# =========================
# Main: guard
# =========================

if uploaded is None:
    st.info("Upload a Kinopt XLSX to start.")
    st.stop()

if uploaded_input1 is None:
    st.info("Upload a kinase/protein time series CSV to start.")
    st.stop()

file_bytes = uploaded.getvalue()
input1_bytes = uploaded_input1.getvalue() if uploaded_input1 is not None else None

params = AppParams(
    kinase_site_source=kinase_site_source,
    beta_bound=float(beta_bound),
    bound_tol=float(bound_tol),
    ko_renormalize_alpha=bool(ko_renorm),
    n_early=int(n_early),
    n_late=int(n_late),
    kinase_label_load_threshold=float(kinase_label_load_threshold),
    outlier_band_quantile=float(outlier_band_quantile),
    outlier_max_labels=int(outlier_max_labels),
    egfr_topk_per_site=int(egfr_topk_per_site),
    min_alpha_filter=float(min_alpha_filter),
)

with st.spinner("Computing analysis from XLSX..."):
    out = compute_all(file_bytes=file_bytes, input1_bytes=input1_bytes, params=params, time_mode=time_mode)

tcfg: TimeConfig = out["time"]
edges: pd.DataFrame = out["edges"]
betas: pd.DataFrame = out["betas"]
latent_summary: pd.DataFrame = out["latent_summary"]
target_dom: pd.DataFrame = out["target_dom"]
kinase_load: pd.DataFrame = out["kinase_load"]
ko: pd.DataFrame = out["ko"]
latent: Dict[str, np.ndarray] = out["latent"]
latent_site_detail: Dict[str, pd.DataFrame] = out["latent_site_detail"]
target_decomp_cache: Dict[Tuple[str, str], dict] = out["target_decomp_cache"]


# =========================
# Layout: tabs
# =========================

tab_overview, tab_tables, tab_viz, tab_drilldown, tab_network = st.tabs(
    ["Overview", "Tables", "Visualizations", "Drilldown", "Networks (gravis)"]
)

with tab_overview:
    section_header("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Edges (α)", len(edges))
    c2.metric("Kinases (β)", betas["Kinase"].nunique() if not betas.empty else 0)
    c3.metric("Targets", target_dom.shape[0])
    c4.metric("Knockout rows", ko.shape[0] if not ko.empty else 0)

    section_header("Key caveats")
    st.write(
        "This dashboard recomputes the mechanistic, model-consistent audit of signal flow: "
        "latent kinase activities A_j(t), target contributions α·A, and in-silico knockouts. "
        "All outputs are derived from the uploaded XLSX only."
    )

    section_header("Quick top lists")
    c1, c2 = st.columns(2)

    with c1:
        st.write("Top kinases by total control load")
        st.dataframe(kinase_load.head(10), use_container_width=True)

    with c2:
        st.write("Most fragile targets (dominant_overall_share)")
        st.dataframe(target_dom.head(10), use_container_width=True)


with tab_tables:
    section_header("Latent kinase activity summary")
    dataframe_block(latent_summary, "kinopt_latent_kinase_activity_summary")

    section_header("Target dominant kinases")
    dataframe_block(target_dom, "kinopt_target_dominant_kinases")

    section_header("Kinase load")
    dataframe_block(kinase_load, "kinopt_kinase_load")

    section_header("Knockout effects")
    dataframe_block(ko, "kinopt_knockout_effects")


with tab_viz:
    section_header("Kinase-level")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_bar_kinase_control_load(kinase_load, top_n=25), use_container_width=True)
    with c2:
        st.plotly_chart(fig_scatter_breadth_vs_load(kinase_load, label_threshold=params.kinase_label_load_threshold), use_container_width=True)

    st.plotly_chart(fig_scatter_boundpressure_vs_load(kinase_load), use_container_width=True)

    section_header("Target-level")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_target_dominance_distribution(target_dom), use_container_width=True)
    with c2:
        st.plotly_chart(fig_fragile_targets_bar(target_dom, top_n=25), use_container_width=True)

    st.plotly_chart(fig_obs_vs_reconstructed(target_dom, outlier_quantile=params.outlier_band_quantile, max_labels=params.outlier_max_labels),
                    use_container_width=True)

    section_header("Knockout landscape")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_knockout_distribution(ko), use_container_width=True)
    with c2:
        st.plotly_chart(fig_knockout_top_edges(ko, direction="activating", top_n=20), use_container_width=True)

    st.plotly_chart(fig_knockout_top_edges(ko, direction="suppressing", top_n=20), use_container_width=True)


with tab_drilldown:
    section_header("Kinase drilldown (latent A_j(t) and β-site composition)")
    kinases = sorted(list(latent.keys()))
    if not kinases:
        st.warning("No latent kinases computed (check Beta Values and kinase-site availability).")
    else:
        c1, c2 = st.columns([0.55, 0.45])

        with c1:
            sel_kin = st.selectbox("Select kinase", kinases, index=0)
            st.plotly_chart(fig_timeseries_kinase(latent, tcfg, sel_kin), use_container_width=True)

        with c2:
            st.write("β-site rows for selected kinase")
            df_sites = latent_site_detail.get(sel_kin, pd.DataFrame())
            if df_sites.empty:
                st.info("No β-site detail available.")
            else:
                st.dataframe(df_sites, use_container_width=True, height=420)

    st.divider()
    section_header("Target drilldown (observed vs predicted, plus top-k contributions)")
    targets = sorted(list(target_decomp_cache.keys()))
    if not targets:
        st.warning("No targets available (check Alpha Values and filtering).")
    else:
        # Build readable target strings
        target_labels = [f"{g} {p}" for (g, p) in targets]
        idx = st.selectbox("Select target (Gene Psite)", list(range(len(target_labels))), format_func=lambda i: target_labels[i])
        gene, psite = targets[idx]

        show_top_k = st.slider("Show top-k kinase contributions", min_value=1, max_value=25, value=8, step=1)
        decomp = target_decomp_cache[(gene, psite)]
        st.plotly_chart(fig_timeseries_target_decomp(tcfg, gene, psite, decomp, show_top_k=show_top_k), use_container_width=True)

        # Provide the per-target knockout subset
        st.write("Knockout effects for selected target")
        ko_sub = ko[(ko["Gene"].astype(str) == gene) & (ko["Psite"].astype(str) == psite)].copy()
        ko_sub = ko_sub.sort_values("delta_auc_abs", ascending=False)
        st.dataframe(ko_sub, use_container_width=True, height=380)


with tab_network:
    section_header("Alpha network (Kinase → Target) via gravis")
    st.caption("Targets are encoded as 'Gene|Psite'. Edge weight is α. Use filters in the sidebar to reduce size.")

    # Optional subgraph filters (interactive)
    all_kin = sorted(edges["Kinase"].unique().tolist()) if not edges.empty else []
    all_gene = sorted(edges["Gene"].unique().tolist()) if not edges.empty else []
    c1, c2, c3 = st.columns(3)
    with c1:
        kin_filter = st.multiselect("Filter by kinase (optional)", all_kin, default=[])
    with c2:
        gene_filter = st.multiselect("Filter by gene (optional)", all_gene, default=[])
    with c3:
        knockout_preview = st.selectbox("Knockout preview (remove kinase)", ["(none)"] + all_kin, index=0)

    edge_mode = st.selectbox("Edge width/color mode", ["alpha", "ko"], index=0)
    st.caption("alpha: width ~ |α|.  ko: width ~ |ΔAUC| and color by sign.")

    edges_net = edges.copy()
    if kin_filter:
        edges_net = edges_net[edges_net["Kinase"].isin(kin_filter)].copy()
    if gene_filter:
        edges_net = edges_net[edges_net["Gene"].isin(gene_filter)].copy()
    if knockout_preview != "(none)":
        edges_net = edges_net[edges_net["Kinase"].astype(str) != str(knockout_preview)].copy()

    G_alpha = build_alpha_network(edges_net, ko=ko, target_dom=target_dom, edge_mode=edge_mode)

    # Render
    html = render_gravis(G_alpha, node_size=node_size, edge_size_scale=edge_size_scale, edge_label=edge_label, directed=True)
    st.components.v1.html(html, height=720, scrolling=True)

    st.divider()
    section_header("Separate activating vs suppressing edges")
    st.caption("Split is based on the selected edge mode: alpha uses sign(α), ko uses sign(KO effect).")

    # Split edges by sign WITHOUT copying all nodes first
    G_pos = nx.DiGraph()
    G_neg = nx.DiGraph()

    for u, v, d in G_alpha.edges(data=True):
        s = d.get("sign", 0)
        if s > 0:
            G_pos.add_edge(u, v, **d)
        elif s < 0:
            G_neg.add_edge(u, v, **d)

    # Copy node attributes only for nodes that remain
    for n in list(G_pos.nodes()):
        if n in G_alpha.nodes:
            G_pos.nodes[n].update(G_alpha.nodes[n])

    for n in list(G_neg.nodes()):
        if n in G_alpha.nodes:
            G_neg.nodes[n].update(G_alpha.nodes[n])

    # (Optional) extra pruning: drop nodes with total degree == 0 (should already be none)
    G_pos.remove_nodes_from([n for n in G_pos.nodes() if G_pos.degree(n) == 0])
    G_neg.remove_nodes_from([n for n in G_neg.nodes() if G_neg.degree(n) == 0])

    st.subheader("Activating edges")
    html = render_gravis(G_pos, node_size=node_size, edge_size_scale=edge_size_scale, edge_label=edge_label, directed=True)
    st.components.v1.html(html, height=720, scrolling=True)
    st.subheader("Suppressing edges")
    html = render_gravis(G_neg, node_size=node_size, edge_size_scale=edge_size_scale, edge_label=edge_label, directed=True)
    st.components.v1.html(html, height=720, scrolling=True)

    st.divider()
    section_header("EGFR control logic DAG via gravis (Gene → site → kinase)")
    gene_for_dag = st.selectbox("Gene for DAG", options=sorted(ko["Gene"].astype(str).unique().tolist()) if not ko.empty else ["EGFR"], index=0)
    G_dag = build_egfr_control_dag(ko, gene_name=gene_for_dag, topk_per_site=params.egfr_topk_per_site)

    if G_dag.number_of_nodes() == 0:
        st.info(f"No DAG available for {gene_for_dag} (no knockout rows or all effects ~0).")
    else:
        # Encode edge style cues
        # gravis doesn't guarantee dashed line support across all renderers; we store sign and show it in hover/labels if needed.
        for u, v, d in G_dag.edges(data=True):
            sign = int(d.get("sign", 1))
            d["label"] = "activating" if sign > 0 else "inhibitory"

        html2 = render_gravis(G_dag, node_size=node_size, edge_size_scale=edge_size_scale, edge_label=False, directed=True)
        st.components.v1.html(html2, height=720, scrolling=True)

