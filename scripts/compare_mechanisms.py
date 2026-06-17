from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import inspect
import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import gravis as gv
import networkx as nx
from pathlib import Path

from networkmodel import config
import logging
import networkmodel.network as network_mod
from networkmodel.network import Index, KinaseInput, System
from networkmodel.simulate import simulate_and_measure
from networkmodel.SteadyStateAnalysis import simulate_until_steady
from networkmodel.io import load_data
from networkmodel.BuildMatrix import build_W_parallel, build_tf_matrix

st.set_page_config(page_title="PhoskinTime Global Knockout", layout="wide")
RESULTS_DIR_PICKED = Path("./results_network_distributive")
MODEL_NAMES = {
    0: "distributive",
    1: "sequential",
    2: "combinatorial",
    4: "saturating",
}

INTENDED_MODEL = 0  # change to 2 for combinatorial

def _standardize_tf_columns(df_tf: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the column names of a DataFrame for transcription factors (TF).

    This function ensures that the DataFrame contains standardized column names
    of "tf" (representing the transcription factor) and"""
    if df_tf is None or df_tf.empty:
        return df_tf
    cols = df_tf.columns.tolist()
    if "tf" in cols and "target" in cols:
        return df_tf

    # common alternates used in exports
    src_col = "Source" if "Source" in cols else ("source" if "source" in cols else None)
    tgt_col = "Target" if "Target" in cols else ("target" if "target" in cols else None)

    if src_col is None or tgt_col is None:
        # last-resort: assume first two columns are (src,tgt)
        src_col = cols[0]
        tgt_col = cols[1] if len(cols) > 1 else cols[0]

    df = df_tf.copy()
    df = df.rename(columns={src_col: "tf", tgt_col: "target"})
    return df


def _build_df_tf_model(
        df_tf_raw: pd.DataFrame,
        df_kin: pd.DataFrame,
        df_prot: pd.DataFrame,
        df_rna: pd.DataFrame,
        df_pho: pd.DataFrame,
        kin_beta_map: dict | None,
        tf_beta_map: dict | None,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Builds a transcription factor (TF) model dataframe and maps orphan transcription factors to
    proxy proteins based on their targets and associated scoring logic.

    This function processes raw transcription factor data and filters it to include only entries
    with valid targets present in"""
    if df_tf_raw is None or df_tf_raw.empty:
        return df_tf_raw, {}

    df_tf = _standardize_tf_columns(df_tf_raw)

    required = {"tf", "target"}
    missing = required - set(df_tf.columns)
    if missing:
        raise ValueError(f"TF net missing columns: {missing}. Found columns: {list(df_tf.columns)}")

    proteins_with_sites = set(df_kin["protein"].astype(str).str.strip().unique())
    kinase_set = set(df_kin["kinase"].astype(str).str.strip().unique())

    target_universe = (
            set(df_kin["protein"].astype(str).str.strip().unique())
            | set(df_kin["kinase"].astype(str).str.strip().unique())
            | set(df_prot["protein"].astype(str).str.strip().unique())
            | set(df_rna["protein"].astype(str).str.strip().unique())
            | set(df_pho["protein"].astype(str).str.strip().unique())
    )

    df_tf_model = df_tf[df_tf["target"].astype(str).str.strip().isin(target_universe)].copy()
    df_tf_model["tf"] = df_tf_model["tf"].astype(str).str.strip()
    df_tf_model["target"] = df_tf_model["target"].astype(str).str.strip()

    orphan_tfs = sorted(set(df_tf_model["tf"].unique()) - proteins_with_sites)

    TF_PROXY_MAP: dict[str, str] = {}

    def _proxy_score(orphan: str, candidate: str) -> float:
        score = 0.0
        if tf_beta_map and orphan in tf_beta_map:
            score += float(tf_beta_map[orphan])
        if kin_beta_map and candidate in kin_beta_map:
            score += float(kin_beta_map[candidate])
        return score

    for orphan in orphan_tfs:
        targets = df_tf_model.loc[df_tf_model["tf"] == orphan, "target"].astype(str)
        cand1 = [t for t in targets if t in kinase_set]  # Priority 1
        cand2 = [t for t in targets if t in proteins_with_sites]  # Priority 2
        candidates = cand1 if cand1 else cand2
        if candidates:
            best = sorted(candidates, key=lambda c: (-_proxy_score(orphan, c), c))[0]
            TF_PROXY_MAP[orphan] = best

    if TF_PROXY_MAP:
        df_tf_model["tf_original"] = df_tf_model["tf"]
        df_tf_model["tf"] = df_tf_model["tf"].replace(TF_PROXY_MAP)

    # enforce representability: TF source must be a signaling protein-with-sites
    df_tf_model = df_tf_model[df_tf_model["tf"].isin(proteins_with_sites)].copy()

    keep_cols = [c for c in df_tf_model.columns if c in ("tf", "target", "alpha", "tf_original")]
    df_tf_model = df_tf_model[keep_cols].drop_duplicates()

    return df_tf_model, TF_PROXY_MAP


def _mechanistic_phospho_filter(df_kin: pd.DataFrame, df_pho: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a phosphorylation site dataset to retain only the rows that match protein and phosphorylation site 
    pairs present in a kinase dataset. This function ensures that only mechanistically relevant phosphorylation 
    site data remains.

    Args:
        df_kin (pd.DataFrame): DataFrame containing kinase information with 'protein' and 'psite' columns. 
            Each row represents a kinase and its associated phosphorylation site.
        df_pho (pd.DataFrame): DataFrame containing phosphorylation site data with 'protein' and 'psite' 
            columns. Rows in this DataFrame are filtered"""
    if df_pho is None or df_pho.empty:
        return df_pho

    df_kin2 = df_kin.copy()
    df_pho2 = df_pho.copy()

    for _df in (df_kin2, df_pho2):
        _df["protein"] = _df["protein"].astype(str).str.strip()
    df_kin2["psite"] = df_kin2["psite"].astype(str).str.strip()
    df_pho2["psite"] = df_pho2["psite"].astype(str).str.strip()

    kin_site_pairs = set(zip(df_kin2["protein"].values, df_kin2["psite"].values))
    pairs = list(zip(df_pho2["protein"].values, df_pho2["psite"].values))
    keep = np.fromiter(((p, s) in kin_site_pairs for (p, s) in pairs), dtype=bool, count=len(pairs))
    return df_pho2.loc[keep].copy()

def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in paths))


def _load_fitted_params_picked(results_dir: Path, defaults: dict) -> dict:
    path = _first_existing(
        results_dir / "fitted_params_picked.json",
        results_dir / "tables" / "fitted_params_picked.json",
    )

    with open(path, "r") as f:
        raw = json.load(f)

    # Handle either flat JSON or nested export JSON.
    if "params" in raw and isinstance(raw["params"], dict):
        raw = raw["params"]
    elif "best_params" in raw and isinstance(raw["best_params"], dict):
        raw = raw["best_params"]
    elif "fitted_params" in raw and isinstance(raw["fitted_params"], dict):
        raw = raw["fitted_params"]

    params = {}

    for key, default_value in defaults.items():
        if key not in raw:
            raise KeyError(f"{path} does not contain fitted parameter key: {key}")

        value = raw[key]

        if isinstance(default_value, np.ndarray):
            arr = np.asarray(value, dtype=float)

            if arr.shape != default_value.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: JSON has {arr.shape}, "
                    f"expected {default_value.shape}"
                )

            params[key] = arr
        else:
            params[key] = float(value)

    return params

def _load_picked_predictions(results_dir: Path):
    wt_dfp = pd.read_csv(results_dir / "pred_prot_picked.csv")
    wt_dfr = pd.read_csv(results_dir / "pred_rna_picked.csv")
    wt_pho = pd.read_csv(results_dir / "pred_phospho_picked.csv")

    for df in (wt_dfp, wt_dfr, wt_pho):
        if "protein" in df.columns:
            df["protein"] = df["protein"].astype(str).str.strip().str.upper()
        if "psite" in df.columns:
            df["psite"] = df["psite"].astype(str).str.strip()
        if "time" in df.columns:
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
        if "pred_fc" in df.columns:
            df["pred_fc"] = pd.to_numeric(df["pred_fc"], errors="coerce")

    return wt_dfp, wt_dfr, wt_pho

@st.cache_resource
def load_system():
    """
    Caches and initializes a System object along with associated parameters, indices, and data models.

    This function performs a comprehensive setup to construct a `System` object for further analysis.
    It loads data, preprocesses various datasets (e.g., kinase, transcription factor, and protein data),
    and compiles the necessary matrices and configurations required by the system. Additionally, it handles
    parameters derived from an optimization run, reconstructing necessary input for reanalysis or dashboarding.

    Returns:
        tuple: A tuple containing the"""
    # IMPORTANT: ensure model selection matches run before building Index/System.
    # networkmodel.network imports MODEL as a module-level value, so update both.
    config.MODEL = INTENDED_MODEL
    network_mod.MODEL = INTENDED_MODEL

    model_name = MODEL_NAMES.get(INTENDED_MODEL, f"unknown_MODEL_{INTENDED_MODEL}")

    logging.getLogger(__name__).warning(
        "[Dashboard] Intended networkmodel MODEL=%d (%s); results_dir=%s",
        INTENDED_MODEL,
        model_name,
        RESULTS_DIR_PICKED,
    )

    results_dir = RESULTS_DIR_PICKED

    class Args:
        kinase_net, tf_net = config.KINASE_NET_FILE, config.TF_NET_FILE
        ms, rna, phospho = config.MS_DATA_FILE, config.RNA_DATA_FILE, config.PHOSPHO_DATA_FILE
        kinopt, tfopt = None, None
        normalize_fc_steady = False

    df_kin, df_tf_raw, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(Args())

    # runner: strict mechanistic phospho filter
    df_pho = _mechanistic_phospho_filter(df_kin, df_pho)

    # runner: TF proxying + representability filtering
    df_tf_model, TF_PROXY_MAP = _build_df_tf_model(
        df_tf_raw=df_tf_raw,
        df_kin=df_kin,
        df_prot=df_prot,
        df_rna=df_rna,
        df_pho=df_pho,
        kin_beta_map=kin_beta_map,
        tf_beta_map=tf_beta_map,
    )

    # runner: build Index with beta maps
    idx = Index(df_kin, tf_interactions=df_tf_model, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)

    # runner: restrict observations to idx.proteins
    if df_prot is not None and not df_prot.empty:
        df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    if df_rna is not None and not df_rna.empty:
        df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    if df_pho is not None and not df_pho.empty:
        df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    # runner: build kinase input from observed kinase trajectories only
    df_prot_kin = df_prot[df_prot["protein"].isin(idx.kinases)].copy() if df_prot is not None else df_prot
    kin_in = KinaseInput(idx.kinases, df_prot_kin)

    # runner: build matrices
    W_global = build_W_parallel(df_kin, idx, n_cores=1)
    tf_mat = build_tf_matrix(df_tf_model, idx, tf_beta_map=tf_beta_map, kin_beta_map=kin_beta_map)

    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    # runner: initialize c_k from beta priors
    c_k_init = np.array([max(0.01, float(kin_beta_map.get(k, 1.0))) for k in idx.kinases])

    defaults = {
        "c_k": c_k_init,
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1,
    }

    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    # Load the exact parameter set used by the pipeline-selected solution.
    # Do not use pareto_X[0]; it may not be the picked/best solution.
    best_params = _load_fitted_params_picked(results_dir, defaults)

    s_rates_path = results_dir / "S_rates_picked.csv"
    s_rates = pd.read_csv(s_rates_path) if s_rates_path.exists() else pd.DataFrame()

    # return df_tf_model for the graph layer (it may include tf_original column)
    return sys, idx, best_params, df_tf_model, s_rates


def run_sim(sys, idx, mod_params):
    """
    Updates a system with given modification parameters and executes a simulation to return measurements.

    This function takes a system object, updates its attributes based on the provided 
    modification parameters, and runs a simulation. Measurements for protein, RNA, and 
    phosphorylation levels are taken at predefined time points.

    Args:
        sys: The system object to be updated and simulated.
        idx: An integer index specifying which part of the system to simulate.
        mod_params: A dictionary containing the modification parameters to update 
            the system with.

    Returns:
        The measurements obtained from the simulation"""
    sys.update(**mod_params)
    return simulate_and_measure(sys, idx, config.TIME_POINTS_PROTEIN, config.TIME_POINTS_RNA,
                                config.TIME_POINTS_PHOSPHO)

def extract_fc_from_Y(Y, idx, t, protein, normalize=True):
    """Extract RNA, total protein, and phospho-site trajectories for one protein."""
    p_idx = idx.p2i[protein]
    st_y = idx.offset_y[p_idx]

    rna_vals = Y[:, st_y]
    prot_vals = Y[:, st_y + 1]

    ns = idx.n_sites[p_idx]
    site_names = idx.sites[p_idx]

    if ns > 0:
        psite_vals = Y[:, st_y + 2: st_y + 2 + ns]
        phos_sum = np.sum(psite_vals, axis=1)
        total_prot = prot_vals + phos_sum
    else:
        psite_vals = np.zeros((len(t), 0))
        total_prot = prot_vals

    if normalize:
        rna_base = rna_vals[0] if abs(rna_vals[0]) > 1e-12 else 1.0
        prot_base = total_prot[0] if abs(total_prot[0]) > 1e-12 else 1.0

        rna_vals = rna_vals / rna_base
        total_prot = total_prot / prot_base

        if ns > 0:
            site_base = psite_vals[0, :].copy()
            site_base[np.abs(site_base) < 1e-12] = 1.0
            psite_vals = psite_vals / site_base

    df = pd.DataFrame(
        {
            "time": t,
            "rna": rna_vals,
            "protein": total_prot,
        }
    )

    if ns > 0:
        df_ps = pd.DataFrame(psite_vals, columns=site_names)
        df_ps["time"] = t
        df_long = df_ps.melt(
            id_vars="time",
            var_name="psite",
            value_name="psite_value",
        )
        df = df.merge(df_long, on="time", how="left")

    return df
def _safe_min_max(values: np.ndarray) -> tuple[float, float]:
    """Return finite min/max values for dashboard diagnostics."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def _simulate_until_steady_optional_y0(sys_obj, t_max: float, n_points: int, y0: np.ndarray | None = None):
    """
    Call simulate_until_steady with a fixed initial state if the installed function exposes
    such an argument. Otherwise fall back to the normal call.

    This keeps compare_mechanisms.py script-only and avoids modifying networkmodel modules.
    """
    sig = inspect.signature(simulate_until_steady)
    kwargs = {
        "t_max": float(t_max),
        "n_points": int(n_points),
    }

    used_fixed_y0 = False

    if y0 is not None:
        for y0_name in ("y0", "initial_state", "initial_conditions", "y_init"):
            if y0_name in sig.parameters:
                kwargs[y0_name] = np.asarray(y0, dtype=float)
                used_fixed_y0 = True
                break

    t, Y = simulate_until_steady(sys_obj, **kwargs)
    return t, Y, used_fixed_y0


def _align_Y_to_reference_initial(
    Y: np.ndarray,
    Y_ref: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Align a trajectory to the reference initial state.

    This removes artificial t=0 inflation caused by separately initialized
    perturbation trajectories. Multiplicative alignment preserves non-negativity
    and relative dynamic shape better than a blind additive shift.
    """
    Y = np.asarray(Y, dtype=float)
    Y_ref = np.asarray(Y_ref, dtype=float)

    if Y.ndim != 2 or Y_ref.ndim != 2:
        raise ValueError(f"Expected 2D trajectories, got Y={Y.shape}, Y_ref={Y_ref.shape}")

    if Y.shape[1] != Y_ref.shape[1]:
        raise ValueError(
            f"State dimension mismatch: KO has {Y.shape[1]} states, WT has {Y_ref.shape[1]} states"
        )

    Y_aligned = Y.copy()

    ko0 = Y[0, :].astype(float)
    wt0 = Y_ref[0, :].astype(float)

    scale = np.ones_like(ko0, dtype=float)
    can_scale = np.abs(ko0) > eps

    scale[can_scale] = wt0[can_scale] / ko0[can_scale]
    Y_aligned[:, can_scale] = Y[:, can_scale] * scale[can_scale]

    # If a KO initial state is numerically zero but WT is not, use additive correction.
    cannot_scale = ~can_scale
    if np.any(cannot_scale):
        Y_aligned[:, cannot_scale] = Y[:, cannot_scale] + (wt0[cannot_scale] - ko0[cannot_scale])

    # Enforce exact equality at t=0 after numerical alignment.
    Y_aligned[0, :] = wt0

    return Y_aligned


def _run_forward_wt_ko_for_display(
    sys_obj,
    best_params: dict,
    ko_params: dict,
    t_max: float,
    n_points: int,
):
    """
    Run WT and KO forward simulations for the dashboard.

    WT is the fitted baseline.
    KO is simulated with perturbed parameters, then displayed from the WT fitted
    initial state. If simulate_until_steady supports y0, that is used directly.
    Otherwise the KO trajectory is state-aligned for display.
    """
    sys_obj.update(**best_params)
    t_wt, Y_wt, _ = _simulate_until_steady_optional_y0(
        sys_obj,
        t_max=t_max,
        n_points=n_points,
        y0=None,
    )

    Y_wt = np.asarray(Y_wt, dtype=float)
    wt_y0 = Y_wt[0, :].copy()

    sys_obj.update(**ko_params)
    t_ko, Y_ko_raw, used_fixed_y0 = _simulate_until_steady_optional_y0(
        sys_obj,
        t_max=t_max,
        n_points=n_points,
        y0=wt_y0,
    )

    Y_ko_raw = np.asarray(Y_ko_raw, dtype=float)

    if used_fixed_y0:
        Y_ko_plot = Y_ko_raw.copy()
        Y_ko_plot[0, :] = wt_y0
        alignment_mode = "fixed y0 passed to simulate_until_steady"
    else:
        Y_ko_plot = _align_Y_to_reference_initial(Y_ko_raw, Y_wt)
        alignment_mode = "post-simulation WT-initial-state alignment"

    sys_obj.update(**best_params)

    return {
        "t_fine": np.asarray(t_wt, dtype=float),
        "Y_wt": Y_wt,
        "t_ko": np.asarray(t_ko, dtype=float),
        "Y_ko_raw": Y_ko_raw,
        "Y_ko_plot": Y_ko_plot,
        "used_fixed_y0": used_fixed_y0,
        "alignment_mode": alignment_mode,
    }


def extract_phosphosite_states_from_Y(
    Y: np.ndarray,
    idx: Index,
    t: np.ndarray,
    protein: str,
    normalize_to_t0: bool = False,
    reference_raw_states: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extract raw or reference-normalized phosphosite ODE states for one protein.

    This intentionally reads directly from the ODE state vector. It does not use
    exported fitted predictions, pred_fc, or simulate_and_measure() output.

    If reference_raw_states is provided, normalization uses its t0 denominator.
    That is required for intervention plots: WT and KO must use the same WT
    t0 phosphosite reference.
    """
    p_idx = idx.p2i[protein]
    ns = int(idx.n_sites[p_idx])
    site_names = list(idx.sites[p_idx])

    if ns <= 0:
        empty = pd.DataFrame(columns=["time", "psite", "psite_value"])
        return empty, np.zeros((len(t), 0)), np.zeros((len(t), 0))

    start = int(idx.offset_y[p_idx]) + 2
    stop = start + ns

    raw_states = np.asarray(Y[:, start:stop], dtype=float)
    plotted_states = raw_states.copy()

    if normalize_to_t0:
        ref = raw_states if reference_raw_states is None else np.asarray(reference_raw_states, dtype=float)

        if ref.shape[1] != raw_states.shape[1]:
            raise ValueError(
                f"Reference phosphosite state dimension mismatch: ref={ref.shape}, raw={raw_states.shape}"
            )

        denom = ref[0, :].copy()
        denom[np.abs(denom) < 1e-12] = 1.0
        plotted_states = raw_states / denom

    df = pd.DataFrame(plotted_states, columns=site_names)
    df["time"] = np.asarray(t, dtype=float)

    df_long = df.melt(
        id_vars="time",
        var_name="psite",
        value_name="psite_value",
    )

    return df_long, raw_states, plotted_states

def _extract_raw_phosphosite_matrix(
    Y: np.ndarray,
    idx: Index,
    protein: str,
) -> tuple[list[str], np.ndarray]:
    """Return raw phosphosite ODE states for one protein as (site_names, matrix)."""
    p_idx = idx.p2i[protein]
    ns = int(idx.n_sites[p_idx])
    site_names = list(idx.sites[p_idx])

    if ns <= 0:
        return [], np.zeros((Y.shape[0], 0), dtype=float)

    start = int(idx.offset_y[p_idx]) + 2
    stop = start + ns

    return site_names, np.asarray(Y[:, start:stop], dtype=float)


def _interp_state_matrix(
    t_src: np.ndarray,
    values_src: np.ndarray,
    t_target: np.ndarray,
) -> np.ndarray:
    """Interpolate each state column onto requested target times."""
    t_src = np.asarray(t_src, dtype=float)
    t_target = np.asarray(t_target, dtype=float)
    values_src = np.asarray(values_src, dtype=float)

    if values_src.size == 0:
        return np.zeros((len(t_target), 0), dtype=float)

    order = np.argsort(t_src)
    t_src = t_src[order]
    values_src = values_src[order, :]

    out = np.empty((len(t_target), values_src.shape[1]), dtype=float)

    for j in range(values_src.shape[1]):
        out[:, j] = np.interp(t_target, t_src, values_src[:, j])

    return out


def _build_inspector_ko_phosphosite_intervention_df(
    sys_obj,
    idx: Index,
    best_params: dict,
    ko_params: dict,
    protein: str,
    wt_pho_data: pd.DataFrame,
    n_points: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build KO phosphosite inspector values on the same scale as WT picked pred_fc.

    Correct scale rule:
        KO_display_fc(t) = WT_picked_pred_fc(t) * KO_raw_state(t) / WT_raw_state(t)

    This avoids the bad t0 normalization:
        KO_raw_state(t) / WT_raw_state(t0)

    which can inflate values to 20-50 when WT t0 state is around 0.01.
    """
    if wt_pho_data is None or wt_pho_data.empty:
        empty = pd.DataFrame(columns=["time", "psite", "psite_value"])
        debug = pd.DataFrame()
        return empty, debug

    required = {"protein", "psite", "time", "pred_fc"}
    missing = required - set(wt_pho_data.columns)
    if missing:
        raise ValueError(
            f"wt_pho_data is missing columns required for phosphosite inspector: {missing}"
        )

    target_times = np.asarray(
        sorted(wt_pho_data["time"].dropna().unique()),
        dtype=float,
    )

    if target_times.size == 0:
        empty = pd.DataFrame(columns=["time", "psite", "psite_value"])
        debug = pd.DataFrame()
        return empty, debug

    t_max_local = float(np.max(target_times))
    n_points_local = max(int(n_points), int(target_times.size) * 50)

    forward = _run_forward_wt_ko_for_display(
        sys_obj=sys_obj,
        best_params=best_params,
        ko_params=ko_params,
        t_max=t_max_local,
        n_points=n_points_local,
    )

    t_wt = np.asarray(forward["t_fine"], dtype=float)
    Y_wt = np.asarray(forward["Y_wt"], dtype=float)

    t_ko = np.asarray(forward["t_ko"], dtype=float)
    Y_ko_plot = np.asarray(forward["Y_ko_plot"], dtype=float)

    site_names, raw_wt_full = _extract_raw_phosphosite_matrix(Y_wt, idx, protein)
    _, raw_ko_full = _extract_raw_phosphosite_matrix(Y_ko_plot, idx, protein)

    if not site_names:
        empty = pd.DataFrame(columns=["time", "psite", "psite_value"])
        debug = pd.DataFrame()
        return empty, debug

    raw_wt_at_obs = _interp_state_matrix(t_wt, raw_wt_full, target_times)
    raw_ko_at_obs = _interp_state_matrix(t_ko, raw_ko_full, target_times)

    rows = []
    eps = 1e-12

    for j, site in enumerate(site_names):
        wt_site = wt_pho_data[wt_pho_data["psite"].astype(str) == str(site)].copy()

        if wt_site.empty:
            continue

        wt_site["time"] = pd.to_numeric(wt_site["time"], errors="coerce")
        wt_site["pred_fc"] = pd.to_numeric(wt_site["pred_fc"], errors="coerce")
        wt_site = wt_site.dropna(subset=["time", "pred_fc"])

        if wt_site.empty:
            continue

        # Map observation time -> row in target_times.
        time_to_i = {float(t): i for i, t in enumerate(target_times)}

        for r in wt_site.itertuples(index=False):
            t_val = float(r.time)
            i = time_to_i[t_val]

            wt_raw = float(raw_wt_at_obs[i, j])
            ko_raw = float(raw_ko_at_obs[i, j])
            wt_picked_fc = float(r.pred_fc)

            if abs(wt_raw) < eps:
                # Do not divide by a near-zero state. If WT raw is too small,
                # treat KO effect ratio as neutral rather than creating inflation.
                effect_ratio = 1.0
            else:
                effect_ratio = ko_raw / wt_raw

            ko_display_fc = wt_picked_fc * effect_ratio

            rows.append(
                {
                    "protein": protein,
                    "psite": str(site),
                    "time": t_val,
                    "psite_value": ko_display_fc,
                    "wt_picked_fc": wt_picked_fc,
                    "wt_raw_state": wt_raw,
                    "ko_raw_state": ko_raw,
                    "effect_ratio_ko_over_wt": effect_ratio,
                }
            )

    ko_long = pd.DataFrame(rows)

    if raw_wt_full.shape == raw_ko_full.shape and raw_wt_full.size:
        t0_max_abs_diff = float(np.max(np.abs(raw_wt_full[0, :] - raw_ko_full[0, :])))
    else:
        t0_max_abs_diff = float("nan")

    if ko_long.empty:
        debug = pd.DataFrame()
    else:
        debug = pd.DataFrame(
            {
                "metric": [
                    "WT raw phosphosite min",
                    "WT raw phosphosite max",
                    "KO displayed raw phosphosite min",
                    "KO displayed raw phosphosite max",
                    "KO/WT raw effect ratio min",
                    "KO/WT raw effect ratio max",
                    "KO inspector displayed FC min",
                    "KO inspector displayed FC max",
                    "max abs WT-vs-KO phosphosite difference at t0",
                ],
                "value": [
                    float(np.nanmin(raw_wt_full)),
                    float(np.nanmax(raw_wt_full)),
                    float(np.nanmin(raw_ko_full)),
                    float(np.nanmax(raw_ko_full)),
                    float(np.nanmin(ko_long["effect_ratio_ko_over_wt"])),
                    float(np.nanmax(ko_long["effect_ratio_ko_over_wt"])),
                    float(np.nanmin(ko_long["psite_value"])),
                    float(np.nanmax(ko_long["psite_value"])),
                    t0_max_abs_diff,
                ],
            }
        )

    return ko_long, debug

# --- UI Setup ---
st.title("🧪 Global Signaling & Transcriptional Knockout Explorer")
st.caption("This dashboard builds the networkmodel system and runs WT/KO simulations on demand.")

with st.sidebar:
    if st.button("Clear cached model", key="clear-compare-cache"):
        st.cache_resource.clear()
        st.session_state.pop("compare_mechanisms_started", None)
        st.rerun()

    start_dashboard = st.button(
        "Load model and start dashboard",
        type="primary",
        key="start-compare-mechanisms",
    )

if start_dashboard:
    st.session_state["compare_mechanisms_started"] = True

if not st.session_state.get("compare_mechanisms_started", False):
    st.info("Click **Load model and start dashboard** in the sidebar. First load may take several minutes.")
    st.stop()

with st.spinner("Loading data, building network matrices, and reconstructing fitted parameters..."):
    sys, idx, best_params, df_tf_model, s_rates = load_system()

model_name = MODEL_NAMES.get(INTENDED_MODEL, f"unknown_MODEL_{INTENDED_MODEL}")

if not st.session_state.get("model_banner_shown", False):
    st.success(
        f"Loaded intended networkmodel: MODEL={INTENDED_MODEL} ({model_name}); "
        f"results_dir={RESULTS_DIR_PICKED}"
    )
    st.session_state["model_banner_shown"] = True

st.sidebar.info(f"Model: MODEL={INTENDED_MODEL} ({model_name})")

st.sidebar.header("🕹️ Control Panel")

ko_params = {
    k: v.copy() if isinstance(v, np.ndarray) else float(v) if np.isscalar(v) else v
    for k, v in best_params.items()
}

st.sidebar.subheader("Perturbations")

def _scale_protein_param(param_name: str, proteins: list[str], factor: float) -> None:
    if param_name not in ko_params or not proteins:
        return
    ids = [idx.p2i[p] for p in proteins if p in idx.p2i]
    if ids:
        ko_params[param_name][ids] *= float(factor)


def _scale_kinase_param(param_name: str, kinases: list[str], factor: float) -> None:
    if param_name not in ko_params or not kinases:
        return
    ids = [idx.k2i[k] for k in kinases if k in idx.k2i]
    if ids:
        ko_params[param_name][ids] *= float(factor)


def _site_rows_for_proteins(proteins: list[str]) -> list[int]:
    rows: list[int] = []
    for p in proteins:
        if p not in idx.p2i:
            continue

        p_i = idx.p2i[p]
        start = int(idx.offset_s[p_i])
        stop = start + int(idx.n_sites[p_i])
        rows.extend(range(start, stop))

    return rows


def _scale_site_param_by_protein(param_name: str, proteins: list[str], factor: float) -> None:
    if param_name not in ko_params or not proteins:
        return
    rows = _site_rows_for_proteins(proteins)
    if rows:
        ko_params[param_name][rows] *= float(factor)


with st.sidebar.expander("Kinase activity", expanded=True):
    kinase_activity_targets = st.multiselect(
        "Inhibit kinase activity",
        options=list(idx.kinases),
        key="ko_kinase_activity_targets",
    )
    kinase_activity_scale = st.slider(
        "Kinase activity scale",
        0.0, 1.0, 0.0, 0.05,
        key="ko_kinase_activity_scale",
        help="0 = full kinase activity inhibition, 1 = WT",
    )

with st.sidebar.expander("mRNA kinetics", expanded=False):
    mrna_synthesis_targets = st.multiselect(
        "Inhibit mRNA synthesis",
        options=list(idx.proteins),
        key="ko_mrna_synthesis_targets",
    )
    mrna_synthesis_scale = st.slider(
        "mRNA synthesis scale",
        0.0, 1.0, 1.0, 0.05,
        key="ko_mrna_synthesis_scale",
    )

    mrna_degradation_targets = st.multiselect(
        "Inhibit mRNA degradation",
        options=list(idx.proteins),
        key="ko_mrna_degradation_targets",
    )
    mrna_degradation_scale = st.slider(
        "mRNA degradation scale",
        0.0, 1.0, 1.0, 0.05,
        key="ko_mrna_degradation_scale",
    )

with st.sidebar.expander("Protein kinetics", expanded=True):
    protein_synthesis_targets = st.multiselect(
        "Inhibit protein synthesis / translation",
        options=list(idx.proteins),
        key="ko_protein_synthesis_targets",
    )
    protein_synthesis_scale = st.slider(
        "Protein synthesis scale",
        0.0, 1.0, 0.0, 0.05,
        key="ko_protein_synthesis_scale",
        help="0 = full translation/protein synthesis inhibition, 1 = WT",
    )

    protein_degradation_targets = st.multiselect(
        "Inhibit protein degradation",
        options=list(idx.proteins),
        key="ko_protein_degradation_targets",
    )
    protein_degradation_scale = st.slider(
        "Protein degradation scale",
        0.0, 1.0, 1.0, 0.05,
        key="ko_protein_degradation_scale",
    )

with st.sidebar.expander("Phospho-site turnover", expanded=False):
    phospho_loss_targets = st.multiselect(
        "Inhibit dephosphorylation / phosphodegradation for proteins",
        options=list(idx.proteins),
        key="ko_phospho_loss_targets",
    )
    phospho_loss_scale = st.slider(
        "Phospho-loss scale",
        0.0, 1.0, 1.0, 0.05,
        key="ko_phospho_loss_scale",
        help="0 = block phospho-site loss, 1 = WT",
    )

with st.sidebar.expander("Transcriptional regulation", expanded=False):
    tf_efficacy_targets = st.multiselect(
        "Inhibit TF/transcriptional efficacy",
        options=list(idx.proteins),
        key="ko_tf_efficacy_targets",
    )
    tf_efficacy_scale = st.slider(
        "TF efficacy scale",
        0.0, 1.0, 1.0, 0.05,
        key="ko_tf_efficacy_scale",
    )

    tf_global_scale = st.slider(
        "Global TF-drive scale",
        0.0, 1.0, 1.0, 0.05,
        key="ko_tf_global_scale",
    )


# Apply perturbations.
# Do not mutate sys.kin.Kmat or W_global.
# Keep topology fixed; perturb only fitted dynamic parameters.

_scale_kinase_param("c_k", kinase_activity_targets, kinase_activity_scale)

_scale_protein_param("A_i", mrna_synthesis_targets, mrna_synthesis_scale)
_scale_protein_param("B_i", mrna_degradation_targets, mrna_degradation_scale)

_scale_protein_param("C_i", protein_synthesis_targets, protein_synthesis_scale)
_scale_protein_param("D_i", protein_degradation_targets, protein_degradation_scale)

_scale_site_param_by_protein("Dp_i", phospho_loss_targets, phospho_loss_scale)

_scale_protein_param("E_i", tf_efficacy_targets, tf_efficacy_scale)

if "tf_scale" in ko_params:
    ko_params["tf_scale"] = float(ko_params["tf_scale"]) * float(tf_global_scale)


all_selected_targets = (
    list(kinase_activity_targets)
    + list(mrna_synthesis_targets)
    + list(mrna_degradation_targets)
    + list(protein_synthesis_targets)
    + list(protein_degradation_targets)
    + list(phospho_loss_targets)
    + list(tf_efficacy_targets)
)

has_global_tf_perturbation = abs(float(tf_global_scale) - 1.0) > 1e-12
has_perturbation = bool(all_selected_targets) or has_global_tf_perturbation

ko_type = "Multi-perturbation" if has_perturbation else "None"
target = all_selected_targets[0] if all_selected_targets else None
scale = 1.0

perturbation_signature = {
    "kinase_activity": (tuple(kinase_activity_targets), float(kinase_activity_scale)),
    "mrna_synthesis": (tuple(mrna_synthesis_targets), float(mrna_synthesis_scale)),
    "mrna_degradation": (tuple(mrna_degradation_targets), float(mrna_degradation_scale)),
    "protein_synthesis": (tuple(protein_synthesis_targets), float(protein_synthesis_scale)),
    "protein_degradation": (tuple(protein_degradation_targets), float(protein_degradation_scale)),
    "phospho_loss": (tuple(phospho_loss_targets), float(phospho_loss_scale)),
    "tf_efficacy": (tuple(tf_efficacy_targets), float(tf_efficacy_scale)),
    "tf_global": float(tf_global_scale),
}

with st.spinner("Loading picked WT predictions and running perturbation..."):
    wt_dfp, wt_dfr, wt_pho = _load_picked_predictions(RESULTS_DIR_PICKED)

    if ko_type == "None":
        ko_dfp = wt_dfp.copy()
        ko_dfr = wt_dfr.copy()
        ko_pho = wt_pho.copy()
    else:
        ko_dfp, ko_dfr, _ko_pho_bad_fc = run_sim(sys, idx, ko_params)

        # Do not trust phospho pred_fc from simulate_and_measure here.
        # It is using a different/too-small FC denominator in this dashboard path.
        ko_pho = _ko_pho_bad_fc.copy()


# Leave the mutable System in a known baseline state after the comparison.
sys.update(**best_params)

# --- Versatile Visualization: The Impact Scatter ---
st.header("🎯 System-Wide Impact Analysis")

final_t = wt_dfr["time"].max()
comparison = pd.DataFrame(
    {
        "protein": wt_dfr[wt_dfr["time"] == final_t]["protein"].values,
        "WT_final": wt_dfr[wt_dfr["time"] == final_t]["pred_fc"].values,
        "KO_final": ko_dfr[ko_dfr["time"] == final_t]["pred_fc"].values,
    }
)

comparison["log2_fc"] = (comparison["KO_final"] + 1e-6) / (comparison["WT_final"] + 1e-6)

sensitivity = []
for p in comparison["protein"]:
    wt_vals = wt_dfr[wt_dfr["protein"] == p]["pred_fc"].values
    ko_vals = ko_dfr[ko_dfr["protein"] == p]["pred_fc"].values
    sensitivity.append(np.sum(np.abs(wt_vals - ko_vals)))
comparison["sensitivity_score"] = sensitivity

fig_scatter = px.scatter(
    comparison,
    x="log2_fc",
    y="sensitivity_score",
    text="protein",
    color="log2_fc",
    color_continuous_scale="RdBu_r",
    labels={"log2_fc": "Log2 Impact (KO/WT)", "sensitivity_score": "Cumulative Perturbation (Area)"},
    title="Sensitivity vs. Magnitude: Which genes are most 'fragile'?",
)
fig_scatter.update_traces(textposition="top center")
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Delta Trajectory ---
st.header("⏱️ Signal Loss Propagation (Δ-Trajectories)")
st.markdown("This shows the **absolute loss of signal** over time ($WT - KO$).")

delta_df = wt_dfr.copy()
delta_df["delta"] = wt_dfr["pred_fc"] - ko_dfr["pred_fc"]

max_deltas = delta_df.groupby("protein")["delta"].transform(lambda x: x.abs().max())
delta_df["is_significant"] = max_deltas > 0.1
delta_df["Impact Group"] = delta_df["is_significant"].map({True: "Significant Effect", False: "Minimal/No Effect"})

fig_delta = px.line(
    delta_df,
    x="time",
    y="delta",
    color="protein",
    line_group="protein",
    hover_name="protein",
    color_discrete_sequence=px.colors.qualitative.Alphabet,
    labels={"time": "Time (min)", "delta": "Δ Signal (WT - KO)", "protein": "Protein"},
    title="Propagation of Signal Loss Over Time",
)

fig_delta.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white",
)

for trace in fig_delta.data:
    p_name = trace.name
    max_d = delta_df[delta_df["protein"] == p_name]["delta"].abs().max()
    if max_d < 0.1:
        trace.line.width = 1
        trace.opacity = 0.2
        trace.showlegend = False
    else:
        trace.line.width = 3
        trace.opacity = 0.9

st.plotly_chart(fig_delta, use_container_width=True)

# --- Delta Logic: Cross-Modality Toggle ---
st.header("⏱️ Signal Loss Propagation (Δ-Trajectories)")

modality_choice = st.radio("Select Modality for Delta Analysis:", ["mRNA", "Protein"], horizontal=True)
active_wt = wt_dfr if modality_choice == "mRNA" else wt_dfp
active_ko = ko_dfr if modality_choice == "mRNA" else ko_dfp

delta_df = active_wt.copy()
delta_df["delta"] = active_wt["pred_fc"] - active_ko["pred_fc"]

max_deltas = delta_df.groupby("protein")["delta"].transform(lambda x: x.abs().max())
delta_df["is_significant"] = max_deltas > (0.05 if modality_choice == "Protein" else 0.1)
delta_df["Impact Group"] = delta_df["is_significant"].map({True: "Significant", False: "Minimal"})

fig_delta = px.line(
    delta_df,
    x="time",
    y="delta",
    color="protein",
    line_group="protein",
    hover_name="protein",
    color_discrete_sequence=px.colors.qualitative.Safe,
    labels={"time": "Time (min)", "delta": f"Δ {modality_choice} Signal (WT - KO)", "protein": "Protein"},
    title=f"Propagation of {modality_choice} Signal Loss",
)

for trace in fig_delta.data:
    p_name = trace.name
    max_d = delta_df[delta_df["protein"] == p_name]["delta"].abs().max()
    if max_d < (0.05 if modality_choice == "Protein" else 0.1):
        trace.line.width = 1
        trace.opacity = 0.2
        trace.showlegend = False
    else:
        trace.line.width = 3
        trace.opacity = 0.9

fig_delta.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_delta, use_container_width=True)

# --- Summary Statistics Table ---
st.header("📋 Knockout Impact Summary")
top_impact = comparison.copy()
top_impact["absolute_impact"] = (top_impact["log2_fc"] - 1.0).abs()
top_impact = top_impact.sort_values("absolute_impact", ascending=False).head(15)

st.dataframe(
    top_impact[["protein", "log2_fc", "sensitivity_score"]],
    column_config={
        "log2_fc": st.column_config.NumberColumn("Log2 Impact", format="%.2f"),
        "sensitivity_score": st.column_config.NumberColumn("Total Area Delta", format="%.2f"),
    },
    use_container_width=True,
)

# --- Steady-State Kinase Drive Analysis ---
st.header("⚙️ Steady-State Kinase Drive")

# Recompute steady-state kinetics
_, Y = simulate_until_steady(sys, t_max=960)
y_last = Y[-1].astype(float, copy=False)
t_last = 960.0

Kt = sys.kin.eval(t_last) * sys.c_k
W = sys.W_global.tocoo()
edge_contrib = W.data * Kt[W.col]

# Sum phospho drive per kinase
nK = len(idx.kinases)
kin_sum = np.zeros(nK, dtype=float)
np.add.at(kin_sum, W.col, edge_contrib)

df_kin = pd.DataFrame({"kinase": idx.kinases, "Kt": Kt, "phospho_drive": kin_sum})
df_kin = df_kin.sort_values("phospho_drive", ascending=False)

# Plot top 25
top_df = df_kin.head(10)
fig_kinase = px.bar(
    top_df,
    x="kinase",
    y="phospho_drive",
    color="Kt",
    title="Top 10 Kinases by Phosphorylation Drive at Steady State",
    labels={"phospho_drive": "Σ W_ik * Kt_k", "Kt": "Kinase Activity"},
    color_continuous_scale="Viridis",
)
fig_kinase.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_kinase, use_container_width=True)

# --- Dominant Kinase Per Site Summary ---
n_sites_total = sys.W_global.shape[0]
best_val = np.full(n_sites_total, -np.inf, dtype=float)
best_k = np.full(n_sites_total, -1, dtype=np.int32)
second_val = np.full(n_sites_total, -np.inf, dtype=float)

for r, c, v in zip(W.row, W.col, edge_contrib):
    if v > best_val[r]:
        second_val[r] = best_val[r]
        best_val[r] = v
        best_k[r] = c
    elif v > second_val[r]:
        second_val[r] = v

# Build kinase labels
dominant_kins = np.asarray(idx.kinases, dtype=object)[best_k[best_k >= 0]]
df_dom = pd.DataFrame({"kinase": dominant_kins})
dom_counts = df_dom["kinase"].value_counts().reset_index()
dom_counts.columns = ["kinase", "n_sites_dominated"]

# Plot top10
fig_dom = px.bar(
    dom_counts.head(10),
    x="kinase",
    y="n_sites_dominated",
    title="Top 10 Kinases by Dominated Phospho-Sites (Steady State)",
    labels={"n_sites_dominated": "# Dominated Sites"},
    color="n_sites_dominated",
    color_continuous_scale="Blues",
)
fig_dom.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_dom, use_container_width=True)

# --- Download Section ---
st.subheader("📥 Export Plot Data")
pivot_delta = delta_df.pivot(index="time", columns="protein", values="delta")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        label=f"Download Δ-{modality_choice} CSV",
        data=pivot_delta.to_csv(),
        file_name=f"delta_{modality_choice.lower()}_knockout.csv",
        mime="text/csv",
    )
with c2:
    st.info(f"The CSV contains the absolute loss (WT-KO) for all {len(idx.proteins)} entities.")

st.divider()

st.header("Data - Fit Inspector")
selected_p = st.selectbox("Select a protein to inspect in detail:", idx.proteins)
wt_p_data = wt_dfp[wt_dfp["protein"] == selected_p]
ko_p_data = ko_dfp[ko_dfp["protein"] == selected_p]
wt_r_data = wt_dfr[wt_dfr["protein"] == selected_p]
ko_r_data = ko_dfr[ko_dfr["protein"] == selected_p]
wt_pho_data = wt_pho[wt_pho["protein"] == selected_p]
ko_pho_data = ko_pho[ko_pho["protein"] == selected_p]

col1, col2, col3 = st.columns(3)

with col1:
    fig_insp_r = go.Figure()
    fig_insp_r.add_trace(
        go.Scatter(x=wt_r_data["time"], y=wt_r_data["pred_fc"], name="Wild Type",
                   line=dict(dash="dash", color="black", width=2))
    )
    fig_insp_r.add_trace(
        go.Scatter(x=ko_r_data["time"], y=ko_r_data["pred_fc"], name="Knockout", line=dict(color="red", width=3))
    )
    fig_insp_r.update_layout(title=f"{selected_p} mRNA Response", xaxis_title="Time (min)", yaxis_title="Fold Change",
                             template="plotly_white")
    st.plotly_chart(fig_insp_r, use_container_width=True)

with col2:
    fig_insp_p = go.Figure()
    fig_insp_p.add_trace(
        go.Scatter(x=wt_p_data["time"], y=wt_p_data["pred_fc"], name="Wild Type",
                   line=dict(dash="dash", color="black", width=2))
    )
    fig_insp_p.add_trace(
        go.Scatter(x=ko_p_data["time"], y=ko_p_data["pred_fc"], name="Knockout", line=dict(color="blue", width=3))
    )
    fig_insp_p.update_layout(title=f"{selected_p} Protein Abundance", xaxis_title="Time (min)",
                             yaxis_title="Fold Change", template="plotly_white")
    st.plotly_chart(fig_insp_p, use_container_width=True)

with col3:
    if wt_pho_data.empty:
        st.info("No phospho-site data available for this protein.")
    else:
        fig_sites = go.Figure()

        phospho_times = wt_pho_data["time"].dropna().unique()

        if ko_type != "None":
            ko_phosphosite_inspector_df, ko_phosphosite_debug_df = (
                _build_inspector_ko_phosphosite_intervention_df(
                    sys_obj=sys,
                    idx=idx,
                    best_params=best_params,
                    ko_params=ko_params,
                    protein=selected_p,
                    wt_pho_data=wt_pho_data,
                    n_points=1000,
                )
            )
        else:
            ko_phosphosite_inspector_df = pd.DataFrame(
                columns=["time", "psite", "psite_value"]
            )
            ko_phosphosite_debug_df = pd.DataFrame()

        for site in wt_pho_data["psite"].dropna().unique():
            color = px.colors.qualitative.Plotly[
                hash(site) % len(px.colors.qualitative.Plotly)
            ]

            site_wt = wt_pho_data[wt_pho_data["psite"] == site]

            fig_sites.add_trace(
                go.Scatter(
                    x=site_wt["time"],
                    y=site_wt["pred_fc"],
                    name=f"Wild Type {site}",
                    mode="lines",
                    line=dict(dash="dash", color=color, width=2),
                )
            )

            if ko_type != "None":
                site_ko = ko_phosphosite_inspector_df[
                    ko_phosphosite_inspector_df["psite"] == site
                ]

                if not site_ko.empty:
                    fig_sites.add_trace(
                        go.Scatter(
                            x=site_ko["time"],
                            y=site_ko["psite_value"],
                            name=f"Knockout {site}",
                            mode="lines",
                            line=dict(color=color, width=3),
                        )
                    )
            else:
                fig_sites.add_trace(
                    go.Scatter(
                        x=site_wt["time"],
                        y=site_wt["pred_fc"],
                        name=f"Knockout {site}",
                        mode="lines",
                        line=dict(color=color, width=3),
                    )
                )

        fig_sites.update_layout(
            title=f"{selected_p} Phosphosite Dynamics",
            xaxis_title="Time (min)",
            yaxis_title="Fold Change",
            template="plotly_white",
        )

        st.plotly_chart(fig_sites, use_container_width=True)

        if ko_type != "None" and not ko_phosphosite_debug_df.empty:
            with st.expander("Data inspector phosphosite intervention debug", expanded=False):
                st.dataframe(
                    ko_phosphosite_debug_df,
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()

# --- Forward Simulation Panel with Finer Resolution ---
st.subheader(f"🔁 Forward Simulation of {selected_p}")

t_max = st.slider(
    "Simulation Time (minutes)",
    min_value=2,
    max_value=14 * 24 * 60,
    value=960,
    step=60,
)

n_points = st.select_slider(
    "Forward simulation resolution",
    options=[250, 500, 1000, 2500, 5000, 10000],
    value=1000,
)

run_forward_panel = st.button(
    "Run detailed WT/KO forward simulation",
    key="run-forward-simulation-panel",
)

forward_cache_key = (
    selected_p,
    float(t_max),
    int(n_points),
    str(perturbation_signature),
)

if run_forward_panel:
    with st.spinner(f"Running WT/KO forward simulation with {n_points} time points..."):
        forward_results = _run_forward_wt_ko_for_display(
            sys_obj=sys,
            best_params=best_params,
            ko_params=ko_params,
            t_max=float(t_max),
            n_points=int(n_points),
        )

    st.session_state["forward_panel_cache_key"] = forward_cache_key
    st.session_state["forward_panel_results"] = forward_results

has_forward_results = (
    st.session_state.get("forward_panel_cache_key") == forward_cache_key
    and "forward_panel_results" in st.session_state
)

if not has_forward_results:
    st.info("Click **Run detailed WT/KO forward simulation** to show the detailed simulation panel.")
else:
    cached = st.session_state["forward_panel_results"]

    t_fine = cached["t_fine"]
    Y_wt = cached["Y_wt"]

    t_ko = cached["t_ko"]
    Y_ko_raw = cached.get("Y_ko_raw", cached.get("Y_ko"))
    Y_ko_plot = cached.get("Y_ko_plot", Y_ko_raw)

    used_fixed_y0 = bool(cached.get("used_fixed_y0", False))
    alignment_mode = cached.get("alignment_mode", "legacy cache / no alignment metadata")

    normalize_forward_states = st.checkbox(
        "Normalize forward states to t0",
        value=False,
        key="normalize_forward_states",
    )

    normalize_forward_phosphosite_states = st.checkbox(
        "Normalize forward phosphosite states to WT t0",
        value=False,
        key="normalize_forward_phosphosite_states",
    )

    # Use Y_ko_plot for all displayed KO forward-state panels.
    # This removes artificial KO t0 inflation while preserving the perturbation trajectory shape.
    df_wt = extract_fc_from_Y(
        Y_wt,
        idx,
        t_fine,
        selected_p,
        normalize=normalize_forward_states,
    )

    df_ko = extract_fc_from_Y(
        Y_ko_plot,
        idx,
        t_ko,
        selected_p,
        normalize=normalize_forward_states,
    )

    df_wt_phosphosite, raw_wt_phosphosite, plotted_wt_phosphosite = extract_phosphosite_states_from_Y(
        Y_wt,
        idx,
        t_fine,
        selected_p,
        normalize_to_t0=normalize_forward_phosphosite_states,
        reference_raw_states=None,
    )

    df_ko_phosphosite, raw_ko_phosphosite, plotted_ko_phosphosite = extract_phosphosite_states_from_Y(
        Y_ko_plot,
        idx,
        t_ko,
        selected_p,
        normalize_to_t0=normalize_forward_phosphosite_states,
        reference_raw_states=raw_wt_phosphosite,
    )

    y_label = (
        "Phosphosite ODE state normalized to WT t0"
        if normalize_forward_phosphosite_states
        else "Raw phosphosite ODE state"
    )

    # --- Plot mRNA and protein
    col1, col2 = st.columns(2)

    with col1:
        fig_fine_r = go.Figure()
        fig_fine_r.add_trace(
            go.Scatter(
                x=df_wt["time"],
                y=df_wt["rna"],
                name="WT",
                line=dict(color="black", dash="dash"),
            )
        )
        fig_fine_r.add_trace(
            go.Scatter(
                x=df_ko["time"],
                y=df_ko["rna"],
                name="KO",
                line=dict(color="red"),
            )
        )
        fig_fine_r.update_layout(
            title="mRNA Simulation",
            xaxis_title="Time",
            yaxis_title="Fold Change" if normalize_forward_states else "Raw mRNA ODE state",
            template="plotly_white",
        )
        fig_fine_r.update_xaxes(type="log")
        st.plotly_chart(fig_fine_r, use_container_width=True)

    with col2:
        fig_fine_p = go.Figure()
        fig_fine_p.add_trace(
            go.Scatter(
                x=df_wt["time"],
                y=df_wt["protein"],
                name="WT",
                line=dict(color="black", dash="dash"),
            )
        )
        fig_fine_p.add_trace(
            go.Scatter(
                x=df_ko["time"],
                y=df_ko["protein"],
                name="KO",
                line=dict(color="blue"),
            )
        )
        fig_fine_p.update_layout(
            title="Protein Simulation",
            xaxis_title="Time",
            yaxis_title="Fold Change" if normalize_forward_states else "Raw total protein ODE state",
            template="plotly_white",
        )
        fig_fine_p.update_xaxes(type="log")
        st.plotly_chart(fig_fine_p, use_container_width=True)

    col1, col2 = st.columns(2)

    # --- Signaling Drive Panel
    with col1:
        if selected_p in idx.p2i:
            t_S, Y_S = t_ko, Y_ko_plot

            kin_vals = Y_S[:, [idx.k2i[k] for k in idx.kinases]].T
            kin_scaled = kin_vals * ko_params["c_k"][:, None]
            S_t = sys.W_global @ kin_scaled

            site_names, site_rows = [], []
            site_counter = 0

            for i, p in enumerate(idx.proteins):
                for site in idx.sites[i]:
                    if p == selected_p:
                        site_names.append(site)
                        site_rows.append(site_counter)
                    site_counter += 1

            fig_s_time = go.Figure()

            for site_name, site_idx in zip(site_names, site_rows):
                color = px.colors.qualitative.Plotly[
                    hash(site_name) % len(px.colors.qualitative.Plotly)
                ]
                fig_s_time.add_trace(
                    go.Scatter(
                        x=t_S,
                        y=S_t[site_idx, :],
                        name=site_name,
                        mode="lines",
                        line=dict(dash="solid", color=color),
                        opacity=0.9,
                    )
                )

            fig_s_time.update_layout(
                title=f"{selected_p} – Phosphorylation (S)",
                xaxis_title="Time (min)",
                yaxis_title="S (Signaling Rate)",
                template="plotly_white",
            )
            fig_s_time.update_xaxes(type="log")
            st.plotly_chart(fig_s_time, use_container_width=True)
        else:
            st.warning(f"{selected_p} not found in protein index.")

    # --- Phosphosite calibrated fine-grid ODE dynamics panel
    with col2:
        fig_sites_fine = go.Figure()

        wt_pho_forward_data = wt_pho[wt_pho["protein"] == selected_p].copy()

        if not df_wt_phosphosite.empty and not wt_pho_forward_data.empty:
            p_idx = idx.p2i[selected_p]
            site_order = list(idx.sites[p_idx])

            raw_wt_matrix = np.asarray(raw_wt_phosphosite, dtype=float)
            raw_ko_matrix = np.asarray(raw_ko_phosphosite, dtype=float)

            t_wt_grid = np.asarray(t_fine, dtype=float)
            t_ko_grid = np.asarray(t_ko, dtype=float)

            eps = 1e-12
            debug_rows = []

            for j, site in enumerate(site_order):
                site_wt_picked = wt_pho_forward_data[
                    wt_pho_forward_data["psite"].astype(str) == str(site)
                ].copy()

                if site_wt_picked.empty:
                    continue

                site_wt_picked["time"] = pd.to_numeric(
                    site_wt_picked["time"], errors="coerce"
                )
                site_wt_picked["pred_fc"] = pd.to_numeric(
                    site_wt_picked["pred_fc"], errors="coerce"
                )
                site_wt_picked = site_wt_picked.dropna(subset=["time", "pred_fc"])
                site_wt_picked = site_wt_picked.sort_values("time")

                if site_wt_picked.empty:
                    continue

                picked_t = site_wt_picked["time"].to_numpy(dtype=float)
                picked_fc = site_wt_picked["pred_fc"].to_numpy(dtype=float)

                wt_raw_fine = raw_wt_matrix[:, j]
                ko_raw_fine = raw_ko_matrix[:, j]

                # Interpolate only the smooth raw WT ODE state to picked model times
                # for estimating one scalar measurement calibration. This does NOT
                # create a time-dependent coarse calibration curve.
                wt_raw_at_picked_t = np.interp(
                    picked_t,
                    t_wt_grid,
                    wt_raw_fine,
                )

                valid = (
                    np.isfinite(wt_raw_at_picked_t)
                    & np.isfinite(picked_fc)
                    & (np.abs(wt_raw_at_picked_t) > eps)
                )

                if np.any(valid):
                    # Robust scalar calibration: median picked_FC / raw_state.
                    # This puts the fine ODE trajectory on the same magnitude scale
                    # as the pipeline-picked phosphosite prediction without using
                    # tiny t0 denominators or piecewise interpolation.
                    ratios = picked_fc[valid] / wt_raw_at_picked_t[valid]
                    ratios = ratios[np.isfinite(ratios)]

                    if ratios.size:
                        calibration = float(np.median(ratios))
                    else:
                        calibration = 1.0
                else:
                    calibration = 1.0

                wt_fc_fine = wt_raw_fine * calibration

                if ko_type != "None":
                    ko_fc_fine = ko_raw_fine * calibration
                else:
                    ko_fc_fine = wt_fc_fine.copy()

                color = px.colors.qualitative.Plotly[
                    hash(site) % len(px.colors.qualitative.Plotly)
                ]

                fig_sites_fine.add_trace(
                    go.Scatter(
                        x=t_wt_grid,
                        y=wt_fc_fine,
                        name=f"Wild Type {site}",
                        mode="lines",
                        line=dict(dash="dash", color=color, width=2),
                    )
                )

                fig_sites_fine.add_trace(
                    go.Scatter(
                        x=t_ko_grid,
                        y=ko_fc_fine,
                        name=f"Knockout {site}",
                        mode="lines",
                        line=dict(color=color, width=3),
                    )
                )

                # Optional picked WT model points, useful for checking alignment.
                fig_sites_fine.add_trace(
                    go.Scatter(
                        x=picked_t,
                        y=picked_fc,
                        name=f"Picked WT model {site}",
                        mode="markers",
                        marker=dict(size=6, color=color, symbol="circle-open"),
                        showlegend=False,
                    )
                )

                debug_rows.append(
                    {
                        "psite": site,
                        "WT raw min": float(np.nanmin(wt_raw_fine)),
                        "WT raw max": float(np.nanmax(wt_raw_fine)),
                        "KO raw min": float(np.nanmin(ko_raw_fine)),
                        "KO raw max": float(np.nanmax(ko_raw_fine)),
                        "WT calibrated FC min": float(np.nanmin(wt_fc_fine)),
                        "WT calibrated FC max": float(np.nanmax(wt_fc_fine)),
                        "KO calibrated FC min": float(np.nanmin(ko_fc_fine)),
                        "KO calibrated FC max": float(np.nanmax(ko_fc_fine)),
                        "picked WT FC min": float(np.nanmin(picked_fc)),
                        "picked WT FC max": float(np.nanmax(picked_fc)),
                        "scalar calibration": calibration,
                    }
                )

            fig_sites_fine.update_layout(
                title=f"{selected_p} Phosphosite Dynamics",
                xaxis_title="Time (min)",
                yaxis_title="Fold Change",
                template="plotly_white",
            )
            fig_sites_fine.update_xaxes(type="log")
            st.plotly_chart(fig_sites_fine, use_container_width=True)

            if raw_wt_matrix.shape == raw_ko_matrix.shape and raw_wt_matrix.size:
                phosphosite_t0_max_abs_diff = float(
                    np.max(np.abs(raw_wt_matrix[0, :] - raw_ko_matrix[0, :]))
                )
            else:
                phosphosite_t0_max_abs_diff = float("nan")

            with st.expander("Forward phosphosite calibrated dynamics debug", expanded=False):
                st.write(
                    {
                        "alignment_mode": alignment_mode,
                        "used_fixed_y0_argument": used_fixed_y0,
                        "max_abs_t0_difference_WT_vs_displayed_KO_phosphosite": phosphosite_t0_max_abs_diff,
                        "calibration_mode": "single robust per-site scalar: median(picked WT FC / WT raw ODE state)",
                    }
                )

                if debug_rows:
                    st.dataframe(
                        pd.DataFrame(debug_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                if Y_ko_raw is not None:
                    _, raw_ko_unaligned_phosphosite, _ = extract_phosphosite_states_from_Y(
                        Y_ko_raw,
                        idx,
                        t_ko,
                        selected_p,
                        normalize_to_t0=False,
                    )
                    unaligned_min, unaligned_max = _safe_min_max(raw_ko_unaligned_phosphosite)
                    st.write(
                        {
                            "raw_unaligned_KO_phosphosite_min": unaligned_min,
                            "raw_unaligned_KO_phosphosite_max": unaligned_max,
                        }
                    )
        else:
            st.info("No phospho site data available for this protein.")

st.divider()

# --- Graph Options ---
st.sidebar.divider()
st.sidebar.header("🕸️ Graph Options")
depth = st.sidebar.slider("Cascade Depth", min_value=1, max_value=3, value=1,
                          help="1: Direct targets only. 2+: Includes targets of targets.")

# --- Functional Hierarchy Map ---
if ko_type != "None" and df_tf_model is not None and not df_tf_model.empty and target is not None:
    st.divider()
    st.header(f"Functional Influence Map: {target}")

    # df_tf_model is standardized to ('tf','target')
    src_col, tgt_col = "tf", "target"

    edges_list = []
    current_layer_nodes = {target}
    processed_nodes = set()

    final_kin_act = sys.kin.Kmat[:, -1] * ko_params["c_k"]
    S_final = sys.W_global.dot(final_kin_act)

    for d in range(depth):
        next_layer_nodes = set()
        if not current_layer_nodes:
            break

        if any(k in idx.kinases for k in current_layer_nodes):
            for k_name in current_layer_nodes:
                if k_name in idx.kinases:
                    k_idx = idx.k2i[k_name]
                    W_csc = sys.W_global.tocsc()
                    col = W_csc.getcol(k_idx)
                    rows, _ = col.nonzero()

                    for site_idx in rows:
                        prot_idx = np.searchsorted(idx.offset_s, site_idx, side="right") - 1
                        target_prot = idx.proteins[prot_idx]

                        beta_val = W_csc[site_idx, k_idx]
                        s_flux = S_final[site_idx]

                        edges_list.append(
                            {
                                "src": k_name,
                                "tgt": target_prot,
                                "type": "signaling",
                                "weight": float(beta_val),
                                "label": f"β (Kin-Site): {float(beta_val):.3f}, S-Flux: {float(s_flux):.3f}",
                            }
                        )
                        next_layer_nodes.add(target_prot)

        tf_hits = df_tf_model[df_tf_model[src_col].isin(current_layer_nodes)]
        for _, row in tf_hits.iterrows():
            s, t = row[src_col], row[tgt_col]
            t_idx = idx.p2i[t]
            alpha_val = sys.E_i[t_idx]

            edges_list.append(
                {
                    "src": s,
                    "tgt": t,
                    "type": "transcription",
                    "weight": float(alpha_val),
                    "label": f"α (TF Efficacy): {float(alpha_val):.3f}",
                }
            )
            next_layer_nodes.add(t)

        processed_nodes.update(current_layer_nodes)
        current_layer_nodes = next_layer_nodes - processed_nodes

    if not edges_list:
        st.warning("No functional connections found.")
    else:
        plot_df = pd.DataFrame(edges_list).drop_duplicates()
        G = nx.from_pandas_edgelist(plot_df, "src", "tgt", create_using=nx.DiGraph())
        pos = nx.spring_layout(G, k=1.0, seed=42)

        edge_traces = []
        midpoint_node_x, midpoint_node_y, midpoint_hover_text = [], [], []

        for _, row in plot_df.iterrows():
            x0, y0 = pos[row["src"]]
            x1, y1 = pos[row["tgt"]]

            width = np.clip(row["weight"] * 5, 1.0, 7)
            color = "#3498db" if row["type"] == "signaling" else "#e67e22"

            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color=color),
                    hoverinfo="none",
                    mode="lines",
                    opacity=0.6,
                )
            )

            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            midpoint_node_x.append(mid_x)
            midpoint_node_y.append(mid_y)
            midpoint_hover_text.append(row["label"])

        edge_hover_trace = go.Scatter(
            x=midpoint_node_x,
            y=midpoint_node_y,
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            hoverinfo="text",
            text=midpoint_hover_text,
            name="Edge Data",
        )

        node_x, node_y, node_color, node_text = [], [], [], []
        # Inside your node plotting loop
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Get impact value if available
            impact = comparison.loc[comparison["protein"] == node, "log2_fc"]
            impact = float(impact.iloc[0]) if not impact.empty else 0.0
            node_color.append(impact)

            # Handle missing nodes in idx.p2i
            if node in idx.p2i:
                n_idx = idx.p2i[node]
                c_val = float(sys.C_i[n_idx])
            else:
                c_val = float('nan')  # or 0.0 or "N/A" depending on display preference

            node_text.append(
                f"<b>{node}</b><br>Log2 Impact: {impact:.2f}<br>Internal α (C_i): {c_val:.3f}"
                if not np.isnan(c_val)
                else f"<b>{node}</b><br>Log2 Impact: {impact:.2f}<br>Internal α (C_i): N/A"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[f"<b>{n}</b>" for n in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale="RdBu_r",
                size=35,
                colorbar=dict(title="Log2 FC", thickness=15),
                color=node_color,
                cmin=0,
                cmax=5,
                line=dict(width=2, color="white"),
            ),
            hovertext=node_text,
            hoverinfo="text",
        )

        fig = go.Figure(
            data=edge_traces + [edge_hover_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="rgba(0,0,0,0)",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info("🔵 Blue Edges: Signaling (S-flux & β) | 🟠 Orange Edges: Transcription (α)")

st.divider()


def _total_protein_from_y(y_last: np.ndarray, idx: Index, protein: str) -> float:
    """
    Calculates the total protein abundance from a given state vector.

    This function computes the total abundance of a specified protein, including its phosphorylated forms,
    based on the provided state vector and index.

    Args:
        y_last (np.ndarray): The state vector representing the abundance of different species.
        idx (Index): An index object containing mappings and offsets for protein data.
        protein (str): The name of the protein for which the total abundance is calculated.

    Returns:
        float: The total abundance of the specified protein, including phosphorylated forms.
    """
    p_i = idx.p2i[protein]
    st_y = idx.offset_y[p_i]
    prot = float(y_last[st_y + 1])
    ns = int(idx.n_sites[p_i])
    if ns > 0:
        phos_sum = float(np.sum(y_last[st_y + 2: st_y + 2 + ns]))
        return prot + phos_sum
    return prot


def _compute_state_snapshot(sys: System, idx: Index, params: dict, t_eval: float = 960.0):
    """
    Updates the system state and computes a snapshot of system variables and metrics at
    a specific evaluation time.

    Args:
        sys (System): The system object representing the model or structure being evaluated.
        idx (Index): The index structure containing mappings for protein indices or similar
            elements in the system.
        params (dict): A dictionary of parameters that will override or update the system's
            current state.
        t_eval (float, optional): The simulation evaluation time. Defaults to 960"""
    # backup current system state minimally by re-applying later outside this helper if needed
    sys.update(**params)

    _, Y = simulate_until_steady(sys, t_max=t_eval)
    y_last = Y[-1].astype(float, copy=False)

    # kinase activities
    Kt = sys.kin.eval(t_eval) * sys.c_k  # (nK,)

    # total protein dictionary (for TF "activity" modulation)
    totalP = {p: _total_protein_from_y(y_last, idx, p) for p in idx.proteins}

    return Kt, y_last, totalP


def _build_global_edge_tables(
        sys: System,
        idx: Index,
        params: dict,
        df_tf_model: pd.DataFrame | None,
        t_eval: float = 960.0,
):
    """
    Builds global edge tables for signaling and transcription interactions based on the provided
    system state, kinase and protein indices, parameters, transcript factor models, and evaluation time.

    This function computes and aggregates signaling edges between kinases and proteins, as well as
    transcription edges between transcription factors and target proteins, forming two distinct edge 
    dataframes. Signaling edge contributions depend on the system's global weight matrix and the state 
    snapshot, while transcription edges are derived from transcription factor interactions.

    Args:
        sys (System): The system object containing global matrices, required weights, and"""
    Kt, y_last, totalP = _compute_state_snapshot(sys, idx, params, t_eval=t_eval)

    # ---- Signaling edges: kinase -> protein (aggregate per target protein) ----
    W = sys.W_global.tocoo()
    # edge contribution per (site,row) from earlier code: W.data * Kt[W.col]
    edge_contrib = W.data * Kt[W.col]  # (nnz,)

    # map site-row -> protein via idx.offset_s (same logic you used earlier)
    prot_idx = np.searchsorted(idx.offset_s, W.row, side="right") - 1
    prot_idx = np.clip(prot_idx, 0, len(idx.proteins) - 1)

    df_sig = pd.DataFrame(
        {
            "src": np.asarray(idx.kinases, dtype=object)[W.col],
            "tgt": np.asarray(idx.proteins, dtype=object)[prot_idx],
            "weight": edge_contrib.astype(float),
            "beta": W.data.astype(float),
            "Kt": Kt[W.col].astype(float),
        }
    )
    # aggregate over all sites per (kinase, protein)
    df_sig = (
        df_sig.groupby(["src", "tgt"], as_index=False)
        .agg(weight=("weight", "sum"), beta=("beta", "mean"), Kt=("Kt", "mean"))
    )
    df_sig["type"] = "signaling"

    # ---- Transcription edges: tf -> target ----
    df_tf_edges = pd.DataFrame(columns=["src", "tgt", "weight", "type"])
    if df_tf_model is not None and not df_tf_model.empty:
        tf_mat = sys.tf_mat  # built in your System; typically sparse
        tf_scale = float(getattr(sys, "tf_scale", 1.0))

        # safer column names
        src_col, tgt_col = "tf", "target"

        # Evaluate each TF interaction as: drive = tf_scale * tf_mat[target, tf] * TF_level
        # (We do not assume sign semantics; keep sign for coloring later.)
        rows = []
        for r in df_tf_model.itertuples(index=False):
            tf = getattr(r, src_col)
            tgt = getattr(r, tgt_col)
            if tf not in idx.p2i or tgt not in idx.p2i:
                continue
            i_tgt = idx.p2i[tgt]
            j_tf = idx.p2i[tf]

            # fetch coefficient
            try:
                coeff = float(tf_mat[i_tgt, j_tf])
            except Exception:
                # if sparse matrix returns 1x1 matrix
                coeff = float(np.asarray(tf_mat[i_tgt, j_tf]).squeeze())

            if abs(coeff) < 1e-14:
                continue

            tf_level = float(totalP.get(tf, 0.0))
            drive = tf_scale * coeff * tf_level

            rows.append((tf, tgt, drive))

        if rows:
            df_tf_edges = pd.DataFrame(rows, columns=["src", "tgt", "weight"])
            df_tf_edges["type"] = "transcription"

    return df_sig, df_tf_edges


def _merge_and_filter_edges(
        df_sig: pd.DataFrame,
        df_tf: pd.DataFrame,
        max_edges: int = 300,
        min_abs_weight: float = 1e-3,
        include_tf: bool = True,
):
    """
    Merges and filters edges from the given dataframes based on specified conditions.

    This function combines two dataframes, applies a filter based on the absolute weight
    of edges, and limits the total number of edges returned. It ensures only significant
    edges, as defined by the minimum absolute weight and a maximum number of edges, are 
    retained.

    Args:
        df_sig (pd.DataFrame): The dataframe containing significant edges.
        df_tf (pd.DataFrame): The dataframe containing transcription factor (TF) edges.
        max_edges (int): The maximum number of edges to include in the"""
    df_all = df_sig.copy()
    if include_tf and df_tf is not None and not df_tf.empty:
        df_all = pd.concat([df_all, df_tf], ignore_index=True)

    if df_all.empty:
        return df_all

    df_all["absw"] = df_all["weight"].abs()
    df_all = df_all[df_all["absw"] >= float(min_abs_weight)].copy()

    # keep only top edges by absolute weight for readability
    df_all = df_all.sort_values("absw", ascending=False).head(int(max_edges)).copy()

    return df_all


def _gravis_html_from_edges(df_edges: pd.DataFrame, title: str):
    """
    Generates an HTML representation of a directed graph visualization based on the provided
    edges. The graph is constructed using the NetworkX library and visualized using Gravis.

    Args:
        df_edges (pd.DataFrame): A DataFrame containing the edges of the graph. Each edge
            should specify the source node, target node, weight, and type.
        title (str): The title to be displayed above the graph visualization.

    Returns:
        str: An HTML string containing the graph visualization with interactivity and styling.
    """
    G = nx.DiGraph()

    for r in df_edges.itertuples(index=False):
        src, tgt = r.src, r.tgt
        w = float(r.weight)
        typ = r.type

        # gravis expects a numeric "size" for edge thickness
        G.add_edge(
            src, tgt,
            weight=w,
            size=float(abs(w)),
            etype=typ,
            color=("#3498db" if typ == "signaling" else "#e67e22"),
        )

    for n in G.nodes():
        G.nodes[n]["label"] = n
        G.nodes[n]["size"] = float(max(1, G.degree(n)))
        G.nodes[n]["hover"] = f"<b>{n}</b><br>deg={G.degree(n)}"

    for u, v, d in G.edges(data=True):
        d["hover"] = (
            f"{u} → {v}"
            f"<br>type={d.get('etype')}"
            f"<br>weight={float(d.get('weight', 0.0)):.4g}"
        )

    fig = gv.d3(
        G,
        graph_height=700,

        # Labels
        node_label_data_source="label",
        show_node_label=True,
        show_node_label_border=True,

        # Node sizes
        node_size_data_source="size",
        use_node_size_normalization=True,
        node_size_normalization_max=18,

        # Edge sizes (correct API)
        edge_size_data_source="size",
        use_edge_size_normalization=True,
        edge_size_normalization_max=6,

        # Tooltips
        node_hover_tooltip=True,
        edge_hover_tooltip=True,

        # UI containers (correct API; replaces `details=...`)
        show_details=False,
        show_details_toggle_button=True,
        show_menu=False,
        show_menu_toggle_button=True,

        zoom_factor=1.0,
    )

    html = fig.to_html()
    html = f"<h3 style='margin:0 0 8px 0'>{title}</h3>" + html
    return html


def _plotly_network_from_edges(df_edges: pd.DataFrame, title: str):
    """
    Creates and returns a Plotly visual representation of a directed graph constructed 
    from the given edges DataFrame.

    This function uses NetworkX to generate a directed graph and visualizes it using 
    Plotly with nodes and edges styled appropriately. The layout of the graph is 
    calculated using a spring layout. Nodes are displayed as markers with labels, and 
    edges are represented as lines with hover information showing details about 
    the connection.

    Args:
        df_edges (pd.DataFrame): A DataFrame representing graph edges. Each row should 
            contain the source node ('src'),"""
    G = nx.DiGraph()
    for r in df_edges.itertuples(index=False):
        G.add_edge(r.src, r.tgt, weight=float(r.weight), etype=r.type)

    pos = nx.spring_layout(G, k=0.8, seed=42)

    # edges
    edge_traces = []
    hover_x, hover_y, hover_text = [], [], []
    for r in df_edges.itertuples(index=False):
        x0, y0 = pos[r.src]
        x1, y1 = pos[r.tgt]
        w = float(r.weight)
        typ = r.type

        width = float(np.clip(np.log10(1 + abs(w)) * 3.0, 0.8, 6.0))
        color = "#3498db" if typ == "signaling" else "#e67e22"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                opacity=0.55,
                hoverinfo="none",
                showlegend=False,
            )
        )

        hover_x.append((x0 + x1) / 2)
        hover_y.append((y0 + y1) / 2)
        hover_text.append(f"{r.src} → {r.tgt}<br>{typ}<br>weight={w:.4g}")

    edge_hover = go.Scatter(
        x=hover_x,
        y=hover_y,
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        text=hover_text,
        showlegend=False,
    )

    # nodes
    node_x, node_y, node_text = [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{n}</b>")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        marker=dict(size=18, line=dict(width=1, color="white")),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(
        data=edge_traces + [edge_hover, node_trace],
        layout=go.Layout(
            title=title,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
        ),
    )
    return fig


# =========================
# Functional Influence (gravis) — KO consistent, WT/KO/Δ like global panel
# =========================

def _totalP_dict_from_snapshot(y_last: np.ndarray, idx: Index) -> dict[str, float]:
    """
    Calculates total protein abundance for each protein in the provided snapshot.

    This function generates a dictionary mapping each protein to its total abundance,
    based on the given simulation snapshot and protein index.

    Args:
        y_last (np.ndarray): The state vector of the system from a simulation snapshot.
        idx (Index): An index object that provides mappings and references to proteins
            in the system.

    Returns:
        dict[str, float]: A dictionary where the keys are protein names and the values
        are the corresponding total protein abundances.
    """
    return {p: _total_protein_from_y(y_last, idx, p) for p in idx.proteins}


def _snapshot_for_params(sys_local: System, idx_local: Index, params: dict, t_eval: float):
    """
    Creates a snapshot for the system based on the given parameters and evaluation time.

    This function computes the state snapshot of the system using the provided system 
    and index objects, parameter dictionary, and a specific evaluation time. It returns 
    three computed values that represent the state snapshot of the system at the specified 
    time.

    Args:
        sys_local (System): The system object representing the current system configuration.
        idx_local (Index): The index object representing the indices or mappings used in 
            computation.
        params (dict): A dictionary of parameters required for the computation of 
    """
    Kt, y_last, totalP = _compute_state_snapshot(sys_local, idx_local, params, t_eval=float(t_eval))
    return Kt, y_last, totalP


def _cascade_edges_from_seed(
        sys_local: System,
        idx_local: Index,
        params: dict,
        df_tf_model_local: pd.DataFrame | None,
        seed: str,
        depth: int = 1,
        t_eval: float = 960.0,
        include_tf: bool = True,
        max_edges: int = 400,
        min_abs_weight: float = 1e-3,
):
    """
    Generates a cascaded network of edges based on a seed node using depth-limited expansion. The
    function calculates kinase-protein signaling edges, evaluates transcription factor (TF)-target
    edges, and performs a depth-limited breadth-first search (BFS) for edge expansion.

    Args:
        sys_local (System): The system instance containing global and localized matrices for
            kinase-protein signaling and transcription factors.
        idx_local (Index): An index mapping object for kinase and protein IDs.
    """
    Kt, y_last, totalP = _snapshot_for_params(sys_local, idx_local, params, t_eval=float(t_eval))

    # Precompute fast signaling lookup:
    # For each kinase k, get all site rows it hits, and map those site rows -> protein
    W = sys_local.W_global.tocoo()
    # drive per site-edge: beta * Kt_k
    edge_drive = W.data * Kt[W.col]

    # site_row -> protein index
    prot_idx = np.searchsorted(idx_local.offset_s, W.row, side="right") - 1
    prot_idx = np.clip(prot_idx, 0, len(idx_local.proteins) - 1)
    prot_name = np.asarray(idx_local.proteins, dtype=object)[prot_idx]
    kin_name = np.asarray(idx_local.kinases, dtype=object)[W.col]

    df_site = pd.DataFrame(
        {"src": kin_name, "tgt": prot_name, "weight": edge_drive.astype(float)}
    )
    # aggregate to kinase->protein
    df_sig = df_site.groupby(["src", "tgt"], as_index=False).agg(weight=("weight", "sum"))
    df_sig["type"] = "signaling"

    # TF edges as in your global table, but evaluated at this snapshot
    df_tf_edges = pd.DataFrame(columns=["src", "tgt", "weight", "type"])
    if include_tf and df_tf_model_local is not None and not df_tf_model_local.empty:
        tf_mat = sys_local.tf_mat
        tf_scale = float(getattr(sys_local, "tf_scale", 1.0))

        rows = []
        for r in df_tf_model_local.itertuples(index=False):
            tf = getattr(r, "tf")
            tgt = getattr(r, "target")
            if tf not in idx_local.p2i or tgt not in idx_local.p2i:
                continue
            i_tgt = idx_local.p2i[tgt]
            j_tf = idx_local.p2i[tf]
            try:
                coeff = float(tf_mat[i_tgt, j_tf])
            except Exception:
                coeff = float(np.asarray(tf_mat[i_tgt, j_tf]).squeeze())
            if abs(coeff) < 1e-14:
                continue
            tf_level = float(totalP.get(tf, 0.0))
            drive = tf_scale * coeff * tf_level
            rows.append((tf, tgt, float(drive)))

        if rows:
            df_tf_edges = pd.DataFrame(rows, columns=["src", "tgt", "weight"])
            df_tf_edges["type"] = "transcription"

    # -------- Depth-limited expansion (BFS on directed edges) --------
    # We expand from current frontier nodes by:
    #  - if node is a kinase: outgoing signaling edges kinase -> proteins
    #  - if node is a protein: outgoing TF edges protein(tf) -> targets (only if it appears as tf)
    edges_out_sig = df_sig
    edges_out_tf = df_tf_edges

    frontier = {seed}
    visited = set()
    keep_rows = []

    for _ in range(int(depth)):
        if not frontier:
            break

        next_frontier = set()
        for node in frontier:
            # Signaling expansion if node is kinase
            if node in idx_local.k2i:
                sub = edges_out_sig[edges_out_sig["src"] == node]
                if not sub.empty:
                    keep_rows.append(sub)
                    next_frontier.update(sub["tgt"].tolist())

            # TF expansion if node is protein and is TF in df_tf_edges
            if include_tf and (node in idx_local.p2i):
                sub = edges_out_tf[edges_out_tf["src"] == node]
                if not sub.empty:
                    keep_rows.append(sub)
                    next_frontier.update(sub["tgt"].tolist())

        visited.update(frontier)
        frontier = next_frontier - visited

    if not keep_rows:
        return pd.DataFrame(columns=["src", "tgt", "type", "weight", "absw"])

    df_edges = pd.concat(keep_rows, ignore_index=True).drop_duplicates()
    df_edges["absw"] = df_edges["weight"].abs()
    df_edges = df_edges[df_edges["absw"] >= float(min_abs_weight)].copy()
    df_edges = df_edges.sort_values("absw", ascending=False).head(int(max_edges)).copy()
    return df_edges


def _functional_influence_edges(mode: str, seed: str, depth: int, t_eval: float,
                                include_tf: bool, max_edges: int, min_abs_w: float):
    """
    Calculates functional influence on edges in a network model under wild-type (WT) conditions,
    knockout (KO) conditions, or their differences. The results can be filtered based on minimum
    absolute weight and limited to a maximum number of edges.

    Args:
        mode (str): Specifies the mode of operation. Can be "WT", "KO", or calculate
            differences ("Δ(KO−WT)").
        seed (str): Initial node or seed for cascading influence calculation.
        depth (int): Maximum depth of propagation in the network.
        t_eval ("""
    # WT
    sys_wt, idx_wt, _, df_tf_wt, _ = load_system()
    df_wt = _cascade_edges_from_seed(
        sys_wt, idx_wt, best_params, df_tf_wt,
        seed=seed, depth=depth, t_eval=t_eval,
        include_tf=include_tf, max_edges=max_edges, min_abs_weight=min_abs_w
    )

    # KO
    sys_ko, idx_ko, _, df_tf_ko, _ = load_system()
    df_ko = _cascade_edges_from_seed(
        sys_ko, idx_ko, ko_params, df_tf_ko,
        seed=seed, depth=depth, t_eval=t_eval,
        include_tf=include_tf, max_edges=max_edges, min_abs_weight=min_abs_w
    )

    if mode == "WT":
        return df_wt, f"Functional influence (WT) — seed={seed} — depth={depth} — t={t_eval:.1f} min"
    if mode == "KO":
        return df_ko, f"Functional influence (KO) — seed={seed} — depth={depth} — t={t_eval:.1f} min"

    # Δ(KO−WT) align edges
    key = ["src", "tgt", "type"]
    df_w = df_wt[key + ["weight"]].rename(columns={"weight": "w_wt"})
    df_k = df_ko[key + ["weight"]].rename(columns={"weight": "w_ko"})
    df_d = df_k.merge(df_w, on=key, how="outer")
    df_d["w_ko"] = df_d["w_ko"].fillna(0.0)
    df_d["w_wt"] = df_d["w_wt"].fillna(0.0)
    df_d["weight"] = df_d["w_ko"] - df_d["w_wt"]
    df_d = df_d[key + ["weight"]]
    df_d["absw"] = df_d["weight"].abs()
    df_d = df_d[df_d["absw"] >= float(min_abs_w)].sort_values("absw", ascending=False).head(int(max_edges)).copy()

    return df_d, f"Functional influence Δ(KO−WT) — seed={seed} — depth={depth} — t={t_eval:.1f} min"


# -------------------------
# UI panel (uses your existing KO selection)
# -------------------------
st.divider()
st.header("🧭 Functional Influence (gravis)")

seed = st.selectbox("Seed node", sorted(set(idx.proteins) | set(idx.kinases)), index=0)

if target is None:
    st.info("Select a KO target (protein or kinase) in the sidebar to enable influence mapping.")
else:
    cI1, cI2, cI3, cI4 = st.columns(4)
    with cI1:
        infl_view = st.selectbox("View", ["KO", "WT", "Δ(KO−WT)"], index=0, key="infl_view")
    with cI2:
        infl_t = st.slider("Evaluation time (min)", min_value=10, max_value=960, value=960, step=10, key="infl_t")
    with cI3:
        infl_depth = st.slider("Cascade depth", min_value=1, max_value=4, value=int(depth), step=1, key="infl_depth")
    with cI4:
        infl_include_tf = st.checkbox("Include TF edges", value=True, key="infl_include_tf")

    cJ1, cJ2 = st.columns(2)
    with cJ1:
        infl_max_edges = st.slider("Max edges (top-|weight|)", 50, 1200, 300, 50, key="infl_max_edges")
    with cJ2:
        infl_min_abs_w = st.number_input("Min |weight| filter", min_value=0.0, value=0.001, step=0.001,
                                         format="%.4f", key="infl_min_abs_w")

    df_infl, infl_title = _functional_influence_edges(
        mode=infl_view,
        seed=seed,
        depth=int(infl_depth),
        t_eval=float(infl_t),
        include_tf=bool(infl_include_tf),
        max_edges=int(infl_max_edges),
        min_abs_w=float(infl_min_abs_w),
    )

    if df_infl.empty:
        st.warning(
            "No edges passed the current filters for this influence graph. Lower the |weight| threshold or increase max edges.")
    else:
        html = _gravis_html_from_edges(df_infl, infl_title)
        components.html(html, height=760, scrolling=True)
        with st.expander("Show influence edge table"):
            st.dataframe(df_infl[["src", "tgt", "type", "weight", "absw"]].head(500), use_container_width=True)

# =========================
# PANEL UI
# =========================
st.divider()
st.header("🕸️ Global KO Network (Whole-System Cascade)")

# Build WT and KO edge tables from a consistent timepoint (default 960 like your steady-state panels)
t_eval = st.slider("Network evaluation time (min)", min_value=10, max_value=960, value=960, step=10)

cA, cB, cC, cD = st.columns(4)
with cA:
    view_mode = st.selectbox("View", ["KO", "WT", "Δ(KO−WT)"], index=0)
with cB:
    include_tf_edges = st.checkbox("Include TF edges", value=True)
with cC:
    max_edges = st.slider("Max edges (top-|weight|)", 50, 1200, 300, 50)
with cD:
    min_abs_w = st.number_input("Min |weight| filter", min_value=0.0, value=0.001, step=0.001, format="%.4f")


# Compute WT / KO edge tables for unmutated global system
# df_sig_wt, df_tf_wt = _build_global_edge_tables(sys, idx, best_params, df_tf_model, t_eval=float(t_eval))
# df_sig_ko, df_tf_ko = _build_global_edge_tables(sys, idx, ko_params, df_tf_model, t_eval=float(t_eval))

def build_network_from_params(params):
    """
    Builds a network from the given parameters and returns the result.

    This function leverages the provided system and index data to construct
    global edge tables and produce the resultant network. It takes in a set
    of parameters necessary for the network-building process.

    Args:
        params: Parameters required to build the network.
    """
    # HARD RESET
    sys_local, idx_local, _, _, _ = load_system()
    return _build_global_edge_tables(
        sys_local, idx_local, params, df_tf_model, t_eval=float(t_eval)
    )


df_sig_wt, df_tf_wt = build_network_from_params(best_params)
df_sig_ko, df_tf_ko = build_network_from_params(ko_params)


# Merge + filter per mode
def _prepare_edges_for_mode(mode: str):
    """
    Prepares and filters edges for a specified network mode. Depending on the mode, this function
    can generate edges for WT (wild type), KO (knockout), or the delta network (difference between
    knockout and wild type). The filtering process may include merging edges, aligning weights, 
    removing redundant data, and applying thresholds based on the provided parameters. The result 
    includes the appropriately filtered edge data and a descriptive title.

    Args:
        mode (str): The network mode"""
    if mode == "WT":
        df_edges = _merge_and_filter_edges(df_sig_wt, df_tf_wt, max_edges=max_edges, min_abs_weight=min_abs_w,
                                           include_tf=include_tf_edges)
        title = f"WT network at t={t_eval} min"
        return df_edges, title

    if mode == "KO":
        df_edges = _merge_and_filter_edges(df_sig_ko, df_tf_ko, max_edges=max_edges, min_abs_weight=min_abs_w,
                                           include_tf=include_tf_edges)
        title = f"KO network at t={t_eval} min (perturbation applied)"
        return df_edges, title

    # Δ(KO−WT): align edges and subtract weights
    df_wt = pd.concat([df_sig_wt, df_tf_wt], ignore_index=True)
    df_ko = pd.concat([df_sig_ko, df_tf_ko], ignore_index=True)

    if df_wt.empty and df_ko.empty:
        return pd.DataFrame(), f"Δ network at t={t_eval} min"

    key_cols = ["src", "tgt", "type"]
    df_wt = df_wt[key_cols + ["weight"]].rename(columns={"weight": "w_wt"})
    df_ko = df_ko[key_cols + ["weight"]].rename(columns={"weight": "w_ko"})

    df_delta = df_ko.merge(df_wt, on=key_cols, how="outer")
    df_delta["w_ko"] = df_delta["w_ko"].fillna(0.0)
    df_delta["w_wt"] = df_delta["w_wt"].fillna(0.0)
    df_delta["weight"] = df_delta["w_ko"] - df_delta["w_wt"]
    df_delta = df_delta[key_cols + ["weight"]]

    # filter + top edges
    df_delta["absw"] = df_delta["weight"].abs()
    df_delta = df_delta[df_delta["absw"] >= float(min_abs_w)].copy()
    df_delta = df_delta.sort_values("absw", ascending=False).head(int(max_edges)).copy()

    title = f"Δ network (KO−WT) at t={t_eval} min"
    return df_delta, title


df_edges, net_title = _prepare_edges_for_mode(view_mode)

if df_edges.empty:
    st.warning("No edges passed the current filters. Lower the |weight| threshold or increase max edges.")
else:
    html = _gravis_html_from_edges(df_edges, net_title)
    components.html(html, height=760, scrolling=True)
    st.caption("Rendering: gravis (interactive force-directed).")

    # Optional: table for reproducibility / debugging
    with st.expander("Show edge table (top edges)"):
        st.dataframe(
            df_edges.sort_values("absw", ascending=False)[["src", "tgt", "type", "weight"]].head(200),
            use_container_width=True,
        )

# =========================
# TIME SWEEP (per-timepoint graphs) — ADD BELOW YOUR WORKING PANEL
# =========================
st.divider()
st.header("🧭 Per-timepoint Network Sweep (grid)")

cS1, cS2, cS3, cS4 = st.columns(4)
with cS1:
    sweep_view_mode = st.selectbox(
        "View (sweep)",
        ["KO", "WT", "Δ(KO−WT)"],
        index=["KO", "WT", "Δ(KO−WT)"].index(view_mode) if "view_mode" in globals() else 0,
        key="sweep_view_mode",
    )
with cS2:
    sweep_t_end = st.slider(
        "Sweep end time (min)",
        min_value=10,
        max_value=14 * 24 * 60,
        value=int(t_eval) if "t_eval" in globals() else 960,
        step=10,
        key="sweep_t_end",
    )
with cS3:
    sweep_n = st.slider(
        "Timepoints (rendered)",
        min_value=4,
        max_value=24,  # keep this modest; gravis HTML is heavy
        value=9,
        step=1,
        key="sweep_n",
    )
with cS4:
    sweep_cols = st.slider(
        "Grid columns",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        key="sweep_cols",
    )

cF1, cF2, cF3 = st.columns(3)
with cF1:
    sweep_include_tf = st.checkbox(
        "Include TF edges (sweep)",
        value=bool(include_tf_edges) if "include_tf_edges" in globals() else True,
        key="sweep_include_tf",
    )
with cF2:
    sweep_max_edges = st.slider(
        "Max edges (sweep)",
        50, 1200,
        int(max_edges) if "max_edges" in globals() else 300,
        50,
        key="sweep_max_edges",
    )
with cF3:
    sweep_min_abs_w = st.number_input(
        "Min |weight| (sweep)",
        min_value=0.0,
        value=float(min_abs_w) if "min_abs_w" in globals() else 0.001,
        step=0.001,
        format="%.4f",
        key="sweep_min_abs_w",
    )

st.caption(
    "This renders multiple gravis graphs on the page. Keep timepoints modest (e.g., ≤ 12) "
    "or the browser will become heavy."
)

from networkmodel.simulate import simulate_diffrax


def _compute_state_snapshot_sweep(sys: System, idx: Index, params: dict, t_eval: float):
    """
    Computes the state snapshot for a system over a specified time range.

    This function evaluates the state of a system after applying specific 
    parameters and computes its time evolution over a defined evaluation 
    time (`t_eval`). It integrates the system's state using numerical 
    integration, ensuring stability and consistency of the results. 
    The function also calculates additional outputs like kinetic evaluations 
    and total protein quantities.

    Args:
        sys (System): The dynamic system object representing the modeled 
            process. Must include methods for updating parameters and 
            state integration, as well as kinetic evaluation.
       """
    sys.update(**params)

    t_eval = float(t_eval)
    if t_eval <= 0.0:
        # Degenerate case: just evaluate at t=0
        t_grid = np.array([0.0], dtype=float)
    else:
        # Choose a reasonable number of integration points for stability
        # (you can increase this if you need more accurate snapshots)
        n_points = int(np.clip(200, 50, 2000))
        t_grid = np.linspace(0.0, t_eval, n_points, dtype=float)

        # Ensure monotonic and no NaNs/Infs
        t_grid = t_grid[np.isfinite(t_grid)]
        # If you ever end up with repeats, allow them but keep order
        # Diffrax requires strictly increasing values for this comparison grid
        t_grid.sort()

    Y = simulate_diffrax(sys, t_grid, rtol=1e-6, atol=1e-8, max_steps=50000)
    y_last = np.asarray(Y[-1], dtype=float)

    Kt = sys.kin.eval(t_eval) * sys.c_k
    totalP = {p: _total_protein_from_y(y_last, idx, p) for p in idx.proteins}
    return Kt, y_last, totalP


def _build_global_edge_tables_at_time_sweep(
        sys: System,
        idx: Index,
        params: dict,
        df_tf_model: pd.DataFrame | None,
        t_eval: float,
):
    """
    Builds global edge tables for signaling and transcription edges at a given time point in a system.

    This function computes the state snapshot at a specific time and constructs two dataframes:
    one for signaling edges and the other for transcriptional regulation edges. The signaling
    edges consider contributions from kinases to proteins based on weighted connectivity, while 
    the transcription edges consider transcription factor activity on their target genes.

    Args:
        sys (System): The system object containing global data, such as the global edge matrix 
            (`W_global`) and transcription factor matrix (`tf_mat`), required for edge"""
    Kt, y_last, totalP = _compute_state_snapshot_sweep(sys, idx, params, t_eval=float(t_eval))

    # ---- Signaling edges ----
    W = sys.W_global.tocoo()
    edge_contrib = W.data * Kt[W.col]

    prot_idx = np.searchsorted(idx.offset_s, W.row, side="right") - 1
    prot_idx = np.clip(prot_idx, 0, len(idx.proteins) - 1)

    df_sig = pd.DataFrame(
        {
            "src": np.asarray(idx.kinases, dtype=object)[W.col],
            "tgt": np.asarray(idx.proteins, dtype=object)[prot_idx],
            "weight": edge_contrib.astype(float),
            "beta": W.data.astype(float),
            "Kt": Kt[W.col].astype(float),
        }
    )
    df_sig = (
        df_sig.groupby(["src", "tgt"], as_index=False)
        .agg(weight=("weight", "sum"), beta=("beta", "mean"), Kt=("Kt", "mean"))
    )
    df_sig["type"] = "signaling"

    # ---- Transcription edges ----
    df_tf_edges = pd.DataFrame(columns=["src", "tgt", "weight", "type"])
    if df_tf_model is not None and not df_tf_model.empty:
        tf_mat = sys.tf_mat
        tf_scale = float(getattr(sys, "tf_scale", 1.0))

        rows = []
        for r in df_tf_model.itertuples(index=False):
            tf = getattr(r, "tf")
            tgt = getattr(r, "target")
            if tf not in idx.p2i or tgt not in idx.p2i:
                continue
            i_tgt = idx.p2i[tgt]
            j_tf = idx.p2i[tf]
            try:
                coeff = float(tf_mat[i_tgt, j_tf])
            except Exception:
                coeff = float(np.asarray(tf_mat[i_tgt, j_tf]).squeeze())

            if abs(coeff) < 1e-14:
                continue

            tf_level = float(totalP.get(tf, 0.0))
            drive = tf_scale * coeff * tf_level
            rows.append((tf, tgt, float(drive)))

        if rows:
            df_tf_edges = pd.DataFrame(rows, columns=["src", "tgt", "weight"])
            df_tf_edges["type"] = "transcription"

    return df_sig, df_tf_edges


def build_network_from_params_at_time(params, t_eval_local: float):
    """
    Builds a network configuration at a specific time using the provided parameters.

    This function constructs a network configuration by loading the system, applying
    parameters, and evaluating at a specified time. It leverages helper functions
    to generate global edge tables during the time sweep process.

    Args:
        params: Network configuration parameters to be applied when building the 
            network.
        t_eval_local (float): The specific evaluation time for applying the 
            parameters while constructing the network.

    Returns:
        The resulting global edge tables after applying the parameters at the 
        specified time.
    """
    sys_local, idx_local, _, _, _ = load_system()
    return _build_global_edge_tables_at_time_sweep(
        sys_local, idx_local, params, df_tf_model, t_eval=float(t_eval_local)
    )


def _prepare_edges_for_mode_at_time(mode: str, t_eval_local: float):
    """
    Prepares edges for visualization or analysis based on the given mode and time.
    Generates a dataframe of network edges and an associated title representing the
    state (wild type, knockout, or differential) at a specific time"""
    df_sig_wt, df_tf_wt = build_network_from_params_at_time(best_params, t_eval_local)
    df_sig_ko, df_tf_ko = build_network_from_params_at_time(ko_params, t_eval_local)

    if mode == "WT":
        df_edges = _merge_and_filter_edges(
            df_sig_wt, df_tf_wt,
            max_edges=int(sweep_max_edges),
            min_abs_weight=float(sweep_min_abs_w),
            include_tf=bool(sweep_include_tf),
        )
        title = f"WT @ t={float(t_eval_local):.1f} min"
        return df_edges, title

    if mode == "KO":
        df_edges = _merge_and_filter_edges(
            df_sig_ko, df_tf_ko,
            max_edges=int(sweep_max_edges),
            min_abs_weight=float(sweep_min_abs_w),
            include_tf=bool(sweep_include_tf),
        )
        title = f"KO @ t={float(t_eval_local):.1f} min"
        return df_edges, title

    # Δ(KO−WT)
    df_wt = pd.concat([df_sig_wt, df_tf_wt], ignore_index=True)
    df_ko = pd.concat([df_sig_ko, df_tf_ko], ignore_index=True)

    if df_wt.empty and df_ko.empty:
        return pd.DataFrame(), f"Δ @ t={float(t_eval_local):.1f} min"

    key_cols = ["src", "tgt", "type"]
    df_wt = df_wt[key_cols + ["weight"]].rename(columns={"weight": "w_wt"})
    df_ko = df_ko[key_cols + ["weight"]].rename(columns={"weight": "w_ko"})

    df_delta = df_ko.merge(df_wt, on=key_cols, how="outer")
    df_delta["w_ko"] = df_delta["w_ko"].fillna(0.0)
    df_delta["w_wt"] = df_delta["w_wt"].fillna(0.0)
    df_delta["weight"] = df_delta["w_ko"] - df_delta["w_wt"]
    df_delta = df_delta[key_cols + ["weight"]]

    df_delta["absw"] = df_delta["weight"].abs()
    df_delta = df_delta[df_delta["absw"] >= float(sweep_min_abs_w)].copy()
    df_delta = df_delta.sort_values("absw", ascending=False).head(int(sweep_max_edges)).copy()

    title = f"Δ(KO−WT) @ t={float(t_eval_local):.1f} min"
    return df_delta, title


# --- sweep execution + rendering ---
if st.button("Generate sweep graphs", key="sweep_run_btn"):
    times = np.linspace(0.0, float(sweep_t_end), int(sweep_n))

    sweep_results = []
    prog = st.progress(0.0)
    status = st.empty()

    for i, tt in enumerate(times):
        status.write(f"Computing {i + 1}/{len(times)} at t={tt:.1f} min")
        df_edges_i, title_i = _prepare_edges_for_mode_at_time(sweep_view_mode, float(tt))
        sweep_results.append((float(tt), df_edges_i, title_i))
        prog.progress((i + 1) / len(times))

    status.empty()
    prog.empty()

    # Render as a grid
    ncols = int(sweep_cols)
    rows = (len(sweep_results) + ncols - 1) // ncols

    k = 0
    for r in range(rows):
        cols = st.columns(ncols)
        for c in range(ncols):
            if k >= len(sweep_results):
                break
            tt, df_edges_i, title_i = sweep_results[k]
            with cols[c]:
                if df_edges_i is None or df_edges_i.empty:
                    st.info(f"{title_i}\n\nNo edges after filtering.")
                else:
                    html_i = _gravis_html_from_edges(df_edges_i, title_i)
                    components.html(html_i, height=420, scrolling=True)
            k += 1

    # Optional: export the sweep as a single CSV for reproducibility/debugging
    with st.expander("Export sweep edge tables (long format)"):
        long_rows = []
        for tt, df_edges_i, _ in sweep_results:
            if df_edges_i is None or df_edges_i.empty:
                continue
            tmp = df_edges_i.copy()
            tmp["time"] = tt
            long_rows.append(tmp)
        df_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
            columns=["src", "tgt", "type", "weight", "absw", "time"]
        )
        st.dataframe(df_long.head(500), use_container_width=True)
        st.download_button(
            "Download sweep edges CSV",
            data=df_long.to_csv(index=False),
            file_name=f"sweep_edges_{sweep_view_mode.lower()}_t0-{int(sweep_t_end)}_n{int(sweep_n)}.csv",
            mime="text/csv",
            key="sweep_edges_csv",
        )

# =========================
# TIME-RESOLVED ANIMATED NETWORK (pre-steady-state)
# Button-driven + session-state cached.
# Does NOT recompute on every Streamlit rerun.
# =========================

st.divider()
st.header("🕸️ Time-Resolved Animated Network")

# -------------------------
# State/vector helpers
# -------------------------
def _extract_mrna_vec_from_y(y_row: np.ndarray, idx: Index) -> np.ndarray:
    out = np.zeros(len(idx.proteins), dtype=float)
    for i in range(len(idx.proteins)):
        st_y = idx.offset_y[i]
        out[i] = float(y_row[st_y])
    return out


def _extract_total_protein_vec_from_y(y_row: np.ndarray, idx: Index) -> np.ndarray:
    out = np.zeros(len(idx.proteins), dtype=float)

    for i in range(len(idx.proteins)):
        st_y = idx.offset_y[i]
        prot = float(y_row[st_y + 1])
        ns = int(idx.n_sites[i])

        if ns > 0:
            phos_sum = float(np.sum(y_row[st_y + 2: st_y + 2 + ns]))
            out[i] = prot + phos_sum
        else:
            out[i] = prot

    return out


def _total_protein_from_Y_row(y_row: np.ndarray, idx: Index, protein: str) -> float:
    p_i = idx.p2i[protein]
    st_y = idx.offset_y[p_i]

    prot = float(y_row[st_y + 1])
    ns = int(idx.n_sites[p_i])

    if ns > 0:
        return prot + float(np.sum(y_row[st_y + 2: st_y + 2 + ns]))

    return prot


def _aligned_log2_ko_wt_for_nodes(
    nodes: list[str],
    idx: Index,
    wt_vec: np.ndarray,
    ko_vec: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return log2(KO/WT), aligned to Plotly/NetworkX node order.

    Unknown nodes are set to 0.0 so they appear neutral instead of breaking
    the color scale.
    """
    out = np.zeros(len(nodes), dtype=float)

    for i, node in enumerate(nodes):
        if node in idx.p2i:
            j = idx.p2i[node]
            out[i] = float(np.log2((float(ko_vec[j]) + eps) / (float(wt_vec[j]) + eps)))

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# -------------------------
# Time-resolved simulation
# -------------------------
def _simulate_state_series(params: dict, t_end: float, n_points: int):
    """Simulate one parameterization once and return a regular time grid."""
    sys_local, idx_local, _, df_tf_local, _ = load_system()
    sys_local.update(**params)

    t_end = float(t_end)
    n_points = int(n_points)

    t_target = np.linspace(0.0, t_end, n_points, dtype=float)

    t_sim, Y = simulate_until_steady(
        sys_local,
        t_max=t_end,
        n_points=n_points,
    )

    t_sim = np.asarray(t_sim, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2:
        raise RuntimeError(f"Expected 2D state matrix from simulation, got shape={Y.shape}")

    if len(t_sim) != len(t_target) or not np.allclose(t_sim, t_target):
        Y_interp = np.empty((len(t_target), Y.shape[1]), dtype=float)

        for j in range(Y.shape[1]):
            Y_interp[:, j] = np.interp(t_target, t_sim, Y[:, j])

        t_sim = t_target
        Y = Y_interp

    return sys_local, idx_local, df_tf_local, t_sim, Y


# -------------------------
# Edge construction at one frame
# -------------------------
def _edge_tables_from_state_at_time(
    sys_local: System,
    idx_local: Index,
    df_tf_local: pd.DataFrame | None,
    y_row: np.ndarray,
    t: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build signaling and TF edge tables from an already-simulated state row."""
    t = float(t)

    # Signaling edges: kinase -> target protein.
    # Kinase activity is represented through external/interpolated kinase input
    # multiplied by fitted/scaled c_k.
    Kt = sys_local.kin.eval(t) * sys_local.c_k

    W = sys_local.W_global.tocoo()
    edge_contrib = W.data * Kt[W.col]

    prot_idx = np.searchsorted(idx_local.offset_s, W.row, side="right") - 1
    prot_idx = np.clip(prot_idx, 0, len(idx_local.proteins) - 1)

    df_sig = pd.DataFrame(
        {
            "src": np.asarray(idx_local.kinases, dtype=object)[W.col],
            "tgt": np.asarray(idx_local.proteins, dtype=object)[prot_idx],
            "weight": edge_contrib.astype(float),
            "type": "signaling",
        }
    )

    if not df_sig.empty:
        df_sig = (
            df_sig.groupby(["src", "tgt", "type"], as_index=False)
            .agg(weight=("weight", "sum"))
        )

    # Transcription edges: TF protein level -> target.
    df_tf_edges = pd.DataFrame(columns=["src", "tgt", "weight", "type"])

    if df_tf_local is not None and not df_tf_local.empty:
        tf_mat = sys_local.tf_mat
        tf_scale = float(getattr(sys_local, "tf_scale", 1.0))

        rows = []

        for r in df_tf_local.itertuples(index=False):
            tf = getattr(r, "tf")
            tgt = getattr(r, "target")

            if tf not in idx_local.p2i or tgt not in idx_local.p2i:
                continue

            i_tgt = idx_local.p2i[tgt]
            j_tf = idx_local.p2i[tf]

            try:
                coeff = float(tf_mat[i_tgt, j_tf])
            except Exception:
                coeff = float(np.asarray(tf_mat[i_tgt, j_tf]).squeeze())

            if abs(coeff) < 1e-14:
                continue

            tf_level = _total_protein_from_Y_row(y_row, idx_local, tf)
            drive = tf_scale * coeff * tf_level

            rows.append((tf, tgt, float(drive), "transcription"))

        if rows:
            df_tf_edges = pd.DataFrame(
                rows,
                columns=["src", "tgt", "weight", "type"],
            )

    return df_sig, df_tf_edges


def _filter_edges_for_animation(
    df_sig: pd.DataFrame,
    df_tf: pd.DataFrame,
    *,
    include_tf: bool,
    min_abs_weight: float,
    top_k: int,
    edge_scale: float,
) -> pd.DataFrame:
    """Merge, clean, scale, threshold, and rank edges for one frame."""
    if df_sig is None or df_sig.empty:
        df_all = pd.DataFrame(columns=["src", "tgt", "weight", "type"])
    else:
        df_all = df_sig.copy()

    if include_tf and df_tf is not None and not df_tf.empty:
        df_all = pd.concat([df_all, df_tf], ignore_index=True)

    if df_all.empty:
        df_all["absw"] = []
        return df_all

    df_all["src"] = df_all["src"].astype(str)
    df_all["tgt"] = df_all["tgt"].astype(str)
    df_all["type"] = df_all["type"].astype(str)
    df_all["weight"] = pd.to_numeric(df_all["weight"], errors="coerce")

    df_all = (
        df_all.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["src", "tgt", "type", "weight"])
        .copy()
    )

    df_all["weight"] = df_all["weight"] * float(edge_scale)
    df_all["absw"] = df_all["weight"].abs()

    df_all = df_all[df_all["absw"] >= float(min_abs_weight)].copy()
    df_all = df_all.sort_values("absw", ascending=False).head(int(top_k)).copy()

    return df_all


def _delta_edges_for_frame(df_wt_all: pd.DataFrame, df_ko_all: pd.DataFrame) -> pd.DataFrame:
    """Align WT and KO edges and compute KO - WT edge weight."""
    key_cols = ["src", "tgt", "type"]

    if df_wt_all.empty and df_ko_all.empty:
        return pd.DataFrame(columns=["src", "tgt", "type", "weight"])

    if df_wt_all.empty:
        df_w = pd.DataFrame(columns=key_cols + ["w_wt"])
    else:
        df_w = df_wt_all[key_cols + ["weight"]].rename(columns={"weight": "w_wt"})

    if df_ko_all.empty:
        df_k = pd.DataFrame(columns=key_cols + ["w_ko"])
    else:
        df_k = df_ko_all[key_cols + ["weight"]].rename(columns={"weight": "w_ko"})

    df_d = df_k.merge(df_w, on=key_cols, how="outer")
    df_d["w_ko"] = df_d["w_ko"].fillna(0.0)
    df_d["w_wt"] = df_d["w_wt"].fillna(0.0)
    df_d["weight"] = df_d["w_ko"] - df_d["w_wt"]

    return df_d[key_cols + ["weight"]].copy()


def _build_animation_frames(
    *,
    view_mode: str,
    best_params: dict,
    ko_params: dict,
    include_tf_edges: bool,
    min_abs_weight: float,
    top_k: int,
    edge_scale: float,
    t_end: float,
    n_frames: int,
    node_metric: str,
):
    """Generate all animation data once."""
    sys_wt, idx_wt, df_tf_wt, t_wt, Y_wt = _simulate_state_series(
        best_params,
        t_end=float(t_end),
        n_points=int(n_frames),
    )

    sys_ko, idx_ko, df_tf_ko, t_ko, Y_ko = _simulate_state_series(
        ko_params,
        t_end=float(t_end),
        n_points=int(n_frames),
    )

    times = np.asarray(t_wt, dtype=float)

    if len(t_ko) != len(times) or not np.allclose(t_ko, times):
        raise RuntimeError("WT and KO simulations produced incompatible time grids.")

    edge_frames = []

    for i, tt in enumerate(times):
        df_sig_wt, df_tf_e_wt = _edge_tables_from_state_at_time(
            sys_wt,
            idx_wt,
            df_tf_wt,
            Y_wt[i],
            float(tt),
        )

        df_sig_ko, df_tf_e_ko = _edge_tables_from_state_at_time(
            sys_ko,
            idx_ko,
            df_tf_ko,
            Y_ko[i],
            float(tt),
        )

        df_wt_all = _filter_edges_for_animation(
            df_sig_wt,
            df_tf_e_wt,
            include_tf=include_tf_edges,
            min_abs_weight=0.0,
            top_k=10_000_000,
            edge_scale=1.0,
        )

        df_ko_all = _filter_edges_for_animation(
            df_sig_ko,
            df_tf_e_ko,
            include_tf=include_tf_edges,
            min_abs_weight=0.0,
            top_k=10_000_000,
            edge_scale=1.0,
        )

        if view_mode == "WT":
            df_e = _filter_edges_for_animation(
                df_sig_wt,
                df_tf_e_wt,
                include_tf=include_tf_edges,
                min_abs_weight=min_abs_weight,
                top_k=top_k,
                edge_scale=edge_scale,
            )

        elif view_mode == "KO":
            df_e = _filter_edges_for_animation(
                df_sig_ko,
                df_tf_e_ko,
                include_tf=include_tf_edges,
                min_abs_weight=min_abs_weight,
                top_k=top_k,
                edge_scale=edge_scale,
            )

        else:
            df_d = _delta_edges_for_frame(df_wt_all, df_ko_all)

            df_sig_d = df_d[df_d["type"] == "signaling"].copy()
            df_tf_d = df_d[df_d["type"] == "transcription"].copy()

            df_e = _filter_edges_for_animation(
                df_sig_d,
                df_tf_d,
                include_tf=include_tf_edges,
                min_abs_weight=min_abs_weight,
                top_k=top_k,
                edge_scale=edge_scale,
            )

        edge_frames.append(df_e)

    # Build union graph for stable node order.
    G_union = nx.DiGraph()

    for df in edge_frames:
        if df is None or df.empty:
            continue

        for r in df.itertuples(index=False):
            G_union.add_edge(str(r.src), str(r.tgt), etype=str(r.type))

    nodes_union = list(G_union.nodes())

    node_color_frames = None

    if node_metric != "None" and len(nodes_union) > 0:
        node_color_frames = []

        for i in range(len(times)):
            if "mRNA" in node_metric:
                wt_vec = _extract_mrna_vec_from_y(Y_wt[i], idx_wt)
                ko_vec = _extract_mrna_vec_from_y(Y_ko[i], idx_ko)
            else:
                wt_vec = _extract_total_protein_vec_from_y(Y_wt[i], idx_wt)
                ko_vec = _extract_total_protein_vec_from_y(Y_ko[i], idx_ko)

            node_color_frames.append(
                _aligned_log2_ko_wt_for_nodes(
                    nodes_union,
                    idx_wt,
                    wt_vec,
                    ko_vec,
                )
            )

    edge_long = []

    for tt, df in zip(times, edge_frames):
        if df is None or df.empty:
            continue

        tmp = df.copy()
        tmp["time"] = float(tt)
        edge_long.append(tmp)

    df_edge_time = (
        pd.concat(edge_long, ignore_index=True)
        if edge_long
        else pd.DataFrame(columns=["src", "tgt", "type", "weight", "absw", "time"])
    )

    return {
        "edge_frames": edge_frames,
        "times": times,
        "nodes_union": nodes_union,
        "node_color_frames": node_color_frames,
        "df_edge_time": df_edge_time,
    }


# -------------------------
# Plotly animated network
# -------------------------
def _build_union_graph_from_edge_frames(edge_frames: list[pd.DataFrame]) -> nx.DiGraph:
    G = nx.DiGraph()

    for df in edge_frames:
        if df is None or df.empty:
            continue

        for r in df.itertuples(index=False):
            G.add_edge(str(r.src), str(r.tgt), etype=str(r.type))

    return G


def _node_activity_frames(edge_frames: list[pd.DataFrame], nodes: list[str]) -> list[np.ndarray]:
    node2i = {n: i for i, n in enumerate(nodes)}
    frames = []

    for df in edge_frames:
        active = np.zeros(len(nodes), dtype=float)

        if df is not None and not df.empty:
            act_nodes = pd.unique(pd.concat([df["src"], df["tgt"]], ignore_index=True).astype(str))

            for n in act_nodes:
                j = node2i.get(n)
                if j is not None:
                    active[j] = 1.0

        frames.append(active)

    return frames


def _plotly_animated_network(
    edge_frames: list[pd.DataFrame],
    times: np.ndarray,
    title: str,
    node_color_frames: list[np.ndarray] | None = None,
    node_cmax: float = 2.0,
) -> go.Figure:
    G = _build_union_graph_from_edge_frames(edge_frames)

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No edges passed the current filters",
            template="plotly_white",
        )
        return fig

    pos = nx.spring_layout(G, k=0.8, seed=42)
    nodes = list(G.nodes())

    activity_frames = _node_activity_frames(edge_frames, nodes)

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]

    base_size = 10.0
    boost_size = 18.0

    union_edges = list(G.edges())
    edge_type = {(u, v): G.edges[u, v].get("etype", "signaling") for (u, v) in union_edges}

    def _edge_traces_for_frame(df_edges: pd.DataFrame):
        traces = []

        if df_edges is None or df_edges.empty:
            for color in ("#3498db", "#e67e22"):
                traces.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="lines",
                        line=dict(width=1, color=color),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
            return traces

        wmap = {
            (str(r.src), str(r.tgt)): float(r.weight)
            for r in df_edges.itertuples(index=False)
        }

        for etype, color, label in [
            ("signaling", "#3498db", "Signaling"),
            ("transcription", "#e67e22", "Transcription"),
        ]:
            x_e, y_e, widths = [], [], []

            for u, v in union_edges:
                if edge_type[(u, v)] != etype:
                    continue

                w = wmap.get((u, v), 0.0)

                if abs(w) <= 0.0:
                    continue

                x0, y0 = pos[u]
                x1, y1 = pos[v]

                x_e += [x0, x1, None]
                y_e += [y0, y1, None]
                widths.append(float(np.clip(np.log10(1.0 + abs(w)) * 3.0, 0.5, 6.0)))

            # Plotly cannot assign different line widths within one trace.
            # Use median width per edge type for stable rendering.
            line_width = float(np.median(widths)) if widths else 0.5

            traces.append(
                go.Scatter(
                    x=x_e,
                    y=y_e,
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    opacity=0.6,
                    hoverinfo="none",
                    name=label,
                    showlegend=False,
                )
            )

        return traces

    def _node_trace_for_frame(i: int):
        activity = activity_frames[i]
        sizes = base_size + boost_size * activity

        marker = dict(
            size=sizes,
            line=dict(width=1, color="black"),
        )

        if node_color_frames is not None:
            colors = np.asarray(node_color_frames[i], dtype=float)

            marker.update(
                color=colors,
                colorscale="RdBu_r",
                cmin=-float(node_cmax),
                cmax=float(node_cmax),
                colorbar=dict(title="log2(KO/WT)", thickness=15),
            )

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            hovertext=[f"<b>{n}</b>" for n in nodes],
            hoverinfo="text",
            marker=marker,
            showlegend=False,
        )

    init_data = _edge_traces_for_frame(edge_frames[0]) + [_node_trace_for_frame(0)]

    frames = []

    for i, (df_e, tt) in enumerate(zip(edge_frames, times)):
        frames.append(
            go.Frame(
                data=_edge_traces_for_frame(df_e) + [_node_trace_for_frame(i)],
                name=str(i),
                layout=go.Layout(
                    title=f"{title} — t={float(tt):.1f} min"
                ),
            )
        )

    fig = go.Figure(
        data=init_data,
        frames=frames,
    )

    fig.update_layout(
        title=f"{title} — t={float(times[0]):.1f} min",
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=250, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Time: ", suffix=" min"),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=f"{float(tt):.0f}",
                    )
                    for i, tt in enumerate(times)
                ],
            )
        ],
    )

    return fig


# -------------------------
# Matplotlib export
# -------------------------
def _export_network_animation_matplotlib(
    edge_frames: list[pd.DataFrame],
    times: np.ndarray,
    *,
    fmt: str = "gif",
    fps: int = 6,
    dpi: int = 150,
    node_cmax: float = 2.0,
    node_color_frames: list[np.ndarray] | None = None,
    figsize: tuple[float, float] = (9.0, 7.0),
) -> bytes:
    import os
    import tempfile

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if not edge_frames:
        raise RuntimeError("No animation frames available. Generate the animated network first.")

    if len(times) != len(edge_frames):
        raise RuntimeError(f"times length {len(times)} != edge_frames length {len(edge_frames)}")

    G = _build_union_graph_from_edge_frames(edge_frames)

    if G.number_of_nodes() == 0:
        raise RuntimeError("No nodes available. Lower Min |weight| or increase top edges.")

    nodes = list(G.nodes())
    pos = nx.spring_layout(G, k=0.8, seed=42)

    node_xy = np.array([pos[n] for n in nodes], dtype=float)
    node2i = {n: i for i, n in enumerate(nodes)}

    def _segments_for_type(df: pd.DataFrame, etype: str):
        if df is None or df.empty:
            return np.zeros((0, 2, 2), dtype=float), np.zeros((0,), dtype=float)

        sub = df[df["type"].astype(str) == etype]

        if sub.empty:
            return np.zeros((0, 2, 2), dtype=float), np.zeros((0,), dtype=float)

        segs = []
        widths = []

        for r in sub.itertuples(index=False):
            u = str(r.src)
            v = str(r.tgt)

            if u not in node2i or v not in node2i:
                continue

            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = float(r.weight)

            segs.append([[x0, y0], [x1, y1]])
            widths.append(float(np.clip(np.log10(1.0 + abs(w)) * 2.5, 0.3, 4.5)))

        if not segs:
            return np.zeros((0, 2, 2), dtype=float), np.zeros((0,), dtype=float)

        return np.asarray(segs, dtype=float), np.asarray(widths, dtype=float)

    sig_segments, sig_widths = [], []
    tf_segments, tf_widths = [], []

    for df in edge_frames:
        s_seg, s_w = _segments_for_type(df, "signaling")
        t_seg, t_w = _segments_for_type(df, "transcription")

        sig_segments.append(s_seg)
        sig_widths.append(s_w)
        tf_segments.append(t_seg)
        tf_widths.append(t_w)

    use_node_colors = (
        node_color_frames is not None
        and len(node_color_frames) == len(edge_frames)
    )

    norm = Normalize(vmin=-float(node_cmax), vmax=float(node_cmax))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    lc_sig = LineCollection([], linewidths=1.0, alpha=0.60, color="#3498db")
    lc_tf = LineCollection([], linewidths=1.0, alpha=0.60, color="#e67e22")

    ax.add_collection(lc_sig)
    ax.add_collection(lc_tf)

    sc = ax.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        s=np.full(len(nodes), 65.0),
        edgecolors="black",
        linewidths=0.6,
    )

    labels = [
        ax.text(
            pos[n][0],
            pos[n][1],
            str(n),
            fontsize=7,
            ha="center",
            va="bottom",
        )
        for n in nodes
    ]

    title_obj = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
    )

    pad = 0.08
    xmin, ymin = node_xy.min(axis=0) - pad
    xmax, ymax = node_xy.max(axis=0) + pad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if use_node_colors:
        mappable = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("log2(KO/WT)")

    def _update(i: int):
        lc_sig.set_segments(sig_segments[i])
        lc_sig.set_linewidths(sig_widths[i] if len(sig_widths[i]) else 0.2)

        lc_tf.set_segments(tf_segments[i])
        lc_tf.set_linewidths(tf_widths[i] if len(tf_widths[i]) else 0.2)

        if use_node_colors:
            colors_i = np.asarray(node_color_frames[i], dtype=float)

            if colors_i.shape[0] == len(nodes):
                sc.set_array(colors_i)
                sc.set_cmap("RdBu_r")
                sc.set_norm(norm)

        title_obj.set_text(f"Network at t={float(times[i]):.1f} min")

        return (lc_sig, lc_tf, sc, title_obj, *labels)

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(edge_frames),
        interval=1000 / max(1, int(fps)),
        blit=False,
    )

    fmt = fmt.lower().strip()

    if fmt not in {"gif", "mp4"}:
        plt.close(fig)
        raise ValueError("fmt must be 'gif' or 'mp4'")

    suffix = f".{fmt}"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if fmt == "gif":
            writer = PillowWriter(fps=int(fps))
        else:
            writer = FFMpegWriter(fps=int(fps), codec="libx264", bitrate=1800)

        anim.save(tmp_path, writer=writer, dpi=int(dpi))

        with open(tmp_path, "rb") as f:
            data = f.read()

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        plt.close(fig)

    return data


# -------------------------
# Streamlit controls
# -------------------------
if target is None:
    st.info("No KO target is selected. WT view works, but KO/Δ views will be identical or uninformative.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    anim_view_mode = st.selectbox(
        "View",
        ["WT", "KO", "Δ(KO−WT)"],
        index=1,
        key="anim_net_view_mode",
    )

with c2:
    anim_t_end = st.slider(
        "End time (min)",
        min_value=120,
        max_value=24 * 7 * 60,
        value=960,
        step=120,
        key="anim_net_t_end",
    )

with c3:
    anim_n_frames = st.slider(
        "Frames",
        min_value=10,
        max_value=120,
        value=30,
        step=5,
        key="anim_net_n_frames",
    )

with c4:
    anim_include_tf = st.checkbox(
        "Include TF edges",
        value=True,
        key="anim_net_include_tf_edges",
    )

c5, c6, c7, c8 = st.columns(4)

with c5:
    anim_min_abs_w = st.number_input(
        "Min |weight| filter",
        min_value=0.0,
        value=0.001,
        step=0.001,
        format="%.4f",
        key="anim_net_min_abs_w",
    )

with c6:
    anim_top_k = st.slider(
        "Top edges per frame",
        min_value=50,
        max_value=800,
        value=250,
        step=50,
        key="anim_net_top_k",
    )

with c7:
    anim_node_metric = st.selectbox(
        "Node color metric",
        ["None", "ΔmRNA (log2 KO/WT)", "ΔProtein (log2 KO/WT)"],
        index=2,
        key="anim_net_node_metric",
    )

with c8:
    anim_node_cmax = st.slider(
        "Node color range (± log2)",
        min_value=1.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        key="anim_net_node_cmax",
    )

anim_edge_scale = st.slider(
    "Edge weight scaling",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.1,
    key="anim_net_edge_scale",
)

anim_cache_key = (
    str(anim_view_mode),
    float(anim_t_end),
    int(anim_n_frames),
    bool(anim_include_tf),
    float(anim_min_abs_w),
    int(anim_top_k),
    str(anim_node_metric),
    float(anim_node_cmax),
    float(anim_edge_scale),
    str(perturbation_signature),
)

generate_anim = st.button(
    "Generate animated network",
    key="anim_net_generate_btn",
    type="primary",
)

if generate_anim:
    with st.spinner("Simulating WT/KO trajectories and building time-resolved network frames..."):
        anim_payload = _build_animation_frames(
            view_mode=str(anim_view_mode),
            best_params=best_params,
            ko_params=ko_params,
            include_tf_edges=bool(anim_include_tf),
            min_abs_weight=float(anim_min_abs_w),
            top_k=int(anim_top_k),
            edge_scale=float(anim_edge_scale),
            t_end=float(anim_t_end),
            n_frames=int(anim_n_frames),
            node_metric=str(anim_node_metric),
        )

    st.session_state["anim_net_cache_key"] = anim_cache_key
    st.session_state["anim_net_payload"] = anim_payload

has_anim_payload = (
    st.session_state.get("anim_net_cache_key") == anim_cache_key
    and "anim_net_payload" in st.session_state
)

if not has_anim_payload:
    st.info("Click **Generate animated network** to compute and display this section.")
else:
    anim_payload = st.session_state["anim_net_payload"]

    edge_frames = anim_payload["edge_frames"]
    times = anim_payload["times"]
    node_color_frames = anim_payload["node_color_frames"]
    df_edge_time = anim_payload["df_edge_time"]

    fig_anim = _plotly_animated_network(
        edge_frames=edge_frames,
        times=times,
        title=f"{anim_view_mode} network",
        node_color_frames=node_color_frames,
        node_cmax=float(anim_node_cmax),
    )

    st.plotly_chart(fig_anim, use_container_width=True, key="anim_net_plot")

    c_dl1, c_dl2 = st.columns(2)

    with c_dl1:
        st.download_button(
            "Download edge dynamics CSV",
            data=df_edge_time.to_csv(index=False),
            file_name=(
                f"edge_dynamics_{str(anim_view_mode).lower()}"
                f"_t0-{int(anim_t_end)}_frames{int(anim_n_frames)}.csv"
            ),
            mime="text/csv",
            key="anim_net_edge_csv",
        )

    with c_dl2:
        st.download_button(
            "Download edge dynamics JSON",
            data=df_edge_time.to_json(orient="records"),
            file_name=(
                f"edge_dynamics_{str(anim_view_mode).lower()}"
                f"_t0-{int(anim_t_end)}_frames{int(anim_n_frames)}.json"
            ),
            mime="application/json",
            key="anim_net_edge_json",
        )

    with st.expander("Preview edge dynamics table"):
        st.dataframe(df_edge_time.head(500), use_container_width=True)

    st.subheader("Export animation file")

    cE1, cE2, cE3, cE4 = st.columns(4)

    with cE1:
        export_fmt = st.selectbox(
            "Format",
            ["gif", "mp4"],
            index=0,
            key="anim_export_fmt",
        )

    with cE2:
        export_fps = st.slider(
            "FPS",
            min_value=2,
            max_value=20,
            value=6,
            step=1,
            key="anim_export_fps",
        )

    with cE3:
        export_dpi = st.slider(
            "DPI",
            min_value=80,
            max_value=300,
            value=150,
            step=10,
            key="anim_export_dpi",
        )

    with cE4:
        export_node_cmax = st.slider(
            "Export node color range (± log2)",
            min_value=1.1,
            max_value=10.0,
            value=float(anim_node_cmax),
            step=0.1,
            key="anim_export_node_cmax",
        )

    if st.button("Render export file", key="anim_export_btn"):
        try:
            with st.spinner("Rendering animation export..."):
                export_bytes = _export_network_animation_matplotlib(
                    edge_frames=edge_frames,
                    times=times,
                    fmt=str(export_fmt),
                    fps=int(export_fps),
                    dpi=int(export_dpi),
                    node_cmax=float(export_node_cmax),
                    node_color_frames=node_color_frames,
                )

            st.download_button(
                f"Download {str(export_fmt).upper()}",
                data=export_bytes,
                file_name=(
                    f"network_{str(anim_view_mode).lower()}"
                    f"_t0-{int(anim_t_end)}_frames{int(anim_n_frames)}.{export_fmt}"
                ),
                mime=("image/gif" if export_fmt == "gif" else "video/mp4"),
                key="anim_export_download_btn",
            )

        except Exception as exc:
            st.error(f"Animation export failed: {exc}")