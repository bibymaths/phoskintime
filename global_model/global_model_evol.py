#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global ODE Dual-Fit Optimization Pipeline.

This script fits a global Ordinary Differential Equation (ODE) model to temporal
Fold Change (FC) data for both Proteins and RNA. It incorporates prior knowledge
from 'Alpha Values' (Transcription Factor/Kinase edges) and 'Beta Values' (Kinase/TF
activities) to constrain the fit.

The model topology includes:
    1. mRNA (R): Driven by Transcription Factors (TFs).
    2. Protein (P): Translated from mRNA.
    3. Phosphosites (Psite): Phosphorylated by Kinases.

Optimization:
    Uses `scipy.optimize.minimize` (Trust-Region Constrained) to estimate:
    - c_k: Kinase specific activity multipliers.
    - D_i: Protein-specific degradation/turnover rates.
    - alpha: Sparse interaction weights for regulatory edges.
"""

# ------------------------------
# Imports
# ------------------------------
import math
import argparse  # CLI parsing
import json  # JSON I/O for fitted params and summaries
import os  # Filesystem utilities
import re  # Regex for time column parsing
import time  # Timings
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any

import numpy as np  # Numerical computing
import pandas as pd  # Data handling
from scipy.integrate import solve_ivp  # ODE solver
from scipy import sparse  # Sparse matrices for α and maps
from scipy.optimize import minimize  # Optimizer for parameter estimation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm  # Progress bars
from numba import njit  # JIT compilation for performance
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.core.problem import StarmapParallelization

# ------------------------------
# Constants & Configuration
# ------------------------------

# Fixed time grids (minutes) for Protein and RNA observations
TIME_POINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    dtype=float
)
TIME_POINTS_RNA = np.array(
    [4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    dtype=float
)


# ------------------------------
# Utility Helpers
# ------------------------------

def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names to snake_case lowercase for robust parsing.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with renormalized column names.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _ensure_dir(p: str) -> None:
    """Create an output directory if it does not exist."""
    if not os.path.exists(p):
        print(f"[SYSTEM] Creating directory: {p}")
        os.makedirs(p, exist_ok=True)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Stable softplus transformation to enforce positivity of parameters.
    y = ln(1 + exp(x))
    """
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def inv_softplus(y: np.ndarray) -> np.ndarray:
    """
    Inverse softplus transformation for initializing raw parameters.
    x = ln(exp(y) - 1)
    """
    return np.log(np.expm1(np.maximum(y, 1e-12)))


def _find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    """
    Find the first column in the DataFrame that matches any candidate name.

    Args:
        df: The DataFrame to search.
        cands: List of candidate column names strings.

    Returns:
        The matching column name or None.
    """
    for c in cands:
        if c in df.columns:
            return c
    return None


def _time_cols(df: pd.DataFrame) -> List[str]:
    """
    Identify columns representing time points.
    Matches columns containing a single numeric token.
    """
    times = []
    for col in df.columns:
        # Regex to find float/int in string
        m = re.findall(r"[-+]?\d*\.?\d+", str(col))
        if len(m) == 1:
            try:
                v = float(m[0])
                times.append((col, v))
            except ValueError:
                pass
    times.sort(key=lambda x: x[1])
    return [c for c, _ in times]


def build_layout(idx, r_site_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Builds the flat state vector layout for the ODE system.

    The state vector 'y' is packed hierarchically per protein:
    Block i: [R (mRNA), P (Protein), Psite_0, ..., Psite_m]

    Args:
        idx: The Index object containing topology counts.
        r_site_list: List of arrays containing site-specific parameters.

    Returns:
        Tuple containing:
        - block_start: Indices where each protein block begins.
        - n_sites: Number of sites per protein.
        - site_start: Indices where site parameters begin in the flat array.
        - r_site_flat: Flattened array of site parameters.
        - total_sites: Total count of phosphorylation sites in the system.
    """
    N = idx.N
    n_sites = np.array(idx.n_sites_per_protein, dtype=np.int32)

    block_start = np.empty(N, dtype=np.int32)
    site_start = np.empty(N, dtype=np.int32)

    bs = 0
    ss = 0
    for i in range(N):
        block_start[i] = bs
        site_start[i] = ss
        bs += 2 + n_sites[i]  # 2 represents R + P states
        ss += n_sites[i]

    total_sites = int(ss)

    # Flatten r_site for Numba consumption
    r_site_flat = np.zeros(total_sites, dtype=np.float64)
    for i in range(N):
        m = int(n_sites[i])
        if m == 0:
            continue
        r_site_flat[site_start[i]:site_start[i] + m] = np.asarray(r_site_list[i], dtype=np.float64)

    return block_start, n_sites, site_start, r_site_flat, total_sites


def normalize_signed_simplex(df: pd.DataFrame, group: str, val: str, clip: float = 2.0) -> pd.DataFrame:
    """
    Enforce constraints on beta values:
      1. Clip values to [-clip, clip].
      2. Normalize within each group (e.g., per Kinase) so sum(beta) = 1.
         If sum is near 0, sets a default basis.
    """
    d = df.copy()
    d[val] = pd.to_numeric(d[val], errors="coerce").fillna(0.0)
    d[val] = d[val].clip(-clip, clip)

    out = []
    for g, sub in d.groupby(group, dropna=False):
        s = float(sub[val].sum())
        sub = sub.copy()
        if abs(s) < 1e-12:
            # Fallback strategy for zero-sum groups
            if "psite" in sub.columns:
                mask0 = sub["psite"].astype(str).fillna("").str.strip().eq("")
                if mask0.any():
                    sub.loc[:, val] = 0.0
                    sub.loc[mask0, val] = 1.0
                else:
                    sub.loc[:, val] = 0.0
                    sub.iloc[0, sub.columns.get_loc(val)] = 1.0
            else:
                sub.loc[:, val] = 0.0
                sub.iloc[0, sub.columns.get_loc(val)] = 1.0
        else:
            sub[val] = sub[val] / s
        out.append(sub)
    return pd.concat(out, ignore_index=True)


# ------------------------------
# Data Loading Functions
# ------------------------------

def load_interactions(path: str) -> pd.DataFrame:
    """
    Load Protein–Psite–Kinase interactions from CSV.
    Requires columns for Protein, Psite, and Kinase.
    """
    print(f"[IO] Loading interaction network from: {path}")
    df = pd.read_csv(path)
    df = _normcols(df)
    pcol = _find_col(df, ["protein", "gene", "geneid", "prot"])
    scol = _find_col(df, ["psite", "site", "phosphosite", "residue"])
    kcol = _find_col(df, ["kinase", "kinases", "k", "enzyme"])

    if not (pcol and scol and kcol):
        raise ValueError(f"Interaction file must have Protein, Psite, Kinase. Found: {df.columns.tolist()}")

    out = pd.DataFrame({
        "protein": df[pcol].astype(str).str.strip(),
        "psite": df[scol].astype(str).str.strip(),
        "kinase": df[kcol].astype(str).str.strip(),
    }).drop_duplicates()

    print(f"     -> Loaded {len(out)} unique interactions.")
    return out.reset_index(drop=True)


def load_kinopt_effects(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Kinase Optimization (kinopt) Alpha and Beta sheets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (kin_alpha, kin_beta)
    """
    print(f"[IO] Loading Kinase Effects (Alpha/Beta) from: {path}")
    xl = pd.ExcelFile(path)

    # --- Parse Alpha Values ---
    dfA = pd.read_excel(xl, sheet_name="Alpha Values")
    dfA = _normcols(dfA)
    gcol = _find_col(dfA, ["gene", "protein", "geneid"])
    scol = _find_col(dfA, ["psite", "site"])
    kcol = _find_col(dfA, ["kinase", "kinases", "k"])
    aval = _find_col(dfA, ["alpha", "value", "effect", "score", "weight"])

    if not (gcol and scol and kcol and aval):
        raise ValueError("kinopt::Alpha Values must have (Gene, Psite, Kinase, Alpha).")

    kin_alpha = dfA[[gcol, scol, kcol, aval]].rename(
        columns={gcol: "protein", scol: "psite", kcol: "kinase", aval: "alpha"})
    kin_alpha["protein"] = kin_alpha["protein"].astype(str).str.strip()
    kin_alpha["psite"] = kin_alpha["psite"].astype(str).str.strip()
    kin_alpha["kinase"] = kin_alpha["kinase"].astype(str).str.strip()
    kin_alpha["alpha"] = pd.to_numeric(kin_alpha["alpha"], errors="coerce").fillna(0.0)

    # --- Parse Beta Values ---
    dfB = pd.read_excel(xl, sheet_name="Beta Values")
    dfB = _normcols(dfB)
    kcolB = _find_col(dfB, ["kinase", "kinases", "k"])
    scolB = _find_col(dfB, ["psite", "site"])
    bval = _find_col(dfB, ["beta", "value", "effect", "score", "weight"])

    if not (kcolB and bval):
        raise ValueError("kinopt::Beta Values must have (Kinase, [Psite], Beta).")

    kin_beta = dfB[[kcolB, bval] + ([scolB] if scolB else [])].rename(
        columns={kcolB: "kinase", bval: "beta", (scolB or "psite"): "psite"})
    kin_beta["kinase"] = kin_beta["kinase"].astype(str).str.strip()
    if "psite" in kin_beta.columns:
        kin_beta["psite"] = kin_beta["psite"].astype(str).str.strip()
    else:
        kin_beta["psite"] = ""

    # Identify unknown sites and normalize
    kin_beta["is_unknown_site"] = kin_beta["psite"].eq("")
    kin_beta["beta"] = pd.to_numeric(kin_beta["beta"], errors="coerce").fillna(0.0)
    kin_beta = normalize_signed_simplex(kin_beta, group="kinase", val="beta", clip=2.0)

    print(f"     -> Loaded {len(kin_alpha)} alpha priors and {len(kin_beta)} beta priors.")
    return kin_alpha, kin_beta


def load_tfopt_effects(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Transcription Factor Optimization (tfopt) Alpha and Beta sheets.
    """
    print(f"[IO] Loading TF Effects (Alpha/Beta) from: {path}")
    xl = pd.ExcelFile(path)

    # --- Alpha Values (TF -> mRNA edges) ---
    dfA = pd.read_excel(xl, sheet_name="Alpha Values")
    dfA = _normcols(dfA)
    mcol = _find_col(dfA, ["mrna", "gene", "target", "protein", "geneid"])
    tcol = _find_col(dfA, ["tf", "source"])
    vcol = _find_col(dfA, ["value", "alpha", "effect", "score", "weight"])
    if not (mcol and tcol and vcol):
        raise ValueError("tfopt::Alpha Values must have (mRNA, TF, Value).")

    tf_alpha_edges = dfA[[mcol, tcol, vcol]].rename(columns={mcol: "mrna", tcol: "tf", vcol: "alpha"})
    tf_alpha_edges["mrna"] = tf_alpha_edges["mrna"].astype(str).str.strip()
    tf_alpha_edges["tf"] = tf_alpha_edges["tf"].astype(str).str.strip()
    tf_alpha_edges["alpha"] = pd.to_numeric(tf_alpha_edges["alpha"], errors="coerce").fillna(0.0)

    # Enforce alpha simplex per mRNA: alpha>=0, sum=1
    tf_alpha_edges["alpha"] = tf_alpha_edges["alpha"].clip(lower=0.0)
    s = tf_alpha_edges.groupby("mrna")["alpha"].transform("sum").replace(0, 1.0)
    tf_alpha_edges["alpha"] = tf_alpha_edges["alpha"] / s

    # --- Beta Values ---
    try:
        dfB = pd.read_excel(xl, sheet_name="Beta Values")
        dfB = _normcols(dfB)
        tcolB = _find_col(dfB, ["tf"])
        scolB = _find_col(dfB, ["psite", "site"])
        vcolB = _find_col(dfB, ["value", "beta", "effect", "score", "weight"])
        if tcolB and vcolB:
            tf_beta = dfB[[tcolB, vcolB] + ([scolB] if scolB else [])].rename(
                columns={tcolB: "tf", vcolB: "beta", (scolB or "psite"): "psite"}
            )
            tf_beta["tf"] = tf_beta["tf"].astype(str).str.strip()
            tf_beta["psite"] = tf_beta["psite"].astype(str).fillna("").str.strip()
            tf_beta["beta"] = pd.to_numeric(tf_beta["beta"], errors="coerce").fillna(0.0)
            tf_beta = normalize_signed_simplex(tf_beta, group="tf", val="beta", clip=2.0)
        else:
            tf_beta = pd.DataFrame(columns=["tf", "psite", "beta"])
    except Exception:
        print("     -> [WARN] No Beta Values found for TF, using empty frame.")
        tf_beta = pd.DataFrame(columns=["tf", "psite", "beta"])

    print(f"     -> Loaded {len(tf_alpha_edges)} TF alpha edges and {len(tf_beta)} TF beta priors.")
    return tf_alpha_edges, tf_beta


def load_estimated_psite_FC(path: str) -> pd.DataFrame:
    """Load phosphosite FC time series from kinopt::Estimated."""
    print(f"[IO] Loading Phosphosite FC estimates from: {path}")
    df = pd.read_excel(path, sheet_name="Estimated")
    df = _normcols(df)

    # Identify columns
    pcol = _find_col(df, ["protein", "gene", "geneid", "target"])
    scol = _find_col(df, ["psite", "site", "phosphosite", "residue"])

    if not (pcol and scol):
        raise ValueError("kinopt::Estimated must have Protein and Psite columns.")

    # Drop rows where psite is missing (these are likely whole-protein data)
    df = df.dropna(subset=[scol])
    df = df[df[scol].astype(str).str.strip() != ""]

    tcols = _time_cols(df.drop(columns=[pcol, scol], errors="ignore"))

    # Melt to tidy format
    tidy = df[[pcol, scol] + tcols].melt(id_vars=[pcol, scol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
    tidy = tidy.drop(columns=["time_col"])
    tidy = tidy.rename(columns={pcol: "protein", scol: "psite"})

    # Clean and Filter
    tidy["protein"] = tidy["protein"].astype(str).str.strip()
    tidy["psite"] = tidy["psite"].astype(str).str.strip()
    tidy = tidy[tidy["time"].isin(TIME_POINTS)].copy()
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    tidy = tidy.dropna(subset=["fc"])

    print(f"     -> Loaded Phosphosite FC data with {len(tidy)} observations.")
    return tidy.sort_values(["protein", "psite", "time"]).reset_index(drop=True)

def load_estimated_protein_FC(path: str) -> pd.DataFrame:
    """
    Load protein Fold Change (FC) time series.
    """
    print(f"[IO] Loading Protein FC estimates from: {path}")
    df = pd.read_excel(path, sheet_name="Estimated")
    df = _normcols(df)
    namecol = _find_col(df, ["protein", "gene", "geneid", "target"])
    if namecol is None:
        raise ValueError("kinopt::Estimated must include a protein/gene identifier column.")

    tcols = _time_cols(df.drop(columns=[namecol], errors="ignore"))
    tidy = df[[namecol] + tcols].melt(id_vars=[namecol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
    tidy = tidy.drop(columns=["time_col"])
    tidy = tidy.rename(columns={namecol: "protein"})
    tidy["protein"] = tidy["protein"].astype(str).str.strip()

    # Filter to specific time points
    tidy = tidy[tidy["time"].isin(TIME_POINTS)].copy()
    tidy = tidy.sort_values(["protein", "time"]).reset_index(drop=True)
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")

    print(f"     -> Loaded Protein FC data with {len(tidy)} observations.")
    return tidy


def load_estimated_rna_FC(path: str) -> pd.DataFrame:
    """
    Load RNA Fold Change (FC) time series.
    """
    print(f"[IO] Loading RNA FC estimates from: {path}")
    df = pd.read_excel(path, sheet_name="Estimated")
    df = _normcols(df)
    namecol = _find_col(df, ["mrna", "protein", "gene", "geneid", "target"])
    if namecol is None:
        raise ValueError("tfopt::Estimated must include an mRNA/protein identifier column.")

    tcols = _time_cols(df.drop(columns=[namecol], errors="ignore"))
    tidy = df[[namecol] + tcols].melt(id_vars=[namecol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
    tidy = tidy.drop(columns=["time_col"])
    tidy = tidy.rename(columns={namecol: "protein"})
    tidy["protein"] = tidy["protein"].astype(str).str.strip()

    # Filter to specific time points
    tidy = tidy[tidy["time"].isin(TIME_POINTS_RNA)].copy()
    tidy = tidy.sort_values(["protein", "time"]).reset_index(drop=True)
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")

    print(f"     -> Loaded RNA FC data with {len(tidy)} observations.")
    return tidy


# ------------------------------
# Indexing & Sparse Maps
# ------------------------------

class Index:
    """
    Manages the global state indexing for Proteins, Sites, and Kinases.

    Attributes:
        proteins (List[str]): Sorted list of unique protein names.
        sites (List[List[str]]): List of sites per protein.
        kinases (List[str]): Sorted list of unique kinase names.
        N (int): Total number of proteins.
        state_dim (int): Total dimension of the ODE state vector y.
    """

    def __init__(self, interactions: pd.DataFrame):
        # Unique proteins
        self.proteins = sorted(interactions["protein"].unique().tolist())
        # Map protein name to index
        self.p2i = {p: i for i, p in enumerate(self.proteins)}
        # Per-protein site lists
        self.sites = [interactions.loc[interactions["protein"] == p, "psite"].unique().tolist() for p in self.proteins]
        # Unique kinases
        self.kinases = sorted(interactions["kinase"].unique().tolist())
        # Map kinase name to index
        self.k2i = {k: i for i, k in enumerate(self.kinases)}
        # Counts
        self.N = len(self.proteins)
        self.n_sites_per_protein = [len(s) for s in self.sites]

        # Offsets for slicing y vector: [R, P, Psite_1..Psite_m]
        # State layout: y is concatenated blocks.
        self.offset = []
        curr = 0
        for nsi in self.n_sites_per_protein:
            self.offset.append(curr)
            curr += 2 + nsi  # 2 = mRNA + Protein
        # Total state dimension across all proteins
        self.state_dim = curr

    def block(self, i: int) -> slice:
        """Return the y-slice for protein i."""
        s = self.offset[i]
        e = s + 2 + self.n_sites_per_protein[i]
        return slice(s, e)


def _build_W_one(args):
    """Worker helper to build one protein's W matrix (rows: sites, cols: kinases)."""
    i, p, interactions, sites, k2i = args
    sub = interactions[interactions["protein"] == p][["psite", "kinase"]]
    site_order = {s: r for r, s in enumerate(sites)}
    rows, cols = [], []
    for _, r in sub.iterrows():
        s = r["psite"];
        k = r["kinase"]
        if s in site_order and k in k2i:
            rows.append(site_order[s]);
            cols.append(k2i[k])
    data = np.ones(len(rows), float)
    return i, sparse.csr_matrix((data, (rows, cols)), shape=(len(sites), len(k2i)))


def build_W(interactions: pd.DataFrame, idx: Index) -> List[sparse.csr_matrix]:
    """
    Build sparse adjacency maps W_i (site x kinase) for all proteins in parallel.
    Each row in W_i represents a phosphosite, columns represent kinases.
    """
    print("[INFO] Building Sparse W (site×kinase) maps ...")
    with mp.get_context("fork").Pool(processes=6) as pool:
        tasks = ((i, p, interactions, idx.sites[i], idx.k2i) for i, p in enumerate(idx.proteins))
        results = list(tqdm(pool.imap_unordered(_build_W_one, tasks), total=idx.N, desc="W maps", ncols=90))
    W = [None] * idx.N
    for i, Wi in results:
        W[i] = Wi
    return W


# ------------------------------
# Parameter Initialization
# ------------------------------

def init_params_from_effects(idx: Index, W_list: List[sparse.csr_matrix],
                             kin_alpha: pd.DataFrame, kin_beta: pd.DataFrame,
                             tf_alpha: pd.DataFrame, alpha_scale=0.2) -> Dict[str, object]:
    """
    Initialize model parameters (c_k, A_i, B_i, C_i, D_i, and α) using the loaded effects tables.

    Returns:
        Dict containing initialized arrays/matrices.
    """
    print("[INIT] Initializing parameters from effect priors ...")

    # 1. c_k: Kinase activity multipliers (from Beta Values)
    # Unknown-site entries treated as general kinase effect; fallback to mean
    base = kin_beta[kin_beta["is_unknown_site"]].groupby("kinase")["beta"].mean()
    c_k = base.reindex(idx.kinases).fillna(kin_beta.groupby("kinase")["beta"].mean()).fillna(0.0).values.astype(float)
    c_k = softplus(c_k)
    c_k /= (np.mean(c_k) + 1e-12)  # Normalize

    # 2. A_i: Baseline RNA synthesis (Transcription)
    # Scaled by TFOPT dynamic drive later
    A_i = np.full(idx.N, 1.0, dtype=float)
    A_i = softplus(A_i)
    A_i /= (np.median(A_i) + 1e-12)

    # 3. Fixed baselines (Degradation/Translation rates)
    # B_i: RNA degradation
    # C_i: Translation rate (RNA -> Protein)
    # D_i: Protein degradation
    N = idx.N
    B_i = np.full(N, 0.2, float)
    C_i = np.full(N, 0.5, float)
    D_i = np.full(N, 0.05, float)
    r_site = [np.zeros(idx.n_sites_per_protein[i], float) for i in range(N)]

    # 4. α (Alpha): Interaction strengths (Kinase -> Psite)
    # Seeded from kin_alpha, falling back to `alpha_scale`.
    print("[INIT] Seeding α from kinopt::Alpha Values (fallback to default where missing) ...")
    kin_alpha_keyed = kin_alpha.copy()
    kin_alpha_keyed["key"] = list(zip(kin_alpha_keyed["protein"].astype(str),
                                      kin_alpha_keyed["psite"].astype(str),
                                      kin_alpha_keyed["kinase"].astype(str)))
    alpha_lookup = {k: v for k, v in zip(kin_alpha_keyed["key"], kin_alpha_keyed["alpha"])}

    alpha_init = []
    for i, p in enumerate(tqdm(idx.proteins, desc="Init α", ncols=90)):
        Wi = W_list[i].tocsr()
        if Wi.nnz == 0:
            alpha_init.append(Wi.copy())
            continue
        rows, cols = Wi.nonzero()
        data = np.zeros_like(rows, dtype=float)
        # Fill α entries
        for n, (r, c) in enumerate(zip(rows, cols)):
            s_name = idx.sites[i][r]
            k_name = idx.kinases[c]
            val = alpha_lookup.get((p, s_name, k_name), np.nan)
            # Use found value or fallback
            data[n] = alpha_scale if np.isnan(val) else max(0.0, float(val))
        alpha_init.append(sparse.csr_matrix((data, (rows, cols)), shape=Wi.shape))

    params = {
        "c_k": c_k,
        "A_i": A_i,
        "B_i": B_i,
        "C_i": C_i,
        "D_i": D_i,
        "r_site": r_site,
        "alpha_list": alpha_init
    }
    return params


# ------------------------------
# Numba-Accelerated Math
# ------------------------------

@njit(cache=True, fastmath=True)
def csr_dot_vec(indptr, indices, data, x, out):
    """
    Standard CSR matrix-vector multiplication: out = A * x.
    Used for computing site phosphorylation rates from kinase activities.
    """
    for i in range(out.shape[0]):
        acc = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            acc += data[jj] * x[indices[jj]]
        out[i] = acc


@njit(cache=True, fastmath=True)
def kinase_activity_numba(a, y, c_k, kb0, kin_P_yidx, kin_site_ki, kin_site_yidx, kin_site_beta):
    """
    Calculates current kinase activity based on state vector `y`.
    Activity = c_k * (Beta_protein * P + sum(Beta_site * Psite))


    """
    # Start with ones (fallback for kinases not modeled in y)
    for ki in range(a.shape[0]):
        a[ki] = 1.0

    # Add protein component (abundance of the kinase protein)
    for ki in range(a.shape[0]):
        yidx = kin_P_yidx[ki]
        if yidx >= 0:
            a[ki] = c_k[ki] * (kb0[ki] * y[yidx])

    # Add site-specific components (activating/inhibiting sites on the kinase)
    for r in range(kin_site_beta.shape[0]):
        ki = kin_site_ki[r]
        yidx = kin_site_yidx[r]
        a[ki] += c_k[ki] * kin_site_beta[r] * y[yidx]


@njit(cache=True, fastmath=True)
def tf_drive_numba(TF_drive, y, tf_mrna_i, tf_tf_i, tf_alpha, block_start):
    """
    Calculates TF drive for mRNA synthesis.
    TF_drive[i] = sum(alpha_edge * P_tf)
    """
    for i in range(TF_drive.shape[0]):
        TF_drive[i] = 0.0
    for e in range(tf_alpha.shape[0]):
        mi = tf_mrna_i[e]  # mRNA index
        ti = tf_tf_i[e]  # TF protein index
        if ti < 0:
            continue
        b = block_start[ti]
        P_tf = y[b + 1]  # P state is at offset +1 in block
        TF_drive[mi] += tf_alpha[e] * P_tf


@njit(cache=True, fastmath=True)
def rhs_numba(y, dy,
              block_start, n_sites, site_start,
              A_i, B_i, C_i, D_i,
              r_site_flat, S_flat, TF_drive):
    """
    Computes the derivatives (RHS) for the ODE system.

    Layout of `y` for protein i (starts at `b`):
      y[b]     = R (mRNA)
      y[b+1]   = P (Protein)
      y[b+2..] = Psite values

    Equations:
      dR/dt = (A_i * TF_drive) - B_i * R
      dP/dt = C_i * R - (D_i + sum(S)) * P + sum(Psite)
      dPsite_j/dt = S_j * P - (1 + kdeg_s) * Psite_j
    """
    # Zero out derivative buffer
    for j in range(dy.shape[0]):
        dy[j] = 0.0

    N = block_start.shape[0]
    for i in range(N):
        b = block_start[i]
        m = int(n_sites[i])
        ss = site_start[i]

        R = y[b]
        P = y[b + 1]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]

        # 1. mRNA Dynamics: dR/dt
        dR = (Ai + TF_drive[i]) - Bi * R

        # Accumulate total phosphorylation rate (sumS) and total phosphosite abundance (sumPs)
        sumS = 0.0
        sumPs = 0.0
        for j in range(m):
            sumS += S_flat[ss + j]
            sumPs += y[b + 2 + j]

        # 2. Protein Dynamics: dP/dt
        # Gains from Translation (Ci*R) and Dephosphorylation (sumPs assuming fast phosphatase)
        # Losses from Degradation (Di*P) and Phosphorylation (sumS*P)
        dP = Ci * R - (Di + sumS) * P + sumPs

        dy[b] = dR
        dy[b + 1] = dP

        # 3. Phosphosite Dynamics: dPsite_j/dt
        for j in range(m):
            Sij = S_flat[ss + j]  # Phosphorylation rate for site j
            Ps = y[b + 2 + j]  # Current Psite abundance
            rsi = r_site_flat[ss + j]  # Relative stability factor
            kdeg_s = (1.0 + rsi) * Di  # Modified degradation for Psite

            dy[b + 2 + j] = Sij * P - (1.0 + kdeg_s) * Ps

    # Clip non-positive values to zero
    np.clip(dy, 0.0, np.inf, out=dy)

# ------------------------------
# ODE System Class
# ------------------------------

class KinaseInput:
    """Encapsulates kinase activity input; currently assumes constant baseline."""

    def __init__(self, kinases: List[str], const_levels: np.ndarray):
        self.kinases = kinases
        self.const = np.asarray(const_levels, float)

    def eval(self, t: float) -> np.ndarray:
        return self.const


class System:
    """
    Global ODE system wrapper.
    Holds parameters, structure, and performs RHS evaluation.
    """

    def __init__(self, idx: Index, W_list: List[sparse.csr_matrix], params: Dict[str, object],
                 kin_input: KinaseInput, kin_beta: pd.DataFrame,
                 tf_alpha_edges: pd.DataFrame = None, tf_beta: pd.DataFrame = None):
        print("[SYSTEM] initializing ODE System structures...")
        self.idx = idx
        self.W_list = [W.tocsr() for W in W_list]
        self.c_k = params["c_k"]
        self.A_i = params["A_i"]
        self.B_i = params["B_i"]
        self.C_i = params["C_i"]
        self.D_i = params["D_i"]
        self.r_site = params["r_site"]
        self.alpha_list = params["alpha_list"]
        self.kin = kin_input
        self._buf_S = [np.zeros(idx.n_sites_per_protein[i], float) for i in range(idx.N)]
        self.block_start, self.n_sites, self.site_start, self.r_site_flat, self.total_sites = build_layout(idx,
                                                                                                           self.r_site)

        self.S_flat = np.zeros(self.total_sites, dtype=np.float64)
        self.dy_buf = np.zeros(idx.state_dim, dtype=np.float64)
        self.TF_drive = np.zeros(idx.N, dtype=np.float64)

        self.kin_beta = kin_beta.copy()

        # --- Precompute Kinase Activity Mappings ---
        self._kb0 = np.zeros(len(idx.kinases), dtype=float)
        self._kb_sites = [dict() for _ in range(len(idx.kinases))]

        K = len(self.idx.kinases)
        self._kin_P_yidx = np.full(K, -1, dtype=np.int32)

        # Map kinase name -> y index for its P state
        for kname, ki in self.idx.k2i.items():
            pi = self.idx.p2i.get(kname, None)
            if pi is None:
                continue
            b = self.block_start[pi]
            self._kin_P_yidx[ki] = b + 1

        # Map (kinase, psite) -> y index for that psite state
        kin_site_rows = []
        for kname, ki in self.idx.k2i.items():
            pi = self.idx.p2i.get(kname, None)
            if pi is None:
                continue
            b = self.block_start[pi]
            site_to_pos = {s: j for j, s in enumerate(self.idx.sites[pi])}
            for ps, bval in self._kb_sites[ki].items():
                j = site_to_pos.get(ps, None)
                if j is None:
                    continue
                kin_site_rows.append((ki, b + 2 + j, float(bval)))

        if kin_site_rows:
            kin_site_rows = np.asarray(kin_site_rows, dtype=np.float64)
            self._kin_site_ki = kin_site_rows[:, 0].astype(np.int32)
            self._kin_site_yidx = kin_site_rows[:, 1].astype(np.int32)
            self._kin_site_beta = kin_site_rows[:, 2].astype(np.float64)
        else:
            self._kin_site_ki = np.zeros(0, dtype=np.int32)
            self._kin_site_yidx = np.zeros(0, dtype=np.int32)
            self._kin_site_beta = np.zeros(0, dtype=np.float64)

        self._kb0 = self._kb0.astype(np.float64)

        # Parse Kinase Beta inputs into map
        k2i = idx.k2i
        for _, r in self.kin_beta.iterrows():
            k = str(r["kinase"]).strip()
            if k not in k2i:
                continue
            ki = k2i[k]
            ps = str(r.get("psite", "")).strip()
            b = float(r["beta"])
            if ps == "":
                self._kb0[ki] += b
            else:
                self._kb_sites[ki][ps] = self._kb_sites[ki].get(ps, 0.0) + b

        # --- TF Maps ---
        self.tf_alpha_edges = tf_alpha_edges if tf_alpha_edges is not None else pd.DataFrame(
            columns=["mrna", "tf", "alpha"])
        self.tf_beta = tf_beta if tf_beta is not None else pd.DataFrame(columns=["tf", "psite", "beta"])

        # Populate TF beta structures
        self._tb0 = {}
        self._tb_sites = {}
        for _, r in self.tf_beta.iterrows():
            tf = str(r["tf"]).strip()
            ps = str(r.get("psite", "")).strip()
            b = float(r["beta"])
            if tf not in self._tb_sites:
                self._tb_sites[tf] = {}
                self._tb0[tf] = 0.0
            if ps == "":
                self._tb0[tf] += b
            else:
                self._tb_sites[tf][ps] = self._tb_sites[tf].get(ps, 0.0) + b

        # Pack TF edges for fast RHS
        tf_rows = []
        for _, rr in self.tf_alpha_edges.iterrows():
            mrna = str(rr["mrna"]).strip()
            tf = str(rr["tf"]).strip()
            if mrna not in self.idx.p2i:
                continue
            mi = self.idx.p2i[mrna]
            ti = self.idx.p2i.get(tf, -1)  # -1 => not modeled (treated as 0 activity)
            a = float(rr["alpha"])
            if a == 0.0:
                continue
            tf_rows.append((mi, ti, a))

        if tf_rows:
            tf_rows = np.asarray(tf_rows, dtype=np.float64)
            self._tf_mrna_i = tf_rows[:, 0].astype(np.int32)
            self._tf_tf_i = tf_rows[:, 1].astype(np.int32)
            self._tf_alpha = tf_rows[:, 2].astype(np.float64)
        else:
            self._tf_mrna_i = np.zeros(0, dtype=np.int32)
            self._tf_tf_i = np.zeros(0, dtype=np.int32)
            self._tf_alpha = np.zeros(0, dtype=np.float64)

        print("[SYSTEM] Ready.")

    def set_c_k(self, c):
        self.c_k = c

    def set_D_i(self, D):
        self.D_i = D

    def set_alpha_from_vals(self, alpha_vals_list: List[np.ndarray]):
        """Update α matrices from flat arrays matching nonzero patterns."""
        new = []
        for i, W in enumerate(self.W_list):
            if W.nnz == 0:
                new.append(W.copy())
                continue
            rows, cols = W.nonzero()
            A = sparse.csr_matrix((alpha_vals_list[i], (rows, cols)), shape=W.shape)
            new.append(A)
        self.alpha_list = new

    def site_rates(self, t: float, y: np.ndarray) -> List[np.ndarray]:
        """Compute phosphorylation rates S for all sites."""
        a_k = np.empty(len(self.idx.kinases), dtype=np.float64)

        # 1. Update Kinase Activities
        kinase_activity_numba(
            a_k, y,
            self.c_k, self._kb0,
            self._kin_P_yidx,
            self._kin_site_ki, self._kin_site_yidx, self._kin_site_beta
        )

        out = self._buf_S
        for i in range(self.idx.N):
            A = self.alpha_list[i]
            if A.nnz == 0:
                out[i].fill(0.0)
                continue
            # 2. Map Activities to Site Rates via Alpha: S = alpha * a_k
            csr_dot_vec(A.indptr, A.indices, A.data, a_k, out[i])

        return out

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Main ODE Right-Hand Side function dy/dt = f(t, y)."""
        # Compute per-protein site rates
        S_list = self.site_rates(t, y)

        # Flatten S into 1D array for Numba
        for i in range(self.idx.N):
            ss = int(self.site_start[i])
            m = int(self.n_sites[i])
            if m > 0:
                self.S_flat[ss:ss + m] = S_list[i]

        np.maximum(self.S_flat, 0.0, out=self.S_flat)
        dy = self.dy_buf

        # Compute TF drive per mRNA: sum(alpha * P_tf)
        if len(self.tf_alpha_edges) > 0:
            tf_drive_numba(self.TF_drive, y, self._tf_mrna_i, self._tf_tf_i, self._tf_alpha, self.block_start)

        # Compute full derivatives
        rhs_numba(y, dy,
                  self.block_start, self.n_sites, self.site_start,
                  self.A_i, self.B_i, self.C_i, self.D_i,
                  self.r_site_flat, self.S_flat, self.TF_drive)

        return dy.copy()

    def y0(self, R0=1.0, P0=1.0, Psite0=0.01) -> np.ndarray:
        """Generates the initial state vector y0."""
        y0 = np.zeros(self.idx.state_dim, float)
        for i in range(self.idx.N):
            sl = self.idx.block(i)
            y0[sl.start + 0] = R0
            y0[sl.start + 1] = P0
            nsi = self.idx.n_sites_per_protein[i]
            if nsi > 0:
                y0[sl.start + 2: sl.start + 2 + nsi] = Psite0
        return y0

class ODEProblem(ElementwiseProblem):
    def __init__(self, cost_fun, n_var, xl, xu, runner=None):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, elementwise_runner=runner)
        self.cost_fun = cost_fun

    def _evaluate(self, x, out, *args, **kwargs):
        loss = self.cost_fun(x)
        out["F"] = loss

# --------------------------------------------

# ------------------------------
# Simulation & Loss Calculation
# ------------------------------

def build_jac_sparsity(idx: Index) -> sparse.csr_matrix:
    """
    Constructs the block-diagonal Jacobian sparsity pattern.
    Used to accelerate the ODE solver.
    """
    rows, cols = [], []
    base = 0
    for m in idx.n_sites_per_protein:
        # local indices in block
        R, P = 0, 1
        # dR/dR
        rows.append(base + R);
        cols.append(base + R)
        # dP/dR, dP/dP
        rows += [base + P, base + P];
        cols += [base + R, base + P]
        # dP/dPsite_j (sum over sites)
        for j in range(m):
            rows.append(base + P);
            cols.append(base + 2 + j)
        # dPsite_j/dP and dPsite_j/dPsite_j
        for j in range(m):
            rows += [base + 2 + j, base + 2 + j]
            cols += [base + P, base + 2 + j]
        base += 2 + m
    n = idx.state_dim
    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def simulate_union(sys: System, jsp: sparse.csr_matrix, times_union: np.ndarray, atol=1e-8, rtol=1e-6, method="BDF"):
    """
    Integrate ODE on the union of protein and RNA time grids.
    """
    t0, t1 = float(times_union.min()), float(times_union.max())
    sol = solve_ivp(sys.rhs, (t0, t1), sys.y0(), t_eval=np.asarray(times_union, float),
                    atol=atol, rtol=rtol, method=method, jac_sparsity=jsp)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y.T  # shape: (T, state_dim)


def psite_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    """Extract Phosphosite Fold Change from state Y."""
    mask = np.isin(t, times_needed)
    t_sel = t[mask]
    rows = []

    # Iterate over proteins
    for i, prot in enumerate(idx.proteins):
        sl = idx.block(i)
        sites = idx.sites[i]
        n_sites = idx.n_sites_per_protein[i]

        if n_sites == 0:
            continue

        # Extract all sites for this protein: Block offset + 2 (skip R, P)
        # shape: (Time, n_sites)
        P_sites = Y[:, sl.start + 2: sl.start + 2 + n_sites]

        # Calculate FC for each site
        for j, s_name in enumerate(sites):
            # Safe normalization (avoid div by zero)
            start_val = max(P_sites[0, j], 1e-12)
            fc_vals = P_sites[:, j] / start_val

            # Store predictions at requested times
            rows.append(pd.DataFrame({
                "time": t_sel,
                "protein": prot,
                "psite": s_name,
                "fc": fc_vals[mask]
            }))

    if not rows:
        return pd.DataFrame(columns=["time", "protein", "psite", "fc"])

    return pd.concat(rows, ignore_index=True)

def protein_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    """Extract Protein Fold Change (Total Protein + Phospho) from state Y."""
    mask = np.isin(t, times_needed)
    t_sel = t[mask]
    rows = []
    # Loop proteins
    for i, prot in enumerate(idx.proteins):
        sl = idx.block(i)
        P = Y[:, sl.start + 1]
        m = idx.n_sites_per_protein[i]
        Ps = Y[:, sl.start + 2: sl.start + 2 + m].sum(axis=1) if m > 0 else 0.0
        total = np.maximum(P + Ps, 1e-12)
        fc = total / total[0]
        rows.append(pd.DataFrame({"time": t_sel, "protein": prot, "fc": fc[mask]}))
    return pd.concat(rows, ignore_index=True)


def rna_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    """Extract RNA Fold Change from state Y."""
    mask = np.isin(t, times_needed)
    t_sel = t[mask]
    rows = []
    for i, prot in enumerate(idx.proteins):
        sl = idx.block(i)
        R = np.maximum(Y[:, sl.start + 0], 1e-12)
        fc = R / R[0]
        rows.append(pd.DataFrame({"time": t_sel, "protein": prot, "fc": fc[mask]}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------
# Optimization Helpers
# ------------------------------

def init_raw_for_stage(params: Dict[str,object], alpha_list: List[sparse.csr_matrix], stage: int):
    """
    Pack parameters into a raw flat vector `theta` for the optimizer.
    Stages:
      1=c_k
      2=alpha
      3=c_k + alpha + D_i (Standard)
      4=ALL (c_k, alpha, D_i, A_i, B_i, C_i, r_site) <--- NEW
    """
    parts: Dict[str, object] = {}
    vecs: List[np.ndarray] = []

    # --- Existing Parameters (Stage 1-3) ---

    # 1. c_k (Kinase Activity)
    if stage in (1, 3, 4):
        rc = inv_softplus(params["c_k"])
        parts["raw_c"] = slice(0, len(rc))
        vecs.append(rc)

    # 2. α (Interaction Weights)
    if stage in (2, 3, 4):
        chunks = []
        sls = []
        start = sum(len(v) for v in vecs)
        for A in alpha_list:
            data = A.data if A.nnz > 0 else np.array([], float)
            raw = inv_softplus(np.maximum(data, 1e-12))
            chunks.append(raw)
            sls.append(slice(start, start+len(raw)))
            start += len(raw)
        ra = np.concatenate(chunks) if len(chunks) > 0 else np.array([], float)
        if ra.size > 0:
            parts["raw_alpha"] = sls
            vecs.append(ra)

    # 3. D_i (Protein Degradation)
    if stage in (3, 4):
        rD = inv_softplus(params["D_i"])
        start = sum(len(v) for v in vecs)
        parts["raw_D"] = slice(start, start+len(rD))
        vecs.append(rD)

    # --- NEW: Full Estimation (Stage 4) ---

    if stage == 4:
        # 4. A_i (Transcription Baselines)
        rA = inv_softplus(params["A_i"])
        start = sum(len(v) for v in vecs)
        parts["raw_A"] = slice(start, start+len(rA))
        vecs.append(rA)

        # 5. B_i (RNA Degradation)
        rB = inv_softplus(params["B_i"])
        start = sum(len(v) for v in vecs)
        parts["raw_B"] = slice(start, start+len(rB))
        vecs.append(rB)

        # 6. C_i (Translation Rates)
        rC = inv_softplus(params["C_i"])
        start = sum(len(v) for v in vecs)
        parts["raw_C"] = slice(start, start+len(rC))
        vecs.append(rC)

        # 7. r_site (Phosphosite Stability)
        # Flatten list of arrays into one vector
        r_site_flat = np.concatenate(params["r_site"])
        r_site_raw = inv_softplus(r_site_flat)
        start = sum(len(v) for v in vecs)
        parts["raw_r_site"] = slice(start, start+len(r_site_raw))
        vecs.append(r_site_raw)

    theta0 = np.concatenate(vecs) if len(vecs) > 0 else np.array([], float)
    return theta0, parts


def assign_theta(theta: np.ndarray, parts: Dict[str, object], sys: System, alpha_init: List[sparse.csr_matrix]):
    """Unpack raw theta vector back into the System object."""
    # c_k
    if "raw_c" in parts:
        sl = parts["raw_c"]
        sys.set_c_k(softplus(theta[sl]))

    # α (alpha)
    if "raw_alpha" in parts:
        vals = []
        for sl in parts["raw_alpha"]:
            raw = theta[sl] if sl.start != sl.stop else np.array([], float)
            vals.append(softplus(raw) if raw.size > 0 else np.array([], float))
        sys.set_alpha_from_vals(vals)
    else:
        # If not optimizing alpha, ensure we keep the initial values
        sys.set_alpha_from_vals([A.data if A.nnz > 0 else np.array([], float) for A in alpha_init])

    # D_i
    if "raw_D" in parts:
        sl = parts["raw_D"]
        sys.set_D_i(softplus(theta[sl]))

    # --- NEW Unpacking Logic ---

    # A_i (Transcription)
    if "raw_A" in parts:
        sl = parts["raw_A"]
        sys.A_i = softplus(theta[sl])  # Direct assignment to system array

    # B_i (RNA Degradation)
    if "raw_B" in parts:
        sl = parts["raw_B"]
        sys.B_i = softplus(theta[sl])

    # C_i (Translation)
    if "raw_C" in parts:
        sl = parts["raw_C"]
        sys.C_i = softplus(theta[sl])

    # r_site (Site stability)
    if "raw_r_site" in parts:
        sl = parts["raw_r_site"]
        full_r = softplus(theta[sl])

        # Re-distribute flat array back to list of arrays per protein
        new_r_site = []
        curr = 0
        for i in range(sys.idx.N):
            n = sys.idx.n_sites_per_protein[i]
            new_r_site.append(full_r[curr: curr + n])
            curr += n

        sys.r_site = new_r_site
        # IMPORTANT: Must re-build layout because r_site_flat in system needs update
        # However, for speed, we can just update the flat array directly:
        sys.r_site_flat[:] = full_r


def build_weights(times: np.ndarray, early_focus: float) -> Dict[float, float]:
    """Generate time-dependent weights to emphasize early dynamics."""
    tmin, tmax = float(times.min()), float(times.max())
    span = (tmax - tmin) if tmax > tmin else 1.0
    return {float(t): 1.0 + early_focus * (tmax - float(t)) / span for t in times}


@njit(cache=True, fastmath=True)
def compute_weighted_squared_error(w, pred, obs):
    """JIT-compiled Weighted Sum of Squared Errors."""
    total = 0.0
    for i in range(w.shape[0]):
        diff = obs[i] - pred[i]
        total += w[i] * diff * diff
    return total


@njit(cache=True, fastmath=True)
def compute_c_prior(c_k, c_k_init):
    """Compute prior regularization penalty for c_k (L2 from init)."""
    reg = 0.0
    for i in range(c_k.shape[0]):
        diff = c_k[i] - c_k_init[i]
        reg += diff * diff
    return reg


def dual_loss(theta: np.ndarray, parts: Dict[str, object], sys: System, idx: Index,
              alpha_init: List[sparse.csr_matrix], c_k_init: np.ndarray,
              df_prot_obs: pd.DataFrame, df_rna_obs: pd.DataFrame, df_psite_obs: pd.DataFrame,
              lam: Dict[str, float], atol: float, rtol: float, jsp: sparse.csr_matrix):
    """
    Global objective function:
    Loss = Prot_LSQ + lambda_rna * RNA_LSQ + lambda_psite * PSITE_LSQ + Regularization terms
    """
    # 1. Update parameters
    assign_theta(theta, parts, sys, alpha_init)

    # 2. Simulate
    times_union = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA, TIME_POINTS]))
    t, Y = simulate_union(sys, jsp, times_union, atol=atol, rtol=rtol)

    # 3. Extract Predictions
    dfp = protein_FC(idx, t, Y, TIME_POINTS).rename(columns={"fc": "pred_fc"})
    dfr = rna_FC(idx, t, Y, TIME_POINTS_RNA).rename(columns={"fc": "pred_fc"})
    dfs = psite_FC(idx, t, Y, TIME_POINTS).rename(columns={"fc": "pred_fc"})

    # 4. Compare with Observations
    mp = df_prot_obs.merge(dfp, on=["protein", "time"], how="inner")
    mr = df_rna_obs.merge(dfr, on=["protein", "time"], how="inner")
    ms = df_psite_obs.merge(dfs, on=["protein", "psite", "time"], how="inner")

    # 5. Compute Data Loss
    prot_loss = compute_weighted_squared_error(
        mp["w"].values.astype(np.float64),
        mp["pred_fc"].values.astype(np.float64),
        mp["fc"].values.astype(np.float64)
    )
    rna_loss = compute_weighted_squared_error(
        mr["w"].values.astype(np.float64),
        mr["pred_fc"].values.astype(np.float64),
        mr["fc"].values.astype(np.float64)
    )
    psite_loss = 0.0
    if not ms.empty:
        psite_loss = compute_weighted_squared_error(
            ms["w"].values.astype(np.float64), ms["pred_fc"].values.astype(np.float64),
            ms["fc"].values.astype(np.float64)
        )

    # 6. Regularization
    reg_alpha_l1 = 0.0
    reg_alpha_prior = 0.0
    if "raw_alpha" in parts:
        cur_alpha = np.concatenate(
            [A.data.astype(np.float64, copy=False) for A in sys.alpha_list if A.nnz > 0],
            axis=0
        ) if any(A.nnz > 0 for A in sys.alpha_list) else np.empty(0, dtype=np.float64)

        init_alpha = np.concatenate(
            [A.data.astype(np.float64, copy=False) for A in alpha_init if A.nnz > 0],
            axis=0
        ) if any(A.nnz > 0 for A in alpha_init) else np.empty(0, dtype=np.float64)

        reg_alpha_l1 = float(cur_alpha.sum()) if cur_alpha.size else 0.0

        if cur_alpha.shape == init_alpha.shape and cur_alpha.size:
            diff = cur_alpha - init_alpha
            reg_alpha_prior = float(np.dot(diff, diff))
        else:
            reg_alpha_prior = 0.0

    reg_c_prior = 0.0
    if "raw_c" in parts:
        reg_c_prior = compute_c_prior(
            sys.c_k.astype(np.float64),
            c_k_init.astype(np.float64)
        )

    total = (prot_loss + lam.get("lambda_rna", 1.0) * rna_loss \
            + lam.get("lambda_psite", 1.0) * psite_loss \
            + lam.get("l1_alpha", 0.0) * reg_alpha_l1 \
            + lam.get("prior_alpha", 0.0) * reg_alpha_prior \
            + lam.get("prior_c", 0.0) * reg_c_prior)

    details = {
        "prot_loss": float(prot_loss),
        "rna_loss": float(rna_loss),
        "psite_loss": float(psite_loss),
        "reg_alpha_l1": float(reg_alpha_l1),
        "reg_alpha_prior": float(reg_alpha_prior),
        "reg_c_prior": float(reg_c_prior),
        "total": float(total)
    }
    return total, details


# ------------------------------
# Reporting & Plotting
# ------------------------------

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))


def _fit_metrics(df: pd.DataFrame, group_cols=("protein",), obs_col="fc", pred_col="pred_fc"):
    """Calculate R2, RMSE, MAE per group."""
    rows = []
    dfp = df.dropna(subset=[obs_col, pred_col])
    for g, sub in dfp.groupby(list(group_cols)):
        y = sub[obs_col].to_numpy(float)
        yhat = sub[pred_col].to_numpy(float)
        resid = y - yhat
        rss = float(np.sum(resid ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - (rss / tss) if (tss and not math.isnan(tss) and tss > 0) else float("nan")
        rmse = float(np.sqrt(np.mean(resid ** 2))) if len(y) else float("nan")
        mae = float(np.mean(np.abs(resid))) if len(y) else float("nan")
        m = {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y))}
        if isinstance(g, tuple):
            for k, v in zip(group_cols, g): m[k] = v
        else:
            m[group_cols[0]] = g
        rows.append(m)
    return pd.DataFrame(rows)


def _plot_series(ax, t_all, y_obs, y_pred, title, xlabel="time (min)", ylabel="FC"):
    if y_pred.size:
        ax.plot(t_all, y_pred, lw=1.0)
        ax.scatter(t_all, y_pred, s=18)
    if y_obs.size:
        ax.scatter(t_all, y_obs, s=24)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_residuals(ax, t_pair, resid, title):
    ax.axhline(0.0, lw=1)
    if resid.size:
        ax.scatter(t_pair, resid, s=18)
    ax.set_title(title)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("residual (obs - pred)")


def _scatter_global(ax, y_obs, y_pred, title):
    if y_obs.size and y_pred.size:
        ax.scatter(y_obs, y_pred, s=10, alpha=0.8)
        lo = float(min(np.min(y_obs), np.min(y_pred)))
        hi = float(max(np.max(y_obs), np.max(y_pred)))
        ax.plot([lo, hi], [lo, hi], lw=1)
    ax.set_title(title)
    ax.set_xlabel("Observed FC")
    ax.set_ylabel("Predicted FC")


def _scatter_pvo(ax, y_obs, y_pred, title):
    if y_obs.size and y_pred.size:
        ax.scatter(y_obs, y_pred, s=20, alpha=0.9)
        lo = float(min(np.min(y_obs), np.min(y_pred)))
        hi = float(max(np.max(y_obs), np.max(y_pred)))
        ax.plot([lo, hi], [lo, hi], lw=1)
    ax.set_title(title)
    ax.set_xlabel("Obs FC")
    ax.set_ylabel("Pred FC")


def plot_fit_report(
        df_prot_obs: pd.DataFrame,
        df_prot_pred: pd.DataFrame,
        df_rna_obs: pd.DataFrame,
        df_rna_pred: pd.DataFrame,
        df_psite_obs: pd.DataFrame,
        df_psite_pred: pd.DataFrame,
        out_dir: str,
        pdf_name: str = "fit_report.pdf",
        metrics_csv: str = "fit_metrics.csv",
        per_page: int = 12
):
    """Generates a multipage PDF report of the fit."""
    print(f"[REPORT] Generating fit report PDF in {out_dir} ...")
    os.makedirs(out_dir, exist_ok=True)

    # Outer merge so no timepoint is dropped
    p = (df_prot_obs[["protein", "time", "fc"]]
         .merge(df_prot_pred[["protein", "time", "pred_fc"]],
                on=["protein", "time"], how="outer")
         .sort_values(["protein", "time"]))
    r = (df_rna_obs[["protein", "time", "fc"]]
         .merge(df_rna_pred[["protein", "time", "pred_fc"]],
                on=["protein", "time"], how="outer")
         .sort_values(["protein", "time"]))
    psite = (df_psite_obs[["protein", "time", "psite", "fc"]]
         .merge(df_psite_pred[["protein", "time", "psite", "pred_fc"]],
                on=["protein", "time", "psite"], how="outer")
         .sort_values(["protein", "time", "psite"]))

    m_prot = _fit_metrics(p, group_cols=("protein",), obs_col="fc", pred_col="pred_fc")
    m_rna = _fit_metrics(r, group_cols=("protein",), obs_col="fc", pred_col="pred_fc")
    m_psite = _fit_metrics(psite, group_cols=("protein", "psite"), obs_col="fc", pred_col="pred_fc")

    def _global_pairs(df):
        dfp = df.dropna(subset=["fc", "pred_fc"])
        if dfp.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}, np.array([]), np.array([])
        y = dfp["fc"].to_numpy(float)
        yhat = dfp["pred_fc"].to_numpy(float)
        resid = y - yhat
        rss = float(np.sum(resid ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - (rss / tss) if (tss and not math.isnan(tss) and tss > 0) else float("nan")
        rmse = float(np.sqrt(np.mean(resid ** 2))) if len(y) else float("nan")
        mae = float(np.mean(np.abs(resid))) if len(y) else float("nan")
        return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y))}, y, yhat

    g_prot, gp_obs, gp_pred = _global_pairs(p)
    g_rna, gr_obs, gr_pred = _global_pairs(r)
    g_psite, gsp_obs, gsp_pred = _global_pairs(psite)

    # Save metrics table
    m_prot["modality"] = "protein"
    m_rna["modality"] = "rna"
    m_psite["modality"] = "psite"
    pd.concat([m_prot, m_rna, m_psite], ignore_index=True).to_csv(
        os.path.join(out_dir, metrics_csv), index=False
    )

    # PDF generation
    pdf_path = os.path.join(out_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        # Overview page
        fig, ax = plt.subplots(1, 3, figsize=(10, 4.2))
        _scatter_global(ax[0], gp_obs, gp_pred,
                        f"Protein: Pred vs Obs (R²={g_prot['r2']:.3f}, RMSE={g_prot['rmse']:.3g}, n={g_prot['n']})")
        _scatter_global(ax[1], gr_obs, gr_pred,
                        f"RNA: Pred vs Obs (R²={g_rna['r2']:.3f}, RMSE={g_rna['rmse']:.3g}, n={g_rna['n']})")
        _scatter_global(ax[2], gsp_obs, gsp_pred,
                        f"PSite: Pred vs Obs (R²={g_psite['r2']:.3f}, RMSE={g_psite['rmse']:.3g}, n={g_psite['n']})")
        fig.suptitle("Goodness of Fit — Global Overview (paired points)", y=1.02)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Helper to plot batches
        def plot_batch(entity_list, data_df, title_prefix):
            for start in range(0, len(entity_list), per_page):
                batch = entity_list[start:start + per_page]
                n = len(batch)
                ncols = 3
                nrows = math.ceil(n * 3 / ncols)
                fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(6, 2.4 * nrows)))
                axes = axes.ravel() if nrows * ncols > 1 else [axes]
                slot = 0
                for prot in batch:
                    sub = data_df[data_df["protein"] == prot]
                    if sub.empty: continue
                    t_all = sub["time"].to_numpy(float)
                    y_obs = sub["fc"].to_numpy(float)
                    y_pred = sub["pred_fc"].to_numpy(float)

                    paired = sub.dropna(subset=["fc", "pred_fc"])
                    t_pair = paired["time"].to_numpy(float)
                    resid = paired["fc"].to_numpy(float) - paired["pred_fc"].to_numpy(float)

                    _plot_series(axes[slot], t_all, y_obs, y_pred, f"{prot} | {title_prefix}");
                    slot += 1
                    _plot_residuals(axes[slot], t_pair, resid, "Residuals")
                    slot += 1
                    _scatter_pvo(axes[slot],
                                 paired["fc"].to_numpy(float),
                                 paired["pred_fc"].to_numpy(float),
                                 "Pred vs Obs")
                    slot += 1

                while slot < len(axes):
                    axes[slot].axis("off")
                    slot += 1
                fig.suptitle(f"{title_prefix} fits", y=1.02)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        plot_batch(p["protein"].dropna().unique().tolist(), p, "Protein FC")
        plot_batch(r["protein"].dropna().unique().tolist(), r, "RNA FC")
        plot_batch(psite["protein"].dropna().unique().tolist(), psite, "PSite FC")

    return pdf_path, os.path.join(out_dir, metrics_csv)


# --- ADD THIS CLASS BEFORE main() ---
class GlobalLoss:
    """
    Pickleable objective function wrapper.
    Stores all necessary data to compute loss so it can be passed to workers.
    """

    def __init__(self, parts, sys, idx, alpha_init, c_k_init,
                 df_prot_obs, df_rna_obs, df_psite_obs,
                 lam, atol, rtol, jsp):
        self.parts = parts
        self.sys = sys
        self.idx = idx
        self.alpha_init = alpha_init
        self.c_k_init = c_k_init
        self.df_prot_obs = df_prot_obs
        self.df_rna_obs = df_rna_obs
        self.df_psite_obs = df_psite_obs
        self.lam = lam
        self.atol = atol
        self.rtol = rtol
        self.jsp = jsp

        # We can try to store details, though they won't sync back
        # from workers to main process easily.
        self.last_details = None

    def __call__(self, theta):
        val, details = dual_loss(
            theta,
            self.parts, self.sys, self.idx,
            self.alpha_init, self.c_k_init,
            self.df_prot_obs, self.df_rna_obs, self.df_psite_obs,
            self.lam, self.atol, self.rtol, self.jsp
        )
        self.last_details = details
        return val

# ------------------------------
# Main Execution Flow
# ------------------------------

def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser(description="Global ODE dual-fit (FC) with effects from 'Alpha Values'/'Beta Values'.")
    ap.add_argument("--interaction", required=True, help="Path to Interaction CSV (Protein, Psite, Kinase)")
    ap.add_argument("--kinopt", required=True, help="Path to Kinase Optimization Excel (Alpha/Beta Values + Estimated)")
    ap.add_argument("--tfopt", required=True, help="Path to TF Optimization Excel (Alpha Values + Estimated)")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=3,
                    help="Optimization stage (1=c_k, 2=alpha, 3=std, 4=ALL)")
    ap.add_argument("--early-focus", type=float, default=1.0, help="Weight factor for early time points")
    ap.add_argument("--l1-alpha", type=float, default=2e-3, help="L1 regularization coefficient for Alpha")
    ap.add_argument("--lambda-prior", type=float, default=1e-2, help="Prior regularization coefficient for Alpha")
    ap.add_argument("--lambda-c", type=float, default=1e-2, help="Prior regularization coefficient for c_k")
    ap.add_argument("--lambda-rna", type=float, default=1.0, help="Weighting for RNA loss term")
    ap.add_argument("--lambda-psite", type=float, default=1.0, help="Weighting for Psite loss")
    ap.add_argument("--maxiter", type=int, default=300, help="Maximum optimizer iterations")
    ap.add_argument("--atol", type=float, default=1e-8, help="ODE Absolute Tolerance")
    ap.add_argument("--rtol", type=float, default=1e-6, help="ODE Relative Tolerance")
    ap.add_argument("--output-dir", default="./out_dual_fc", help="Output directory")
    ap.add_argument("--log-every", type=int, default=25,
                    help="Print objective every N function evaluations (0 = silent)")
    args = ap.parse_args()

    # Timer start
    T0 = time.time()
    _ensure_dir(args.output_dir)

    print("=" * 70)
    print("GLOBAL ODE DUAL-FIT OPTIMIZATION")
    print("=" * 70)

    # 1. Load inputs
    print("\n--- [STEP 1/6] LOADING DATA ---")
    interactions = load_interactions(args.interaction)
    kin_alpha, kin_beta = load_kinopt_effects(args.kinopt)
    tf_alpha, tf_beta = load_tfopt_effects(args.tfopt)

    # 2. Load FC targets
    print("\n--- [STEP 2/6] LOADING TARGETS ---")
    df_prot_obs = load_estimated_protein_FC(args.kinopt)
    df_rna_obs = load_estimated_rna_FC(args.tfopt)
    df_psite_obs = load_estimated_psite_FC(args.kinopt)

    # 3. Build index and W
    print("\n--- [STEP 3/6] BUILDING TOPOLOGY ---")
    idx = Index(interactions)
    print(f"     -> Proteins: {idx.N}, Kinases: {len(idx.kinases)}")
    print(f"     -> State Dimension: {idx.state_dim}")
    W_list = build_W(interactions, idx)
    JSP = build_jac_sparsity(idx)

    # Filter observations to modeled proteins
    df_prot_obs = df_prot_obs[df_prot_obs["protein"].isin(idx.proteins)].reset_index(drop=True)
    df_rna_obs = df_rna_obs[df_rna_obs["protein"].isin(idx.proteins)].reset_index(drop=True)
    df_psite_obs = df_psite_obs[df_psite_obs["protein"].isin(idx.proteins)].reset_index(drop=True)

    # 4. Initialize parameters from effects
    print("\n--- [STEP 4/6] PARAMETER INITIALIZATION ---")
    params = init_params_from_effects(idx, W_list, kin_alpha, kin_beta, tf_alpha, alpha_scale=0.2)

    # Build weights (early emphasis on protein series)

    wmap_protein = build_weights(TIME_POINTS, early_focus=args.early_focus)
    wmap_rna = build_weights(TIME_POINTS_RNA, early_focus=args.early_focus)
    df_prot_obs["w"] = df_prot_obs["time"].map(wmap_protein).astype(float)
    df_rna_obs["w"] = df_rna_obs["time"].map(wmap_rna).astype(float)
    df_psite_obs["w"] = df_psite_obs["time"].map(wmap_protein).astype(float)

    # 5. Construct ODE system
    print("\n--- [STEP 5/6] SYSTEM CONSTRUCTION ---")
    kin_input = KinaseInput(idx.kinases, params["c_k"].copy())  # c_k-scaled baseline
    sys = System(idx, W_list, params, kin_input, kin_beta, tf_alpha, tf_beta)

    # Prepare optimizer variables
    theta0, parts = init_raw_for_stage(params, params["alpha_list"], stage=args.stage)
    c_k_init = sys.c_k.copy()
    alpha_init = [A.copy() for A in params["alpha_list"]]
    lam = {
        "l1_alpha": args.l1_alpha,
        "prior_alpha": args.lambda_prior,
        "prior_c": args.lambda_c,
        "lambda_rna": args.lambda_rna,
        "lambda_psite": args.lambda_psite
    }

    # 6. Optimization
    print("\n--- [STEP 6/6] OPTIMIZATION LOOP ---")
    print(f"     -> Max Iterations: {args.maxiter}")
    print(f"     -> Parameter count: {len(theta0)}")

    print("[STEP 6/6] Starting optimization ...")
    progress = {"iter": 0}
    last_details = None

    # 1. Instantiate the pickleable cost function
    cost_fn = GlobalLoss(
        parts, sys, idx, alpha_init, c_k_init,
        df_prot_obs, df_rna_obs, df_psite_obs,
        lam, args.atol, args.rtol, JSP
    )

    # 2. New Callback Function (Handles Iteration Logging)
    def on_iteration(xk, state=None):
        """Callback executed by minimize() after every iteration."""
        progress["iter"] += 1

        # Check if we should log this iteration
        if args.log_every and (progress["iter"] % args.log_every == 0):
            d = last_details
            if d:
                print(
                    f"[iter {progress['iter']:04d}] "
                    f"total={d['total']:.6f} | "
                    f"prot={d['prot_loss']:.4f} | "
                    f"rna={d['rna_loss']:.4f} | "
                    f"psite={d.get('psite_loss', 0.0):.4f} | "
                    f"reg={d['reg_alpha_l1'] + d['reg_alpha_prior'] + d['reg_c_prior']:.4f}",
                    flush=True
                )

    # ---- Bounds configuration ----
    def _sp_inv(x):
        # safe inverse softplus for bounds
        x = np.asarray(x, float)
        return np.log(np.expm1(np.maximum(x, 1e-12)))

    # 1. Define Biological Limits (Actual parameter scale)
    ALPHA_MIN, ALPHA_MAX = 0, 2.0  # Interaction strength
    C_K_MIN, C_K_MAX = 0, 20.0  # Kinase activity multipliers
    D_MIN, D_MAX = 0, 20.0  # Protein degradation

    A_MIN, A_MAX = 0, 20.0  # Transcription baseline
    B_MIN, B_MAX = 0, 20.0  # RNA degradation
    C_TRANS_MIN, C_TRANS_MAX = 0, 20.0  # Translation rate (RNA->Prot)
    R_SITE_MIN, R_SITE_MAX = 0, 20.0  # Phosphosite relative stability

    # Initialize bounds list with (-inf, inf)
    bounds = [(-np.inf, np.inf)] * len(theta0)

    # 2. Apply Bounds based on 'parts' keys

    # c_k (Kinase Activity)
    if "raw_c" in parts:
        sl = parts["raw_c"]
        lo = _sp_inv(np.full(sl.stop - sl.start, C_K_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, C_K_MAX))
        for j in range(sl.start, sl.stop):
            bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # α (Alpha / Interactions)
    if "raw_alpha" in parts:
        for sl in parts["raw_alpha"]:
            if sl.start == sl.stop: continue
            lo = _sp_inv(np.full(sl.stop - sl.start, max(ALPHA_MIN, 1e-12)))
            hi = _sp_inv(np.full(sl.stop - sl.start, ALPHA_MAX))
            for j in range(sl.start, sl.stop):
                bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # D_i (Protein Degradation)
    if "raw_D" in parts:
        sl = parts["raw_D"]
        lo = _sp_inv(np.full(sl.stop - sl.start, D_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, D_MAX))
        for j in range(sl.start, sl.stop):
            bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # A_i (Transcription)
    if "raw_A" in parts:
        sl = parts["raw_A"]
        lo = _sp_inv(np.full(sl.stop - sl.start, A_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, A_MAX))
        for j in range(sl.start, sl.stop):
            bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # B_i (RNA Degradation)
    if "raw_B" in parts:
        sl = parts["raw_B"]
        lo = _sp_inv(np.full(sl.stop - sl.start, B_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, B_MAX))
        for j in range(sl.start, sl.stop):
            bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # C_i (Translation)
    if "raw_C" in parts:
        sl = parts["raw_C"]
        lo = _sp_inv(np.full(sl.stop - sl.start, C_TRANS_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, C_TRANS_MAX))
        for j in range(sl.start, sl.stop):
            bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # r_site (Site Stability)
    if "raw_r_site" in parts:
        sl = parts["raw_r_site"]
        lo = _sp_inv(np.full(sl.stop - sl.start, R_SITE_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, R_SITE_MAX))
        for j in range(sl.start, sl.stop):
            bounds[j] = (float(lo[j - sl.start]), float(hi[j - sl.start]))

    # JIT Warm-up
    print("     -> Compiling and Warming up ODE solver...")
    _ = sys.rhs(float(TIME_POINTS[0]), sys.y0())

    # Run optimizer
    # if theta0.size > 0:
    #     res = minimize(
    #         fun,
    #         theta0,
    #         method="trust-constr",
    #         bounds=bounds,
    #         callback=on_iteration,
    #         options={"maxiter": args.maxiter}
    #     )
    #     if last_details is not None:
    #         print(
    #             f"\n[FINAL @ {progress['iter']} iterations]\n"
    #             f"  Total Loss : {last_details['total']:.6f}\n"
    #             f"  Prot LSQ   : {last_details['prot_loss']:.6f}\n"
    #             f"  RNA LSQ    : {last_details['rna_loss']:.6f}\n"
    #             f"  PSite LSQ  : {last_details['psite_loss']:.6f}\n"
    #             f"  Reg Terms  : {last_details['reg_alpha_l1'] + last_details['reg_alpha_prior'] + last_details['reg_c_prior']:.6f}"
    #         )
    #     if not res.success:
    #         print(f"[WARN] Optimizer: {res.message}")
    #     theta_opt = res.x
    # else:
    #     theta_opt = theta0

    # --- Differential Evolution ---

    if theta0.size > 0:
        # 1. Setup Bounds (clamping infinite bounds for DE)
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])

        # 2. Initialize Parallelization
        # Use N-1 cores to keep system responsive
        n_procs = max(1, mp.cpu_count() - 1)
        print(f"     -> Initializing StarmapParallelization with {n_procs} cores...")

        pool = mp.Pool(n_procs)
        runner = StarmapParallelization(pool.starmap)

        # 3. Instantiate Problem with Runner
        problem = ODEProblem(cost_fn, n_var=len(theta0), xl=xl, xu=xu, runner=runner)

        # 4. Define Algorithm
        algorithm = DE(
            pop_size=40,  # Increased pop_size since we have parallel power
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.7,
            dither="vector",
            jitter=False,
            verbose=True,
        )

        # 6. Run Optimization
        print(f"     -> Starting Pymoo DE for {args.maxiter} generations...")

        try:
            res = pymoo_minimize(
                problem,
                algorithm,
                get_termination("n_gen", args.maxiter),
                seed=1,
                verbose=True
            )
        finally:
            # Ensure the pool is closed even if optimization crashes
            pool.close()
            pool.join()

        theta_opt = res.X
        if last_details is not None:
            print(f"\n[FINAL] Best Loss: {res.F[0]:.6f}")

    else:
        theta_opt = theta0
    # --------------------------------------------------------

    # Final objective breakdown
    f_opt, comps = dual_loss(theta_opt, parts, sys, idx, alpha_init, c_k_init,
                             df_prot_obs, df_rna_obs, df_psite_obs, lam, args.atol, args.rtol, JSP)

    # Final simulation for outputs
    times_union = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    t, Y = simulate_union(sys, JSP, times_union, atol=args.atol, rtol=args.rtol)

    # Assemble predictions
    df_out_prot = protein_FC(idx, t, Y, TIME_POINTS)
    df_out_rna = rna_FC(idx, t, Y, TIME_POINTS_RNA)
    df_out_psite = psite_FC(idx, t, Y, TIME_POINTS)

    # Save artifacts
    out_prot_path = os.path.join(args.output_dir, "predicted_protein_fc.csv")
    out_rna_path = os.path.join(args.output_dir, "predicted_rna_fc.csv")
    out_psite_path = os.path.join(args.output_dir, "predicted_psite_fc.csv")

    fit_params_path = os.path.join(args.output_dir, "fitted_params.json")
    fit_summary_path = os.path.join(args.output_dir, "fit_summary.json")

    df_out_prot.to_csv(out_prot_path, index=False)
    df_out_rna.to_csv(out_rna_path, index=False)
    df_out_psite.to_csv(out_psite_path, index=False)

    # Generate Report
    dfp_obs = df_prot_obs.copy()
    dfp_pred = df_out_prot.rename(columns={"fc": "pred_fc"})
    dfr_obs = df_rna_obs.copy()
    dfr_pred = df_out_rna.rename(columns={"fc": "pred_fc"})
    dps_obs = df_psite_obs.copy()
    dps_pred = df_out_psite.rename(columns={"fc": "pred_fc"})

    pdf_path, metrics_path = plot_fit_report(
        df_prot_obs=dfp_obs,
        df_prot_pred=dfp_pred,
        df_rna_obs=dfr_obs,
        df_rna_pred=dfr_pred,
        df_psite_obs=dps_obs,
        df_psite_pred=dps_pred,
        out_dir=args.output_dir,
        pdf_name="fit_report.pdf",
        metrics_csv="fit_metrics.csv",
        per_page=12
    )
    print(f"\n[OUTPUT] Saved Report: {pdf_path}")

    # Save fitted parameters
    fitted = {
        "proteins": idx.proteins,
        "kinases": idx.kinases,
        "c_k": sys.c_k.tolist(),
        "D_i": sys.D_i.tolist(),
        "alpha_nonzeros": [
            {"protein": idx.proteins[i],
             "rows": sys.alpha_list[i].nonzero()[0].tolist(),
             "cols": sys.alpha_list[i].nonzero()[1].tolist(),
             "data": sys.alpha_list[i].data.tolist()
             } for i in range(idx.N)
        ],
        "stage": args.stage
    }
    with open(fit_params_path, "w") as f:
        json.dump(fitted, f, indent=2)

    # Save summary
    summary = {
        "objective": float(f_opt),
        "components": comps,
        "stage": args.stage,
        "hyperparams": {
            "early_focus": args.early_focus,
            "l1_alpha": args.l1_alpha,
            "lambda_prior": args.lambda_prior,
            "lambda_c": args.lambda_c,
            "lambda_rna": args.lambda_rna,
            "maxiter": args.maxiter,
            "atol": args.atol,
            "rtol": args.rtol
        },
        "timing_sec": float(time.time() - T0)
    }
    with open(fit_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 50)
    print(f"DONE. Duration: {summary['timing_sec']:.2f} sec")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    # Use 'fork' for multiprocessing context where supported (faster for numpy)
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()