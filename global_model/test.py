#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global ODE dual-fit (FC targets) with effects from 'Alpha Values' and 'Beta Values'.

Targets are FC (not log2FC):
  Protein FC_total(t) = (P(t) + Σ Psite(t)) / (P(0) + Σ Psite(0))
  RNA FC_RNA(t)       = R(t) / R(0)

Effects inputs:
  kinopt::Alpha Values  -> (Gene, Psite, Kinase, Alpha) initialize α_{i,s,k}
  kinopt::Beta Values   -> (Kinase, Psite, Beta)         initialize c_k (empty Psite row = general kinase effect)
  tfopt::Alpha Values   -> (mRNA, TF, Value)             initialize A_i by mRNA/protein name
  tfopt::Beta Values    -> (TF, Psite, Value)            parsed, reserved for future use

Time grids (fixed):
  Protein (MS): 14 pts  [0, 0.5, 0.75, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480, 960]
  RNA:           9 pts  [4, 8, 15, 30, 60, 120, 240, 480, 960]
"""

# ------------------------------
# Imports
# ------------------------------
import math
import argparse                             # CLI parsing
import json                                  # JSON I/O for fitted params and summaries
import os                                    # Filesystem utilities
import re                                    # Regex for time column parsing
import time                                  # Timings
from typing import Dict, List, Optional  # Type hints

import numpy as np                           # Numerical computing
import pandas as pd                          # Data handling
from scipy.integrate import solve_ivp        # ODE solver
from scipy import sparse                     # Sparse matrices for α and maps
from scipy.optimize import minimize          # Optimizer for parameter estimation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
except Exception:  # Fallback if tqdm not available
    def tqdm(iterable=None, total=None, desc=None, ncols=None):
        return iterable if iterable is not None else range(total or 0)

# Optional: numba for JIT
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

# Multiprocessing
import multiprocessing as mp

# ------------------------------
# Fixed time grids (minutes)
# ------------------------------
TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0], dtype=float)
TIME_POINTS_RNA = np.array([4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0], dtype=float)

# ------------------------------
# Utility helpers
# ------------------------------

def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case lowercase for robust parsing."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _ensure_dir(p: str):
    """Create an output directory if it does not exist."""
    os.makedirs(p, exist_ok=True)

def softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus to enforce positivity of parameters."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def inv_softplus(y: np.ndarray) -> np.ndarray:
    """Inverse transform for initializing raw parameters from positive values."""
    return np.log(np.expm1(np.maximum(y, 1e-12)))

def _find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    """Find the first column in df that matches any candidate name."""
    for c in cands:
        if c in df.columns:
            return c
    return None

def _time_cols(df: pd.DataFrame) -> List[str]:
    """Pick columns whose names contain a single numeric token, sorted by numeric value."""
    times = []
    for col in df.columns:
        m = re.findall(r"[-+]?\d*\.?\d+", str(col))
        if len(m) == 1:
            try:
                v = float(m[0])
                times.append((col, v))
            except ValueError:
                pass
    times.sort(key=lambda x: x[1])
    return [c for c, _ in times]

# ------------------------------
# Load interaction network
# ------------------------------

def load_interactions(path: str) -> pd.DataFrame:
    """Load Protein–Psite–Kinase triples from CSV."""
    df = pd.read_csv(path)
    df = _normcols(df)
    pcol = _find_col(df, ["protein","gene","geneid","prot"])
    scol = _find_col(df, ["psite","site","phosphosite","residue"])
    kcol = _find_col(df, ["kinase","kinases","k","enzyme"])
    if not (pcol and scol and kcol):
        raise ValueError(f"Interaction file must have Protein, Psite, Kinase. Found: {df.columns.tolist()}")
    out = pd.DataFrame({
        "protein": df[pcol].astype(str).str.strip(),
        "psite": df[scol].astype(str).str.strip(),
        "kinase": df[kcol].astype(str).str.strip(),
    }).drop_duplicates()
    return out.reset_index(drop=True)

# ------------------------------
# Effects loaders
# ------------------------------

def load_kinopt_effects(path: str):
    """Parse kinopt Alpha/Beta sheets into α triplets and c_k priors."""
    xl = pd.ExcelFile(path)
    # Parse Alpha Values
    dfA = pd.read_excel(xl, sheet_name="Alpha Values")
    dfA = _normcols(dfA)
    gcol = _find_col(dfA, ["gene","protein","geneid"])
    scol = _find_col(dfA, ["psite","site"])
    kcol = _find_col(dfA, ["kinase","kinases","k"])
    aval = _find_col(dfA, ["alpha","value","effect","score","weight"])
    if not (gcol and scol and kcol and aval):
        raise ValueError("kinopt::Alpha Values must have (Gene, Psite, Kinase, Alpha).")
    kin_alpha = dfA[[gcol,scol,kcol,aval]].rename(
        columns={gcol:"protein",scol:"psite",kcol:"kinase",aval:"alpha"})
    kin_alpha["protein"] = kin_alpha["protein"].astype(str).str.strip()
    kin_alpha["psite"]   = kin_alpha["psite"].astype(str).str.strip()
    kin_alpha["kinase"]  = kin_alpha["kinase"].astype(str).str.strip()
    kin_alpha["alpha"]   = pd.to_numeric(kin_alpha["alpha"], errors="coerce").fillna(0.0)

    # Parse Beta Values
    dfB = pd.read_excel(xl, sheet_name="Beta Values")
    dfB = _normcols(dfB)
    kcolB = _find_col(dfB, ["kinase","kinases","k"])
    scolB = _find_col(dfB, ["psite","site"])
    bval  = _find_col(dfB, ["beta","value","effect","score","weight"])
    if not (kcolB and bval):
        raise ValueError("kinopt::Beta Values must have (Kinase, [Psite], Beta).")
    kin_beta = dfB[[kcolB, bval] + ([scolB] if scolB else [])].rename(
        columns={kcolB:"kinase", bval:"beta", (scolB or "psite"):"psite"})
    kin_beta["kinase"] = kin_beta["kinase"].astype(str).str.strip()
    if "psite" in kin_beta.columns:
        kin_beta["psite"] = kin_beta["psite"].astype(str).str.strip()
    else:
        kin_beta["psite"] = ""
    kin_beta["is_unknown_site"] = kin_beta["psite"].isna() | (kin_beta["psite"].astype(str).str.len()==0)
    kin_beta["beta"] = pd.to_numeric(kin_beta["beta"], errors="coerce").fillna(0.0)
    return kin_alpha, kin_beta

def load_tfopt_effects(path: str):
    """Parse tfopt Alpha/Beta sheets. Alpha is used to initialize A_i."""
    xl = pd.ExcelFile(path)
    # Alpha: (mRNA, TF, Value) -> we only need mRNA/value here
    dfA = pd.read_excel(xl, sheet_name="Alpha Values")
    dfA = _normcols(dfA)
    mcol = _find_col(dfA, ["mrna","gene","protein","geneid"])
    vcol = _find_col(dfA, ["value","alpha","effect","score","weight"])
    if not (mcol and vcol):
        raise ValueError("tfopt::Alpha Values must have (mRNA, Value).")
    tf_alpha = dfA[[mcol, vcol]].rename(columns={mcol:"protein", vcol:"value"})
    tf_alpha["protein"] = tf_alpha["protein"].astype(str).str.strip()
    tf_alpha["value"]   = pd.to_numeric(tf_alpha["value"], errors="coerce").fillna(0.0)

    # Beta: (TF, Psite, Value) -> parsed for completeness; not used in current ODE
    try:
        dfB = pd.read_excel(xl, sheet_name="Beta Values")
        dfB = _normcols(dfB)
        tcolB = _find_col(dfB, ["tf"])
        scolB = _find_col(dfB, ["psite","site"])
        vcolB = _find_col(dfB, ["value","beta","effect","score","weight"])
        if tcolB and vcolB:
            tf_beta = dfB[[tcolB, vcolB] + ([scolB] if scolB else [])].rename(
                columns={tcolB:"tf", vcolB:"value", (scolB or "psite"):"psite"})
            tf_beta["psite"] = tf_beta["psite"].astype(str).fillna("").str.strip()
            tf_beta["value"] = pd.to_numeric(tf_beta["value"], errors="coerce").fillna(0.0)
        else:
            tf_beta = pd.DataFrame(columns=["tf","psite","value"])
    except Exception:
        tf_beta = pd.DataFrame(columns=["tf","psite","value"])
    return tf_alpha, tf_beta

# ------------------------------
# Estimated targets (FC)
# ------------------------------

def load_estimated_protein_FC(path: str) -> pd.DataFrame:
    """Load protein FC time series from kinopt::Estimated and filter to the 14-point grid."""
    df = pd.read_excel(path, sheet_name="Estimated")
    df = _normcols(df)
    namecol = _find_col(df, ["protein","gene","geneid","target"])
    if namecol is None:
        raise ValueError("kinopt::Estimated must include a protein/gene identifier column.")
    tcols = _time_cols(df.drop(columns=[namecol], errors="ignore"))
    tidy = df[[namecol]+tcols].melt(id_vars=[namecol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
    tidy = tidy.drop(columns=["time_col"])
    tidy = tidy.rename(columns={namecol:"protein"})
    tidy["protein"] = tidy["protein"].astype(str).str.strip()
    tidy = tidy[tidy["time"].isin(TIME_POINTS)].copy()
    tidy = tidy.sort_values(["protein","time"]).reset_index(drop=True)
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    return tidy

def load_estimated_rna_FC(path: str) -> pd.DataFrame:
    """Load RNA FC time series from tfopt::Estimated and filter to the 9-point grid."""
    df = pd.read_excel(path, sheet_name="Estimated")
    df = _normcols(df)
    namecol = _find_col(df, ["mrna","protein","gene","geneid","target"])
    if namecol is None:
        raise ValueError("tfopt::Estimated must include an mRNA/protein identifier column.")
    tcols = _time_cols(df.drop(columns=[namecol], errors="ignore"))
    tidy = df[[namecol]+tcols].melt(id_vars=[namecol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
    tidy = tidy.drop(columns=["time_col"])
    tidy = tidy.rename(columns={namecol:"protein"})
    tidy["protein"] = tidy["protein"].astype(str).str.strip()
    tidy = tidy[tidy["time"].isin(TIME_POINTS_RNA)].copy()
    tidy = tidy.sort_values(["protein","time"]).reset_index(drop=True)
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    return tidy

# ------------------------------
# Index and sparse maps
# ------------------------------

class Index:
    """Holds indexing for proteins, sites, kinases and state offsets."""
    def __init__(self, interactions: pd.DataFrame):
        # Unique proteins
        self.proteins = sorted(interactions["protein"].unique().tolist())
        # Map protein name to index
        self.p2i = {p:i for i,p in enumerate(self.proteins)}
        # Per-protein site lists
        self.sites = [interactions.loc[interactions["protein"]==p,"psite"].unique().tolist() for p in self.proteins]
        # Unique kinases
        self.kinases = sorted(interactions["kinase"].unique().tolist())
        # Map kinase name to index
        self.k2i = {k:i for i,k in enumerate(self.kinases)}
        # Counts
        self.N = len(self.proteins)
        self.n_sites_per_protein = [len(s) for s in self.sites]
        # Offsets for slicing y vector: [R, P, Psite_1..Psite_m]
        self.offset = []
        curr = 0
        for nsi in self.n_sites_per_protein:
            self.offset.append(curr)
            curr += 2 + nsi
        # Total state dimension across all proteins
        self.state_dim = curr

    def block(self, i: int) -> slice:
        """Return the y-slice for protein i."""
        s = self.offset[i]
        e = s + 2 + self.n_sites_per_protein[i]
        return slice(s, e)

def _build_W_one(args):
    """Helper to build one protein's W (rows: sites, cols: kinases)."""
    i, p, interactions, sites, k2i = args
    sub = interactions[interactions["protein"]==p][["psite","kinase"]]
    site_order = {s:r for r,s in enumerate(sites)}
    rows, cols = [], []
    for _, r in sub.iterrows():
        s = r["psite"]; k = r["kinase"]
        if s in site_order and k in k2i:
            rows.append(site_order[s]); cols.append(k2i[k])
    data = np.ones(len(rows), float)
    return i, sparse.csr_matrix((data,(rows,cols)), shape=(len(sites), len(k2i)))

def build_W(interactions: pd.DataFrame, idx: Index) -> List[sparse.csr_matrix]:
    """Build sparse maps W_i (site x kinase) in parallel."""
    print("[INFO] Building W (site×kinase) maps ...")
    with mp.get_context("fork").Pool(processes=6) as pool:
        tasks = ((i, p, interactions, idx.sites[i], idx.k2i) for i,p in enumerate(idx.proteins))
        results = list(tqdm(pool.imap_unordered(_build_W_one, tasks), total=idx.N, desc="W maps", ncols=90))
    W = [None] * idx.N
    for i, Wi in results:
        W[i] = Wi
    return W

# ------------------------------
# Initialize parameters from effects
# ------------------------------

def init_params_from_effects(idx: Index, W_list: List[sparse.csr_matrix],
                             kin_alpha: pd.DataFrame, kin_beta: pd.DataFrame,
                             tf_alpha: pd.DataFrame, alpha_scale=0.2) -> Dict[str, object]:
    """Initialize c_k, A_i, B_i, C_i, D_i and α from provided effects tables."""
    print("[INFO] Initializing parameters from effects ...")
    # c_k from kin_beta (unknown-site entries as general kinase effect; fallback to mean)
    base = kin_beta[kin_beta["is_unknown_site"]].groupby("kinase")["beta"].mean()
    c_k = base.reindex(idx.kinases).fillna(kin_beta.groupby("kinase")["beta"].mean()).fillna(0.0).values.astype(float)
    c_k = softplus(c_k); c_k /= (np.mean(c_k)+1e-12)

    # A_i from tf_alpha (aggregate duplicates before reindex)
    if tf_alpha is not None and len(tf_alpha):
        tf_alpha = tf_alpha.copy()
        tf_alpha["protein"] = tf_alpha["protein"].astype(str).str.strip()
        tf_alpha["value"]   = pd.to_numeric(tf_alpha["value"], errors="coerce")
        tf_alpha_series = tf_alpha.groupby("protein", as_index=True)["value"].mean().sort_index()
        default_A = float(tf_alpha_series.mean()) if len(tf_alpha_series) else 1.0
        A_i = tf_alpha_series.reindex(idx.proteins).fillna(default_A).to_numpy(dtype=float)
    else:
        A_i = np.full(idx.N, 1.0, dtype=float)
    A_i = softplus(A_i); A_i /= (np.median(A_i)+1e-12)

    # Fixed baselines (kept as in original logic)
    N = idx.N
    B_i = np.full(N, 0.2, float)
    C_i = np.full(N, 0.5, float)
    D_i = np.full(N, 0.05, float)
    r_site = [np.zeros(idx.n_sites_per_protein[i], float) for i in range(N)]

    # α from kin_alpha where available, else fallback to alpha_scale; optionally scaled by site-specific beta
    print("[INFO] Seeding α from kinopt::Alpha Values (fallback to alpha_scale where missing) ...")
    kin_alpha_keyed = kin_alpha.copy()
    kin_alpha_keyed["key"] = list(zip(kin_alpha_keyed["protein"].astype(str),
                                      kin_alpha_keyed["psite"].astype(str),
                                      kin_alpha_keyed["kinase"].astype(str)))
    alpha_lookup = {k: v for k, v in zip(kin_alpha_keyed["key"], kin_alpha_keyed["alpha"])}

    # Precompute site-specific beta map (optional multiplier)
    site_beta = kin_beta[~kin_beta["is_unknown_site"]].copy()
    if not site_beta.empty:
        site_beta["key"] = list(zip(site_beta["kinase"].astype(str), site_beta["psite"].astype(str)))
        beta_map = {k: v for k, v in zip(site_beta["key"], site_beta["beta"])}
    else:
        beta_map = {}

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
            data[n] = alpha_scale if np.isnan(val) else max(0.0, float(val))
            b = beta_map.get((k_name, s_name), np.nan)
            if not np.isnan(b):  # optional site-specific scaling
                data[n] *= softplus(b)
        alpha_init.append(sparse.csr_matrix((data, (rows, cols)), shape=Wi.shape))

    params = {"c_k": c_k, "A_i": A_i, "B_i": B_i, "C_i": C_i, "D_i": D_i, "r_site": r_site, "alpha_list": alpha_init}
    return params

# ------------------------------
# Numba-accelerated helpers (inner math)
# ------------------------------

@njit(cache=True, fastmath=True)
def _dot_alpha_kt(rows: np.ndarray, indptr: np.ndarray, data: np.ndarray, Kt: np.ndarray, out: np.ndarray):
    """Compute alpha.dot(Kt) for a CSR matrix into out (length = n_sites)."""
    # For each row (site), sum data * Kt[col]
    for i in range(out.shape[0]):
        acc = 0.0
        for jj in range(indptr[i], indptr[i+1]):
            acc += data[jj] * Kt[rows[jj]]
        out[i] = acc

# ------------------------------
# ODE system
# ------------------------------

class KinaseInput:
    """Encapsulates kinase activity input; here: constant baseline."""
    def __init__(self, kinases: List[str], const_levels: np.ndarray):
        self.kinases = kinases
        self.const = np.asarray(const_levels, float)
    def eval(self, t: float) -> np.ndarray:
        return self.const

class System:
    """Global ODE system holding parameters, structure, and RHS evaluation."""
    def __init__(self, idx: Index, W_list: List[sparse.csr_matrix], params: Dict[str,object], kin_input: KinaseInput):
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
        # Buffers for per-protein site phosphorylation rates S_i(t)
        self._buf_S = [np.zeros(idx.n_sites_per_protein[i], float) for i in range(idx.N)]

    def set_c_k(self, c): self.c_k = c
    def set_D_i(self, D): self.D_i = D

    def set_alpha_from_vals(self, alpha_vals_list: List[np.ndarray]):
        """Set α from per-protein dense arrays matching nonzero pattern of W_i."""
        new = []
        for i, W in enumerate(self.W_list):
            if W.nnz == 0:
                new.append(W.copy())
                continue
            rows, cols = W.nonzero()
            A = sparse.csr_matrix((alpha_vals_list[i], (rows, cols)), shape=W.shape)
            new.append(A)
        self.alpha_list = new

    def site_rates(self, t: float) -> List[np.ndarray]:
        """Compute per-protein site phosphorylation rates S_i(t) = α_i · (c_k ⊙ K(t))."""
        Kt = self.kin.eval(t) * self.c_k
        out = self._buf_S
        for i in range(self.idx.N):
            if self.W_list[i].nnz == 0 or self.alpha_list[i].nnz == 0:
                out[i].fill(0.0)
                continue
            A = self.alpha_list[i]
            # JIT-accelerated CSR multiply with dense vector Kt
            if NUMBA_OK:
                _dot_alpha_kt(A.indices, A.indptr, A.data, Kt, out[i])
            else:
                out[i][:] = A.dot(Kt)
        return out

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """RHS for the global ODE; structure unchanged."""
        dy = np.zeros_like(y)
        S_list = self.site_rates(t)
        for i in range(self.idx.N):
            sl = self.idx.block(i)
            block = y[sl]
            R, P = block[0], block[1]
            Ps = block[2:]
            Ai, Bi, Ci, Di = self.A_i[i], self.B_i[i], self.C_i[i], self.D_i[i]
            if self.idx.n_sites_per_protein[i] > 0:
                rsi = self.r_site[i]
                kdeg_s = (1.0 + rsi) * Di
                S_i = S_list[i]
            else:
                kdeg_s = np.array([], float)
                S_i = np.array([], float)
            # dR/dt
            dR = Ai - Bi * R
            # dP/dt
            sumS = S_i.sum() if S_i.size else 0.0
            dP = Ci * R - (Di + sumS) * P + Ps.sum()
            # dPsite/dt
            dPs = S_i * P - (1.0 + kdeg_s) * Ps
            # Write back
            dy[sl.start+0] = dR
            dy[sl.start+1] = dP
            if dPs.size:
                dy[sl.start+2: sl.start+2+dPs.size] = dPs
        return dy

    def y0(self, R0=1.0, P0=1.0, Psite0=0.01) -> np.ndarray:
        """Biologically grounded initial state."""
        y0 = np.zeros(self.idx.state_dim, float)
        for i in range(self.idx.N):
            sl = self.idx.block(i)
            y0[sl.start+0] = R0
            y0[sl.start+1] = P0
            nsi = self.idx.n_sites_per_protein[i]
            if nsi > 0:
                y0[sl.start+2: sl.start+2+nsi] = Psite0
        return y0

# ------------------------------
# Simulation + observables (FC)
# ------------------------------

def build_jac_sparsity(idx: Index) -> sparse.csr_matrix:
    """Block-diagonal sparsity for [R_i, P_i, P_i,1..P_i,m]."""
    rows, cols = [], []
    base = 0
    for m in idx.n_sites_per_protein:
        # local indices in block
        R, P = 0, 1
        # dR/dR
        rows.append(base + R); cols.append(base + R)
        # dP/dR, dP/dP
        rows += [base + P, base + P]; cols += [base + R, base + P]
        # dP/dPsite_j (sum over sites)
        for j in range(m):
            rows.append(base + P); cols.append(base + 2 + j)
        # dPsite_j/dP and dPsite_j/dPsite_j
        for j in range(m):
            rows += [base + 2 + j, base + 2 + j]
            cols += [base + P,     base + 2 + j]
        base += 2 + m
    n = idx.state_dim
    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

def simulate_union(sys: System, jsp:sparse.csr_matrix, times_union: np.ndarray, atol=1e-8, rtol=1e-6, method="BDF"):
    """Integrate ODE once on the union of protein and RNA time grids."""
    t0, t1 = float(times_union.min()), float(times_union.max())
    # print(f"[INFO] Integrating ODE from t={t0} to t={t1} (|states|={sys.idx.state_dim}) ...")
    sol = solve_ivp(sys.rhs, (t0, t1), sys.y0(), t_eval=np.asarray(times_union, float),
                    atol=atol, rtol=rtol, method=method, jac_sparsity=jsp)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y.T  # shape: (T, state_dim)

def _protein_fc_one(args):
    """Compute protein FC row-block for a single protein index."""
    i, prot, idx, t, Y, times_needed = args
    sl = idx.block(i)
    P = Y[:, sl.start+1]
    nsi = idx.n_sites_per_protein[i]
    Ps = Y[:, sl.start+2: sl.start+2+nsi].sum(axis=1) if nsi>0 else 0.0
    total = np.maximum(P + Ps, 1e-12)
    fc = total / total[0]
    mask = np.isin(t, times_needed)
    return prot, t[mask], fc[mask]

def _rna_fc_one(args):
    """Compute RNA FC row-block for a single protein index."""
    i, prot, idx, t, Y, times_needed = args
    sl = idx.block(i)
    R = np.maximum(Y[:, sl.start+0], 1e-12)
    fc = R / R[0]
    mask = np.isin(t, times_needed)
    return prot, t[mask], fc[mask]

# def protein_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
#     """Assemble protein FC dataframe using multiprocessing"""
#     rows = []
#     ctx = mp.get_context("fork")
#     with ctx.Pool(processes=6) as pool:
#         tasks = ((i, prot, idx, t, Y, times_needed) for i, prot in enumerate(idx.proteins))
#         for prot, tt, fc in pool.imap_unordered(_protein_fc_one, tasks):
#             rows.append(pd.DataFrame({"time": tt, "protein": prot, "fc": fc}))
#     return pd.concat(rows, ignore_index=True)
#
#
# def rna_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
#     """Assemble RNA FC dataframe using multiprocessing"""
#     rows = []
#     ctx = mp.get_context("fork")
#     with ctx.Pool(processes=6) as pool:
#         tasks = ((i, prot, idx, t, Y, times_needed) for i, prot in enumerate(idx.proteins))
#         for prot, tt, fc in pool.imap_unordered(_rna_fc_one, tasks):
#             rows.append(pd.DataFrame({"time": tt, "protein": prot, "fc": fc}))
#     return pd.concat(rows, ignore_index=True)

def protein_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    mask = np.isin(t, times_needed)
    t_sel = t[mask]
    rows = []
    # single Python loop over proteins, heavy work is NumPy
    for i, prot in enumerate(idx.proteins):
        sl = idx.block(i)
        P  = Y[:, sl.start+1]
        m  = idx.n_sites_per_protein[i]
        Ps = Y[:, sl.start+2: sl.start+2+m].sum(axis=1) if m > 0 else 0.0
        total = np.maximum(P + Ps, 1e-12)
        fc = total / total[0]
        rows.append(pd.DataFrame({"time": t_sel, "protein": prot, "fc": fc[mask]}))
    return pd.concat(rows, ignore_index=True)

def rna_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    mask = np.isin(t, times_needed)
    t_sel = t[mask]
    rows = []
    for i, prot in enumerate(idx.proteins):
        sl = idx.block(i)
        R  = np.maximum(Y[:, sl.start+0], 1e-12)
        fc = R / R[0]
        rows.append(pd.DataFrame({"time": t_sel, "protein": prot, "fc": fc[mask]}))
    return pd.concat(rows, ignore_index=True)

# ------------------------------
# Parameter packing
# ------------------------------

def init_raw_for_stage(params: Dict[str,object], alpha_list: List[sparse.csr_matrix], stage: int):
    """Pack current parameters into raw vector theta according to chosen stage."""
    parts: Dict[str, object] = {}
    vecs: List[np.ndarray] = []

    # c_k block
    if stage in (1, 3):
        rc = inv_softplus(params["c_k"])
        parts["raw_c"] = slice(0, len(rc))
        vecs.append(rc)

    # α block
    if stage in (2, 3):
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

    # D_i block
    if stage == 3:
        rD = inv_softplus(params["D_i"])
        start = sum(len(v) for v in vecs)
        parts["raw_D"] = slice(start, start+len(rD))
        vecs.append(rD)

    theta0 = np.concatenate(vecs) if len(vecs) > 0 else np.array([], float)
    return theta0, parts

def assign_theta(theta: np.ndarray, parts: Dict[str,object], sys: System, alpha_init: List[sparse.csr_matrix]):
    """Unpack raw theta back into system parameters."""
    # c_k
    if "raw_c" in parts:
        sl = parts["raw_c"]
        sys.set_c_k(softplus(theta[sl]))
    # α
    if "raw_alpha" in parts:
        vals = []
        for sl in parts["raw_alpha"]:
            raw = theta[sl] if sl.start != sl.stop else np.array([], float)
            vals.append(softplus(raw) if raw.size > 0 else np.array([], float))
        sys.set_alpha_from_vals(vals)
    else:
        sys.set_alpha_from_vals([A.data if A.nnz>0 else np.array([], float) for A in alpha_init])
    # D_i
    if "raw_D" in parts:
        sl = parts["raw_D"]
        sys.set_D_i(softplus(theta[sl]))

# ------------------------------
# Loss (FC)
# ------------------------------

def build_weights(times: np.ndarray, early_focus: float) -> Dict[float,float]:
    """Build time-dependent weights emphasizing early dynamics if requested."""
    tmin, tmax = float(times.min()), float(times.max())
    span = (tmax - tmin) if tmax > tmin else 1.0
    return {float(t): 1.0 + early_focus * (tmax - float(t)) / span for t in times}

def dual_loss(theta: np.ndarray, parts: Dict[str,object], sys: System, idx: Index,
              alpha_init: List[sparse.csr_matrix], c_k_init: np.ndarray,
              df_prot_obs: pd.DataFrame, df_rna_obs: pd.DataFrame,
              lam: Dict[str,float], atol: float, rtol: float, jsp: sparse.csr_matrix):
    """Total objective = protein FC LSQ + λ_rna·RNA FC LSQ + regularization terms."""
    # Update system parameters
    assign_theta(theta, parts, sys, alpha_init)

    # Simulate once on union grid
    times_union = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    t, Y = simulate_union(sys, jsp, times_union, atol=atol, rtol=rtol)

    # Model observables on their respective grids
    dfp = protein_FC(idx, t, Y, TIME_POINTS).rename(columns={"fc":"pred_fc"})
    dfr = rna_FC(idx, t, Y, TIME_POINTS_RNA).rename(columns={"fc":"pred_fc"})

    # Join predictions with observations
    mp = df_prot_obs.merge(dfp, on=["protein","time"], how="inner")
    mr = df_rna_obs.merge(dfr,  on=["protein","time"], how="inner")

    # Data terms (weighted LSQ)
    prot_loss = np.sum(mp["w"].values * (mp["pred_fc"].values - mp["fc"].values)**2)
    rna_loss  = np.sum(mr["w"].values * (mr["pred_fc"].values - mr["fc"].values)**2)

    # Regularization
    reg_alpha_l1 = 0.0
    reg_alpha_prior = 0.0
    if "raw_alpha" in parts:
        cur_alpha = []
        for A in sys.alpha_list:
            cur_alpha.append(A.data if A.nnz > 0 else np.array([], float))
        cur_alpha = np.concatenate(cur_alpha) if len(cur_alpha) > 0 else np.array([], float)

        init_alpha = []
        for A in alpha_init:
            init_alpha.append(A.data if A.nnz > 0 else np.array([], float))
        init_alpha = np.concatenate(init_alpha) if len(init_alpha) > 0 else np.array([], float)

        if cur_alpha.size > 0:
            reg_alpha_l1 = np.sum(cur_alpha)  # α >= 0, so |α|_1 = sum(α)
            if init_alpha.size == cur_alpha.size:
                reg_alpha_prior = np.sum((cur_alpha - init_alpha)**2)

    reg_c_prior = 0.0
    if "raw_c" in parts:
        reg_c_prior = np.sum((sys.c_k - c_k_init)**2)

    total = prot_loss + lam.get("lambda_rna", 1.0) * rna_loss \
            + lam.get("l1_alpha", 0.0) * reg_alpha_l1 \
            + lam.get("prior_alpha", 0.0) * reg_alpha_prior \
            + lam.get("prior_c", 0.0) * reg_c_prior

    details = {
        "prot_loss": float(prot_loss),
        "rna_loss": float(rna_loss),
        "reg_alpha_l1": float(reg_alpha_l1),
        "reg_alpha_prior": float(reg_alpha_prior),
        "reg_c_prior": float(reg_c_prior),
        "total": float(total)
    }
    return total, details

# ------------------------------
# Fit report
# ------------------------------
# ------------------------------
# Fit report (plots every single point)
# ------------------------------
def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))

def _fit_metrics(df: pd.DataFrame, group_cols=("protein",), obs_col="fc", pred_col="pred_fc"):
    rows = []
    # metrics require pairs; filter to where both exist
    dfp = df.dropna(subset=[obs_col, pred_col])
    for g, sub in dfp.groupby(list(group_cols)):
        y = sub[obs_col].to_numpy(float)
        yhat = sub[pred_col].to_numpy(float)
        resid = y - yhat
        rss = float(np.sum(resid**2))
        tss = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - (rss / tss) if (tss and not math.isnan(tss) and tss > 0) else float("nan")
        rmse = float(np.sqrt(np.mean(resid**2))) if len(y) else float("nan")
        mae = float(np.mean(np.abs(resid))) if len(y) else float("nan")
        m = {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y))}
        if isinstance(g, tuple):
            for k, v in zip(group_cols, g): m[k] = v
        else:
            m[group_cols[0]] = g
        rows.append(m)
    return pd.DataFrame(rows)

def _plot_series(ax, t_all, y_obs, y_pred, title, xlabel="time (min)", ylabel="FC"):
    # plot both as points so every datum is visible; connect pred with a thin line for guidance
    if y_pred.size:
        ax.plot(t_all, y_pred, lw=1.0)
        ax.scatter(t_all, y_pred, s=18)
    if y_obs.size:
        ax.scatter(t_all, y_obs, s=24)  # slightly larger so obs are clearly visible
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
    # per-entity pred-vs-obs (paired only)
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
    out_dir: str,
    pdf_name: str = "fit_report.pdf",
    metrics_csv: str = "fit_metrics.csv",
    per_page: int = 12
):
    """
    Plots every single data point:
      - Series panels: obs points AND pred points at every time (no downsampling)
      - Residuals: for paired points (obs & pred present)
      - Per-entity Pred-vs-Obs scatter: all paired points
      - Global Pred-vs-Obs scatter: all paired points across entities
    """
    os.makedirs(out_dir, exist_ok=True)

    # Outer merge so no timepoint is dropped; keep all data & all preds
    p = (df_prot_obs[["protein","time","fc"]]
         .merge(df_prot_pred[["protein","time","pred_fc"]],
                on=["protein","time"], how="outer")
         .sort_values(["protein","time"]))
    r = (df_rna_obs[["protein","time","fc"]]
         .merge(df_rna_pred[["protein","time","pred_fc"]],
                on=["protein","time"], how="outer")
         .sort_values(["protein","time"]))

    # Metrics on paired points only
    m_prot = _fit_metrics(p, group_cols=("protein",), obs_col="fc", pred_col="pred_fc")
    m_rna  = _fit_metrics(r, group_cols=("protein",), obs_col="fc", pred_col="pred_fc")

    # Global metrics and paired data for overview
    def _global_pairs(df):
        dfp = df.dropna(subset=["fc","pred_fc"])
        if dfp.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}, np.array([]), np.array([])
        y = dfp["fc"].to_numpy(float)
        yhat = dfp["pred_fc"].to_numpy(float)
        resid = y - yhat
        rss = float(np.sum(resid**2))
        tss = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - (rss / tss) if (tss and not math.isnan(tss) and tss > 0) else float("nan")
        rmse = float(np.sqrt(np.mean(resid**2))) if len(y) else float("nan")
        mae = float(np.mean(np.abs(resid))) if len(y) else float("nan")
        return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y))}, y, yhat

    g_prot, gp_obs, gp_pred = _global_pairs(p)
    g_rna, gr_obs, gr_pred = _global_pairs(r)

    # Save metrics table
    m_prot["modality"] = "protein"
    m_rna["modality"]  = "rna"
    pd.concat([m_prot, m_rna], ignore_index=True).to_csv(
        os.path.join(out_dir, metrics_csv), index=False
    )

    # PDF report
    pdf_path = os.path.join(out_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        # Overview page (paired only)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        _scatter_global(ax[0], gp_obs, gp_pred,
                        f"Protein: Pred vs Obs (R²={g_prot['r2']:.3f}, RMSE={g_prot['rmse']:.3g}, n={g_prot['n']})")
        _scatter_global(ax[1], gr_obs, gr_pred,
                        f"RNA: Pred vs Obs (R²={g_rna['r2']:.3f}, RMSE={g_rna['rmse']:.3g}, n={g_rna['n']})")
        fig.suptitle("Goodness of Fit — Global Overview (paired points)", y=1.02)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Per-protein pages: Protein series + residuals + Pred-vs-Obs
        prot_list = p["protein"].dropna().unique().tolist()
        for start in range(0, len(prot_list), per_page):
            batch = prot_list[start:start+per_page]
            n = len(batch)
            ncols = 3
            nrows = math.ceil(n * 3 / ncols)  # 3 panels per entity
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(6, 2.4*nrows)))
            axes = axes.ravel() if nrows*ncols > 1 else [axes]
            slot = 0
            for prot in batch:
                sub = p[p["protein"] == prot]
                if sub.empty: continue
                t_all = sub["time"].to_numpy(float)
                y_obs = sub["fc"].to_numpy(float)
                y_pred = sub["pred_fc"].to_numpy(float)

                # residuals for paired times only
                paired = sub.dropna(subset=["fc","pred_fc"])
                t_pair = paired["time"].to_numpy(float)
                resid  = paired["fc"].to_numpy(float) - paired["pred_fc"].to_numpy(float)

                _plot_series(axes[slot], t_all, y_obs, y_pred, f"{prot} | Protein FC"); slot += 1
                _plot_residuals(axes[slot], t_pair, resid, "Residuals"); slot += 1
                _scatter_pvo(axes[slot],
                             paired["fc"].to_numpy(float),
                             paired["pred_fc"].to_numpy(float),
                             "Pred vs Obs"); slot += 1

            while slot < len(axes):
                axes[slot].axis("off"); slot += 1
            fig.suptitle("Protein fits (all points shown)", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Per-protein pages: RNA series + residuals + Pred-vs-Obs
        rna_list = r["protein"].dropna().unique().tolist()
        for start in range(0, len(rna_list), per_page):
            batch = rna_list[start:start+per_page]
            n = len(batch)
            ncols = 3
            nrows = math.ceil(n * 3 / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(6, 2.4*nrows)))
            axes = axes.ravel() if nrows*ncols > 1 else [axes]
            slot = 0
            for prot in batch:
                sub = r[r["protein"] == prot]
                if sub.empty: continue
                t_all = sub["time"].to_numpy(float)
                y_obs = sub["fc"].to_numpy(float)
                y_pred = sub["pred_fc"].to_numpy(float)

                paired = sub.dropna(subset=["fc","pred_fc"])
                t_pair = paired["time"].to_numpy(float)
                resid  = paired["fc"].to_numpy(float) - paired["pred_fc"].to_numpy(float)

                _plot_series(axes[slot], t_all, y_obs, y_pred, f"{prot} | RNA FC"); slot += 1
                _plot_residuals(axes[slot], t_pair, resid, "Residuals"); slot += 1
                _scatter_pvo(axes[slot],
                             paired["fc"].to_numpy(float),
                             paired["pred_fc"].to_numpy(float),
                             "Pred vs Obs"); slot += 1

            while slot < len(axes):
                axes[slot].axis("off"); slot += 1
            fig.suptitle("RNA fits (all points shown)", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    return pdf_path, os.path.join(out_dir, metrics_csv)

# ------------------------------
# Main
# ------------------------------

def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser(description="Global ODE dual-fit (FC) with effects from 'Alpha Values'/'Beta Values'.")
    ap.add_argument("--interaction", required=True)
    ap.add_argument("--kinopt", required=True)
    ap.add_argument("--tfopt", required=True)
    ap.add_argument("--stage", type=int, choices=[1,2,3], default=3)
    ap.add_argument("--early-focus", type=float, default=1.0)
    ap.add_argument("--l1-alpha", type=float, default=2e-3)
    ap.add_argument("--lambda-prior", type=float, default=1e-2)
    ap.add_argument("--lambda-c", type=float, default=1e-2)
    ap.add_argument("--lambda-rna", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--output-dir", default="./out_dual_fc")
    ap.add_argument("--log-every", type=int, default=25,
                    help="Print objective every N function evaluations (0 = silent)")
    args = ap.parse_args()

    # Timer start
    T0 = time.time()
    _ensure_dir(args.output_dir)

    # Load inputs
    print("[STEP 1/6] Loading interactions and effects ...")
    interactions = load_interactions(args.interaction)
    kin_alpha, kin_beta = load_kinopt_effects(args.kinopt)
    tf_alpha, tf_beta = load_tfopt_effects(args.tfopt)

    # Load FC targets
    print("[STEP 2/6] Loading Estimated FC time series ...")
    df_prot_obs = load_estimated_protein_FC(args.kinopt)
    df_rna_obs  = load_estimated_rna_FC(args.tfopt)

    # Build index and W
    print("[STEP 3/6] Building global index and W maps ...")
    idx = Index(interactions)
    W_list = build_W(interactions, idx)
    JSP = build_jac_sparsity(idx)

    # Filter observations to modeled proteins
    df_prot_obs = df_prot_obs[df_prot_obs["protein"].isin(idx.proteins)].reset_index(drop=True)
    df_rna_obs  = df_rna_obs[df_rna_obs["protein"].isin(idx.proteins)].reset_index(drop=True)

    # Initialize parameters from effects
    print("[STEP 4/6] Initializing parameters (A, B, C, D, c_k, α) ...")
    params = init_params_from_effects(idx, W_list, kin_alpha, kin_beta, tf_alpha, alpha_scale=0.2)

    # Build weights (early emphasis on protein series)
    wmap = build_weights(TIME_POINTS, early_focus=args.early_focus)
    df_prot_obs["w"] = df_prot_obs["time"].map(wmap).astype(float)
    df_rna_obs["w"] = 1.0

    # Construct ODE system
    print("[STEP 5/6] Constructing ODE system ...")
    kin_input = KinaseInput(idx.kinases, params["c_k"].copy())  # c_k-scaled baseline
    sys = System(idx, W_list, params, kin_input)

    # Prepare optimizer variables
    theta0, parts = init_raw_for_stage(params, params["alpha_list"], stage=args.stage)
    c_k_init = sys.c_k.copy()
    alpha_init = [A.copy() for A in params["alpha_list"]]
    lam = {"l1_alpha": args.l1_alpha, "prior_alpha": args.lambda_prior, "prior_c": args.lambda_c, "lambda_rna": args.lambda_rna}

    # Progress tracking for evaluations
    print("[STEP 6/6] Starting optimization ...")
    progress = {"evals": 0}
    last_details = None  # keep the latest breakdown for a final summary print

    def fun(theta):
        """Objective wrapper with optional periodic prints"""
        nonlocal last_details
        val, details = dual_loss(
            theta, parts, sys, idx, alpha_init, c_k_init,
            df_prot_obs, df_rna_obs, lam, args.atol, args.rtol, JSP
        )
        progress["evals"] += 1
        last_details = details
        # Print every N evaluations if requested
        if args.log_every and (progress["evals"] % args.log_every == 0):
            print(
                f"[eval {progress['evals']}] "
                f"total={details['total']:.6f} "
                f"prot={details['prot_loss']:.6f} "
                f"rna={details['rna_loss']:.6f} "
                f"L1α={details['reg_alpha_l1']:.6f} "
                f"αprior={details['reg_alpha_prior']:.6f} "
                f"cprior={details['reg_c_prior']:.6f}",
                flush=True
            )
        return val

    # ---- Bounds (on RAW parameters; map desired actual bounds via inv_softplus) ----
    def _sp_inv(x):
        # safe inverse softplus for scalars/arrays
        x = np.asarray(x, float)
        return np.log(np.expm1(np.maximum(x, 1e-12)))

    # Choose your biological boxes on the *actual* params:
    ALPHA_MIN, ALPHA_MAX = 0, 2  # α ≥ 0, cap to keep search sane
    C_MIN, C_MAX = 0, 20.0  # kinase strength
    D_MIN, D_MAX = 0, 20.0  # protein turnover baseline

    bounds = [(-np.inf, np.inf)] * (
        0 if 'raw_c' not in parts and 'raw_alpha' not in parts and 'raw_D' not in parts else len(theta0))

    # c_k bounds
    if "raw_c" in parts:
        sl = parts["raw_c"]
        lo = _sp_inv(np.full(sl.stop - sl.start, C_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, C_MAX))
        for j in range(sl.start, sl.stop):
            k = j - sl.start
            bounds[j] = (float(lo[k]), float(hi[k]))

    # α bounds (list of slices, one per protein block)
    if "raw_alpha" in parts:
        for sl in parts["raw_alpha"]:
            if sl.start == sl.stop:
                continue
            # α ≥ ALPHA_MIN; α ≤ ALPHA_MAX
            lo = _sp_inv(np.full(sl.stop - sl.start, max(ALPHA_MIN, 1e-12)))
            hi = _sp_inv(np.full(sl.stop - sl.start, ALPHA_MAX))
            for j in range(sl.start, sl.stop):
                k = j - sl.start
                bounds[j] = (float(lo[k]), float(hi[k]))

    # D_i bounds
    if "raw_D" in parts:
        sl = parts["raw_D"]
        lo = _sp_inv(np.full(sl.stop - sl.start, D_MIN))
        hi = _sp_inv(np.full(sl.stop - sl.start, D_MAX))
        for j in range(sl.start, sl.stop):
            k = j - sl.start
            bounds[j] = (float(lo[k]), float(hi[k]))
    # ---- end bounds ----

    # Run optimizer
    if theta0.size > 0:
        res = minimize(fun, theta0, method="trust-constr", bounds=bounds, options={"maxiter": args.maxiter})
        if last_details is not None:
            print(
                f"[final @ {progress['evals']} evals] "
                f"total={last_details['total']:.6f} "
                f"prot={last_details['prot_loss']:.6f} "
                f"rna={last_details['rna_loss']:.6f} "
                f"L1α={last_details['reg_alpha_l1']:.6f} "
                f"αprior={last_details['reg_alpha_prior']:.6f} "
                f"cprior={last_details['reg_c_prior']:.6f}"
            )
        if not res.success:
            print(f"[WARN] Optimizer: {res.message}")
        theta_opt = res.x
    else:
        theta_opt = theta0

    # Final objective breakdown
    f_opt, comps = dual_loss(theta_opt, parts, sys, idx, alpha_init, c_k_init,
                             df_prot_obs, df_rna_obs, lam, args.atol, args.rtol, JSP)

    # Final simulation for outputs
    times_union = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    t, Y = simulate_union(sys, JSP, times_union, atol=args.atol, rtol=args.rtol)

    # Assemble predictions
    df_out_prot = protein_FC(idx, t, Y, TIME_POINTS)
    df_out_rna  = rna_FC(idx, t, Y, TIME_POINTS_RNA)

    # Save artifacts
    out_prot_path = os.path.join(args.output_dir, "predicted_protein_fc.csv")
    out_rna_path  = os.path.join(args.output_dir, "predicted_rna_fc.csv")
    fit_params_path = os.path.join(args.output_dir, "fitted_params.json")
    fit_summary_path = os.path.join(args.output_dir, "fit_summary.json")

    df_out_prot.to_csv(out_prot_path, index=False)
    df_out_rna.to_csv(out_rna_path, index=False)

    # After you write predicted_protein_fc.csv and predicted_rna_fc.csv:
    dfp_obs = df_prot_obs.copy()
    dfp_pred = df_out_prot.rename(columns={"fc": "pred_fc"})
    dfr_obs = df_rna_obs.copy()
    dfr_pred = df_out_rna.rename(columns={"fc": "pred_fc"})

    pdf_path, metrics_path = plot_fit_report(
        df_prot_obs=dfp_obs,
        df_prot_pred=dfp_pred,
        df_rna_obs=dfr_obs,
        df_rna_pred=dfr_pred,
        out_dir=args.output_dir,
        pdf_name="fit_report.pdf",
        metrics_csv="fit_metrics.csv",
        per_page=12
    )
    print(f"[REPORT] Saved {pdf_path} and {metrics_path}")

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

    # Final report
    print(json.dumps(summary, indent=2))
    print(f"[DONE] Saved:\n  - {out_prot_path}\n  - {out_rna_path}\n  - {fit_params_path}\n  - {fit_summary_path}")

if __name__ == "__main__":
    # On some platforms, using 'fork' is faster for numpy. Adjust if needed.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        # Already set or not supported on Windows; ignore.
        pass
    main()
