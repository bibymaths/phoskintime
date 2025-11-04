
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global phosphorylation ODE simulator (autonomous or input-driven kinases).

Inputs
------
1) Interaction file (CSV): columns contain at least Protein, Psite, Kinase (case-insensitive).
2) Kinase effect file (XLSX): table with a 'Kinase' column and a numeric effect column (e.g., 'Beta', 'Effect').
3) TF effect file (XLSX, optional): may contain per-Protein synthesis effects; will be mapped to A_i.

Optional
--------
- Kinase timecourse CSV: first column 'time', subsequent columns per kinase name; overrides constant K.

Outputs
-------
- global_states.csv: trajectories for all states (R_i, P_i, P_{i,s}).
- global_observables_fc.csv: per-protein fold-change trajectories (log2FC of total protein: P_i + sum_s P_{i,s}).

Model
-----
For each protein i with n_i sites s=0..n_i-1:

dR_i/dt        = A_i - B_i * R_i
dP_i/dt        = C_i * R_i - (D_i + sum_s S_{i,s}(t)) * P_i + sum_s P_{i,s}
dP_{i,s}/dt    = S_{i,s}(t) * P_i - (1 + D_{i,s}) * P_{i,s}

Shared kinase drive:
S_{i,s}(t) = sum_k c_k * alpha_{i,s,k} * K_k(t)

Where c_k is global catalytic strength per kinase, alpha are sparse substrate weights
derived from the interaction table (and optionally regularized by kinopt effects).

Author: generated for a milestone; designed to be self-contained and readable.
"""

import argparse
import sys
import os
import math
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy import sparse


# ------------------------------
# Utilities
# ------------------------------

def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _smart_pick_numeric_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns and np.issubdtype(df[c].dropna().astype(float, errors="ignore").dtype, np.number):
            return c
    # fallback: first numeric column
    for c in df.columns:
        try:
            if np.issubdtype(df[c].dropna().astype(float).dtype, np.number):
                return c
        except Exception:
            continue
    return None

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------------
# Data loading
# ------------------------------

def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_colnames(df)
    # Flexible mappings for column names
    colmap = {}
    # protein
    for cand in ["protein", "gene", "geneid", "prot"]:
        if cand in df.columns:
            colmap["protein"] = cand
            break
    # psite
    for cand in ["psite", "site", "phosphosite", "residue"]:
        if cand in df.columns:
            colmap["psite"] = cand
            break
    # kinase
    for cand in ["kinase", "kinases", "k", "enzyme"]:
        if cand in df.columns:
            colmap["kinase"] = cand
            break
    required = ["protein", "psite", "kinase"]
    if not all(k in colmap for k in required):
        raise ValueError(f"Interaction file needs columns for protein/psite/kinase; found columns={df.columns.tolist()}")
    # standardize
    out = pd.DataFrame({
        "protein": df[colmap["protein"]].astype(str).str.strip(),
        "psite": df[colmap["psite"]].astype(str).str.strip(),
        "kinase": df[colmap["kinase"]].astype(str).str.strip(),
    })
    # deduplicate cleanly
    out = out.drop_duplicates().reset_index(drop=True)
    return out

def load_kinopt(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = _normalize_colnames(df)
    # identify kinase name col
    kcol = None
    for cand in ["kinase", "kinases", "k"]:
        if cand in df.columns:
            kcol = cand
            break
    if kcol is None:
        # attempt alternative: an identifier-like column
        for c in df.columns:
            if "kin" in c:
                kcol = c
                break
    if kcol is None:
        raise ValueError("Kinase results must contain a 'Kinase' column.")
    # numeric effect
    effcol = _smart_pick_numeric_column(df, ["beta", "effect", "score", "weight"])
    if effcol is None:
        raise ValueError("Kinase results must contain a numeric effect column (e.g., Beta/Effect).")
    out = df[[kcol, effcol]].rename(columns={kcol: "kinase", effcol: "effect"})
    out["kinase"] = out["kinase"].astype(str).str.strip()
    out = out.groupby("kinase", as_index=False)["effect"].mean()
    return out

def load_tfopt(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    df = _normalize_colnames(df)
    # heuristic: look for protein/gene column and numeric effect
    pcol = None
    for cand in ["protein", "gene", "geneid", "target"]:
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        return None
    effcol = _smart_pick_numeric_column(df, ["beta", "effect", "score", "weight"])
    if effcol is None:
        return None
    out = df[[pcol, effcol]].rename(columns={pcol: "protein", effcol: "effect"})
    out["protein"] = out["protein"].astype(str).str.strip()
    out = out.groupby("protein", as_index=False)["effect"].mean()
    return out

def load_kinase_timecourse(path: Optional[str], kinases: List[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = _normalize_colnames(df)
    # require a 'time' column
    tcol = "time" if "time" in df.columns else None
    if tcol is None:
        # try first column as time
        tcol = df.columns[0]
    df = df.rename(columns={tcol: "time"})
    # Make sure columns cover provided kinases; missing columns filled with 0.
    for k in kinases:
        if k.lower() not in df.columns:
            df[k.lower()] = 0.0
    # subset to time + given kinases (case-insensitive)
    keep_cols = ["time"] + [k.lower() for k in kinases]
    df = df[[c for c in df.columns if c in keep_cols]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ------------------------------
# Indexing and sparse maps
# ------------------------------

class NetworkIndex:
    def __init__(self, interactions: pd.DataFrame):
        # proteins
        self.proteins = sorted(interactions["protein"].unique().tolist())
        self.p2i: Dict[str, int] = {p: i for i, p in enumerate(self.proteins)}
        # per-protein sites
        self.sites: List[List[str]] = []
        for p in self.proteins:
            ps = interactions.loc[interactions["protein"] == p, "psite"].unique().tolist()
            self.sites.append(ps)
        # kinases
        self.kinases = sorted(interactions["kinase"].unique().tolist())
        self.k2i: Dict[str, int] = {k: i for i, k in enumerate(self.kinases)}
        # offsets for global state vector
        self.N = len(self.proteins)
        self.n_sites_per_protein = [len(slist) for slist in self.sites]
        self.total_site_states = sum(self.n_sites_per_protein)
        # For each protein i, the block starts at offset[i]
        self.offset: List[int] = []
        curr = 0
        for i in range(self.N):
            # each protein has: R_i, P_i, and n_i site states
            self.offset.append(curr)
            curr += 2 + self.n_sites_per_protein[i]
        self.state_dim = curr

    def protein_block_slice(self, i: int) -> slice:
        start = self.offset[i]
        end = start + 2 + self.n_sites_per_protein[i]
        return slice(start, end)

    def site_indices_for_protein(self, i: int) -> np.ndarray:
        # indices relative to protein block: positions of site states after the first two entries (R, P)
        return np.arange(2, 2 + self.n_sites_per_protein[i], dtype=int)


def build_sparse_alpha_maps(interactions: pd.DataFrame, index: NetworkIndex) -> List[sparse.csr_matrix]:
    """
    Build per-protein sparse matrices W_i (n_i x K) with 1 where kinase k can modify site s.
    Later these will be multiplied elementwise by learnable alpha weights.
    """
    mats: List[sparse.csr_matrix] = []
    for i, p in enumerate(index.proteins):
        sub = interactions.loc[interactions["protein"] == p, ["psite", "kinase"]]
        if sub.empty:
            mats.append(sparse.csr_matrix((index.n_sites_per_protein[i], len(index.kinases))))
            continue
        # rows = site order in index.sites[i]
        site_order = {s: r for r, s in enumerate(index.sites[i])}
        rows = []
        cols = []
        data = []
        for _, row in sub.iterrows():
            s = row["psite"]
            k = row["kinase"]
            if s not in site_order or k not in index.k2i:
                continue
            rows.append(site_order[s])
            cols.append(index.k2i[k])
            data.append(1.0)
        if len(rows) == 0:
            mats.append(sparse.csr_matrix((index.n_sites_per_protein[i], len(index.kinases))))
        else:
            W = sparse.csr_matrix((data, (rows, cols)), shape=(index.n_sites_per_protein[i], len(index.kinases)))
            mats.append(W)
    return mats


# ------------------------------
# Parameter initialization
# ------------------------------

def init_parameters(index: NetworkIndex,
                    kinopt: pd.DataFrame,
                    tfopt: Optional[pd.DataFrame],
                    alpha_scale: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Initialize parameters with sane defaults, using kinopt/tfopt if available.
    - c_k from kinopt effects (normalized to mean 1, then softplus to keep positive).
    - A_i from tfopt effects if present else constant baseline.
    - B_i, C_i, D_i shared defaults.
    - D_{i,s} small multipliers of D_i (initialized to 0).
    - alpha_{i,s,k}: start as uniform small weights where mapping exists (scaled by alpha_scale).
    """
    K = len(index.kinases)
    N = index.N
    # c_k from kinopt
    k_eff = kinopt.set_index("kinase")["effect"].reindex(index.kinases).fillna(0.0).values.astype(float)
    # normalize effects to mean 1 after softplus
    c_k = np.log1p(np.exp(k_eff))  # softplus
    c_k /= (np.mean(c_k) + 1e-12)
    # A_i from tfopt if available, else 1.0
    if tfopt is not None:
        A_i = tfopt.set_index("protein")["effect"].reindex(index.proteins).fillna(tfopt["effect"].mean()).values.astype(float)
        # make positive synthesis rates
        A_i = np.log1p(np.exp(A_i))
        A_i /= (np.median(A_i) + 1e-12)
    else:
        A_i = np.full(N, 1.0, dtype=float)
    # Other rates (can be tuned later or fitted)
    B_i = np.full(N, 0.2, dtype=float)      # mRNA decay
    C_i = np.full(N, 0.5, dtype=float)      # translation
    D_i = np.full(N, 0.05, dtype=float)     # base protein turnover
    # per-site extra turnover multiplier (starts 0 => k_deg_i,s = D_i)
    # store as r_i_s so that k_deg_is = (1 + r_i_s)*D_i; start r=0
    r_site: List[np.ndarray] = []
    for i in range(N):
        r_site.append(np.zeros(index.n_sites_per_protein[i], dtype=float))
    # alpha weights per protein: dense arrays aligned with W_i nonzeros
    # We'll store as a list of CSR matrices matching W_i with values initialized to alpha_scale at nonzero positions.
    alpha_vals: List[sparse.csr_matrix] = []
    for i in range(N):
        n_i = index.n_sites_per_protein[i]
        if n_i == 0 or K == 0:
            alpha_vals.append(sparse.csr_matrix((n_i, K)))
            continue
        # create ones for each nonzero then scale
        alpha_vals.append(None)  # placeholder; will be set later
    params = {
        "c_k": c_k,
        "A_i": A_i,
        "B_i": B_i,
        "C_i": C_i,
        "D_i": D_i,
        "r_site_list": r_site,
        "alpha_list": alpha_vals,  # to be filled after we know W_i
    }
    return params


def init_alpha_from_W(alpha_list: List[Optional[sparse.csr_matrix]], W_list: List[sparse.csr_matrix],
                      alpha_scale: float = 0.2) -> List[sparse.csr_matrix]:
    out: List[sparse.csr_matrix] = []
    for i, W in enumerate(W_list):
        W = W.tocsr()
        if W.nnz == 0:
            out.append(W.copy())
            continue
        # for each nonzero in W, set alpha to alpha_scale / (#kinases touching that site)
        # This keeps total per-site incoming weight modest.
        rows, cols = W.nonzero()
        data = np.ones_like(rows, dtype=float)
        # Compute degree per (i, row)
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        data = alpha_scale / deg[rows]
        A = sparse.csr_matrix((data, (rows, cols)), shape=W.shape)
        out.append(A)
    return out


# ------------------------------
# Kinase input handling
# ------------------------------

class KinaseInput:
    def __init__(self, kinases: List[str], const_levels: np.ndarray, timecourse: Optional[pd.DataFrame] = None):
        self.kinases = kinases
        self.const = np.asarray(const_levels, dtype=float).copy()
        self.has_tc = timecourse is not None
        if self.has_tc:
            # Expect columns: time and kinases in lower-case
            self.tc = timecourse.copy()
            # unify column case
            cols_map = {c: c for c in self.tc.columns}
            # ensure 'time'
            if "time" not in self.tc.columns:
                raise ValueError("Kinase timecourse must have a 'time' column.")
            # reorder columns
            keep = ["time"] + [k.lower() for k in kinases]
            for k in kinases:
                if k.lower() not in self.tc.columns:
                    self.tc[k.lower()] = 0.0
            self.tc = self.tc[keep]
            self.tvals = self.tc["time"].values.astype(float)
            self.kmat = self.tc.drop(columns=["time"]).values.astype(float)
        else:
            self.tc = None

    def eval(self, t: float) -> np.ndarray:
        if not self.has_tc:
            return self.const
        # linear interpolate per kinase
        # outside the provided range, hold nearest value
        if t <= self.tvals[0]:
            return self.kmat[0]
        if t >= self.tvals[-1]:
            return self.kmat[-1]
        idx = np.searchsorted(self.tvals, t)  # first index > t
        t0, t1 = self.tvals[idx-1], self.tvals[idx]
        w = (t - t0) / (t1 - t0 + 1e-12)
        return (1.0 - w) * self.kmat[idx-1] + w * self.kmat[idx]


# ------------------------------
# ODE system
# ------------------------------

class GlobalPhosphoODE:
    def __init__(self,
                 index: NetworkIndex,
                 W_list: List[sparse.csr_matrix],
                 params: Dict[str, np.ndarray],
                 kinase_input: KinaseInput):
        self.index = index
        self.W_list = [W.tocsr() for W in W_list]
        self.c_k = params["c_k"]
        self.A_i = params["A_i"]
        self.B_i = params["B_i"]
        self.C_i = params["C_i"]
        self.D_i = params["D_i"]
        self.r_site_list: List[np.ndarray] = params["r_site_list"]
        # fill alpha_list if needed
        alpha_list = params["alpha_list"]
        if any(a is None for a in alpha_list):
            alpha_list = init_alpha_from_W(alpha_list, self.W_list, alpha_scale=0.2)
        self.alpha_list: List[sparse.csr_matrix] = [A.tocsr() for A in alpha_list]
        self.kin_input = kinase_input
        # pre-allocate buffers
        self._buf_S = [np.zeros(index.n_sites_per_protein[i], dtype=float) for i in range(index.N)]

    def site_phos_rates(self, t: float) -> List[np.ndarray]:
        Kt = self.kin_input.eval(t) * self.c_k  # elementwise scale by c_k
        out = self._buf_S
        for i in range(self.index.N):
            if self.W_list[i].nnz == 0:
                out[i].fill(0.0)
                continue
            # S_i(t) = (alpha_i ⊙ W_i) @ (c_k ⊙ K(t))
            # but alpha_i already has values only where W has 1s; so we just use alpha_i as weights matrix
            out[i][:] = self.alpha_list[i].dot(Kt)
        return out

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        dy = np.zeros_like(y)
        S_list = self.site_phos_rates(t)
        for i in range(self.index.N):
            sl = self.index.protein_block_slice(i)
            block = y[sl]
            R = block[0]
            P = block[1]
            Psites = block[2:]
            # parameters for protein i
            Ai, Bi, Ci, Di = self.A_i[i], self.B_i[i], self.C_i[i], self.D_i[i]
            # per-site deg multipliers
            if self.index.n_sites_per_protein[i] > 0:
                rsi = self.r_site_list[i]
                kdeg_sites = (1.0 + rsi) * Di
                S_i = S_list[i]
            else:
                kdeg_sites = np.array([], dtype=float)
                S_i = np.array([], dtype=float)
            # equations
            dR = Ai - Bi * R
            # protein pool
            sumS = S_i.sum() if S_i.size else 0.0
            dP = Ci * R - (Di + sumS) * P + Psites.sum()
            # sites
            dPs = S_i * P - (1.0 + kdeg_sites) * Psites
            # write back
            dy[sl.start + 0] = dR
            dy[sl.start + 1] = dP
            if dPs.size:
                dy[sl.start + 2: sl.start + 2 + dPs.size] = dPs
        return dy

    def initial_state(self,
                      R0: float = 1.0,
                      P0: float = 1.0,
                      Psite0: float = 0.0) -> np.ndarray:
        y0 = np.zeros(self.index.state_dim, dtype=float)
        for i in range(self.index.N):
            sl = self.index.protein_block_slice(i)
            y0[sl.start + 0] = R0
            y0[sl.start + 1] = P0
            nsi = self.index.n_sites_per_protein[i]
            if nsi > 0:
                y0[sl.start + 2: sl.start + 2 + nsi] = Psite0
        return y0


# ------------------------------
# Simulation and outputs
# ------------------------------

def simulate_system(sys: GlobalPhosphoODE,
                    t_span: Tuple[float, float],
                    dt: float,
                    atol: float = 1e-8,
                    rtol: float = 1e-6,
                    method: str = "BDF") -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(t_span[0], t_span[1] + 1e-12, dt, dtype=float)
    y0 = sys.initial_state()
    sol = solve_ivp(fun=sys.rhs,
                    t_span=t_span,
                    y0=y0,
                    method=method,
                    t_eval=t_eval,
                    atol=atol,
                    rtol=rtol)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y.T  # shape (T, state_dim)


def export_states(index: NetworkIndex, t: np.ndarray, Y: np.ndarray, outdir: str):
    rows = []
    # Build a tidy table with columns: time, protein, state, value
    for i, prot in enumerate(index.proteins):
        sl = index.protein_block_slice(i)
        # names for states
        names = [f"R:{prot}", f"P:{prot}"]
        names += [f"Psite:{prot}:{site}" for site in index.sites[i]]
        block = Y[:, sl]
        for j, nm in enumerate(names):
            rows.append(pd.DataFrame({
                "time": t,
                "state": nm,
                "value": block[:, j]
            }))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(outdir, "global_states.csv"), index=False)


def export_observables_fc(index: NetworkIndex, t: np.ndarray, Y: np.ndarray, outdir: str):
    # total protein per protein = P_i + sum_s P_{i,s}
    # log2FC relative to t0
    obs = []
    for i, prot in enumerate(index.proteins):
        sl = index.protein_block_slice(i)
        P = Y[:, sl.start + 1]
        nsi = index.n_sites_per_protein[i]
        if nsi > 0:
            Ps = Y[:, sl.start + 2: sl.start + 2 + nsi].sum(axis=1)
        else:
            Ps = 0.0
        total = P + Ps
        # avoid log of zero
        total = np.maximum(total, 1e-12)
        fc = np.log2(total / total[0])
        obs.append(pd.DataFrame({"time": t, "protein": prot, "log2FC_total": fc}))
    df = pd.concat(obs, ignore_index=True)
    df.to_csv(os.path.join(outdir, "global_observables_fc.csv"), index=False)


# ------------------------------
# Main CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Global ODE for phosphorylation network with shared kinases.")
    ap.add_argument("--interaction", required=True, help="CSV file with Protein, Psite, Kinase columns.")
    ap.add_argument("--kinopt", required=True, help="XLSX file with kinase effects per Kinase.")
    ap.add_argument("--tfopt", default=None, help="XLSX file with TF/protein effects (optional).")
    ap.add_argument("--kinase-timecourse", default=None, help="CSV with columns: time,<kinase1>,<kinase2>,... (optional).")
    ap.add_argument("--tmax", type=float, default=120.0, help="Simulation end time (same units as timecourse if provided).")
    ap.add_argument("--dt", type=float, default=1.0, help="Output time step.")
    ap.add_argument("--output-dir", default="./out", help="Directory for outputs.")
    ap.add_argument("--atol", type=float, default=1e-8, help="ODE absolute tolerance.")
    ap.add_argument("--rtol", type=float, default=1e-6, help="ODE relative tolerance.")
    args = ap.parse_args()

    # 1) Load data
    interactions = load_interactions(args.interaction)
    kinopt = load_kinopt(args.kinopt)
    tfopt = load_tfopt(args.tfopt)

    # 2) Build indices and sparse maps
    index = NetworkIndex(interactions)
    if index.N == 0 or len(index.kinases) == 0:
        raise ValueError("Empty network after parsing interactions.")
    W_list = build_sparse_alpha_maps(interactions, index)

    # 3) Initialize parameters
    params = init_parameters(index, kinopt, tfopt, alpha_scale=0.2)
    params["alpha_list"] = init_alpha_from_W(params["alpha_list"], W_list, alpha_scale=0.2)

    # 4) Kinase input (constant by default; can be overridden by timecourse)
    # constants from kinopt effects, normalized to mean 1
    k_const = kinopt.set_index("kinase")["effect"].reindex(index.kinases).fillna(0.0).values.astype(float)
    # convert to positive levels and normalize
    k_const = np.log1p(np.exp(k_const))
    k_const /= (np.mean(k_const) + 1e-12)
    tc = load_kinase_timecourse(args.kinase_timecourse, index.kinases)
    kin_input = KinaseInput(index.kinases, k_const, tc)

    # 5) Build system and simulate
    system = GlobalPhosphoODE(index, W_list, params, kin_input)
    t, Y = simulate_system(system, (0.0, args.tmax), dt=args.dt, atol=args.atol, rtol=args.rtol, method="BDF")

    # 6) Export
    _ensure_dir(args.output_dir)
    export_states(index, t, Y, args.output_dir)
    export_observables_fc(index, t, Y, args.output_dir)

    # 7) Write a small manifest with model sizes
    manifest = {
        "proteins": index.N,
        "kinases": len(index.kinases),
        "total_site_states": index.total_site_states,
        "state_dim": index.state_dim,
        "t_points": len(t),
        "outputs": [
            os.path.join(args.output_dir, "global_states.csv"),
            os.path.join(args.output_dir, "global_observables_fc.csv")
        ]
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Simulation complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
