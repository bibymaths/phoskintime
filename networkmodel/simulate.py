"""Simulate a System with Diffrax and extract protein, RNA, and phospho measurement tables; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.jax_backend."""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from networkmodel.config import MODEL, ODE_ABS_TOL, ODE_REL_TOL, ODE_MAX_STEPS
from networkmodel.backend import DiffraxSolverConfig, make_networkmodel_rhs, solve_diffrax


def simulate_diffrax(sys, t_eval, rtol=None, atol=None, max_steps=None, solver_name="Kvaerno4"):
    """Simulate a System over requested time points with Diffrax
    
    Args:
        sys: Input value used by this routine.
        t_eval: Input value used by this routine.
        rtol: Input value used by this routine.
        atol: Input value used by this routine.
        max_steps: Input value used by this routine.
        solver_name: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    y0 = np.asarray(sys.y0(), dtype=np.float64)
    cfg = DiffraxSolverConfig(
        solver_name=solver_name,
        rtol=float(ODE_REL_TOL if rtol is None else rtol),
        atol=float(ODE_ABS_TOL if atol is None else atol),
        max_steps=int(ODE_MAX_STEPS if max_steps is None else max_steps),
        root_max_steps=20,
    )
    params = (sys.c_k, sys.A_i, sys.B_i, sys.C_i, sys.D_i, sys.Dp_i, sys.E_i,
              np.asarray([sys.tf_scale], dtype=np.float64))
    rhs = getattr(sys, "_cached_jax_rhs", None)
    if rhs is None:
        rhs = make_networkmodel_rhs(sys)
        sys._cached_jax_rhs = rhs

    return np.asarray(
        solve_diffrax(
            y0,
            np.asarray(t_eval, dtype=np.float64),
            params=params,
            rhs=rhs,
            config=cfg,
        ),
        dtype=np.float64,
    )


def simulate_and_measure(sys, idx, t_points_p, t_points_r, t_points_pho):
    """Simulate a System and return measured output tables
    
    Args:
        sys: Input value used by this routine.
        idx: Input value used by this routine.
        t_points_p: Input value used by this routine.
        t_points_r: Input value used by this routine.
        t_points_pho: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    # 1. Create master time grid
    times = np.unique(np.concatenate([t_points_p, t_points_r, t_points_pho]).astype(np.float64))

    # 2. Run simulation
    Y = simulate_diffrax(sys, times, rtol=1e-5, atol=1e-7, max_steps=5000)

    # Helper to find index of a specific time (for normalization baseline)
    def _bidx(t0: float) -> int:
        return int(np.argmin(np.abs(times - float(t0))))

    prot_b = _bidx(0.0)
    rna_b = _bidx(4.0)  # RNA often normalized to a later baseline if t=0 is noisy or absent
    pho_b = _bidx(0.0)

    rows_p, rows_r, rows_pho = [], [], []

    # 3. Iterate over every protein to extract observables
    for i, gene in enumerate(idx.proteins):
        st = int(idx.offset_y[i])

        # --- RNA ---
        # State index 'st' is always RNA
        R = Y[:, st]
        fc_r = np.maximum(R, 1e-12) / np.maximum(R[rna_b], 1e-12)
        rows_r.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_r}))

        if MODEL == 2:
            # --- Combinatorial Model Extraction ---
            ns = int(idx.n_states[i])
            n_sites = int(idx.n_sites[i])
            p0 = st + 1

            # Total Protein: Sum of all 2^n states
            states = Y[:, p0:p0 + ns]  # (T, ns)
            tot = states.sum(axis=1)  # (T,)
            fc_p = np.maximum(tot, 1e-12) / np.maximum(tot[prot_b], 1e-12)
            rows_p.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_p}))

            # Phospho Sites: Bitwise aggregation
            # We map states to sites using a matrix multiplication (State x Bitmask)
            if n_sites > 0:
                m = np.arange(ns, dtype=np.uint32)[:, None]
                j = np.arange(n_sites, dtype=np.uint32)[None, :]
                bits = ((m >> j) & 1).astype(np.float64)  # (ns, n_sites)
                pho_sites = states @ bits  # (T, n_sites)

                for s_idx, psite in enumerate(idx.sites[i]):
                    sig = pho_sites[:, s_idx]
                    fc = np.maximum(sig, 1e-12) / np.maximum(sig[pho_b], 1e-12)
                    rows_pho.append(pd.DataFrame({
                        "protein": gene, "psite": psite, "time": times, "pred_fc": fc
                    }))

        else:
            # --- Standard Model Extraction (Distributive/Sequential) ---
            ns = int(idx.n_sites[i])

            P0 = Y[:, st + 1]  # Unphosphorylated
            if ns > 0:
                P_sites = Y[:, st + 2: st + 2 + ns]  # (T, ns)
                pho_total = P_sites.sum(axis=1)
            else:
                P_sites = None
                pho_total = np.zeros_like(P0)

            # Total Protein
            tot = P0 + pho_total
            fc_p = np.maximum(tot, 1e-12) / np.maximum(tot[prot_b], 1e-12)
            rows_p.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_p}))

            # Phospho Sites
            if P_sites is not None:
                for s_idx, psite in enumerate(idx.sites[i]):
                    sig = P_sites[:, s_idx]
                    fc = np.maximum(sig, 1e-12) / np.maximum(sig[pho_b], 1e-12)
                    rows_pho.append(pd.DataFrame({
                        "protein": gene, "psite": psite, "time": times, "pred_fc": fc
                    }))

    # 4. Assemble DataFrames
    df_p = pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame(columns=["protein", "time", "pred_fc"])
    df_r = pd.concat(rows_r, ignore_index=True) if rows_r else pd.DataFrame(columns=["protein", "time", "pred_fc"])
    df_pho = pd.concat(rows_pho, ignore_index=True) if rows_pho else pd.DataFrame(
        columns=["protein", "psite", "time", "pred_fc"])

    # 5. Filter to requested timepoints
    tp = np.asarray(t_points_p, dtype=np.float64)
    tr = np.asarray(t_points_r, dtype=np.float64)
    tph = np.asarray(t_points_pho, dtype=np.float64)

    if not df_p.empty:
        df_p = df_p[df_p["time"].isin(tp)]
    if not df_r.empty:
        df_r = df_r[df_r["time"].isin(tr)]
    if not df_pho.empty:
        df_pho = df_pho[df_pho["time"].isin(tph)]

    return df_p, df_r, df_pho
