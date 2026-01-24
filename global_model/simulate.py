"""
Simulation and Measurement Extraction Module.

This module orchestrates the numerical integration of the ODE system and processes
the raw state trajectories into biologically meaningful observables (Fold Changes).

**Key Responsibilities:**
1.  **Solver Wrapper:** Abstracting the choice between `scipy.integrate.odeint` (LSODA)
    and a custom Numba-accelerated RK45 solver.
2.  **State Aggregation:** Converting raw state vectors $Y$ into:
    * **Total Protein:** Sum of unphosphorylated and phosphorylated forms.
    * **Site Phosphorylation:** Sum of specific phospho-states (handling combinatorial logic if needed).
    * **RNA:** Direct extraction from state vector.
3.  **Data Alignment:** Interpolating or slicing the simulated timepoints to match experimental grids.


"""

import warnings
import numpy as np
import pandas as pd
from scipy.integrate import odeint, ODEintWarning

# Suppress warnings from odeint that might flood logs during large optimization runs
warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", message="Excess work done on this call")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from global_model.config import MODEL, USE_CUSTOM_SOLVER
from global_model.jacspeedup import fd_jacobian_odeint, rhs_odeint, build_S_cache_into, solve_custom


def simulate_odeint(sys, t_eval, rtol, atol, mxstep):
    """
    Runs the ODE solver for the given system and timepoints.

    Dispatches to either a custom Numba RK45 solver (faster for small/stiff systems with JIT)
    or SciPy's LSODA (robust standard).

    Args:
        sys (System): The system object containing parameters and topology.
        t_eval (np.ndarray): Time points to return the solution at.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        mxstep (int): Maximum number of internal steps per time interval.

    Returns:
        np.ndarray: The solution matrix $Y$ of shape (len(t_eval), state_dim).
    """
    # Ensure y0 and t_eval are C-contiguous float64 for Numba compatibility
    y0 = sys.y0().astype(np.float64, copy=False)
    t_eval = t_eval.astype(np.float64)

    if USE_CUSTOM_SOLVER:
        # Use the Numba-accelerated Adaptive RK45
        xs = solve_custom(sys, y0, t_eval, rtol=rtol, atol=atol)
        return np.ascontiguousarray(xs, dtype=np.float64)

    # Standard SciPy odeint path (LSODA method)
    if MODEL == 2:
        # For combinatorial models, pre-compute the S-matrix cache to speed up RHS
        build_S_cache_into(sys.S_cache, sys.W_indptr, sys.W_indices, sys.W_data, sys.kin_Kmat, sys.c_k)
        args = sys.odeint_args(sys.S_cache)
    else:
        # For standard models, pack arguments directly
        args = sys.odeint_args()

    xs = odeint(
        rhs_odeint,
        y0,
        t_eval,
        args=args,
        Dfun=fd_jacobian_odeint,  # Use finite-difference Jacobian for stiff solver stability
        col_deriv=False,
        rtol=rtol,
        atol=atol,
        mxstep=mxstep,
    )
    return np.ascontiguousarray(xs, dtype=np.float64)


def simulate_and_measure(sys, idx, t_points_p, t_points_r, t_points_pho):
    """
    Simulates the system and extracts 'Fold Change' (FC) predictions aligned with data.



    Process:
    1.  **Union Grid:** Creates a master time grid containing all experimental timepoints.
    2.  **Simulate:** Integrates the system once over this master grid.
    3.  **Extract & Normalize:**
        -   Calculates raw observables (e.g., Total Protein = Unphos + Phos).
        -   Normalizes by the value at the baseline timepoint (t=0 for protein/phospho, t=4 for RNA).
    4.  **Slice:** Filters the result to match the specific timepoints requested for each modality.

    Args:
        sys: System object.
        idx: Index object (topology map).
        t_points_*: Arrays of timepoints for Protein, RNA, and Phospho data.

    Returns:
        tuple: (df_prot, df_rna, df_phos) - Pandas DataFrames with columns [protein, time, pred_fc].
    """
    # 1. Create master time grid
    times = np.unique(np.concatenate([t_points_p, t_points_r, t_points_pho]).astype(np.float64))

    # 2. Run simulation
    Y = simulate_odeint(sys, times, rtol=1e-5, atol=1e-7, mxstep=5000)

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
