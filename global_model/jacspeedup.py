"""
Numerical Solver Wrappers and JIT Kernels.

This module acts as the high-performance computational core for the simulation.
It bridges the gap between the high-level `System` definitions and the low-level
numerical integrators (Runge-Kutta methods).

Key responsibilities:
1.  **Solver Interface**: Provides `solve_custom` to dispatch the correct integration
    routine based on the selected kinetic model (Distributive, Sequential, etc.).
2.  **JIT Compilation**: Uses Numba to compile time-critical functions (RHS evaluations,
    matrix-vector multiplications) into machine code for speed.
3.  **Sparse Matrix Operations**: Implements fast custom kernels for calculating signaling
    inputs ($S = W \cdot K$) and transcriptional regulation ($TF_{in} = A_{tf} \cdot P$).
4.  **Jacobian Approximation**: Provides finite-difference routines for estimating
    the Jacobian matrix, which is essential for stiff solvers (though mostly used here for
    diagnostics or implicit stepping if enabled).


"""

import numpy as np
from numba import njit, prange

from global_model.config import MODEL
from global_model.models import distributive_rhs, sequential_rhs, combinatorial_rhs, saturating_rhs
from global_model.solvers import adaptive_rk45_model01, adaptive_rk45_model2
from global_model.utils import time_bucket


def solve_custom(sys, y0, t_eval, rtol, atol):
    """
    Dispatches the simulation to the appropriate adaptive Runge-Kutta solver based on the global MODEL type.

    This function handles the setup of model-specific arguments, such as building the
    phosphorylation cache for the combinatorial model (Model 2) or packing the
    argument tuples for the standard models (0, 1, 4).

    Args:
        sys (System): The system object containing parameters and network matrices.
        y0 (np.ndarray): Initial state vector at t=0.
        t_eval (np.ndarray): Time points where the solution is required.
        rtol (float): Relative tolerance for error control.
        atol (float): Absolute tolerance for error control.

    Returns:
        np.ndarray: The integrated solution matrix Y of shape (len(t_eval), n_states).
    """
    if MODEL == 2:
        # 
        # Model 2 (Combinatorial) is computationally heavy. We pre-calculate the
        # "S" matrix (Signaling drive per site per time-bucket) to avoid repeated
        # sparse matrix multiplies inside the tight integrator loop.
        build_S_cache_into(sys.S_cache, sys.W_indptr, sys.W_indices, sys.W_data, sys.kin_Kmat, sys.c_k)

        args_full = sys.odeint_args(sys.S_cache)
        # Model 2 args structure is different, ensuring driver_map is passed if present in tuple
        # args_full usually: (params..., kin_grid, S_cache, TF_mats..., offsets..., tf_deg, driver_map, work_arrays...)
        # We need to drop kin_grid (index 8) for the solver wrapper if it expects that.
        # Based on solver.py: args2 = (c_k... tf_scale, S_cache, ... driver_map, P_vec, TF_in)
        args2 = args_full[:8] + args_full[9:]
        return adaptive_rk45_model2(y0, t_eval, sys.kin_grid, args2, rtol=rtol, atol=atol)
    else:
        # Standard models (0: Distributive, 1: Sequential, 4: Saturating)
        # The arguments are packed directly from the system object.
        args = sys.odeint_args()
        return adaptive_rk45_model01(MODEL, y0, t_eval, sys.kin_grid, args, rtol=rtol, atol=atol)


@njit(cache=True, fastmath=True, nogil=True)
def csr_matvec(indptr, indices, data, x, n_rows):
    """
    Performs a fast sparse matrix-vector multiplication: $y = A \cdot x$.



    This is used to calculate:
    1. Signaling Input: $S_{all} = W_{global} \cdot Kt$ (Kinase -> Site)
    2. TF Input: $TF_{in} = TF_{matrix} \cdot P_{vec}$ (TF -> Gene)

    Args:
        indptr, indices, data: Arrays defining the CSR matrix A.
        x (np.ndarray): The dense vector to multiply.
        n_rows (int): Number of rows in A.

    Returns:
        np.ndarray: The result vector y.
    """
    out = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        s = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            # Dot product of row i with vector x
            s += data[p] * x[indices[p]]
        out[i] = s
    return out


@njit(cache=True, fastmath=True, nogil=True)
def csr_matvec_into(out, indptr, indices, data, x, n_rows):
    """
    In-place version of sparse matrix-vector multiplication to reduce memory allocation.
    Writes the result directly into `out`.
    """
    for i in range(n_rows):
        s = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            s += data[p] * x[indices[p]]
        out[i] = s


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def build_S_cache_into(S_out, W_indptr, W_indices, W_data, kin_Kmat, c_k):
    """
    Pre-computes the signaling drive 'S' for every site at every time bucket.

    For Model 2, calculating $S = W \cdot (K(t) \cdot c_k)$ at every micro-step
    is too slow. Instead, since $K(t)$ is discretized into buckets, we can
    pre-calculate the resulting S for each bucket.

    Parallelized using `prange` for performance.

    Args:
        S_out (np.ndarray): Output cache (n_sites, n_time_buckets).
        W_*: CSR arrays for the kinase-substrate interaction matrix.
        kin_Kmat (np.ndarray): Kinase activity profiles (n_kinases, n_buckets).
        c_k (np.ndarray): Optimized kinase activity multipliers.
    """
    n_rows = S_out.shape[0]
    n_bins = S_out.shape[1]
    for i in prange(n_rows):
        row_start = W_indptr[i]
        row_end = W_indptr[i + 1]
        # For every time bucket b...
        for b in range(n_bins):
            s = 0.0
            # ...compute the weighted sum of kinase activities acting on site i
            for p in range(row_start, row_end):
                k = W_indices[p]
                s += W_data[p] * (kin_Kmat[k, b] * c_k[k])
            S_out[i, b] = s


@njit(cache=True, fastmath=True, nogil=True)
def kin_eval_step(t, grid, Kmat):
    """
    Evaluates kinase activity at time t using step interpolation (Nearest Neighbor / Bucket).

    Args:
        t (float): Current simulation time.
        grid (np.ndarray): Time grid boundaries.
        Kmat (np.ndarray): Kinase data matrix.

    Returns:
        np.ndarray: Vector of kinase activities at time t.
    """
    if t <= grid[0]:
        return Kmat[:, 0].copy()
    if t >= grid[-1]:
        return Kmat[:, -1].copy()

    # Binary search to find the correct time bucket
    j = np.searchsorted(grid, t, side="right") - 1
    if j < 0:
        j = 0
    if j >= grid.size:
        j = grid.size - 1
    return Kmat[:, j].copy()


@njit(cache=True, fastmath=True, nogil=True)
def rhs_nb_distributive(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg, driver_map
):
    """
    Right-Hand Side (RHS) function for Model 0 (Distributive / Independent Binding).

    Calculates dy/dt = f(t, y) including:
    1.  **Kinase inputs**: Interpolates K(t) and maps to sites via sparse W.
    2.  **TF inputs**: Aggregates protein levels, maps to genes via sparse TF matrix,
        and applies Hill-like squashing.
    3.  **Local dynamics**: Calls `distributive_rhs` for the specific kinetic equations.
    """
    dy = np.zeros_like(y)

    # 1. Update Kinase Activity (Kt)
    # Get raw kinase profile at time t and scale by parameter c_k
    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size):
        Kt[i] *= c_k[i]

    # 2. Signaling Inputs (Kinase -> Site)
    # S_all[site_j] = Sum(W_jk * Kt_k)
    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)

    # 3. P_vec with LIVE-DRIVE logic
    # Constructs the vector of regulators (TFs).
    # If a regulator is a "Proxy" (driver_map >= 0), use the Kinase activity directly.
    # Otherwise, sum the total protein abundance (Unphos + Sum(Phos)).
    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        d_idx = driver_map[i]
        if d_idx >= 0:
            P_vec[i] = Kt[d_idx]
        else:
            y_start = offset_y[i]
            ns = n_sites[i]
            tot = y[y_start + 1]
            for j in range(ns):
                tot += y[y_start + 2 + j]
            P_vec[i] = tot

    # 4. TF Inputs & Squashing (Regulator -> Target Gene)
    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        # Apply normalization and saturation: x / (1 + |x|)
        val = TF_inputs[i] / tf_deg[i]
        TF_inputs[i] = val / (1.0 + abs(val))

    # 5. Compute ODE derivatives
    distributive_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs, S_all,
        offset_y, offset_s, n_sites
    )
    return dy


@njit(cache=True, fastmath=True, nogil=True)
def rhs_nb_sequential(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg, driver_map
):
    """
    Right-Hand Side (RHS) function for Model 1 (Sequential Binding).
    Almost identical to Distributive, but calls `sequential_rhs`.
    """
    dy = np.zeros_like(y)

    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size):
        Kt[i] *= c_k[i]

    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)

    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        d_idx = driver_map[i]
        if d_idx >= 0:
            P_vec[i] = Kt[d_idx]
        else:
            y_start = offset_y[i]
            ns = n_sites[i]
            tot = y[y_start + 1]
            for j in range(ns):
                tot += y[y_start + 2 + j]
            P_vec[i] = tot

    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        val = TF_inputs[i] / tf_deg[i]
        TF_inputs[i] = val / (1.0 + abs(val))

    sequential_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs, S_all,
        offset_y, offset_s, n_sites
    )
    return dy


@njit(cache=True, fastmath=True, nogil=True)
def rhs_nb_combinatorial(
        y,
        t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid,
        S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg, driver_map,
        P_vec, TF_inputs
):
    """
    Right-Hand Side (RHS) function for Model 2 (Combinatorial Binding).

    Differences from other models:
    1.  Uses `S_cache` instead of recomputing $S_{all}$.
    2.  Uses pre-allocated work arrays (`P_vec`, `TF_inputs`) passed as arguments.
    3.  Iterates over `n_states` (2^n_sites) rather than just linear sites.
    """
    dy = np.zeros_like(y)

    # pick bucket (stepwise hold) to index into S_cache
    jb = time_bucket(t, kin_grid)

    # build protein totals into preallocated P_vec
    # Note: Live-Drive for Model 2 is limited without Kt access here.
    # We fallback to standard simulation for now (summing all combinatorial states).
    for i in range(n_TF_rows):
        y_start = offset_y[i]
        nst = n_states[i]
        p0 = y_start + 1
        tot = 0.0
        for m in range(nst):
            tot += y[p0 + m]
        P_vec[i] = tot

    # TF_inputs = TF_mat * P_vec (in-place, no allocation)
    csr_matvec_into(TF_inputs, TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        val = TF_inputs[i] / tf_deg[i]
        TF_inputs[i] = val / (1.0 + abs(val))

    # core dynamics using cached phosphorylation rates and transition tables
    combinatorial_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs,
        S_cache, jb,
        offset_y, offset_s,
        n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n
    )
    return dy


@njit(cache=True, fastmath=True, nogil=True)
def rhs_nb_saturating(y, t, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
                      kin_grid, kin_Kmat, W_indptr, W_indices, W_data, n_W_rows,
                      TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y, offset_s, n_sites,
                      tf_deg, driver_map):
    """
    Right-Hand Side (RHS) for Model 4 (Saturating/Michaelis-Menten).
    Identical setup to distributive, but calls `saturating_rhs`.
    """
    dy = np.zeros_like(y)
    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size): Kt[i] *= c_k[i]
    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)
    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        dk = driver_map[i]
        if dk >= 0:
            P_vec[i] = Kt[dk]
        else:
            y_start = offset_y[i]
            ns = n_sites[i]
            tot = y[y_start + 1]
            for j in range(ns): tot += y[y_start + 2 + j]
            P_vec[i] = tot
    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        val = TF_inputs[i] / tf_deg[i]
        TF_inputs[i] = val  # Squashing handled in RHS for saturating model
    saturating_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all, offset_y, offset_s, n_sites)
    return dy


# Dispatch table: Select the correct RHS kernel at import time
if MODEL == 0:
    RHS_NB = rhs_nb_distributive
elif MODEL == 1:
    RHS_NB = rhs_nb_sequential
elif MODEL == 2:
    RHS_NB = rhs_nb_combinatorial
elif MODEL == 4:
    RHS_NB = rhs_nb_saturating
else:
    raise ValueError(f"Invalid MODEL={MODEL}. Expected 0, 1, 2, or 4.")


# --- THIS IS THE CHUNK YOU NEEDED RESTORED ---
def rhs_odeint(y, t, *args):
    """Standard python wrapper required for `scipy.integrate.odeint` or similar API."""
    return RHS_NB(y, t, *args)


@njit(cache=True, fastmath=True, nogil=True)
def fd_jacobian_nb_core_distributive(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg, driver_map,
        eps=1e-8
):
    """
    Finite Difference Jacobian calculation for Distributive Model.
    Approximates J[i,j] = df_i/dy_j using forward differences.
    Useful for testing or if an implicit solver is required without an analytical Jacobian.
    """
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    # Base evaluation
    f0 = rhs_nb_distributive(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg, driver_map
    )

    # Perturb each variable y[j]
    for j in range(n):
        y_pert = y.copy()
        aj = y[j]
        h = eps * (1.0 if abs(aj) < 1.0 else abs(aj))
        y_pert[j] = aj + h

        fj = rhs_nb_distributive(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
            kin_grid, kin_Kmat,
            W_indptr, W_indices, W_data, n_W_rows,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            offset_y, offset_s, n_sites,
            tf_deg, driver_map
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J


@njit(cache=True, fastmath=True, nogil=True)
def fd_jacobian_nb_core_sequential(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg, driver_map,
        eps=1e-8
):
    """Finite Difference Jacobian for Sequential Model."""
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb_sequential(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg, driver_map
    )

    for j in range(n):
        y_pert = y.copy()
        aj = y[j]
        h = eps * (1.0 if abs(aj) < 1.0 else abs(aj))
        y_pert[j] = aj + h

        fj = rhs_nb_sequential(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
            kin_grid, kin_Kmat,
            W_indptr, W_indices, W_data, n_W_rows,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            offset_y, offset_s, n_sites,
            tf_deg, driver_map
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J


@njit(cache=True, fastmath=True)
def fd_jacobian_nb_core_combinatorial(
        y,
        t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg, driver_map,
        P_vec, TF_inputs,
        eps=1e-8
):
    """Finite Difference Jacobian for Combinatorial Model."""
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb_combinatorial(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg, driver_map,
        P_vec, TF_inputs
    )

    for j in range(n):
        y_pert = y.copy()
        h = eps * (1.0 if abs(y[j]) < 1.0 else abs(y[j]))
        y_pert[j] += h

        fj = rhs_nb_combinatorial(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
            kin_grid, S_cache,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            offset_y, offset_s, n_sites, n_states,
            trans_from, trans_to, trans_site, trans_off, trans_n,
            tf_deg, driver_map,
            P_vec, TF_inputs
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J


@njit(cache=True, fastmath=True, nogil=True)
def fd_jacobian_nb_core_saturating(y, t, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_grid, kin_Kmat, W_indptr,
                                   W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                   offset_s, n_sites, tf_deg, driver_map, eps=1e-8):
    """Finite Difference Jacobian for Saturating Model."""
    n = y.size
    J = np.empty((n, n), dtype=np.float64)
    f0 = rhs_nb_saturating(y, t, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_grid, kin_Kmat, W_indptr, W_indices,
                           W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y, offset_s, n_sites,
                           tf_deg, driver_map)
    for j in range(n):
        y_pert = y.copy()
        h = eps * (1.0 if abs(y[j]) < 1.0 else abs(y[j]))
        y_pert[j] += h
        fj = rhs_nb_saturating(y_pert, t, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_grid, kin_Kmat, W_indptr,
                               W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                               offset_s, n_sites, tf_deg, driver_map)
        invh = 1.0 / h
        for i in range(n): J[i, j] = (fj[i] - f0[i]) * invh
    return J


# Dispatch table for Jacobian: Select the correct Jacobian kernel at import time
if MODEL == 0:
    FD_JAC_NB = fd_jacobian_nb_core_distributive
elif MODEL == 1:
    FD_JAC_NB = fd_jacobian_nb_core_sequential
elif MODEL == 2:
    FD_JAC_NB = fd_jacobian_nb_core_combinatorial
elif MODEL == 4:
    FD_JAC_NB = fd_jacobian_nb_core_saturating
else:
    raise ValueError(f"Invalid MODEL={MODEL}. Expected 0, 1, 2, or 4.")


def fd_jacobian_odeint(y, t, *args):
    """Python wrapper for the Jacobian function."""
    y_arr = np.asarray(y, dtype=np.float64)
    return FD_JAC_NB(y_arr, t, *args)
