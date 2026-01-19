import numpy as np
from numba import njit, prange

from phoskintime_global.config import MODEL
from phoskintime_global.models import distributive_rhs, sequential_rhs, combinatorial_rhs
from phoskintime_global.solvers import adaptive_rk45_model01, adaptive_rk45_model2
from phoskintime_global.utils import time_bucket


def solve_custom(sys, y0, t_eval, rtol, atol):
    if MODEL == 2:
        build_S_cache_into(sys.S_cache, sys.W_indptr, sys.W_indices, sys.W_data, sys.kin_Kmat, sys.c_k)
        args_full = sys.odeint_args(sys.S_cache)
        args2 = args_full[:8] + args_full[9:]  # drop kin_grid
        return adaptive_rk45_model2(y0, t_eval, sys.kin_grid, args2, rtol=rtol, atol=atol)
    else:
        args = sys.odeint_args()
        return adaptive_rk45_model01(MODEL, y0, t_eval, sys.kin_grid, args, rtol=rtol, atol=atol)


@njit(cache=True, fastmath=True, nogil=True)
def csr_matvec(indptr, indices, data, x, n_rows):
    out = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        s = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            s += data[p] * x[indices[p]]
        out[i] = s
    return out


@njit(cache=True, fastmath=True, nogil=True)
def csr_matvec_into(out, indptr, indices, data, x, n_rows):
    for i in range(n_rows):
        s = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            s += data[p] * x[indices[p]]
        out[i] = s


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def build_S_cache_into(S_out, W_indptr, W_indices, W_data, kin_Kmat, c_k):
    n_rows = S_out.shape[0]
    n_bins = S_out.shape[1]
    for i in prange(n_rows):
        row_start = W_indptr[i]
        row_end = W_indptr[i + 1]
        for b in range(n_bins):
            s = 0.0
            for p in range(row_start, row_end):
                k = W_indices[p]
                s += W_data[p] * (kin_Kmat[k, b] * c_k[k])
            S_out[i, b] = s


@njit(cache=True, fastmath=True, nogil=True)
def kin_eval_step(t, grid, Kmat):
    if t <= grid[0]:
        return Kmat[:, 0].copy()
    if t >= grid[-1]:
        return Kmat[:, -1].copy()
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
        tf_deg,
):
    dy = np.zeros_like(y)

    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size):
        Kt[i] *= c_k[i]

    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)

    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        y_start = offset_y[i]
        ns = n_sites[i]
        tot = y[y_start + 1]
        for j in range(ns):
            tot += y[y_start + 2 + j]
        P_vec[i] = tot

    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        TF_inputs[i] /= tf_deg[i]

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
        tf_deg,
):
    dy = np.zeros_like(y)

    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size):
        Kt[i] *= c_k[i]

    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)

    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        y_start = offset_y[i]
        ns = n_sites[i]
        tot = y[y_start + 1]
        for j in range(ns):
            tot += y[y_start + 2 + j]
        P_vec[i] = tot

    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        TF_inputs[i] /= tf_deg[i]

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
        tf_deg,
        P_vec, TF_inputs
):
    dy = np.zeros_like(y)

    # pick bucket (stepwise hold)
    jb = time_bucket(t, kin_grid)

    # build protein totals into preallocated P_vec
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
        TF_inputs[i] /= tf_deg[i]

    # core dynamics using cached phosphorylation rates
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


if MODEL == 0:
    RHS_NB = rhs_nb_distributive
elif MODEL == 1:
    RHS_NB = rhs_nb_sequential
elif MODEL == 2:
    RHS_NB = rhs_nb_combinatorial
else:
    raise ValueError(f"Invalid MODEL={MODEL}. Expected 0 (distributive) or 1 (sequential) or 2 (combinatorial).")


def rhs_odeint(y, t, *args):
    return RHS_NB(y, t, *args)


@njit(cache=True, fastmath=True, nogil=True)
def fd_jacobian_nb_core_distributive(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg,
        eps=1e-8
):
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb_distributive(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg
    )

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
            tf_deg
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
        tf_deg,
        eps=1e-8
):
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb_sequential(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg
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
            tf_deg
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
        tf_deg,
        P_vec, TF_inputs,
        eps=1e-8
):
    n = y.size
    J = np.empty((n, n), dtype=np.float64)
    # P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    # TF_inputs = np.zeros(n_TF_rows, dtype=np.float64)

    f0 = rhs_nb_combinatorial(
        y, t,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg,
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
            tf_deg,
            P_vec, TF_inputs
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J


if MODEL == 0:
    FD_JAC_NB = fd_jacobian_nb_core_distributive
elif MODEL == 1:
    FD_JAC_NB = fd_jacobian_nb_core_sequential
elif MODEL == 2:
    FD_JAC_NB = fd_jacobian_nb_core_combinatorial
else:
    raise ValueError(f"Invalid MODEL={MODEL}. Expected 0 (distributive) or 1 (sequential) or 2 (combinatorial).")


def fd_jacobian_odeint(y, t, *args):
    y_arr = np.asarray(y, dtype=np.float64)
    return FD_JAC_NB(y_arr, t, *args)
