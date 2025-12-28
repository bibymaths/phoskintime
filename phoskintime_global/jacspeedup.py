import numpy as np
from numba import njit

from phoskintime_global.config import MODEL
from phoskintime_global.models import distributive_rhs, sequential_rhs, combinatorial_rhs
from phoskintime_global.utils import time_bucket


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
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        A_i, B_i, C_i, D_i, E_i, tf_scale,
        TF_inputs, S_all,
        offset_y, offset_s, n_sites
    )
    return dy


@njit(cache=True, fastmath=True, nogil=True)
def rhs_nb_sequential(
        y, t,
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        A_i, B_i, C_i, D_i, E_i, tf_scale,
        TF_inputs, S_all,
        offset_y, offset_s, n_sites
    )
    return dy

@njit(cache=True, fastmath=True)
def rhs_nb_combinatorial(
    y,
    t,
    c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
    kin_grid,
    S_cache,
    TF_indptr, TF_indices, TF_data, n_TF_rows,
    offset_y, offset_s, n_sites, n_states,
    trans_from, trans_to, trans_site, trans_off, trans_n,
    tf_deg,
):
    dy = np.zeros_like(y)

    # pick bucket (stepwise hold)
    jb = time_bucket(t, kin_grid)

    # build protein totals (P_vec)
    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        y_start = offset_y[i]
        nst = n_states[i]
        p0 = y_start + 1
        tot = 0.0
        for m in range(nst):
            tot += y[p0 + m]
        P_vec[i] = tot

    # TF inputs
    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        TF_inputs[i] /= tf_deg[i]

    # core dynamics using cached phosphorylation rates
    combinatorial_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
            c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
            c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
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
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg,
        eps=1e-8
):
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb_combinatorial(
        y, t,
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg
    )

    for j in range(n):
        y_pert = y.copy()
        h = eps * (1.0 if abs(y[j]) < 1.0 else abs(y[j]))
        y_pert[j] += h

        fj = rhs_nb_combinatorial(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
            kin_grid, S_cache,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            offset_y, offset_s, n_sites, n_states,
            trans_from, trans_to, trans_site, trans_off, trans_n,
            tf_deg
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
