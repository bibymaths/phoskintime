import numpy as np
from numba import njit

from phoskintime_global.models import distributive_rhs


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def kin_eval_step(t, grid, Kmat):
    """
    Stepwise hold (no interpolation), matching your KinaseInput.eval behavior.
    Returns vector K(t) of size n_kinases.
    """
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


@njit(cache=True, fastmath=True)
def rhs_nb(
        y,
        t,
        # params
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        # kinase input (stepwise)
        kin_grid, kin_Kmat,
        # W_global CSR (rows = total_sites, cols = n_kinases)
        W_indptr, W_indices, W_data, n_W_rows,
        # TF CSR (rows = n_proteins, cols = n_proteins)
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        # maps
        p_indices,
        offset_y, offset_s, n_sites,
        tf_deg,
):
    dy = np.zeros_like(y)

    # Kinase vector at time t, scaled by c_k
    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size):
        Kt[i] *= c_k[i]

    # Site phosphorylation rates S_all = W_global * Kt   (len = total_sites)
    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)

    # TF inputs per protein from protein abundance P_vec
    # P_vec is y at the per-protein protein index positions (length = n_proteins)
    P_vec = np.zeros(n_TF_rows, dtype=np.float64)

    for i in range(n_TF_rows):
        y_start = offset_y[i]
        ns = n_sites[i]

        # unphosphorylated protein
        tot = y[y_start + 1]

        # add all phosphorylated states
        for j in range(ns):
            tot += y[y_start + 2 + j]

        P_vec[i] = tot

    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)

    for i in range(n_TF_rows):
        TF_inputs[i] /= tf_deg[i]
        # TF_inputs[i] /= np.sqrt(tf_deg[i])

    # Core dynamics (your hot loop)
    distributive_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, E_i, tf_scale,
        TF_inputs, S_all,
        offset_y, offset_s, n_sites
    )
    return dy


def rhs_odeint(y, t, *args):
    # odeint expects f(y, t, ...)
    return rhs_nb(y, t, *args)


def fd_jacobian_odeint(y, t, *args):
    # odeint expects J(y, t, ...) with J[i, j] = df_i/dy_j
    y_arr = np.asarray(y, dtype=np.float64)
    return fd_jacobian_nb_core(y_arr, t, *args)


@njit(cache=True, fastmath=True)
def fd_jacobian_nb_core(
        y,
        t,
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        p_indices,
        offset_y, offset_s, n_sites,
        tf_deg,
        eps=1e-8
):
    """
    Forward-diff Jacobian of rhs_nb wrt y: J[i, j] = d f_i / d y_j
    """
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb(
        y, t,
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        p_indices,
        offset_y, offset_s, n_sites,
        tf_deg
    )

    for j in range(n):
        y_pert = y.copy()
        h = eps * (1.0 if abs(y[j]) < 1.0 else abs(y[j]))
        y_pert[j] += h

        fj = rhs_nb(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
            kin_grid, kin_Kmat,
            W_indptr, W_indices, W_data, n_W_rows,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            p_indices,
            offset_y, offset_s, n_sites,
            tf_deg
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J
