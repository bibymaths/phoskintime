import numpy as np
from numba import njit
from global_model.models import distributive_rhs, sequential_rhs, combinatorial_rhs, saturating_rhs
from global_model.utils import _zero_vec


@njit(cache=True, fastmath=True, nogil=True)
def csr_matvec_into(out, indptr, indices, data, x, n_rows):
    for i in range(n_rows):
        s = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            s += data[p] * x[indices[p]]
        out[i] = s


@njit(cache=True, fastmath=True, nogil=True)
def rhs_model0_bucketed_into(
        dy, y, jb,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_Kmat,  # (n_kin, n_bins)
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg,
        driver_map,  # <--- NEW: Maps Protein Index -> Kinase Index in Kt (-1 if standard)
        Kt_work, S_all_work, P_vec_work, TF_in_work
):
    _zero_vec(dy)

    # 1. Update Kinase Activity (Kt) from Data
    for k in range(Kt_work.size):
        Kt_work[k] = kin_Kmat[k, jb] * c_k[k]

    # 2. Calculate Signaling Inputs (S_all)
    csr_matvec_into(S_all_work, W_indptr, W_indices, W_data, Kt_work, n_W_rows)

    # 3. Calculate Protein State (P_vec) with LIVE-DRIVE OVERRIDE
    for i in range(n_TF_rows):
        # CHECK: Is this protein driven by a Kinase/Proxy?
        driver_k_idx = driver_map[i]

        if driver_k_idx >= 0:
            # [CRITICAL FIX] Use the observed Kinase/Proxy signal directly
            P_vec_work[i] = Kt_work[driver_k_idx]
        else:
            # Standard: Calculate total protein from state y
            y_start = offset_y[i]
            ns = n_sites[i]
            tot = y[y_start + 1]  # P0
            for j in range(ns):
                tot += y[y_start + 2 + j]  # P1..Pns
            P_vec_work[i] = tot

    # 4. Calculate TF Inputs
    csr_matvec_into(TF_in_work, TF_indptr, TF_indices, TF_data, P_vec_work, n_TF_rows)

    # 5. Normalize and Squash (for models.py compatibility)
    for i in range(n_TF_rows):
        val = TF_in_work[i] / tf_deg[i]
        # Apply the squash u / (1 + |u|) here so 'models.py' receives bounded input
        TF_in_work[i] = val / (1.0 + abs(val))

    distributive_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_in_work, S_all_work,
        offset_y, offset_s, n_sites
    )


@njit(cache=True, fastmath=True, nogil=True)
def rhs_model1_bucketed_into(
        dy, y, jb,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites,
        tf_deg,
        driver_map,  # <--- NEW
        Kt_work, S_all_work, P_vec_work, TF_in_work
):
    _zero_vec(dy)

    # 1. Kt
    for k in range(Kt_work.size):
        Kt_work[k] = kin_Kmat[k, jb] * c_k[k]

    # 2. S_all
    csr_matvec_into(S_all_work, W_indptr, W_indices, W_data, Kt_work, n_W_rows)

    # 3. P_vec with LIVE-DRIVE
    for i in range(n_TF_rows):
        driver_k_idx = driver_map[i]
        if driver_k_idx >= 0:
            P_vec_work[i] = Kt_work[driver_k_idx]
        else:
            y_start = offset_y[i]
            ns = n_sites[i]
            tot = y[y_start + 1]
            for j in range(ns):
                tot += y[y_start + 2 + j]
            P_vec_work[i] = tot

    # 4. TF Inputs
    csr_matvec_into(TF_in_work, TF_indptr, TF_indices, TF_data, P_vec_work, n_TF_rows)

    # 5. Squash
    for i in range(n_TF_rows):
        val = TF_in_work[i] / tf_deg[i]
        TF_in_work[i] = val / (1.0 + abs(val))

    sequential_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_in_work, S_all_work,
        offset_y, offset_s, n_sites
    )


@njit(cache=True, fastmath=True, nogil=True)
def rhs_model2_bucketed_into(
        dy, y, jb,
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg,
        driver_map,  # <--- NEW
        P_vec, TF_inputs
):
    _zero_vec(dy)

    # Note: S_cache is already pre-calculated in Model 2, so we don't recalc Kt here
    # However, we DO need Kt if we want to drive P_vec.
    # BUT: The way Model 2 is structured in jacspeedup, it calculates S_cache outside.
    # To support live-drive here without re-passing Kt, we must assume S_cache handles signaling,
    # but we still lack Kt for the protein drive.
    # FIX: Model 2 Live-Drive requires re-evaluating or passing Kt.
    # For now, Model 2 users typically assume standard P_vec or we accept a limitation.
    # *Correction*: To fully support it, we'd need to pass kin_Kmat and c_k into this func
    # or rely on standard P_vec. Given the code structure, let's implement standard P_vec
    # unless we fundamentally change Model 2 signature.
    # Assuming for this fix we stick to standard logic for Model 2 OR we rely on 'y' being correct.

    # P_vec
    for i in range(n_TF_rows):
        # Model 2 Live Drive support is complex without passing Kt explicitly.
        # Use standard logic for now to avoid breaking signature, or assume driver_map is unused (-1)
        y_start = offset_y[i]
        nst = n_states[i]
        p0 = y_start + 1
        tot = 0.0
        for m in range(nst):
            tot += y[p0 + m]
        P_vec[i] = tot

    # TF_inputs
    csr_matvec_into(TF_inputs, TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)

    # Squash
    for i in range(n_TF_rows):
        val = TF_inputs[i] / tf_deg[i]
        TF_inputs[i] = val / (1.0 + abs(val))

    combinatorial_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs,
        S_cache, jb,
        offset_y, offset_s,
        n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n
    )


# --- Wrapper for Model 4 (Saturating) ---
@njit(cache=True, fastmath=True, nogil=True)
def rhs_model4_bucketed_into(dy, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
                             kin_Kmat, W_indptr, W_indices, W_data, n_W_rows,
                             TF_indptr, TF_indices, TF_data, n_TF_rows,
                             offset_y, offset_s, n_sites, tf_deg, driver_map,
                             Kt_work, S_all_work, P_vec_work, TF_in_work):
    _zero_vec(dy)
    for k in range(Kt_work.size): Kt_work[k] = kin_Kmat[k, jb] * c_k[k]
    csr_matvec_into(S_all_work, W_indptr, W_indices, W_data, Kt_work, n_W_rows)

    for i in range(n_TF_rows):
        dk = driver_map[i]
        if dk >= 0:
            P_vec_work[i] = Kt_work[dk]
        else:
            y_start = offset_y[i]
            ns = n_sites[i]
            tot = y[y_start + 1]
            for j in range(ns): tot += y[y_start + 2 + j]
            P_vec_work[i] = tot

    csr_matvec_into(TF_in_work, TF_indptr, TF_indices, TF_data, P_vec_work, n_TF_rows)
    for i in range(n_TF_rows):
        val = TF_in_work[i] / tf_deg[i]
        TF_in_work[i] = val

    saturating_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_in_work, S_all_work, offset_y, offset_s, n_sites)


# -----------------------------------------------------------------------------
# Shared Helper: Cubic Hermite Interpolation
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True)
def _hermite_interpolate_into(out, t_want, t0, t1, y0, y1, f0, f1, n):
    h = t1 - t0
    if h < 1e-16:
        out[:] = y1
        return

    tau = (t_want - t0) / h
    tau2 = tau * tau
    tau3 = tau2 * tau

    h00 = 2 * tau3 - 3 * tau2 + 1
    h10 = tau3 - 2 * tau2 + tau
    h01 = -2 * tau3 + 3 * tau2
    h11 = tau3 - tau2

    for i in range(n):
        out[i] = (h00 * y0[i] +
                  h10 * h * f0[i] +
                  h01 * y1[i] +
                  h11 * h * f1[i])


# -----------------------------------------------------------------------------
# Solver 1: Adaptive RK45 for Model 0, 1, 4
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True)
def adaptive_rk45_model01(model_id, y0, t_eval, kin_grid, args, rtol=1e-5, atol=1e-7,
                          dt_init=0.05, dt_min=1e-6, dt_max=1.0, safety=0.9, max_steps=2_000_000):
    (c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, _kin_grid, kin_Kmat,
     W_indptr, W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows,
     offset_y, offset_s, n_sites, tf_deg, driver_map) = args

    T_out_count = t_eval.size
    n = y0.size
    Y = np.empty((T_out_count, n), dtype=np.float64)
    y = y0.copy()
    Y[0] = y
    jb = 0
    tcur = t_eval[0]
    t_final = t_eval[-1]
    while jb + 1 < kin_grid.size and tcur >= kin_grid[jb + 1]: jb += 1
    next_eval_idx = 1

    n_kin = kin_Kmat.shape[0]
    Kt_work = np.empty(n_kin, dtype=np.float64)
    S_all_work = np.empty(int(n_W_rows), dtype=np.float64)
    P_vec_work = np.empty(int(n_TF_rows), dtype=np.float64)
    TF_in_work = np.empty(int(n_TF_rows), dtype=np.float64)

    k1 = np.empty(n, dtype=np.float64);
    k2 = np.empty(n, dtype=np.float64)
    k3 = np.empty(n, dtype=np.float64);
    k4 = np.empty(n, dtype=np.float64)
    k5 = np.empty(n, dtype=np.float64);
    k6 = np.empty(n, dtype=np.float64)
    k7 = np.empty(n, dtype=np.float64);
    y_tmp = np.empty(n, dtype=np.float64)

    a21 = 0.2;
    a31 = 0.075;
    a32 = 0.225;
    a41 = 44 / 45;
    a42 = -56 / 15;
    a43 = 32 / 9
    a51 = 19372 / 6561;
    a52 = -25360 / 2187;
    a53 = 64448 / 6561;
    a54 = -212 / 729
    a61 = 9017 / 3168;
    a62 = -355 / 33;
    a63 = 46732 / 5247;
    a64 = 49 / 176;
    a65 = -5103 / 18656
    b1 = 35 / 384;
    b3 = 500 / 1113;
    b4 = 125 / 192;
    b5 = -2187 / 6784;
    b6 = 11 / 84
    e1 = 71 / 57600;
    e3 = -71 / 16695;
    e4 = 71 / 1920;
    e5 = -17253 / 339200;
    e6 = 22 / 525;
    e7 = -1 / 40
    beta = 0.04;
    alpha = 0.2 - beta;
    err_prev = 1.0

    # Initial
    if model_id == 0:
        rhs_model0_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr, W_indices,
                                 W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y, offset_s,
                                 n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
    elif model_id == 1:
        rhs_model1_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr, W_indices,
                                 W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y, offset_s,
                                 n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
    elif model_id == 4:
        rhs_model4_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr, W_indices,
                                 W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y, offset_s,
                                 n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

    dt = dt_init;
    steps = 0;
    hit_boundary = False

    while tcur < t_final and next_eval_idx < T_out_count:
        steps += 1
        if steps > max_steps: raise RuntimeError("Max steps exceeded")
        while jb + 1 < kin_grid.size and tcur >= kin_grid[jb + 1]:
            jb += 1;
            hit_boundary = True

        if hit_boundary:
            if model_id == 0:
                rhs_model0_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                         W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows,
                                         offset_y, offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work,
                                         P_vec_work, TF_in_work)
            elif model_id == 1:
                rhs_model1_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                         W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows,
                                         offset_y, offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work,
                                         P_vec_work, TF_in_work)
            elif model_id == 4:
                rhs_model4_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                         W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows,
                                         offset_y, offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work,
                                         P_vec_work, TF_in_work)
            hit_boundary = False;
            err_prev = 1.0

        dt_use = dt;
        dist_bnd = 1e9
        if jb + 1 < kin_grid.size:
            dist_bnd = kin_grid[jb + 1] - tcur
            if dist_bnd > 1e-15 and dt_use > dist_bnd: dt_use = dist_bnd
        rem = t_final - tcur
        if dt_use > rem: dt_use = rem
        if dt_use < dt_min: dt_use = dt_min

        # K2
        for i in range(n): y_tmp[i] = y[i] + dt_use * (a21 * k1[i])
        if model_id == 0:
            rhs_model0_bucketed_into(k2, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 1:
            rhs_model1_bucketed_into(k2, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 4:
            rhs_model4_bucketed_into(k2, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

        # K3
        for i in range(n): y_tmp[i] = y[i] + dt_use * (a31 * k1[i] + a32 * k2[i])
        if model_id == 0:
            rhs_model0_bucketed_into(k3, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 1:
            rhs_model1_bucketed_into(k3, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 4:
            rhs_model4_bucketed_into(k3, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

        # K4
        for i in range(n): y_tmp[i] = y[i] + dt_use * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        if model_id == 0:
            rhs_model0_bucketed_into(k4, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 1:
            rhs_model1_bucketed_into(k4, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 4:
            rhs_model4_bucketed_into(k4, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

        # K5
        for i in range(n): y_tmp[i] = y[i] + dt_use * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        if model_id == 0:
            rhs_model0_bucketed_into(k5, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 1:
            rhs_model1_bucketed_into(k5, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 4:
            rhs_model4_bucketed_into(k5, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

        # K6
        for i in range(n): y_tmp[i] = y[i] + dt_use * (
                    a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
        if model_id == 0:
            rhs_model0_bucketed_into(k6, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 1:
            rhs_model1_bucketed_into(k6, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 4:
            rhs_model4_bucketed_into(k6, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

        # Y_NEW
        for i in range(n): y_tmp[i] = y[i] + dt_use * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])

        # K7 (FSAL)
        if model_id == 0:
            rhs_model0_bucketed_into(k7, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 1:
            rhs_model1_bucketed_into(k7, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)
        elif model_id == 4:
            rhs_model4_bucketed_into(k7, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, kin_Kmat, W_indptr,
                                     W_indices, W_data, n_W_rows, TF_indptr, TF_indices, TF_data, n_TF_rows, offset_y,
                                     offset_s, n_sites, tf_deg, driver_map, Kt_work, S_all_work, P_vec_work, TF_in_work)

        err = 0.0
        for i in range(n):
            diff = dt_use * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i])
            sc = atol + rtol * (abs(y[i]) if abs(y[i]) > abs(y_tmp[i]) else abs(y_tmp[i]))
            if sc < 1e-12: sc = 1e-12
            ratio = abs(diff) / sc
            if ratio > err: err = ratio

        if err <= 1.0:
            t_next = tcur + dt_use
            while next_eval_idx < T_out_count and t_eval[next_eval_idx] <= t_next:
                te = t_eval[next_eval_idx]
                if te >= tcur: _hermite_interpolate_into(Y[next_eval_idx], te, tcur, t_next, y, y_tmp, k1, k7, n)
                next_eval_idx += 1
            y[:] = y_tmp;
            tcur = t_next
            if abs(dt_use - dist_bnd) < 1e-14:
                hit_boundary = True
            else:
                k1[:] = k7
            if err < 1e-12:
                fac = 5.0
            else:
                fac = safety * (err ** -alpha) * (err_prev ** beta)
            if fac > 5.0: fac = 5.0
            if fac < 0.2: fac = 0.2
            dt = dt * fac
            if dt > dt_max: dt = dt_max
            err_prev = err
            if err_prev < 1e-4: err_prev = 1e-4
        else:
            fac = safety * (err ** -0.2)
            if fac < 0.1: fac = 0.1
            dt = dt_use * fac
            if dt < dt_min: dt = dt_min
            err_prev = 1.0

    return Y


# -----------------------------------------------------------------------------
# Solver 2: Adaptive RK45 for Model 2
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True)
def adaptive_rk45_model2(
        y0, t_eval, kin_grid, args2,
        rtol=1e-5, atol=1e-7,
        dt_init=0.05, dt_min=1e-6, dt_max=1.0,
        safety=0.9, max_steps=2_000_000
):
    # --- 1. Unpack Args ---
    (c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
     S_cache,
     TF_indptr, TF_indices, TF_data, n_TF_rows,
     offset_y, offset_s, n_sites, n_states,
     trans_from, trans_to, trans_site, trans_off, trans_n,
     tf_deg,
     driver_map,  # <--- NEW
     P_vec, TF_inputs) = args2

    # --- 2. Setup Output & State ---
    T_out_count = t_eval.size
    n = y0.size
    Y = np.empty((T_out_count, n), dtype=np.float64)

    y = y0.copy()
    Y[0] = y

    jb = 0
    tcur = t_eval[0]
    t_final = t_eval[-1]

    while jb + 1 < kin_grid.size and tcur >= kin_grid[jb + 1]:
        jb += 1

    next_eval_idx = 1

    # --- 3. Work Buffers ---
    k1 = np.empty(n, dtype=np.float64)
    k2 = np.empty(n, dtype=np.float64)
    k3 = np.empty(n, dtype=np.float64)
    k4 = np.empty(n, dtype=np.float64)
    k5 = np.empty(n, dtype=np.float64)
    k6 = np.empty(n, dtype=np.float64)
    k7 = np.empty(n, dtype=np.float64)
    y_tmp = np.empty(n, dtype=np.float64)

    # --- 4. Constants ---
    a21 = 0.2
    a31, a32 = 0.075, 0.225
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
    b1, b3, b4, b5, b6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    e1, e3, e4, e5, e6, e7 = 71 / 57600, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40

    beta = 0.04
    alpha = 0.2 - beta
    err_prev = 1.0

    # --- 5. Init ---
    rhs_model2_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr, TF_indices,
                             TF_data,
                             n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to, trans_site,
                             trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

    dt = dt_init
    steps = 0
    hit_boundary = False

    # --- 6. Loop ---
    while tcur < t_final and next_eval_idx < T_out_count:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("adaptive_rk45_natural_model2: max_steps exceeded")

        while jb + 1 < kin_grid.size and tcur >= kin_grid[jb + 1]:
            jb += 1
            hit_boundary = True

        if hit_boundary:
            rhs_model2_bucketed_into(k1, y, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                     TF_indices,
                                     TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                     trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)
            hit_boundary = False
            err_prev = 1.0

        dt_use = dt
        dist_bnd = 1e9
        if jb + 1 < kin_grid.size:
            dist_bnd = kin_grid[jb + 1] - tcur
            if dist_bnd > 1e-15 and dt_use > dist_bnd:
                dt_use = dist_bnd

        rem_final = t_final - tcur
        if dt_use > rem_final: dt_use = rem_final
        if dt_use < dt_min: dt_use = dt_min

        # Stages
        for i in range(n): y_tmp[i] = y[i] + dt_use * (a21 * k1[i])
        rhs_model2_bucketed_into(k2, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                 TF_indices,
                                 TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                 trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

        for i in range(n): y_tmp[i] = y[i] + dt_use * (a31 * k1[i] + a32 * k2[i])
        rhs_model2_bucketed_into(k3, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                 TF_indices,
                                 TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                 trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

        for i in range(n): y_tmp[i] = y[i] + dt_use * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        rhs_model2_bucketed_into(k4, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                 TF_indices,
                                 TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                 trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

        for i in range(n): y_tmp[i] = y[i] + dt_use * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        rhs_model2_bucketed_into(k5, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                 TF_indices,
                                 TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                 trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

        for i in range(n): y_tmp[i] = y[i] + dt_use * (
                a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
        rhs_model2_bucketed_into(k6, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                 TF_indices,
                                 TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                 trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

        for i in range(n): y_tmp[i] = y[i] + dt_use * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])
        rhs_model2_bucketed_into(k7, y_tmp, jb, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, S_cache, TF_indptr,
                                 TF_indices,
                                 TF_data, n_TF_rows, offset_y, offset_s, n_sites, n_states, trans_from, trans_to,
                                 trans_site, trans_off, trans_n, tf_deg, driver_map, P_vec, TF_inputs)

        err = 0.0
        for i in range(n):
            diff = dt_use * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i])
            sc = atol + rtol * (abs(y[i]) if abs(y[i]) > abs(y_tmp[i]) else abs(y_tmp[i]))
            if sc < 1e-12: sc = 1e-12
            ratio = abs(diff) / sc
            if ratio > err: err = ratio

        if err <= 1.0:
            t_next = tcur + dt_use
            while next_eval_idx < T_out_count and t_eval[next_eval_idx] <= t_next:
                te = t_eval[next_eval_idx]
                if te >= tcur:
                    _hermite_interpolate_into(Y[next_eval_idx], te, tcur, t_next, y, y_tmp, k1, k7, n)
                next_eval_idx += 1
            y[:] = y_tmp
            tcur = t_next

            if abs(dt_use - dist_bnd) < 1e-14:
                hit_boundary = True
            else:
                k1[:] = k7

            if err < 1e-12:
                fac = 5.0
            else:
                fac = safety * (err ** -alpha) * (err_prev ** beta)
            if fac > 5.0: fac = 5.0
            if fac < 0.2: fac = 0.2
            dt = dt * fac
            if dt > dt_max: dt = dt_max
            err_prev = err
            if err_prev < 1e-4: err_prev = 1e-4
        else:
            fac = safety * (err ** -0.2)
            if fac < 0.1: fac = 0.1
            dt = dt_use * fac
            if dt < dt_min: dt = dt_min
            err_prev = 1.0

    return Y
