import numpy as np

from global_model.models import distributive_rhs, sequential_rhs, combinatorial_rhs, saturating_rhs


def _c(a, dtype=None):
    """Ensure contiguous arrays (helps Numba + reduces hidden copies)."""
    return np.ascontiguousarray(a if dtype is None else np.asarray(a, dtype=dtype))


def _wrap_tf_input(tf_input):
    """
    Normalize TF input provider:
      - None => constant zeros (must be overwritten by caller if needed)
      - ndarray => constant TF inputs
      - callable(t) or callable(t, y) => time/state dependent TF inputs
    """
    if tf_input is None:
        return None

    if callable(tf_input):
        return tf_input

    tf_const = _c(tf_input, np.float64)
    return lambda t, y=None: tf_const


def make_solve_ivp_fun_saturating(
    *,
    A_i, B_i, C_i, D_i, Dp_i, E_i,
    tf_scale,
    tf_input,          # callable(t) / callable(t,y) / ndarray
    S_all,
    offset_y, offset_s,
    n_sites,
):
    A_i = _c(A_i, np.float64); B_i = _c(B_i, np.float64); C_i = _c(C_i, np.float64); D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64); E_i = _c(E_i, np.float64)
    S_all = _c(S_all, np.float64)
    offset_y = _c(offset_y, np.int32); offset_s = _c(offset_s, np.int32); n_sites = _c(n_sites, np.int32)

    tfp = _wrap_tf_input(tf_input)

    def fun(t, y):
        TF_inputs = tfp(t, y) if tfp is not None else np.zeros(A_i.shape[0], dtype=np.float64)
        TF_inputs = _c(TF_inputs, np.float64)

        dy = np.empty_like(y, dtype=np.float64)
        dy.fill(0.0)

        saturating_rhs(
            y, dy,
            A_i, B_i, C_i, D_i, Dp_i, E_i,
            float(tf_scale),
            TF_inputs,
            S_all,
            offset_y, offset_s,
            n_sites,
        )
        return dy

    return fun


def make_solve_ivp_fun_distributive(
    *,
    A_i, B_i, C_i, D_i, Dp_i, E_i,
    tf_scale,
    tf_input,
    S_all,
    offset_y, offset_s,
    n_sites,
):
    A_i = _c(A_i, np.float64); B_i = _c(B_i, np.float64); C_i = _c(C_i, np.float64); D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64); E_i = _c(E_i, np.float64)
    S_all = _c(S_all, np.float64)
    offset_y = _c(offset_y, np.int32); offset_s = _c(offset_s, np.int32); n_sites = _c(n_sites, np.int32)

    tfp = _wrap_tf_input(tf_input)

    def fun(t, y):
        TF_inputs = tfp(t, y) if tfp is not None else np.zeros(A_i.shape[0], dtype=np.float64)
        TF_inputs = _c(TF_inputs, np.float64)

        dy = np.empty_like(y, dtype=np.float64)
        dy.fill(0.0)

        distributive_rhs(
            y, dy,
            A_i, B_i, C_i, D_i, Dp_i, E_i,
            float(tf_scale),
            TF_inputs,
            S_all,
            offset_y, offset_s,
            n_sites,
        )
        return dy

    return fun


def make_solve_ivp_fun_sequential(
    *,
    A_i, B_i, C_i, D_i, Dp_i, E_i,
    tf_scale,
    tf_input,
    S_all,
    offset_y, offset_s,
    n_sites,
):
    A_i = _c(A_i, np.float64); B_i = _c(B_i, np.float64); C_i = _c(C_i, np.float64); D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64); E_i = _c(E_i, np.float64)
    S_all = _c(S_all, np.float64)
    offset_y = _c(offset_y, np.int32); offset_s = _c(offset_s, np.int32); n_sites = _c(n_sites, np.int32)

    tfp = _wrap_tf_input(tf_input)

    def fun(t, y):
        TF_inputs = tfp(t, y) if tfp is not None else np.zeros(A_i.shape[0], dtype=np.float64)
        TF_inputs = _c(TF_inputs, np.float64)

        dy = np.empty_like(y, dtype=np.float64)
        dy.fill(0.0)

        sequential_rhs(
            y, dy,
            A_i, B_i, C_i, D_i, Dp_i, E_i,
            float(tf_scale),
            TF_inputs,
            S_all,
            offset_y, offset_s,
            n_sites,
        )
        return dy

    return fun


def make_solve_ivp_fun_combinatorial(
    *,
    A_i, B_i, C_i, D_i, Dp_i, E_i,
    tf_scale,
    tf_input,
    S_cache, jb,
    offset_y, offset_s,
    n_sites, n_states,
    trans_from, trans_to, trans_site, trans_off, trans_n,
):
    A_i = _c(A_i, np.float64); B_i = _c(B_i, np.float64); C_i = _c(C_i, np.float64); D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64); E_i = _c(E_i, np.float64)
    S_cache = _c(S_cache, np.float64)
    offset_y = _c(offset_y, np.int32); offset_s = _c(offset_s, np.int32)
    n_sites = _c(n_sites, np.int32); n_states = _c(n_states, np.int32)

    trans_from = _c(trans_from, np.int32)
    trans_to = _c(trans_to, np.int32)
    trans_site = _c(trans_site, np.int32)
    trans_off = _c(trans_off, np.int32)
    trans_n = _c(trans_n, np.int32)

    tfp = _wrap_tf_input(tf_input)
    jb = int(jb)

    def fun(t, y):
        TF_inputs = tfp(t, y) if tfp is not None else np.zeros(A_i.shape[0], dtype=np.float64)
        TF_inputs = _c(TF_inputs, np.float64)

        dy = np.empty_like(y, dtype=np.float64)
        dy.fill(0.0)  # REQUIRED for combinatorial (it uses += in several places)

        combinatorial_rhs(
            y, dy,
            A_i, B_i, C_i, D_i, Dp_i, E_i,
            float(tf_scale),
            TF_inputs,
            S_cache,
            jb,
            offset_y, offset_s,
            n_sites, n_states,
            trans_from, trans_to, trans_site, trans_off, trans_n,
        )
        return dy

    return fun
