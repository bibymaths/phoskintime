"""
IVP Solver Interface Module.

This module provides factory functions that generate callable Right-Hand Side (RHS)
functions compatible with `scipy.integrate.solve_ivp`.

These factories bridge the gap between high-level Python solvers and the low-level,
JIT-compiled Numba kernels defined in `global_model.models`. They handle:
1.  **Type Safety:** Ensuring all arrays are C-contiguous and float64 for Numba.
2.  **Input Normalization:** Wrapping transcription factor (TF) inputs whether they are constants or time-dependent functions.
3.  **Closure Creation:** Returning a simple `fun(t, y)` that closes over all static model parameters, optimizing solver overhead.


"""

import numpy as np

from global_model.models import distributive_rhs, sequential_rhs, combinatorial_rhs, saturating_rhs

def _c(a, dtype=None):
    """
    Ensure contiguous arrays (helps Numba + reduces hidden copies).

    Numba functions run significantly faster on C-contiguous arrays due to
    better SIMD vectorization opportunities.
    """
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

    # If input is a static array, wrap it in a lambda to mimic a time-dependent function
    tf_const = _c(tf_input, np.float64)
    return lambda t, y=None: tf_const


def make_solve_ivp_fun_saturating(
        *,
        A_i, B_i, C_i, D_i, Dp_i, E_i,
        tf_scale,
        tf_input,  # callable(t) / callable(t,y) / ndarray
        S_all,
        offset_y, offset_s,
        n_sites,
):
    """
    Creates a `fun(t, y)` closure for the Saturating (Michaelis-Menten) Model.

    Args:
        A_i, B_i, C_i, D_i, Dp_i, E_i (np.ndarray): Kinetic parameters arrays.
        tf_scale (float): Global scaling factor for TF activity.
        tf_input: Transcription factor input (array or callable).
        S_all (np.ndarray): Pre-calculated Signaling drive matrix.
        offset_y, offset_s (np.ndarray): Index offsets for state vectors and sites.
        n_sites (np.ndarray): Number of phospho-sites per protein.

    Returns:
        callable: A function `fun(t, y) -> dy` compatible with ODE solvers.
    """
    # Enforce C-contiguity and correct types before creating the closure
    A_i = _c(A_i, np.float64);
    B_i = _c(B_i, np.float64);
    C_i = _c(C_i, np.float64);
    D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64);
    E_i = _c(E_i, np.float64)
    S_all = _c(S_all, np.float64)
    offset_y = _c(offset_y, np.int32);
    offset_s = _c(offset_s, np.int32);
    n_sites = _c(n_sites, np.int32)

    tfp = _wrap_tf_input(tf_input)

    def fun(t, y):
        # Calculate dynamic TF inputs or default to zero
        TF_inputs = tfp(t, y) if tfp is not None else np.zeros(A_i.shape[0], dtype=np.float64)
        TF_inputs = _c(TF_inputs, np.float64)

        dy = np.empty_like(y, dtype=np.float64)
        dy.fill(0.0)

        # Call the JIT-compiled kernel
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
    """
    Creates a `fun(t, y)` closure for the Distributive (Independent) Model.

    This model assumes phosphorylation sites on the same protein behave independently.
    """
    # Enforce C-contiguity and correct types
    A_i = _c(A_i, np.float64);
    B_i = _c(B_i, np.float64);
    C_i = _c(C_i, np.float64);
    D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64);
    E_i = _c(E_i, np.float64)
    S_all = _c(S_all, np.float64)
    offset_y = _c(offset_y, np.int32);
    offset_s = _c(offset_s, np.int32);
    n_sites = _c(n_sites, np.int32)

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
    """
    Creates a `fun(t, y)` closure for the Sequential Binding Model.

    This model enforces an order of phosphorylation (e.g., Site 1 must be phos before Site 2).
    """
    # Enforce C-contiguity and correct types
    A_i = _c(A_i, np.float64);
    B_i = _c(B_i, np.float64);
    C_i = _c(C_i, np.float64);
    D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64);
    E_i = _c(E_i, np.float64)
    S_all = _c(S_all, np.float64)
    offset_y = _c(offset_y, np.int32);
    offset_s = _c(offset_s, np.int32);
    n_sites = _c(n_sites, np.int32)

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
    """
    Creates a `fun(t, y)` closure for the Combinatorial Model.



    This model handles the full 2^n state space complexity using pre-computed transition tables.

    Args:
        ... (Standard params)
        S_cache (np.ndarray): Cached signaling matrix (pre-computed inputs).
        jb (int): Time bucket index (for static step evaluation).
        trans_*: Arrays defining the sparse state transition graph.

    Returns:
        callable: A function `fun(t, y) -> dy`.
    """
    # Enforce C-contiguity and correct types
    A_i = _c(A_i, np.float64);
    B_i = _c(B_i, np.float64);
    C_i = _c(C_i, np.float64);
    D_i = _c(D_i, np.float64)
    Dp_i = _c(Dp_i, np.float64);
    E_i = _c(E_i, np.float64)
    S_cache = _c(S_cache, np.float64)
    offset_y = _c(offset_y, np.int32);
    offset_s = _c(offset_s, np.int32)
    n_sites = _c(n_sites, np.int32);
    n_states = _c(n_states, np.int32)

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
        # CRITICAL: Combinatorial logic uses += accumulation for transitions.
        # We must zero-initialize dy to prevent garbage values.
        dy.fill(0.0)

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