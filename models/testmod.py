import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT
TEST_MODEL = 'neg_feedback'  # Change this to the desired model type
NEGATIVE_FEEDBACK_CONSTANT = 0.1 # Feedback constant for negative feedback model
PROCESSIVITY = 0.1 # Processivity rate for semi-processive model
HILL_N = 4 # Hill coefficient for cooperative model
K_HALF = 0.5 # Half-maximal activation constant for cooperative model
# Works well for hill_n =< 0.5 and K_half >= 2

# ======================================================
# 1. Semi-processive / Hybrid Model
#
# Differential Equations:
#   dR/dt = A - B·R
#
#   dP/dt = C·R - D·P - p_proc·S₁·P + P₁
#
# For a single phosphorylation site (n = 1):
#   dP₁/dt = p_proc·S₁·P - (1 + D₁)·P₁
#
# For multiple sites (n > 1):
#   dP₁/dt = p_proc·S₁·P - (1 + p_proc·S₂ + D₁)·P₁ + P₂
#   dPᵢ/dt = p_proc·Sᵢ·Pᵢ₋₁ - (1 + p_proc·Sᵢ₊₁ + Dᵢ)·Pᵢ + Pᵢ₊₁, for i = 2, …, n-1
#   dPₙ/dt = p_proc·Sₙ·Pₙ₋₁ - (1 + Dₙ)·Pₙ
# (Here, P₁, P₂, …, Pₙ correspond to y[2], y[3], …, y[2+n-1].)
# ======================================================
@njit
def ode_core_semi_processive(y, A, B, C, D, S_rates, D_rates, p_proc):
    R = y[0]
    P = y[1]
    n = S_rates.shape[0]
    dR = A - B * R
    dP = C * R - D * P
    if n > 0:
        dP -= p_proc * S_rates[0] * P
        dP += y[2]
    dydt = np.empty_like(y)
    dydt[0] = dR
    dydt[1] = dP
    if n == 1:
        dydt[2] = p_proc * S_rates[0] * P - (1.0 + D_rates[0]) * y[2]
    elif n > 1:
        for i in range(n):
            if i == 0:
                dydt[2] = p_proc * S_rates[0] * P - (1.0 + p_proc * S_rates[1] + D_rates[0]) * y[2] + y[3]
            elif i < n - 1:
                dydt[2+i] = p_proc * S_rates[i] * y[1+i] - (1.0 + p_proc * S_rates[i+1] + D_rates[i]) * y[2+i] + y[3+i]
            else:
                dydt[2+i] = p_proc * S_rates[i] * y[1+i] - (1.0 + D_rates[i]) * y[2+i]
    return dydt

# ======================================================
# 2. Site-specific Binding Affinity Model
#
# Differential Equations:
#   dR/dt = A - B·R
#
#   dP/dt = C·R - D·P - S₁·P + P₁
#
# For n = 1:
#   dP₁/dt = S₁·P - (1 + D₁)·P₁
#
# For n > 1:
#   dP₁/dt = S₁·P - (1 + S₂ + D₁)·P₁ + P₂
#   dPᵢ/dt = Sᵢ·Pᵢ₋₁ - (1 + Sᵢ₊₁ + Dᵢ)·Pᵢ + Pᵢ₊₁, for i = 2, …, n-1
#   dPₙ/dt = Sₙ·Pₙ₋₁ - (1 + Dₙ)·Pₙ
# ======================================================
@njit
def ode_core_site_affinity(y, A, B, C, D, S_rates, D_rates):
    R = y[0]
    P = y[1]
    n = S_rates.shape[0]
    dR = A - B * R
    dP = C * R - D * P
    if n > 0:
        dP -= S_rates[0] * P
        dP += y[2]
    dydt = np.empty_like(y)
    dydt[0] = dR
    dydt[1] = dP
    if n == 1:
        dydt[2] = S_rates[0] * P - (1.0 + D_rates[0]) * y[2]
    elif n > 1:
        for i in range(n):
            if i == 0:
                dydt[2] = S_rates[0] * P - (1.0 + S_rates[1] + D_rates[0]) * y[2] + y[3]
            elif i < n - 1:
                dydt[2+i] = S_rates[i] * y[1+i] - (1.0 + S_rates[i+1] + D_rates[i]) * y[2+i] + y[3+i]
            else:
                dydt[2+i] = S_rates[i] * y[1+i] - (1.0 + D_rates[i]) * y[2+i]
    return dydt

# ======================================================
# 7. Negative Feedback Phosphorylation Model
#
# Differential Equations:
#   Let f = 1/(1 + k_fb·(P₁+...+Pₙ))
#
#   dR/dt = A - B·R
#   dP/dt = C·R - D·P - f·S₁·P + P₁
#
# For n = 1:
#   dP₁/dt = f·S₁·P - (1 + D₁)·P₁
#
# For n > 1:
#   dP₁/dt = f·S₁·P - (1 + f·S₂ + D₁)·P₁ + P₂
#   dPᵢ/dt = f·Sᵢ·Pᵢ₋₁ - (1 + f·Sᵢ₊₁ + Dᵢ)·Pᵢ + Pᵢ₊₁, for i = 2,…,n-1
#   dPₙ/dt = f·Sₙ·Pₙ₋₁ - (1 + Dₙ)·Pₙ
# ======================================================
@njit
def ode_core_negative_feedback(y, A, B, C, D, S_rates, D_rates, k_fb):
    R = y[0]
    P = y[1]
    n = S_rates.shape[0]
    # Compute total phosphorylation to define feedback factor
    total = 0.0
    for i in range(n):
        total += y[2+i]
    f = 1.0 / (1.0 + k_fb * total)
    dR = A - B * R
    dP = C * R - D * P - f * S_rates[0] * P + y[2]
    dydt = np.empty_like(y)
    dydt[0] = dR
    dydt[1] = dP
    if n == 1:
        dydt[2] = f * S_rates[0] * P - (1.0 + D_rates[0]) * y[2]
    elif n > 1:
        for i in range(n):
            if i == 0:
                dydt[2] = f * S_rates[0] * P - (1.0 + f * S_rates[1] + D_rates[0]) * y[2] + y[3]
            elif i < n - 1:
                dydt[2+i] = f * S_rates[i] * y[1+i] - (1.0 + f * S_rates[i+1] + D_rates[i]) * y[2+i] + y[3+i]
            else:
                dydt[2+i] = f * S_rates[i] * y[1+i] - (1.0 + D_rates[i]) * y[2+i]
    return dydt

# ======================================================
# 3. Crosstalk Between Sites Model
#
# Differential Equations:
#   dR/dt = A - B·R
#
#   dP/dt = C·R - D·P - S₁·P·(1 + φ) + P₁,
#            where φ = (P₁ + ... + Pₙ)/n is the average phosphorylation level.
#
# For n = 1:
#   dP₁/dt = S₁·P·(1 + φ) - (1 + D₁)·P₁
#
# For n > 1:
#   dP₁/dt = S₁·P·(1 + φ) - (1 + S₂·(1 + φ) + D₁)·P₁ + P₂
#   dPᵢ/dt = Sᵢ·Pᵢ₋₁·(1 + φ) - (1 + Sᵢ₊₁·(1 + φ) + Dᵢ)·Pᵢ + Pᵢ₊₁, for i = 2, …, n-1
#   dPₙ/dt = Sₙ·Pₙ₋₁·(1 + φ) - (1 + Dₙ)·Pₙ
# ======================================================
@njit
def ode_core_crosstalk(y, A, B, C, D, S_rates, D_rates):
    R = y[0]
    P = y[1]
    n = S_rates.shape[0]
    # Calculate average phosphorylation (feedback term)
    sum_P = 0.0
    for i in range(n):
        sum_P += y[2+i]
    mod_factor = sum_P / n if n > 0 else 0.0
    dR = A - B * R
    dP = C * R - D * P
    if n > 0:
        dP -= S_rates[0] * P * (1.0 + mod_factor)
        dP += y[2]
    dydt = np.empty_like(y)
    dydt[0] = dR
    dydt[1] = dP
    if n == 1:
        dydt[2] = S_rates[0] * P * (1.0 + mod_factor) - (1.0 + D_rates[0]) * y[2]
    elif n > 1:
        for i in range(n):
            if i == 0:
                dydt[2] = S_rates[0] * P * (1.0 + mod_factor) - (1.0 + S_rates[1] * (1.0 + mod_factor) + D_rates[0]) * y[2] + y[3]
            elif i < n - 1:
                dydt[2+i] = S_rates[i] * y[1+i] * (1.0 + mod_factor) - (1.0 + S_rates[i+1] * (1.0 + mod_factor) + D_rates[i]) * y[2+i] + y[3+i]
            else:
                dydt[2+i] = S_rates[i] * y[1+i] * (1.0 + mod_factor) - (1.0 + D_rates[i]) * y[2+i]
    return dydt

# ======================================================
# 10. Cooperative (Hill-type) Phosphorylation Model
#
# Differential Equations:
#   Let φ = (P₁ + ... + Pₙ)^hill_n / (K_half^hill_n + (P₁ + ... + Pₙ)^hill_n + ε)
#
#   dR/dt = A - B·R
#   dP/dt = C·R - D·P - S₁·P·φ + P₁
#
# For n = 1:
#   dP₁/dt = S₁·P·φ - (1 + D₁)·P₁
#
# For n > 1:
#   dP₁/dt = S₁·P·φ - (1 + S₂·φ + D₁)·P₁ + P₂
#   dPᵢ/dt = Sᵢ·Pᵢ₋₁·φ - (1 + Sᵢ₊₁·φ + Dᵢ)·Pᵢ + Pᵢ₊₁, for i = 2, …, n-1
#   dPₙ/dt = Sₙ·Pₙ₋₁·φ - (1 + Dₙ)·Pₙ
# ======================================================
@njit
def ode_core_cooperative(y, A, B, C, D, S_rates, D_rates, hill_n, K_half):
    R = y[0]
    P = y[1]
    n = S_rates.shape[0]
    sum_P = 0.0
    if n > 0:
        for i in range(n):
            sum_P += y[2+i]
    activation = (sum_P ** hill_n) / (K_half ** hill_n + sum_P ** hill_n + 1e-8)
    dR = A - B * R
    dP = C * R - D * P
    if n > 0:
        dP -= S_rates[0] * P * activation
        dP += y[2]
    dydt = np.empty_like(y)
    dydt[0] = dR
    dydt[1] = dP
    if n == 1:
        dydt[2] = S_rates[0] * P * activation - (1.0 + D_rates[0]) * y[2]
    elif n > 1:
        for i in range(n):
            if i == 0:
                dydt[2] = S_rates[0] * P * activation - (1.0 + S_rates[1] * activation + D_rates[0]) * y[2] + y[3]
            elif i < n - 1:
                dydt[2+i] = S_rates[i] * y[1+i] * activation - (1.0 + S_rates[i+1] * activation + D_rates[i]) * y[2+i] + y[3+i]
            else:
                dydt[2+i] = S_rates[i] * y[1+i] * activation - (1.0 + D_rates[i]) * y[2+i]
    return dydt

# ======================================================
# Unified ode_system switch
#
# The parameter vector is assumed to be arranged as:
#   [A, B, C, D, S₁, …, Sₙ, D₁, …, Dₙ]
#
# Extra parameters for each model are now hardcoded:
#   semi_processive: p_proc = 0.8
#   cooperative:      hill_n = 2, K_half = 0.5
# ======================================================
def ode_system(y, t, params, num_psites, model_type=TEST_MODEL):
    # Extract common parameters: A, B, C, D
    A = params[0]
    B = params[1]
    C = params[2]
    D = params[3]
    base_offset = 4
    S_rates = np.array([params[base_offset + i] for i in range(num_psites)])
    D_rates = np.array([params[base_offset + num_psites + i] for i in range(num_psites)])

    if model_type == 'semi_processive':
        p_proc = PROCESSIVITY
        return ode_core_semi_processive(y, A, B, C, D, S_rates, D_rates, p_proc)
    elif model_type == 'site_affinity':
        return ode_core_site_affinity(y, A, B, C, D, S_rates, D_rates)
    elif model_type == 'crosstalk':
        return ode_core_crosstalk(y, A, B, C, D, S_rates, D_rates)
    elif model_type == 'cooperative':
        hill_n = HILL_N
        K_half = K_HALF
        return ode_core_cooperative(y, A, B, C, D, S_rates, D_rates, hill_n, K_half)
    elif model_type == 'neg_feedback':
        k_fb = NEGATIVE_FEEDBACK_CONSTANT
        return ode_core_negative_feedback(y, A, B, C, D, S_rates, D_rates, k_fb)
    else:
        raise ValueError("Unknown model_type: " + model_type)

# ======================================================
# ODE Solver Wrapper
#
# Solves the ODE using odeint, clips negative values,
# optionally normalizes the output, and extracts the
# phosphorylated states (always starting at index 2).
# ======================================================
def solve_ode(params, init_cond, num_psites, t, model_type=TEST_MODEL):
    sol = np.asarray(odeint(ode_system, init_cond, t, args=(params, num_psites, model_type)))
    np.clip(sol, 0, None, out=sol)
    if NORMALIZE_MODEL_OUTPUT:
        norm_init = np.array(init_cond, dtype=sol.dtype)
        recip = 1.0 / norm_init
        sol *= recip[np.newaxis, :]
    # Phosphorylated states always start at index 2.
    p_index = 2
    P_fitted = sol[:, p_index:].T
    return sol, P_fitted