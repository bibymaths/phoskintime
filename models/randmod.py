import numpy as np
from numba import njit
from scipy.integrate import odeint

def prepare_vectorized_arrays(num_sites):
    num_states = 2 ** num_sites - 1
    binary_states = np.empty((num_states, num_sites), dtype=np.int32)
    for i in range(num_states):
        state = i + 1
        for j in range(num_sites):
            binary_states[i, j] = 1 if (state & (1 << j)) != 0 else 0
    PHOSPHO_TARGET = -np.ones((num_states, num_sites), dtype=np.int32)
    for i in range(num_states):
        state = i + 1
        for j in range(num_sites):
            if binary_states[i, j] == 0:
                target_state = state | (1 << j)
                if target_state <= num_states:
                    PHOSPHO_TARGET[i, j] = target_state - 1
                else:
                    PHOSPHO_TARGET[i, j] = -1
            else:
                PHOSPHO_TARGET[i, j] = -1
    DEPHOSPHO_TARGET = -np.ones((num_states, num_sites), dtype=np.int32)
    for i in range(num_states):
        state = i + 1
        for j in range(num_sites):
            if binary_states[i, j] == 1:
                lower_state = state & ~(1 << j)
                if lower_state == 0:
                    DEPHOSPHO_TARGET[i, j] = -2
                else:
                    DEPHOSPHO_TARGET[i, j] = lower_state - 1
            else:
                DEPHOSPHO_TARGET[i, j] = -1
    return binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET

@njit
def ode_system(y, t, A, B, C, D, num_sites, binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET, *params):
    num_states = 2 ** num_sites - 1
    S = np.empty(num_sites)
    for j in range(num_sites):
        S[j] = params[j]
    D_deg = np.empty(num_states)
    for i in range(num_states):
        D_deg[i] = params[num_sites + i]
    R = y[0]
    P = y[1]
    X = y[2:]
    dR_dt = A - B * R
    sum_S = 0.0
    for j in range(num_sites):
        sum_S += S[j]
    gain_1site = 0.0
    for i in range(num_states):
        cnt = 0
        for j in range(num_sites):
            cnt += binary_states[i, j]
        if cnt == 1:
            gain_1site += X[i]
    dP_dt = C * R - D * P - sum_S * P + gain_1site
    dX_dt = np.zeros(num_states)
    for i in range(num_states):
        for j in range(num_sites):
            if binary_states[i, j] == 0:
                target = PHOSPHO_TARGET[i, j]
                if target >= 0:
                    dX_dt[target] += S[j] * X[i]
    for i in range(num_states):
        loss = 0.0
        for j in range(num_sites):
            if binary_states[i, j] == 0:
                loss += S[j]
        dX_dt[i] -= loss * X[i]
    for i in range(num_states):
        cnt = 0
        for j in range(num_sites):
            cnt += binary_states[i, j]
        dX_dt[i] -= cnt * X[i]
    for i in range(num_states):
        for j in range(num_sites):
            if binary_states[i, j] == 1:
                lower = DEPHOSPHO_TARGET[i, j]
                if lower == -2:
                    dP_dt += S[j] * X[i]
                elif lower >= 0:
                    dX_dt[lower] += S[j] * X[i]
    for i in range(num_states):
        dX_dt[i] -= D_deg[i] * X[i]
    dydt = np.empty(2 + num_states)
    dydt[0] = dR_dt
    dydt[1] = dP_dt
    for i in range(num_states):
        dydt[2 + i] = dX_dt[i]
    return dydt


def solve_ode(popt, initial_conditions, num_psites, time_points):
    binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET = prepare_vectorized_arrays(num_psites)
    ode_params = popt
    A_val, B_val, C_val, D_val = ode_params[:4]
    remaining = ode_params[4:]
    sol = odeint(ode_system, initial_conditions, time_points,
                 args=(A_val, B_val, C_val, D_val, num_psites,
                       binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET, *remaining))
    np.clip(sol, 0, None, out=sol)
    norm_ic = np.array(initial_conditions, dtype=sol.dtype)
    recip = 1.0 / norm_ic
    sol *= recip[np.newaxis, :]
    if num_psites > 1:
        P_fitted = sol[:, 2:2 + num_psites].T
    else:
        P_fitted = sol[:, 2]
    return sol, P_fitted
