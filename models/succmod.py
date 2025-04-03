import numpy as np
from numba import njit
from scipy.integrate import odeint

from config.constants import NORMALIZE_MODEL_OUTPUT


@njit
def ode_core(y, A, B, C, D, S_rates, D_rates):
    R = y[0]
    P = y[1]
    num_psites = S_rates.shape[0]
    dR_dt = A - B * R
    dP_dt = C * R - D * P
    if num_psites > 0:
        dP_dt -= S_rates[0] * P
        dP_dt += y[2]
    dydt = np.empty_like(y)
    dydt[0] = dR_dt
    dydt[1] = dP_dt
    for i in range(num_psites):
        if num_psites == 1:
            dydt[2] = S_rates[0] * P - (1 + D_rates[0]) * y[2]
        else:
            if i == 0:
                dydt[2] = S_rates[0] * P - (1 + S_rates[1] + D_rates[0]) * y[2] + y[3]
            elif i < num_psites - 1:
                dydt[2 + i] = S_rates[i] * y[1 + i] - (1 + S_rates[i + 1] + D_rates[i]) * y[2 + i] + y[3 + i]
            else:
                dydt[2 + i] = S_rates[i] * y[1 + i] - (1 + D_rates[i]) * y[2 + i]
    return dydt

def ode_system(y, t, params, num_psites):
    A, B, C, D = params[0], params[1], params[2], params[3]
    S_rates = np.array([params[4 + i] for i in range(num_psites)])
    D_rates = np.array([params[4 + num_psites + i] for i in range(num_psites)])
    return ode_core(y, A, B, C, D, S_rates, D_rates)

def solve_ode(params, init_cond, num_psites, t):
    sol = np.asarray(odeint(ode_system, init_cond, t, args=(params, num_psites)))
    np.clip(sol, 0, None, out=sol)
    if NORMALIZE_MODEL_OUTPUT:
        norm_init = np.array(init_cond, dtype=sol.dtype)
        recip = 1.0 / norm_init
        sol *= recip[np.newaxis, :]
    P_fitted = sol[:, 2:].T
    return sol, P_fitted