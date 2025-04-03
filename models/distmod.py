
import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT

@njit
def ode_core(y, A, B, C, D, S_rates, D_rates): 
    R = y[0]
    P = y[1]
    n = S_rates.shape[0]
    dydt = np.empty_like(y)
    dydt[0] = A - B * R
    sum_S = 0.0
    # sum_S is the sum of S_rates
    for i in range(n):
        sum_S += S_rates[i]
    sum_P_sites = 0.0
    # sum_P_sites is the sum of P sites
    for i in range(n):
        sum_P_sites += y[2 + i]
    # dydt[1] is the rate of change of P
    dydt[1] = C * R - (D + sum_S) * P + sum_P_sites
    for i in range(n): 
        # dydt[2 + i] is the rate of change of each P site
        dydt[2 + i] = S_rates[i] * P - (1.0 + D_rates[i]) * y[2 + i]
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

