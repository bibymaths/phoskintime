import numpy as np
from scipy.optimize import minimize

from config.logging_config import setup_logger
logger = setup_logger(__name__)

def initial_condition(num_psites: int) -> list:
    A, B, C, D = 1, 1, 1, 1
    S_rates = np.ones(num_psites)
    D_rates = np.ones(num_psites)

    def steady_state_equations(y):
        R, P, *P_sites = y
        dR_dt = A - B * R
        dP_dt = C * R - (D + np.sum(S_rates)) * P + np.sum(P_sites)
        dP_sites_dt = [S_rates[i] * P - (1 + D_rates[i]) * P_sites[i] for i in range(num_psites)]
        return [dR_dt, dP_dt] + dP_sites_dt

    y0_guess = np.ones(num_psites + 2)
    bounds_local = [(1e-6, None)] * (num_psites + 2)
    result = minimize(lambda y: 0, y0_guess, method='SLSQP', bounds=bounds_local,
                      constraints={'type': 'eq', 'fun': steady_state_equations})
    logger.info("Steady-State conditions calculated")
    if result.success:
        return result.x.tolist()
    else:
        raise ValueError("Failed to find steady-state conditions")