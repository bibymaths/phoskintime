import numpy as np
from scipy.optimize import minimize

from config.logconf import setup_logger
logger = setup_logger()

def initial_condition(num_psites: int) -> list:
    """
    Calculates the initial steady-state conditions for a given number of phosphorylation sites
    for distributive phosphorylation model.

    This function defines a system of equations representing the steady-state conditions
    of an ODE model and solves it using numerical optimization. The steady-state conditions
    are used as initial conditions for further simulations.

    Args:
        num_psites (int): Number of phosphorylation sites in the model.

    Returns:
        list: A list of steady-state values for the variables [R, P, P_sites].

    Raises:
        ValueError: If the optimization fails to find a solution for the steady-state conditions.
    """
    A, B, C, D = 1, 1, 1, 1
    S_rates = np.ones(num_psites)
    D_rates = np.ones(num_psites)

    def steady_state_equations(y):
        """
        Defines the system of equations for the steady-state conditions.

        Args:
            y (list or np.ndarray): Current values of the variables [R, P, P_sites].

        Returns:
            list: Residuals of the steady-state equations.
        """
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