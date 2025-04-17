import numpy as np
from scipy.optimize import minimize
from itertools import combinations

from config.logconf import setup_logger
logger = setup_logger()

def initial_condition(num_psites: int) -> list:
    """
    Calculates the initial steady-state conditions for a given number of phosphorylation sites
    for random phosphorylation model.

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
    subsets = []
    for k in range(1, num_psites + 1):
        for comb in combinations(range(1, num_psites + 1), k):
            subsets.append(comb)
    subset_to_index = {subset: i + 2 for i, subset in enumerate(subsets)}

    def steady_state_equations(y):
        """
        Defines the system of equations for the steady-state conditions.

        Args:
            y (list or np.ndarray): Current values of the variables [R, P, P_sites].

        Returns:
            list: Residuals of the steady-state equations.
        """
        R, P, *phos = y
        A, B, C, D = 1, 1, 1, 1
        S_rates = np.ones(num_psites)
        D_params = np.ones(len(subsets))
        eq_R = A - B * R
        gain_from_dephos = sum(phos[i] for i, subset in enumerate(subsets) if len(subset) == 1)
        eq_P = C * R - D * P - np.sum(S_rates) * P + gain_from_dephos
        eqs_phos = []
        for i, subset in enumerate(subsets):
            P_state = phos[i]
            gain_phos = 0
            for site in subset:
                if len(subset) == 1:
                    gain_phos += S_rates[site - 1] * P
                else:
                    reduced = tuple(sorted(set(subset) - {site}))
                    if reduced in subset_to_index:
                        gain_phos += S_rates[site - 1] * phos[subset_to_index[reduced] - 2]
            loss_phos = sum(S_rates[site - 1] for site in range(1, num_psites + 1) if site not in subset) * P_state
            basal_loss = len(subset) * P_state
            additional_loss = D_params[i] * P_state
            gain_from_further = 0
            for site in range(1, num_psites + 1):
                if site not in subset:
                    new_state = tuple(sorted(subset + (site,)))
                    if new_state in subset_to_index:
                        gain_from_further += phos[subset_to_index[new_state] - 2]
            eqs_phos.append(gain_phos - loss_phos - basal_loss - additional_loss + gain_from_further)
        return [eq_R, eq_P] + eqs_phos

    y0_guess = np.ones(len(subsets) + 2)
    bounds_local = [(1e-6, None)] * (len(subsets) + 2)
    result = minimize(lambda y: 0, y0_guess, method='SLSQP', bounds=bounds_local,
                      constraints={'type': 'eq', 'fun': steady_state_equations})
    logger.info("Steady-State conditions calculated")
    if result.success:
        return result.x.tolist()
    else:
        raise ValueError("Failed to find steady-state conditions")