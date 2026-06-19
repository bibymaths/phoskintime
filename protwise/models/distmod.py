import numpy as np
from numba import njit
from config.constants import NORMALIZE_MODEL_OUTPUT


@njit(cache=True)
def ode_core(y, t, A, B, C, D, S_rates, D_rates):
    """
    The core ODE system for the distributive phosphorylation model.

    Args:
        y: array of concentrations
        t: time
        A: mRNA production rate
        B: mRNA degradation rate
        C: protein production rate
        D: protein degradation rate
        S_rates: phosphorylation rates for each site
        D_rates: dephosphorylation rates for each site

    Returns:
        dydt: array of derivatives
    """
    # y[0] is the concentration of the mRNA
    R = y[0]

    # y[1] is the concentration of the protein
    P = y[1]

    # Number of phosphorylation sites
    n = S_rates.shape[0]

    # Derivative of y
    dydt = np.empty_like(y)

    # dydt[0] is the rate of change of R

    dydt[0] = A - B * R
    # S_rates

    sum_S = 0.0
    # sum_S is the sum of S_rates

    for i in range(n):
        # S_rates[i] is the rate of phosphorylation of site i
        sum_S += S_rates[i]

    # sum_P_sites is the sum of P sites
    sum_P_sites = 0.0

    # Loop over the number of phosphorylation sites
    for i in range(n):
        # y[2:] are the concentrations of the phosphorylated sites
        sum_P_sites += y[2 + i]

    # dydt[1] is the rate of change of P
    dydt[1] = C * R - (D + sum_S) * P + sum_P_sites

    # Loop over the number of phosphorylation sites
    for i in range(n):
        # dydt[2 + i] is the rate of change of each P site
        dydt[2 + i] = S_rates[i] * P - (1.0 + D_rates[i]) * y[2 + i]

    return dydt


@njit(cache=True)
def unpack_params(params, num_psites):
    """
    Function to unpack the parameters for the distributive ODE system.

    Args:
        params(np.array): Parameter vector containing A, B, C, D, S_1.S_n, Ddeg_1.Ddeg_m.
        num_psites(int): Number of phosphorylation sites.

    Returns:
        A (float): mRNA production rate.
        B (float): mRNA degradation rate.
        C (float): protein production rate.
        D (float): protein degradation rate.
        S_rates (np.array): Phosphorylation rates for each site.
        D_rates (np.array): Dephosphorylation rates for each site.
    """
    params = np.asarray(params)
    A = params[0]
    B = params[1]
    C = params[2]
    D = params[3]
    S_rates = params[4: 4 + num_psites]
    D_rates = params[4 + num_psites: 4 + 2 * num_psites]
    return A, B, C, D, S_rates, D_rates


def solve_ode(params, init_cond, num_psites, t, **kwargs):
    """Solve this mechanism with the centralized Diffrax Kvaerno backend."""
    from protwise.models.diffrax_solver import solve_protwise_ode

    # Pass an explicit model name so direct calls to this module are not affected
    # by the global config.constants.ODE_MODEL selected for a different mechanism.
    return solve_protwise_ode(params, init_cond, num_psites, t, model_name="distmod", **kwargs)
