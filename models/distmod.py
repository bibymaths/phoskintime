import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT

@njit(cache=True)
def ode_core(y, t, A, B, C, D, S_rates, D_rates):
    """
    The core ODE system for the distributive phosphorylation model.

    The system is defined by the following equations:

    dR/dt = A - B * R
    dP/dt = C * R - (D + sum(S_rates)) * P + sum(P_sites)
    dP_sites[i]/dt = S_rates[i] * P - (1.0 + D_rates[i]) * P_sites[i]

    where:

    R: the concentration of the mRNA
    P: the concentration of the protein
    P_sites: the concentration of the phosphorylated sites
    A: the rate of production of the mRNA
    B: the rate of degradation of the mRNA
    C: the rate of production of the protein
    D: the rate of degradation of the protein
    S_rates: the rates of phosphorylation of each site
    D_rates: the rates of dephosphorylation of each site

    :param y:
    :param A:
    :param B:
    :param C:
    :param D:
    :param S_rates:
    :param D_rates:
    :return: Derivative of y
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
    Function to unpack the parameters for the ODE system.
    The parameters are expected to be in the following order:
    A, B, C, D, S_rates, D_rates
    where:
    A: mRNA production rate
    B: mRNA degradation rate
    C: protein production rate
    D: protein degradation rate
    S_rates: phosphorylation rates for each site
    D_rates: dephosphorylation rates for each site
    :param params: array of parameters
    :param num_psites: number of phosphorylation sites
    :return: A, B, C, D, S_rates, D_rates
    """
    params = np.asarray(params)
    A = params[0]
    B = params[1]
    C = params[2]
    D = params[3]
    S_rates = params[4 : 4 + num_psites]
    D_rates = params[4 + num_psites : 4 + 2 * num_psites]
    return A, B, C, D, S_rates, D_rates

def solve_ode(params, init_cond, num_psites, t):
    """
    Solve the ODE system for the distributive phosphorylation model.

    :param params:
    :param init_cond:
    :param num_psites:
    :param t:
    :return: solution of the ODE system, solution of phosphorylated sites
    """

    # Unpack the parameters
    A, B, C, D, S_rates, D_rates = unpack_params(params, num_psites)

    # Call the odeint function to solve the ODE system
    sol = np.clip(np.asarray(odeint(ode_core, init_cond, t,
                                    args=(A, B, C, D, S_rates, D_rates))),  0, None)

    # Normalize the solution if NORMALIZE_MODEL_OUTPUT is True
    if NORMALIZE_MODEL_OUTPUT:
        # Normalize the solution to the initial condition
        norm_init = np.array(init_cond, dtype=sol.dtype)
        # Calculate the reciprocal of the norm_init
        recip = 1.0 / norm_init
        # Normalize the solution by multiplying by the reciprocal of the norm_init
        sol *= recip[np.newaxis, :]

    # Extract the mRNA from the solution
    R_fitted = sol[5:, 0].T

    # Extract the phosphorylated sites from the solution
    P_fitted = sol[:, 2:].T

    # Return the solution and the phosphorylated sites
    return sol, np.concatenate((R_fitted.flatten(), P_fitted.flatten()))