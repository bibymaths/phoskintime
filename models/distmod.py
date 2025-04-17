
import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT

@njit
def ode_core(y, A, B, C, D, S_rates, D_rates):
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

def ode_system(y, t, params, num_psites):
    """
    The ODE system for the distributive phosphorylation model which calls the core ODE system.

    :param y:
    :param t:
    :param params:
    :param num_psites:
    :return: ode_core(y, A, B, C, D, S_rates, D_rates)
    """
    # Unpack the parameters
    # params[0] is A
    # params[1] is B
    # params[2] is C
    # params[3] is D
    A, B, C, D = params[0], params[1], params[2], params[3]
    # params[4 + i] is the S_rate for site i
    S_rates = np.array([params[4 + i] for i in range(num_psites)])
    # params[4 + num_psites + i] is the D_rate for site i
    D_rates = np.array([params[4 + num_psites + i] for i in range(num_psites)])
    return ode_core(y, A, B, C, D, S_rates, D_rates)


def solve_ode(params, init_cond, num_psites, t):
    """
    Solve the ODE system for the distributive phosphorylation model.

    :param params:
    :param init_cond:
    :param num_psites:
    :param t:
    :return: solution of the ODE system, solution of phosphorylated sites
    """
    # Call the odeint function to solve the ODE system
    sol = np.asarray(odeint(ode_system, init_cond, t, args=(params, num_psites)))
    # Clip the solution to be non-negative
    np.clip(sol, 0, None, out=sol)
    # Normalize the solution if NORMALIZE_MODEL_OUTPUT is True
    if NORMALIZE_MODEL_OUTPUT:
        # Normalize the solution to the initial condition
        norm_init = np.array(init_cond, dtype=sol.dtype)
        # Calculate the reciprocal of the norm_init
        recip = 1.0 / norm_init
        # Normalize the solution by multiplying by the reciprocal of the norm_init
        sol *= recip[np.newaxis, :]
    # Extract the phosphorylated sites from the solution
    P_fitted = sol[:, 2:].T
    # Return the solution and the phosphorylated sites
    return sol, P_fitted

