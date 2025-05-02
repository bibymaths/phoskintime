import numpy as np
from numba import njit
from scipy.integrate import odeint

from config.constants import NORMALIZE_MODEL_OUTPUT


@njit(cache=True)
def ode_core(y, t, A, B, C, D, S_rates, D_rates):
    """
    The core of the ODE system for the successive ODE model.

    The system is defined by the following equations:
    dR/dt = A - B * R
    dP/dt = C * R - D * P - S_rates[0] * P + sum(P_sites)
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
    :return: derivative of y
    """
    # mRNA
    R = y[0]
    # Protein
    P = y[1]
    # Number of phosphorylated sites
    num_psites = S_rates.shape[0]

    # mRNA dynamics
    dR_dt = A - B * R
    # Protein dynamics
    dP_dt = C * R - D * P

    # Adjust protein dynamics by phosphorylation/dephosphorylation of the first site if exists
    if num_psites > 0:
        # Subtract phosphorylation contribution from the protein at site 0
        dP_dt -= S_rates[0] * P
        # Add dephosphorylation feedback from the first phosphorylated site
        dP_dt += y[2]

    # Prepare output array for derivatives
    dydt = np.empty_like(y)
    dydt[0] = dR_dt
    dydt[1] = dP_dt

    # Phosphorylated sites dynamics loop
    for i in range(num_psites):
        # When there is only one site, handle it separately
        if num_psites == 1:
            # For one phosphorylated site:
            # Calculate the site's rate: phosphorylation from the protein minus its degradation (combined rate).
            dydt[2] = S_rates[0] * P - (1 + D_rates[0]) * y[2]
        else:
            if i == 0:
                # For the first site:
                # Phosphorylation of the protein contributes to the site dynamics.
                # Site feedback: rate from the second site affects the current site.
                # The term y[3] provides dephosphorylation feedback.
                dydt[2] = S_rates[0] * P - (1 + S_rates[1] + D_rates[0]) * y[2] + y[3]
            elif i < num_psites - 1:
                # For intermediate sites:
                # Phosphorylation from the preceding phosphorylated species (y[1+i]) drives the site.
                # The degradation rate is increased by the phosphorylation rate of the next site (S_rates[i+1]).
                # The term y[3+i] provides dephosphorylation feedback from the next site.
                dydt[2 + i] = S_rates[i] * y[1 + i] - (1 + S_rates[i + 1] + D_rates[i]) * y[2 + i] + y[3 + i]
            else:
                # For the last site:
                # Phosphorylation from the preceding site (y[1+i]) drives the site.
                # There is no next phosphorylation term, so only include the dephosphorylation degradation.
                dydt[2 + i] = S_rates[i] * y[1 + i] - (1 + D_rates[i]) * y[2 + i]
    return dydt


@njit(cache=True)
def unpack_params(params, num_psites):
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
    Solve the ODE system using the given parameters and initial conditions.
    The function integrates the ODE system over time and returns the solution.

    :param params:
    :param init_cond:
    :param num_psites:
    :param t:
    :return: solution, solution of phosphorylated sites
    """
    # Unpack the parameters
    A, B, C, D, S_rates, D_rates = unpack_params(params, num_psites)

    # Call the odeint function to solve the ODE system
    sol = np.asarray(odeint(ode_core, init_cond, t, args=(A, B, C, D, S_rates, D_rates)))

    # Clip the solution to be non-negative
    np.clip(sol, 0, None, out=sol)

    # Solve separately at 15.0 minutes
    # sol_15 = np.asarray(odeint(ode_core, init_cond, [15.0], args=(A, B, C, D, S_rates, D_rates)))

    # Clip the solution to be non-negative
    # np.clip(sol_15, 0, None, out=sol_15)

    # Normalize the solution if NORMALIZE_MODEL_OUTPUT is True
    if NORMALIZE_MODEL_OUTPUT:
        # Normalize the solution to the initial condition
        norm_init = np.array(init_cond, dtype=sol.dtype)
        # Calculate the reciprocal of the norm_init
        recip = 1.0 / norm_init
        # Normalize the solution by multiplying by the reciprocal of the norm_init
        sol *= recip[np.newaxis, :]
        # Now, replace the 8th value (index 7) with the solution at 15
        # sol_15 *= recip[np.newaxis, :]
        # Now, replace the 8th value of sol (index 7) with the solution at 15
        # sol[7] = sol_15[0]

    # Extract the mRNA from the solution
    R_fitted = sol[5:, 0].T
    # Now replace the 8th value of R_fitted (index 7 after slicing at 5:)
    # R_fitted[2] = sol_15[0, 0]
    # Extract the phosphorylated sites from the solution
    P_fitted = sol[:, 2:].T

    # Return the solution and the phosphorylated sites
    return sol, np.concatenate((R_fitted.flatten(), P_fitted.flatten()))
