import numpy as np
from numba import njit
from scipy.integrate import odeint

from config.constants import NORMALIZE_MODEL_OUTPUT


@njit(cache=True)
def ode_core(y, t, A, B, C, D, S_rates, D_rates):
    """
    The core of the ODE system for the successive ODE model.

    Args:
        y (np.array): The current state of the system.
        t (float): The current time.
        A (float): The mRNA production rate.
        B (float): The mRNA degradation rate.
        C (float): The protein production rate.
        D (float): The protein degradation rate.
        S_rates (np.array): The phosphorylation rates for each site.
        D_rates (np.array): The dephosphorylation rates for each site.
    Returns:
        dydt (np.array): The derivatives of the state variables.
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
    """
    Function to unpack the parameters for the ODE system.
    The parameters are expected to be in the following order:
    A, B, C, D, S_rates, D_rates
    where S_rates and D_rates are arrays of length num_psites.
    The function returns the unpacked parameters as separate variables.
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

    # Extract the protein from the solution
    Pr_fitted = sol[:, 1].T

    # Extract the phosphorylated sites from the solution
    P_fitted = sol[:, 2:].T

    # Return the solution and the phosphorylated sites
    return sol, np.concatenate((R_fitted.flatten(), Pr_fitted.flatten(), P_fitted.flatten()))
