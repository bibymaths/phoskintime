import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT


@njit(cache=True)
def ode_system(y, t,
               A, B, C, D,
               num_sites,
               *params):
    """
    The ODE system for the Random ODE model.
    The system is defined by the following equations:
    dR/dt = A - B * R
    dP/dt = C * R - D * P
    dX_j/dt = S_j * P - Ddeg_j * X_j
    where:
    R: the concentration of the mRNA
    P: the concentration of the protein
    X_j: the concentration of the phosphorylated state j
    A: the rate of production of the mRNA
    B: the rate of degradation of the mRNA
    C: the rate of production of the protein
    D: the rate of degradation of the protein
    S_j: the rate of phosphorylation of site j
    Ddeg_j: the rate of degradation of state j

    Args:
        y (array): Current state of the system.
        t (float): Current time.
        A (float): Rate of production of the mRNA.
        B (float): Rate of degradation of the mRNA.
        C (float): Rate of production of the protein.
        D (float): Rate of degradation of the protein.
        num_sites (int): Number of phosphorylation sites.
        params (array): Parameters for the ODE system.
        *params (float): Additional parameters for the ODE system.

    Returns:
        dydt (array): Derivatives of the system at the current state.
    """
    n = num_sites
    m = (1 << n) - 1

    # unpack rates
    S = np.empty(n)
    Ddeg = np.empty(m)
    # phosphorylation rates
    for j in range(n):
        S[j] = params[j]
    # pure-degradation rates
    for i in range(m):
        Ddeg[i] = params[n + i]

    # unpack variables
    R = y[0]
    P = y[1]
    # X lives in y[2..2+m)
    # index as X[i] = y[2+i]

    # initialize derivatives
    dR = A - B * R
    dP = C * R - D * P
    dX = np.zeros(m)

    # 1) P → X_j (mono-phospho)
    for j in range(n):
        rate = S[j] * P
        idx = (1 << j) - 1
        dX[idx] += rate
        dP -= rate

    # transitions among X's + dephosphorylation (unit rate)
    for state in range(1, m + 1):
        xi = y[2 + state - 1]

        # forward phospho on each unmodified bit
        for j in range(n):
            if (state & (1 << j)) == 0:
                tgt = state | (1 << j)
                rate = S[j] * xi
                dX[tgt - 1] += rate
                dX[state - 1] -= rate

        # dephosphorylation at unit rate
        for j in range(n):
            if (state & (1 << j)) != 0:
                lower = state & ~(1 << j)
                rate = xi
                if lower == 0:
                    dP += rate
                else:
                    dX[lower - 1] += rate
                dX[state - 1] -= rate

        # ure degradation of this X[state-1]
        dX[state - 1] -= Ddeg[state - 1] * xi

    # pack into dydt
    dydt = np.empty(2 + m)
    dydt[0] = dR
    dydt[1] = dP
    for i in range(m):
        dydt[2 + i] = dX[i]

    return dydt

@njit(cache=True)
def unpack_params(params, num_sites):
    """
    Unpack parameters for the Random model.
    Returns: A, B, C, D, S (phosphorylation rates), Ddeg (degradation rates)
    """
    params = np.asarray(params)
    A = params[0]
    B = params[1]
    C = params[2]
    D = params[3]
    n = num_sites
    m = (1 << n) - 1
    S = np.empty(n)
    Ddeg = np.empty(m)

    # should be length num_sites + (2^n -1)
    for i in range(n):
        S[i] = params[4 + i]
    for i in range(m):
        Ddeg[i] = params[4 + n + i]
    return A, B, C, D, S, Ddeg

def solve_ode(popt, y0, num_sites, t):
    """
    Solve the Random ODE system using the provided parameters and initial conditions.
    The function integrates the ODE system over the specified time points and returns
    the solution.

    Args:
        popt (array): Parameters for the ODE system.
        y0 (array): Initial conditions for the ODE system.
        num_sites (int): Number of phosphorylation sites.
        t (array): Time points for the integration.
    Returns:
        sol (array): Solution of the ODE system.
        mono (array): Solution of phosphorylation states for each site.
    """
    A, B, C, D, S, Ddeg = unpack_params(popt, num_sites)

    sol = np.asarray(odeint(ode_system, y0, t, args=(A, B, C, D, num_sites, *S, *Ddeg)))

    np.clip(sol, 0, None, out=sol)

    # sol_15 = np.asarray(odeint(ode_system, y0, [15.0], args=(A, B, C, D, num_sites, *S, *Ddeg)))
    #
    # np.clip(sol_15, 0, None, out=sol_15)

    if NORMALIZE_MODEL_OUTPUT:
        ic = np.array(y0, dtype=sol.dtype)
        sol *= (1.0 / ic)[None, :]
        # sol_15 *= (1.0 / ic)[None, :]
        # sol[7] = sol_15[0]

    OFFSET = 5
    R_fitted = sol[OFFSET:, 0].copy()
    # R_fitted[2] = sol_15[0, 0]  # Replace 3rd point (index 2)

    if num_sites > 1:
        P_fitted = sol[:, 2:2 + num_sites].T
    else:
        P_fitted = sol[:, 2].reshape(1, -1)

    return sol, np.concatenate((R_fitted, P_fitted.flatten()))
