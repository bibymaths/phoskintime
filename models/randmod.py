
import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT
from functools import lru_cache

@lru_cache(maxsize=None)
def _precompute_indices(num_sites):
    """
    Precompute and cache all bitwise state transition indices needed for the random ODE system.

    Args:
        num_sites (int): Number of phosphorylation sites.
    Returns:
        mono_idx (np.array): Precomputed indices for mono-phosphorylated states.
        forward (np.array): Forward phosphorylation target states.
        drop (np.array): Dephosphorylation target states.
        fcounts (np.array): Number of valid forward transitions for each state.
        dcounts (np.array): Number of valid dephosphorylation transitions for each state.
    """

    # Determine number of phosphorylation sites
    n = num_sites

    # Calculate number of non-zero phosphorylated states: 2^n - 1
    m = (1 << n) - 1

    # Compute mono-phospho indices: (1 << j) - 1 gives the index of state with only site j phosphorylated
    mono_idx = np.array([(1 << j) - 1 for j in range(n)], dtype=np.int64)

    # Maximum number of transitions per state = number of sites
    max_trans = n

    # Initialize forward transition matrix with -1 (invalid entries)
    forward = -np.ones((m, max_trans), dtype=np.int64)

    # Initialize dephosphorylation transition matrix with -1
    drop = -np.ones((m, max_trans), dtype=np.int64)

    # Initialize counters for number of valid forward transitions per state
    fcounts = np.zeros(m, dtype=np.int64)

    # Initialize counters for number of valid dephosphorylation transitions per state
    dcounts = np.zeros(m, dtype=np.int64)

    # Iterate over each phosphorylated state from 1 to m (1-based indexing)
    for state in range(1, m + 1):

        # Initialize counters for current state's transitions
        fi = di = 0

        # Iterate over each phosphorylation site
        for j in range(n):

            # If site j is NOT phosphorylated in current state
            if not (state & (1 << j)):

                # Compute new state by phosphorylating site j (set bit j)
                forward[state - 1, fi] = state | (1 << j)

                # Increment forward transition count
                fcounts[state - 1] += 1

                # Move to next forward slot
                fi += 1

            else:
                # If site j is phosphorylated, we can dephosphorylate

                # Compute new state by clearing bit j
                lower = state & ~(1 << j)

                # Store the resulting dephosphorylated state
                drop[state - 1, di] = lower

                # Increment drop transition count
                dcounts[state - 1] += 1

                # Move to next drop slot
                di += 1

    # Return all precomputed arrays for reuse in ODE integration
    return mono_idx, forward, drop, fcounts, dcounts

@njit(cache=True)
def unpack_params(params, num_sites):
    """
    Unpack parameters for the Random model.

    Args:
        params (np.array): Parameter vector containing A, B, C, D, S_1.S_n, Ddeg_1.Ddeg_m.
        num_sites (int): Number of phosphorylation sites.

    Returns:
        A (float): mRNA production rate.
        B (float): mRNA degradation rate.
        C (float): protein production rate.
        D (float): protein degradation rate.
        S (np.array): Phosphorylation rates for each site.
        Ddeg (np.array): Degradation rates for phosphorylated states.
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

@njit(cache=True)
def ode_system(y, t,
               A, B, C, D,
               num_sites,
               S, Ddeg,
               mono_idx,
               forward, drop, fcounts, dcounts):
    """
    Compute the time derivatives of a random phosphorylation ODE system.

    This function supports a large number of phosphorylation states by using
    precomputed transition indices to optimize speed.

    Args:
        y (np.array): Current state vector [R, P, X_1, ..., X_m].
        t (float): Time (unused; present for compatibility with ODE solvers).
        A (float): mRNA production rate.
        B (float): mRNA degradation rate.
        C (float): protein production rate.
        D (float): protein degradation rate.
        num_sites (int): Number of phosphorylation sites.
        S (np.array): Phosphorylation rates for each site.
        Ddeg (np.array): Degradation rates for phosphorylated states.
        mono_idx (np.array): Precomputed indices for mono-phosphorylated states.
        forward (np.array): Forward phosphorylation target states.
        drop (np.array): Dephosphorylation target states.
        fcounts (np.array): Number of valid forward transitions for each state.
        dcounts (np.array): Number of valid dephosphorylation transitions for each state.

    Returns:
        out (np.array): Derivatives [dR, dP, dX_1, ..., dX_m].
    """

    # Compute number of states (2^n - 1)
    n = num_sites
    m = (1 << n) - 1

    # Extract mRNA concentration
    R = y[0]

    # Extract protein concentration
    P = y[1]

    # Compute derivative of mRNA
    dR = A - B * R

    # Compute derivative of protein
    dP = C * R - D * P

    # Initialize derivatives of phosphorylated states to zero
    dX = np.zeros(m)

    # Loop over each phosphorylation site to handle mono-phosphorylation
    for k in range(n):
        # Index of mono-phosphorylation state for site k
        idx = mono_idx[k]

        # Phosphorylation rate for site k
        rate = S[k] * P

        # Increase corresponding mono-phosphorylation state
        dX[idx] += rate

        # Decrease available protein P
        dP -= rate

    # Loop over each multi-phosphorylated state
    for state in range(1, m + 1):
        # Extract concentration of current state
        xi = y[1 + state]

        # Convert to zero-based index
        base = state - 1

        # Handle forward phosphorylation transitions
        for k in range(fcounts[base]):
            # Get target state index (1-based, convert to 0-based)
            tgt = forward[base, k] - 1

            # Get index of phosphorylated bit (least significant bit)
            j = int(np.log2((tgt + 1) & -(tgt + 1)))

            # Compute phosphorylation rate
            rate = S[j] * xi

            # Increase target state
            dX[tgt] += rate

            # Decrease current state
            dX[base] -= rate

        # Handle dephosphorylation transitions
        for k in range(dcounts[base]):
            # Get target state after dephosphorylation
            lower = drop[base, k]

            # Rate is simply proportional to xi
            rate = xi

            # If dephosphorylation returns to unmodified protein
            if lower == 0:
                dP += rate
            else:
                # Increase lower phosphorylated state
                dX[lower - 1] += rate

            # Decrease current state
            dX[base] -= rate

        # Apply degradation to this phosphorylated state
        dX[base] -= Ddeg[base] * xi

    # Allocate output np.array for all derivatives
    out = np.empty(2 + m)

    # Store derivative of R
    out[0] = dR

    # Store derivative of P
    out[1] = dP

    # Store derivatives of phosphorylated states
    for i in range(m):
        out[2 + i] = dX[i]

    # Return full derivative vector
    return out

def solve_ode(popt, y0, num_sites, t):
    """
    Integrate the ODE system for phosphorylation dynamics in random phosphorylation model.

    Args:
        popt (np.array): Optimized parameter vector [A, B, C, D, S_1.S_n, Ddeg_1.Ddeg_m].
        y0 (np.array): Initial condition vector [R0, P0, X1_0, ..., Xm_0].
        num_sites (int): Number of phosphorylation sites.
        t (np.array): Time points to integrate over.

    Returns:
        sol (ndarray): Full ODE solution of shape (len(t), len(y0)).
        mono (ndarray): 1D array of fitted values for R (after OFFSET) and P states.
    """
    # Unpack kinetic parameters and rate arrays
    A, B, C, D, S, Ddeg = unpack_params(popt, num_sites)

    # Load precomputed transition indices for the given number of sites
    mono_idx, forward, drop, fcounts, dcounts = _precompute_indices(num_sites)

    # Solve the ODE system using scipy's odeint
    sol = np.clip(np.asarray(
        odeint(
            ode_system,                # ODE system function
            y0,                        # Initial state
            t,                         # Time points
            args=(                     # Extra arguments to the ODE function
                A, B, C, D, num_sites,
                S, Ddeg,
                mono_idx, forward, drop, fcounts, dcounts
            )
        )
    ), 0, None)  # Ensure non-negative concentrations

    # If normalization is enabled, divide solution by initial condition
    if NORMALIZE_MODEL_OUTPUT:
        ic = np.array(y0, dtype=sol.dtype)   # convert initial state to same dtype
        sol *= (1.0 / ic)[None, :]           # element-wise normalization

    # Offset for removing early time points for R fitting
    OFFSET = 5

    # Extract mRNA trajectory after OFFSET
    R_fitted = sol[OFFSET:, 0].copy()

    # Extract phosphorylated protein states (transpose to shape: num_sites x time)
    if num_sites > 1:
        P_fitted = sol[:, 2:2 + num_sites].T
    else:
        # Special case when only one site is present
        P_fitted = sol[:, 2].reshape(1, -1)

    # Return full ODE solution and concatenated fit vector (R followed by P states)
    return sol, np.concatenate((R_fitted, P_fitted.flatten()))