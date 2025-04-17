import numpy as np
from numba import njit
from scipy.integrate import odeint
from config.constants import NORMALIZE_MODEL_OUTPUT

def prepare_vectorized_arrays(num_sites):
    """
    Prepare vectorized arrays for the Random ODE system.
    It creates binary states, phosphorylation targets, and dephosphorylation targets.

    Binary states are represented as a 2D array where each row corresponds to a state
    and each column corresponds to a phosphorylation site.

    Phosphorylation and Dephosphorylation targets are represented as 2D arrays where
    each row corresponds to a state and each column corresponds to a phosphorylation site.

    Target values indicate the resulting state after phosphorylation or dephosphorylation.

    :param num_sites: Number of phosphorylation sites
    :return: binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET
    """
    # Calculate the number of states
    num_states = 2 ** num_sites - 1
    # Create binary states
    binary_states = np.empty((num_states, num_sites), dtype=np.int32)
    # Fill the binary states array
    for i in range(num_states):
        # Convert the state number to binary representation
        state = i + 1
        # Fill the binary states array
        for j in range(num_sites):
            # Check if the j-th bit is set in the state number
            binary_states[i, j] = 1 if (state & (1 << j)) != 0 else 0
    # Create phosphorylation target arrays
    PHOSPHO_TARGET = -np.ones((num_states, num_sites), dtype=np.int32)
    # Fill the phosphorylation target array
    for i in range(num_states):
        # Convert the state number to binary representation
        state = i + 1
        # Fill the phosphorylation target array
        for j in range(num_sites):
            # Check if the j-th bit is not set in the state number
            if binary_states[i, j] == 0:
                # Set the j-th bit in the state number
                target_state = state | (1 << j)
                # Check if the target state is within the valid range
                if target_state <= num_states:
                    # Set the target state in the phosphorylation target array
                    PHOSPHO_TARGET[i, j] = target_state - 1
                else:
                    # Set the target state to -1 if out of range
                    PHOSPHO_TARGET[i, j] = -1
            else:
                # If the j-th bit is set, set the target state to -1
                PHOSPHO_TARGET[i, j] = -1
    # Create dephosphorylation target array
    DEPHOSPHO_TARGET = -np.ones((num_states, num_sites), dtype=np.int32)
    # Fill the dephosphorylation target array
    for i in range(num_states):
        # Convert the state number to binary representation
        state = i + 1
        # Fill the dephosphorylation target array
        for j in range(num_sites):
            # Check if the j-th bit is set in the state number
            if binary_states[i, j] == 1:
                # Set the j-th bit to 0 in the state number
                lower_state = state & ~(1 << j)
                # Check if the lower state is within the valid range
                if lower_state == 0:
                    # Set the target state to -2 if the lower state is 0
                    DEPHOSPHO_TARGET[i, j] = -2
                else:
                    # Set the target state in the dephosphorylation target array
                    DEPHOSPHO_TARGET[i, j] = lower_state - 1
            else:
                # If the j-th bit is not set, set the target state to -1
                DEPHOSPHO_TARGET[i, j] = -1

    return binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET

@njit
def ode_system(y, t, A, B, C, D, num_sites, binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET, *params):
    """
    The ODE system for the random phosphorylation model. This function computes the derivatives of the
    variables R, P, and X at each time step.

    - R is the concentration of the receptor
    - P is the concentration of the protein
    - X is the concentration of the phosphorylated states
    - S is the phosphorylation rate for each site
    - D_deg is the degradation rate for each state
    - A, B, C, D are parameters for the system
    - num_sites is the number of phosphorylation sites
    - binary_states is a 2D array representing the binary states of the system
    - PHOSPHO_TARGET is a 2D array representing the phosphorylation targets
    - DEPHOSPHO_TARGET is a 2D array representing the dephosphorylation targets

    :param y:
        y[0] = R
        y[1] = P
        y[2:] = X
    :param t:
    :param A:
    :param B:
    :param C:
    :param D:
    :param num_sites:
    :param binary_states:
    :param PHOSPHO_TARGET:
    :param DEPHOSPHO_TARGET:
    :param params:
    :return:
    """
    # Number of states
    num_states = 2 ** num_sites - 1
    # Phosphorylation rates
    S = np.empty(num_sites)
    for j in range(num_sites):
        # Convert the j-th parameter to a phosphorylation rate
        S[j] = params[j]
    # Dephosphorylation rates
    D_deg = np.empty(num_states)
    for i in range(num_states):
        # Convert the i-th parameter to a dephosphorylation rate
        D_deg[i] = params[num_sites + i]
    # Initialize the derivatives
    R = y[0]
    P = y[1]
    X = y[2:]
    dR_dt = A - B * R
    sum_S = 0.0
    for j in range(num_sites):
        sum_S += S[j]
    # Gain from phosphorylation
    gain_1site = 0.0
    # Loop over all states
    for i in range(num_states):
        cnt = 0
        # Count the number of phosphorylated sites in the state
        for j in range(num_sites):
            # Check if the j-th site is phosphorylated in the state
            cnt += binary_states[i, j]
        # If the state has only one phosphorylated site, add the gain to the total
        if cnt == 1:
            gain_1site += X[i]
    # Protein dynamics
    dP_dt = C * R - D * P - sum_S * P + gain_1site
    # Initialize the derivative of X
    dX_dt = np.zeros(num_states)
    for i in range(num_states):
        for j in range(num_sites):
            # Check if the j-th site is not phosphorylated in the i-th state
            if binary_states[i, j] == 0:
                # Get the phosphorylation target for the i-th state and j-th site
                target = PHOSPHO_TARGET[i, j]
                # If the target state is valid, add the phosphorylation rate to the derivative
                if target >= 0:
                    dX_dt[target] += S[j] * X[i]

    for i in range(num_states):
        # Check if the i-th state is phosphorylated
        loss = 0.0
        for j in range(num_sites):
            # Check if the j-th site is phosphorylated in the i-th state
            if binary_states[i, j] == 0:
                # Add loss to the total
                loss += S[j]
        # If the state has only one phosphorylated site, add the loss to the derivative
        dX_dt[i] -= loss * X[i]

    for i in range(num_states):
        cnt = 0
        for j in range(num_sites):
            # Count the number of phosphorylated sites in the state
            cnt += binary_states[i, j]
        # Add the dephosphorylation rate to the derivative
        dX_dt[i] -= cnt * X[i]

    for i in range(num_states):
        # Check if the i-th state is phosphorylated
        for j in range(num_sites):
            # Check if the j-th site is phosphorylated in the i-th state
            if binary_states[i, j] == 1:
                # Get the dephosphorylation target for the i-th state and j-th site
                lower = DEPHOSPHO_TARGET[i, j]
                # If the target state is valid, add the dephosphorylation rate to the derivative of P
                if lower == -2:
                    dP_dt += S[j] * X[i]
                # If the target state is valid, add the dephosphorylation rate to the derivative of X
                elif lower >= 0:
                    dX_dt[lower] += S[j] * X[i]

    for i in range(num_states):
        # Add the degradation rates to the derivative of X
        dX_dt[i] -= D_deg[i] * X[i]
    # Pack the derivatives into a single array
    dydt = np.empty(2 + num_states)
    dydt[0] = dR_dt
    dydt[1] = dP_dt
    # Pack the derivatives of X into the array
    for i in range(num_states):
        dydt[2 + i] = dX_dt[i]
    # Return the derivatives
    return dydt

def solve_ode(popt, initial_conditions, num_psites, time_points):
    """
    Solve the Random ODE system using the provided parameters and initial conditions.
    The function integrates the ODE system over the specified time points and returns
    the solution.

    :param popt:
    :param initial_conditions:
    :param num_psites:
    :param time_points:
    :return: solution, solution of phosphorylated states for each site
    """
    binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET = prepare_vectorized_arrays(num_psites)
    ode_params = popt
    A_val, B_val, C_val, D_val = ode_params[:4]
    remaining = ode_params[4:]
    sol = np.asarray(odeint(ode_system, initial_conditions, time_points,
                 args=(A_val, B_val, C_val, D_val, num_psites,
                       binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET, *remaining)))
    np.clip(sol, 0, None, out=sol)
    if NORMALIZE_MODEL_OUTPUT:
        norm_ic = np.array(initial_conditions, dtype=sol.dtype)
        recip = 1.0 / norm_ic
        sol *= recip[np.newaxis, :]
    if num_psites > 1:
        P_fitted = sol[:, 2:2 + num_psites].T
    else:
        P_fitted = sol[:, 2]
    return sol, P_fitted
