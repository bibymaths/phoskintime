import numpy as np
import pytest
from models.randmod import solve_ode, prepare_vectorized_arrays

def test_randmod_prepare_vectorized_arrays():
    num_sites = 2
    binary_states, PHOSPHO_TARGET, DEPHOSPHO_TARGET = prepare_vectorized_arrays(num_sites)
    expected_states = 2 ** num_sites - 1
    assert binary_states.shape == (expected_states, num_sites)
    assert PHOSPHO_TARGET.shape == (expected_states, num_sites)
    assert DEPHOSPHO_TARGET.shape == (expected_states, num_sites)

def test_randmod_solution_shape_multiple():
    # For num_sites=2, parameters length = 4 + 2 + 2 = 8.
    params = np.array([1, 1, 1, 1, 0.5, 0.5, 0.2, 0.2])
    # For num_sites=2, initial condition length = 2 + (2^2 - 1) = 5.
    init_cond = [1, 0, 0, 0, 0]
    t = np.linspace(0, 10, 5)
    sol, P_fitted = solve_ode(params, init_cond, num_psites=2, time_points=t)
    expected_sol_cols = 2 + (2 ** 2 - 1)
    assert sol.shape == (5, expected_sol_cols)
    # For num_sites > 1, P_fitted shape is (num_sites, time_points).
    assert P_fitted.shape == (2, 5)
    assert np.all(P_fitted >= 0)

def test_randmod_solution_shape_single_site():
    # For num_sites=1, length = 4+1+1 = 6.
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.array([0])
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, time_points=t)
    assert sol.shape == (1, 3)
    # For a single site, P_fitted can be a scalar; ensure it is wrapped as an array with shape (1,1).
    if np.ndim(P_fitted) == 0:
        P_fitted = np.array([P_fitted])
    assert P_fitted.shape == (1, 1)
    assert np.all(P_fitted >= 0)

def test_randmod_invalid_init_cond():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0]  # Invalid initial condition length.
    t = np.linspace(0, 10, 3)
    with pytest.raises(ValueError):
        solve_ode(params, init_cond, num_psites=1, time_points=t)