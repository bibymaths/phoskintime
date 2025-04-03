import numpy as np
import pytest
from models.distmod import solve_ode

def verify_solve_ode_returns_correct_shapes_for_multiple_time_points():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.linspace(0, 10, 7)
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (7, 3)
    assert P_fitted.shape == (1, 7)
    assert np.all(P_fitted >= 0)

def verify_solve_ode_handles_single_time_point():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.array([0])
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (1, 3)
    assert P_fitted.shape == (1, 1)
    assert np.all(P_fitted >= 0)

def verify_solve_ode_raises_error_for_invalid_initial_conditions():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    # Wrong length of initial conditions
    init_cond = [1, 0]
    t = np.linspace(0, 10, 5)
    with pytest.raises(Exception):
        solve_ode(params, init_cond, num_psites=1, t=t)

def verify_solve_ode_outputs_non_negative_phosphorylation_for_negative_parameters():
    params = np.array([-1, 1, 1, 1, -0.5, -0.5])
    init_cond = [1, 0, 0]
    t = np.linspace(0, 10, 5)
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (5, 3)
    assert P_fitted.shape == (1, 5)
    assert np.all(P_fitted >= 0)