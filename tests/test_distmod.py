import numpy as np
import pytest
from models.distmod import solve_ode

def distmod_handles_zero_time_points():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.array([])
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (0, 3)
    assert P_fitted.shape == (1, 0)

def distmod_handles_negative_time_points():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.array([-1, -2, -3])
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (3, 3)
    assert P_fitted.shape == (1, 3)

def distmod_raises_error_for_negative_num_psites():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.linspace(0, 10, 5)
    with pytest.raises(ValueError):
        solve_ode(params, init_cond, num_psites=-1, t=t)

def distmod_handles_large_number_of_psites():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0, 0, 0, 0]
    t = np.linspace(0, 10, 5)
    sol, P_fitted = solve_ode(params, init_cond, num_psites=4, t=t)
    assert sol.shape == (5, 6)
    assert P_fitted.shape == (4, 5)
    assert np.all(P_fitted >= 0)

def distmod_raises_error_for_mismatched_init_cond_length():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0, 0]  # Incorrect length
    t = np.linspace(0, 10, 5)
    with pytest.raises(ValueError):
        solve_ode(params, init_cond, num_psites=2, t=t)