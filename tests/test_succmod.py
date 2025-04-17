import numpy as np
import pytest
from models.succmod import solve_ode

def test_succmod_solution_shape_multiple():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.linspace(0, 10, 5)
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (5, 3)
    assert P_fitted.shape == (1, 5)
    assert np.all(P_fitted >= 0)

def test_succmod_solution_shape_single():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.array([0])
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (1, 3)
    assert P_fitted.shape == (1, 1)
    assert np.all(P_fitted >= 0)