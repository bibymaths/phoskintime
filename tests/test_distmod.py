import numpy as np
import pytest
from models.distmod import solve_ode

def test_distmod_solution_shape_multiple():
    # For num_psites=1, parameters length = 4+1+1 = 6.
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]  # mRNA, protein, one phosphorylated site.
    t = np.linspace(0, 10, 5)
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (5, 3)
    assert P_fitted.shape == (1, 5)
    assert np.all(P_fitted >= 0)

def test_distmod_solution_shape_single():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0, 0]
    t = np.array([0])
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)
    assert sol.shape == (1, 3)
    assert P_fitted.shape == (1, 1)
    assert np.all(P_fitted >= 0)

def test_distmod_invalid_init_cond():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])
    init_cond = [1, 0]  # Too few elements.
    t = np.linspace(0, 10, 3)
    with pytest.raises(ValueError):
        solve_ode(params, init_cond, num_psites=1, t=t)