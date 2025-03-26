import numpy as np
from models.ode_model import solve_ode

def test_solve_ode():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])  # example for 1 psite
    init_cond = [1, 0, 0]
    t = np.linspace(0, 10, 5)
    _, P = solve_ode(params, init_cond, 1, t)
    assert P.shape == (5, 1)