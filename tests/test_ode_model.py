import numpy as np
from models.distmod import solve_ode


def test_solve_ode():
    params = np.array([1, 1, 1, 1, 0.5, 0.5])  # A, B, C, D, Ssite, Dsite for 1 psite
    init_cond = [1, 0, 0]  # R, P, P1
    t = np.linspace(0, 10, 5)  # time points
    sol, P_fitted = solve_ode(params, init_cond, num_psites=1, t=t)

    # Assertions
    assert sol.shape == (5, 3)         # 5 time points, 3 variables (R, P, P1)
    assert P_fitted.shape == (1, 5)    # 1 phosphorylation site, 5 time points
    assert np.all(P_fitted >= 0)       # Clipped non-negative