
import numpy as np
import pytest
from paramest.normest import normest

# Dummy ODE solver that returns predictable outputs.
def dummy_solve_ode(param_vec, init_cond, num_psites, time_points):
    # Return time_points as solution and a constant prediction (flattened).
    sol = np.array(time_points)
    p_fitted = np.full((len(time_points), 1), 1.0)
    return sol, p_fitted


# Dummy functions for early_emphasis and get_weight_options.
def dummy_early_emphasis(data, time_points, num_psites):
    return np.ones(np.shape(data))


def dummy_get_weight_options(target, time_points, num_psites, use_reg, n_params, early_weights):
    # Return one key with constant sigma.
    return {"dummy": np.ones(n_params)}


@pytest.fixture(autouse=True)
def patch_functions(monkeypatch):
    # For normest, seqest and adapest tests patch the common functions.
    monkeypatch.setattr("paramest.normest.solve_ode", dummy_solve_ode)
    monkeypatch.setattr("paramest.normest.early_emphasis", dummy_early_emphasis)
    monkeypatch.setattr("paramest.normest.get_weight_options", dummy_get_weight_options)
    monkeypatch.setattr("paramest.normest.score_fit", lambda target, pred, p: 0.0)


def test_normest_returns_valid_output():
    # Create dummy measurement data: shape (n,1) where n is the number of time points.
    time_points = np.linspace(0, 10, 5)
    p_data = np.full((5, 1), 1.0)
    r_data = np.full((5, 1), 1.0)
    init_cond = [1.0, 0.0, 0.0, 0.0]
    num_psites = 2
    bounds = {
        "A": (0.1, 10), "B": (0.1, 10), "C": (0.1, 10), "D": (0.1, 10),
        "Ssite": (0.1, 10), "Dsite": (0.1, 10)
    }
    bootstraps = 0

    est_params, model_fits, error_vals = normest("gene_dummy", p_data, r_data, init_cond,
                                                 num_psites, time_points, bounds,
                                                 bootstraps)
    # Check that we have a result for each output.
    assert isinstance(est_params, list) and len(est_params) > 0
    assert isinstance(model_fits, list) and len(model_fits) > 0
    assert isinstance(error_vals, list) and error_vals[0] >= 0