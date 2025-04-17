
import numpy as np
import pandas as pd
import pytest
from paramest.normest import normest
from paramest.seqest import prepare_model_func, sequential_estimation
from paramest.adapest import adaptive_estimation


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
    monkeypatch.setattr("paramest.seqest.solve_ode", dummy_solve_ode)
    monkeypatch.setattr("paramest.adapest.solve_ode", dummy_solve_ode)
    monkeypatch.setattr("paramest.normest.early_emphasis", dummy_early_emphasis)
    monkeypatch.setattr("paramest.seqest.early_emphasis", dummy_early_emphasis)
    monkeypatch.setattr("paramest.adapest.early_emphasis", dummy_early_emphasis)
    monkeypatch.setattr("paramest.normest.get_weight_options", dummy_get_weight_options)
    monkeypatch.setattr("paramest.seqest.get_weight_options", dummy_get_weight_options)
    monkeypatch.setattr("paramest.adapest.get_weight_options", dummy_get_weight_options)
    # Patch score_fit to always return 0.
    monkeypatch.setattr("paramest.normest.score_fit", lambda target, pred, p: 0.0)
    monkeypatch.setattr("paramest.seqest.score_fit", lambda target, pred, p: 0.0)
    monkeypatch.setattr("paramest.adapest.score_fit", lambda target, pred, p: 0.0)


def test_normest_returns_valid_output():
    # Create dummy measurement data: shape (n,1) where n is the number of time points.
    time_points = np.linspace(0, 10, 5)
    p_data = np.full((5, 1), 1.0)
    init_cond = [1.0, 0.0, 0.0, 0.0]
    num_psites = 2
    bounds = {
        "A": (0.1, 10), "B": (0.1, 10), "C": (0.1, 10), "D": (0.1, 10),
        "Ssite": (0.1, 10), "Dsite": (0.1, 10)
    }
    bootstraps = 0

    est_params, model_fits, error_vals = normest("gene_dummy", p_data, init_cond,
                                                 num_psites, time_points, bounds,
                                                 bootstraps)
    # Check that we have a result for each output.
    assert isinstance(est_params, list) and len(est_params) > 0
    assert isinstance(model_fits, list) and len(model_fits) > 0
    assert isinstance(error_vals, list) and error_vals[0] >= 0


def test_sequential_estimation_returns_expected_length():
    # Generate dummy time series data.
    time_points = np.linspace(0, 10, 5)
    p_data = np.full((1, 5), 1.0)
    init_cond = [1.0, 0.0, 0.0, 0.0]
    num_psites = 1
    bounds = {
        "A": (0.1, 10), "B": (0.1, 10), "C": (0.1, 10), "D": (0.1, 10),
        "Ssite": (0.1, 10), "Dsite": (0.1, 10)
    }
    fixed_params = {}  # All parameters free.
    gene = "gene_seq"

    # Obtain model function (prepare_model_func simply returns extra items)
    model_func, free_indices, free_bounds, fixed_values, num_total_params = prepare_model_func(
        num_psites, init_cond, bounds, fixed_params
    )

    est_params, model_fits, error_vals = sequential_estimation(
        p_data, time_points, init_cond, bounds, fixed_params, num_psites, gene
    )
    # The number of estimated parameter sets should equal the number of time points.
    assert len(est_params) == len(time_points)
    assert len(model_fits) == len(time_points)
    assert len(error_vals) == len(time_points)


def test_adaptive_estimation_produces_valid_params():
    # Create dummy data as a dataframe to simulate p_data.
    time_points = np.linspace(0, 10, 5)
    # Dummy p_data: two phosphorylation sites so make a DataFrame with dummy data.
    dummy_data = pd.DataFrame({
        "Gene": ["gene_adapt"] * 2,
        "SomeInfo": [0, 0],
        "TP1": [1.0, 1.0],
        "TP2": [1.0, 1.0],
        "TP3": [1.0, 1.0],
        "TP4": [1.0, 1.0],
        "TP5": [1.0, 1.0]
    })
    init_cond = [1.0, 0.0, 0.0, 0.0]
    num_psites = 2
    bounds = {
        "A": (0.1, 10), "B": (0.1, 10), "C": (0.1, 10), "D": (0.1, 10),
        "Ssite": (0.1, 10), "Dsite": (0.1, 10)
    }
    fixed_params = {}  # No fixed parameters.
    gene = "gene_adapt"
    t = 5.0  # target time

    params = adaptive_estimation(dummy_data, init_cond, num_psites, time_points, t,
                                 bounds, fixed_params, gene)
    # Check that the parameter vector length is positive.
    assert isinstance(params, np.ndarray)
    assert params.size > 0