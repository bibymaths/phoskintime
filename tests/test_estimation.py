import numpy as np
import pytest
from paramest.seqest import prepare_model_func

def test_prepare_model_func_returns_empty_free_indices_when_all_params_fixed():
    num_psites = 2
    init_cond = [1.0, 0.0, 0.0, 0.0]
    bounds = {
        "A": (0, 10), "B": (0, 10), "C": (0, 10), "D": (0, 10),
        "Ssite": (0, 10), "Dsite": (0, 10)
    }
    fixed_params = {
        "A": 1, "B": 1, "C": 1, "D": 1,
        "Ssite_1": 2, "Ssite_2": 2,
        "Dsite_1": 1, "Dsite_2": 1
    }
    time_points = np.linspace(0, 10, 5)
    _, free_indices, _, fixed_values, num_total_params = prepare_model_func(
        num_psites=num_psites,
        init_cond=init_cond,
        bounds=bounds,
        fixed_params=fixed_params,
        time_points=time_points
    )
    assert free_indices == []
    assert len(fixed_values) == num_total_params

def test_prepare_model_func_raises_error_for_empty_time_points():
    num_psites = 2
    init_cond = [1.0, 0.0, 0.0, 0.0]
    bounds = {
        "A": (0, 10), "B": (0, 10), "C": (0, 10), "D": (0, 10),
        "Ssite": (0, 10), "Dsite": (0, 10)
    }
    fixed_params = {
        "A": None, "B": 1, "C": 1, "D": 1,
        "Ssite_1": None, "Ssite_2": None,
        "Dsite_1": 1, "Dsite_2": 1
    }
    empty_time_points = np.array([])
    with pytest.raises(ValueError):
        prepare_model_func(
            num_psites=num_psites,
            init_cond=init_cond,
            bounds=bounds,
            fixed_params=fixed_params,
            time_points=empty_time_points
        )

def test_prepare_model_func_raises_error_for_invalid_bounds():
    num_psites = 1
    init_cond = [1.0, 0.0, 0.0, 0.0]
    bounds = {
        "A": (5, 5), "B": (0, 10), "C": (0, 10), "D": (0, 10),
        "Ssite": (0, 10), "Dsite": (0, 10)
    }
    fixed_params = {
        "A": None, "B": 1, "C": 1, "D": 1,
        "Ssite_1": None,
        "Dsite_1": 1
    }
    time_points = np.linspace(0, 10, 5)
    with pytest.raises(ValueError):
        prepare_model_func(
            num_psites=num_psites,
            init_cond=init_cond,
            bounds=bounds,
            fixed_params=fixed_params,
            time_points=time_points
        )