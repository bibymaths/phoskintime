import numpy as np
from estimation.estimation import prepare_model_func


def test_prepare_model_func_structure():
    num_psites = 2
    init_cond = [1.0, 0.0, 0.0, 0.0]  # R, P, P1, P2 for 2 psites
    bounds = {
        "A": (0, 10), "B": (0, 10), "C": (0, 10), "D": (0, 10),
        "Ssite": (0, 10), "Dsite": (0, 10)
    }
    fixed_params = {
        "A": None, "B": 1, "C": 1, "D": 1,
        "Ssite_1": None, "Ssite_2": None,
        "Dsite_1": 1, "Dsite_2": 1
    }
    time_points = np.linspace(0, 10, 5)

    model_func, free_indices, free_bounds, fixed_values, num_total_params = prepare_model_func(
        num_psites=num_psites,
        init_cond=init_cond,
        bounds=bounds,
        fixed_params=fixed_params,
        time_points=time_points
    )

    # Assertions
    assert callable(model_func)
    assert isinstance(free_indices, list)
    assert isinstance(free_bounds, tuple) and len(free_bounds) == 2
    assert isinstance(fixed_values, dict)
    assert isinstance(num_total_params, int)
    assert all(lb < ub for lb, ub in zip(*free_bounds))
    assert len(free_indices) + len(fixed_values) == num_total_params
