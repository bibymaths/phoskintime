import numpy as np
from phoskintime.estimation.estimation import prepare_model_func

def test_prepare_model_func_structure():
    num_psites = 1
    init_cond = [1, 0, 0]
    bounds = {k: (0, 2) for k in ["A", "B", "C", "D", "Ssite", "Dsite"]}
    fixed = {"B": 1, "C": 1, "D": 1, "Dsite": 1}
    t = np.linspace(0, 10, 5)
    model_func, free_idx, free_bounds, fixed_vals, total = prepare_model_func(num_psites, init_cond, bounds, fixed, t)
    assert callable(model_func)
    assert len(free_idx) > 0