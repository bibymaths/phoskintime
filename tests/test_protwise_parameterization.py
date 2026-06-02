from __future__ import annotations

import pytest

from config.constants import get_num_params, get_param_names
from protwise.paramest.normest import _normalize_bounds, to_opt_space, from_opt_space


def test_parameter_names_match_counts_for_all_local_models():
    for model in ["protwise", "distmod", "succmod", "randmod", "dist", "succ"]:
        names = get_param_names(2, model)
        assert len(names) == get_num_params(model, 2)


def test_dict_bounds_expand_in_parameter_order_without_resize():
    bounds = {"A": (0.1, 1.0), "B": (0.2, 2.0), "C": (0.3, 3.0), "D": (0.4, 4.0), "S(i)": (0.5, 5.0), "D(i)": (0.6, 6.0)}
    lower, upper = _normalize_bounds(bounds, "protwise", 2)
    assert lower.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.6])
    assert upper.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0])


def test_optimizer_space_is_physical_space_for_projected_gradient():
    params = [0.2, 0.3, 0.4]
    theta = to_opt_space(params, "randmod")
    assert theta.tolist() == pytest.approx(params)
    assert list(from_opt_space(theta, "randmod")) == pytest.approx(params)
