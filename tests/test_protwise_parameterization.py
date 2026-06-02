from __future__ import annotations

import numpy as np
import pytest

from config.constants import get_num_params, get_param_names
from config.helpers import generate_randmod_subsets, randmod_subset_masks
from protwise.paramest.normest import _normalize_bounds, aggregate_randmod_phospho, to_opt_space, from_opt_space
from protwise.models.diffrax_solver import aggregate_randmod_site_phospho


def test_parameter_names_match_counts_for_all_local_models():
    for model in ["protwise", "distmod", "succmod", "randmod", "dist", "succ"]:
        names = get_param_names(2, model)
        assert len(names) == get_num_params(model, 2)


def test_dict_bounds_expand_in_parameter_order_without_resize():
    bounds = {"A": (0.1, 1.0), "B": (0.2, 2.0), "C": (0.3, 3.0), "D": (0.4, 4.0), "S(i)": (0.5, 5.0), "D(i)": (0.6, 6.0)}
    lower, upper = _normalize_bounds(bounds, "protwise", 2)
    assert lower.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.6])
    assert upper.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0])



def test_bounds_preserve_zero_fixed_and_map_infinite_upper():
    bounds = {"A": (0.0, np.inf), "B": (0.0, 0.0), "C": (0.3, 3.0), "D": (0.4, 4.0), "S(i)": (0.0, 0.0), "D(i)": (0.6, 6.0)}
    lower, upper = _normalize_bounds(bounds, "protwise", 1)
    assert lower[0] == pytest.approx(0.0)
    assert upper[0] > 1e5
    assert (lower[1], upper[1]) == pytest.approx((0.0, 0.0))
    assert (lower[4], upper[4]) == pytest.approx((0.0, 0.0))

def test_optimizer_space_is_physical_space_for_projected_gradient():
    params = [0.2, 0.3, 0.4]
    theta = to_opt_space(params, "randmod")
    assert theta.tolist() == pytest.approx(params)
    assert list(from_opt_space(theta, "randmod")) == pytest.approx(params)


def test_randmod_three_site_canonical_subset_and_parameter_order():
    assert generate_randmod_subsets(3) == ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
    assert randmod_subset_masks(3) == (1, 2, 4, 3, 5, 6, 7)
    assert get_param_names(3, "randmod")[-7:] == ["D1", "D2", "D3", "D12", "D13", "D23", "D123"]
    bounds = {"A": (0.1, 1.0), "B": (0.2, 2.0), "C": (0.3, 3.0), "D": (0.4, 4.0), "S(i)": (0.5, 5.0), "D(i)": (0.6, 6.0)}
    lower, _ = _normalize_bounds(bounds, "randmod", 3)
    assert lower[7:].tolist() == pytest.approx([0.6] * 7)


def test_randmod_phospho_aggregation_includes_multisite_states():
    # State order is [R, P, P1, P2, P3, P12, P13, P23, P123].
    sol = np.asarray([[0.0, 0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0]])
    ph = np.asarray(aggregate_randmod_phospho(sol, 3))
    assert ph.tolist() == pytest.approx([[131.0], [142.0], [153.0]])


def test_randmod_public_solve_helper_matches_objective_aggregation():
    # Public solve_ode p_fit uses the same site-level aggregation as the objective.
    sol = np.asarray([[0.0, 0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0]])
    assert aggregate_randmod_site_phospho(sol, 3).tolist() == pytest.approx([[131.0], [142.0], [153.0]])


def test_randmod_solve_output_normalizes_after_site_aggregation(monkeypatch):
    import protwise.models.diffrax_solver as dfs

    raw_sol = np.asarray([
        [2.0, 3.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0],
        [4.0, 6.0, 2.0, 4.0, 6.0, 20.0, 40.0, 60.0, 200.0],
    ])

    monkeypatch.setattr(dfs, "NORMALIZE_MODEL_OUTPUT", True)
    monkeypatch.setattr(dfs, "solve_diffrax", lambda *args, **kwargs: raw_sol)
    sol_site, flat = dfs.solve_protwise_ode(np.ones(get_num_params("randmod", 3)), raw_sol[0], 3, [0.0, 1.0], model_name="randmod")
    # r has 2 points, protein has 2 points, then 3 site-level phospho curves.
    phospho_flat = flat[4:]
    expected = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    assert phospho_flat.reshape(3, 2).tolist() == pytest.approx(expected)
    assert sol_site[:, 2:5].T.tolist() == pytest.approx(expected)

def test_mechanism_wrappers_pass_explicit_model_name(monkeypatch):
    calls = []

    def fake_dispatcher(params, init_cond, num_psites, t, model_name=None, **kwargs):
        calls.append(model_name)
        return None, None

    import protwise.models.distmod as distmod
    import protwise.models.randmod as randmod
    import protwise.models.succmod as succmod
    import protwise.models.protwise as protwise

    monkeypatch.setattr("protwise.models.diffrax_solver.solve_protwise_ode", fake_dispatcher)
    for module in (protwise, distmod, randmod, succmod):
        module.solve_ode([1.0], [1.0], 0, [0.0, 1.0])

    assert calls == ["protwise", "distmod", "randmod", "succmod"]
