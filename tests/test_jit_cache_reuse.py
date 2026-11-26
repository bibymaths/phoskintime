"""PERF-1: repeated ODE solves reuse compiled code instead of recompiling (issue X13)."""
import numpy as np
import pytest

from protwise.models.diffrax_solver import make_local_model_rhs, solve_protwise_ode


@pytest.mark.parametrize("model,n", [("distmod", 1), ("succmod", 2), ("randmod", 3), ("distributive", 2)])
def test_local_rhs_factory_returns_same_object(model, n):
    assert make_local_model_rhs(model, n) is make_local_model_rhs(model, n)


def test_local_rhs_factory_distinguishes_models_and_sites():
    assert make_local_model_rhs("distmod", 1) is not make_local_model_rhs("distmod", 2)
    assert make_local_model_rhs("distmod", 2) is not make_local_model_rhs("succmod", 2)
    # aliases resolve to the canonical model and share the cached object
    assert make_local_model_rhs("dist", 2) is make_local_model_rhs("distmod", 2)


def test_local_rhs_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported local ODE model"):
        make_local_model_rhs("nope", 1)


def test_repeated_protwise_solve_is_identical():
    t = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0], dtype=float)
    p = np.asarray([0.3, 0.2, 0.35, 0.1, 0.18, 0.12], dtype=float)
    y0 = np.asarray([1.0, 1.0, 1.0], dtype=float)
    a, fa = solve_protwise_ode(p, y0, 1, t, model_name="distmod")
    b, fb = solve_protwise_ode(p, y0, 1, t, model_name="distmod")
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(fb))
