"""Pre-merge item 1: the protwise optimizer keeps the best iterate and reports a status (see .sci R11: ABL2's objective
jumped to 2.06 while jaxopt reported error 0)."""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _x64():
    from networkmodel.backend import ensure_jax_float64

    ensure_jax_float64()


def _tracked(*a, **k):
    from networkmodel.backend import optimize_scalar_objective_tracked

    return optimize_scalar_objective_tracked(*a, **k)


def test_same_iterates_as_jaxopt_run_when_the_last_iterate_is_best():
    from networkmodel.backend import optimize_scalar_objective

    g = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2  # noqa: E731
    a, state, _ = optimize_scalar_objective(g, np.array([0.1, 0.9]), -2 * np.ones(2), 2 * np.ones(2), maxiter=50,
                                            tol=1e-7)
    b, info = _tracked(g, np.array([0.1, 0.9]), -2 * np.ones(2), 2 * np.ones(2), maxiter=50, tol=1e-7)
    assert info["best_iteration"] == info["iterations"] == int(state.iter_num)
    np.testing.assert_array_equal(a, b)  # bit-identical to jaxopt's run


def test_converged_status_on_an_easy_problem():
    f = lambda x: jnp.sum((x - 0.3) ** 2)  # noqa: E731
    x, info = _tracked(f, np.ones(2) * 0.9, np.zeros(2), np.ones(2), maxiter=200, tol=1e-7)
    assert info["status"] == "converged" and info["reasons"] == []
    np.testing.assert_allclose(x, 0.3, atol=1e-6)
    assert info["iterations"] < 200 and info["grad_norm"] < 1e-5


def test_max_iterations_is_not_converged():
    f = lambda x: jnp.sum((x - jnp.array([0.25, 0.7])) ** 2 * jnp.array([1.0, 30.0]))  # noqa: E731
    _, info = _tracked(f, np.ones(2) * 0.9, np.zeros(2), np.ones(2), maxiter=2, tol=1e-12)
    assert info["status"] == "not_converged" and info["iterations"] == 2
    assert any("max iterations" in r for r in info["reasons"])


def test_jump_after_the_best_iterate_returns_the_best_and_is_not_converged():
    # A discontinuity the line search steps over: the run ends at 5.0 after having reached 0.16.
    h = lambda x: jnp.sum((x - 0.9) ** 2) + jnp.where(x[0] > 0.5, 5.0, 0.0)  # noqa: E731
    x, info = _tracked(h, np.array([0.0]), np.zeros(1), np.ones(1), maxiter=50, tol=1e-9)
    assert info["status"] == "not_converged"
    assert info["final_objective"] > info["best_objective"]
    assert float(h(jnp.asarray(x))) == pytest.approx(info["best_objective"], rel=0, abs=0)
    assert any("above the best iterate" in r for r in info["reasons"])


def test_protwise_records_optimizer_info_in_metadata():
    import importlib
    import inspect
    import sys

    import protwise.paramest.core as core

    importlib.import_module("protwise.runner.main")
    assert '"optimizer": opt_info' in inspect.getsource(core)
    assert '"optimizer":' in inspect.getsource(sys.modules["protwise.runner.main"].main)
