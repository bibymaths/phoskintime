"""DECISIONS_C1 item 6: ODE solver step counts and rejections are recorded (metadata "solver_stats"); the
trajectory is unchanged when the statistics are requested."""
import numpy as np
import pytest


def test_solve_diffrax_returns_step_statistics_without_changing_the_trajectory():
    import jax.numpy as jnp

    from networkmodel.backend import DiffraxSolverConfig, solve_diffrax

    def stiff(t, y, args):  # a stiff linear system: rates 1 and 1e4
        return jnp.array([-y[0], -1e4 * (y[1] - jnp.cos(t))])

    t = np.linspace(0.0, 5.0, 11)
    y0 = np.array([1.0, 0.0])
    cfg = DiffraxSolverConfig()
    ys = solve_diffrax(y0, t, rhs=stiff, config=cfg)
    ys2, stats = solve_diffrax(y0, t, rhs=stiff, config=cfg, return_stats=True)
    np.testing.assert_array_equal(np.asarray(ys), np.asarray(ys2))
    for k in ("num_steps", "num_accepted_steps", "num_rejected_steps", "max_steps", "solver", "rtol", "atol"):
        assert k in stats, k
    assert stats["num_steps"] == stats["num_accepted_steps"] + stats["num_rejected_steps"]
    assert 0 < stats["num_steps"] <= stats["max_steps"]
    assert stats["max_steps_fraction"] == pytest.approx(stats["num_steps"] / stats["max_steps"])
