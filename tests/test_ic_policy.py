"""Round 2 item 13: initial-condition policy. Per species: measured t0 value; else "steady_state" (the model's own
steady state from the current parameters, inside the objective; e.g. R(0) = A/B); "estimate" is opt-in."""
import numpy as np
import pytest



def test_steady_state_fills_unmeasured_species():
    import jax.numpy as jnp

    from networkmodel.backend import ensure_jax_float64

    ensure_jax_float64()
    from protwise.steady.initdata import InitialCondition

    ic = InitialCondition("G", "distmod", 2, np.array([[1.3, 1.0]]), np.array([[0.7, 1.0], [np.nan, 1.0]]),
                          policy="steady_state")
    assert ic.rules == ["steady_state", "measured_t0", "measured_t0", "steady_state"]
    p = np.array([0.6, 0.3, 0.5, 0.2, 0.4, 0.8, 0.1, 0.3])  # A, B, C, D, S1, S2, D1, D2
    y0 = np.asarray(ic(jnp.asarray(p)))
    assert y0[0] == pytest.approx(0.6 / 0.3, rel=1e-12)  # R(0) = A/B
    np.testing.assert_allclose(y0[1:3], [1.3, 0.7])
    # X2 is at steady state given the measured P(0) = 1.3: dX2/dt = S2·P(0) − (1 + D2)·X2 = 0
    assert y0[3] == pytest.approx(0.8 * 1.3 / (1 + 0.3), rel=1e-12)
    from protwise.models.diffrax_solver import make_local_model_rhs

    dy = np.asarray(make_local_model_rhs("distmod", 2)(0.0, jnp.asarray(y0), jnp.asarray(p)))
    assert abs(dy[0]) < 1e-12 and abs(dy[3]) < 1e-12  # unmeasured species are stationary at t = 0


def test_singular_protein_block_does_not_spoil_R_and_final_check_raises():
    import jax.numpy as jnp

    from networkmodel.backend import ensure_jax_float64
    from protwise.steady.initdata import InitialCondition

    ensure_jax_float64()
    ic = InitialCondition("G", "distmod", 1, np.array([[1.0]]), np.array([[0.5]]), policy="steady_state")
    p = np.array([0.6, 0.3, 0.5, 0.0, 0.0, 0.0])  # D = S = D1 = 0: the protein block is singular
    assert float(ic(jnp.asarray(p))[0]) == pytest.approx(2.0)  # R(0) = A/B still defined
    bad = np.array([0.6, 0.0, 0.5, 0.2, 0.4, 0.1])  # B = 0: R has no steady state
    with pytest.raises(RuntimeError, match="not finite"):
        ic.value(bad)


def test_estimate_policy_adds_bounded_parameters():
    from protwise.steady.initdata import InitialCondition

    ic = InitialCondition("G", "distmod", 1, np.array([[1.0]]), np.array([[0.5]]), policy="estimate")
    assert ic.rules == ["estimate", "measured_t0", "measured_t0"] and ic.n_estimate == 1


def test_policy_recorded_and_configurable():
    from config.constants import IC_POLICY

    assert IC_POLICY == "steady_state"
