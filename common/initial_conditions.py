"""Initial-condition policy shared by the workflows (Round 2, item 13).

Per species, in this order:
1. measured at t = 0 → that value (P4b);
2. otherwise the policy:
   - "steady_state" (default): the model's own steady state under the t = 0 inputs, from the current parameters
     (inside the objective, so it moves with the parameters). For models that are linear in the states with constant
     inputs (all protwise models) the steady state is the exact solution of J·y* = −f(0) (J = ∂f/∂y, f(0) = the
     constant input term), e.g. R* = A/B for dR/dt = A − B·R.
   - "estimate" (opt-in): the initial value is a bounded free parameter.
The first observation is never shifted to t = 0 and nothing is back-extrapolated.
"""
from __future__ import annotations

import numpy as np

POLICIES = ("steady_state", "estimate")


def rules_for(measured, policy: str) -> list[str]:
    """Rule used per species: "measured_t0" or the policy name."""
    if policy not in POLICIES:
        raise ValueError(f"ic_policy must be one of {POLICIES}, got {policy!r}.")
    m = np.asarray(measured, dtype=np.float64)
    return ["measured_t0" if np.isfinite(v) else policy for v in m]


def linear_steady_state(rhs, n_states, xp):
    """Steady state of a model f(y) = J·y + f(0) that is linear in y: y* = −J⁻¹ f(0)."""
    import jax

    zero = xp.zeros(n_states, dtype=xp.float64)
    J = jax.jacfwd(rhs)(zero)
    return -xp.linalg.solve(J, rhs(zero))


def initial_state(measured, *, policy: str, steady=None, estimated=None, xp=np):
    """y0 from the measured t0 values (NaN = not measured) and, for the others, the steady state or the estimate."""
    m = np.asarray(measured, dtype=np.float64)
    fill = steady if policy == "steady_state" else estimated
    if fill is None:
        raise ValueError(f"ic_policy {policy!r} needs its values (steady state or estimates).")
    return xp.where(xp.asarray(np.isfinite(m)), xp.asarray(np.nan_to_num(m)), fill)
