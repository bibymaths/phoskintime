"""Initial-condition policy shared by the workflows (Round 2, item 13).

Per species, in this order:
1. measured at t = 0 → that value (P4b);
2. otherwise the policy:
   - "steady_state" (default): the model's own steady state under the t = 0 inputs, from the current parameters
     (inside the objective, so it moves with the parameters): the unmeasured species at steady state with the
     measured species held at their t0 values. For models linear in the states (all protwise models) this is the exact
     solution of J_uu·y_u = −(f(0)_u + J_um·y_m), e.g. R* = A/B for dR/dt = A − B·R. Note: when R is unmeasured and
     nothing drives it, R stays at A/B for all t (the mRNA prediction is flat).
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


def conditional_linear_steady_state(rhs, measured, xp):
    """Steady state of the UNMEASURED species with the measured species held at their t0 values, for a model linear
    in the states: J_uu·y_u = −(f(0)_u + J_um·y_m). This is the model's own steady state under the t = 0 inputs
    (measured values are inputs at t = 0). It only involves the unmeasured block, so a singular block elsewhere (e.g.
    a protein block with all decay rates at a zero bound) does not affect, e.g., R(0) = A/B (item 13 review)."""
    import jax

    m = np.asarray(measured, dtype=np.float64)
    meas = np.isfinite(m)
    u, k = np.flatnonzero(~meas), np.flatnonzero(meas)
    zero = xp.zeros(m.size, dtype=xp.float64)
    J = jax.jacfwd(rhs)(zero)
    f0 = rhs(zero)
    ym = xp.asarray(m[k])
    rhs_u = f0[u] + (J[xp.ix_(u, k)] @ ym if k.size else 0.0)
    y_u = -xp.linalg.solve(J[xp.ix_(u, u)], rhs_u) if u.size else xp.zeros(0)
    return xp.zeros(m.size, dtype=xp.float64).at[u].set(y_u).at[k].set(ym)


def initial_state(measured, *, policy: str, steady=None, estimated=None, xp=np):
    """y0 from the measured t0 values (NaN = not measured) and, for the others, the steady state or the estimate."""
    m = np.asarray(measured, dtype=np.float64)
    fill = steady if policy == "steady_state" else estimated
    if fill is None:
        raise ValueError(f"ic_policy {policy!r} needs its values (steady state or estimates).")
    return xp.where(xp.asarray(np.isfinite(m)), xp.asarray(np.nan_to_num(m)), fill)
