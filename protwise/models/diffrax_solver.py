"""Diffrax-backed local phosphorylation model solver helpers."""
from __future__ import annotations

from functools import partial
import numpy as np
import jax.numpy as jnp

from networkmodel.jax_backend import solve_diffrax, DiffraxSolverConfig
from config.constants import NORMALIZE_MODEL_OUTPUT, get_num_params


def _canonical_model_name(model_name: str | None) -> str:
    model = str(model_name or "protwise").strip().lower()
    aliases = {"dist": "distmod", "distributive": "distmod", "succ": "succmod", "successive": "succmod", "random": "randmod"}
    return aliases.get(model, model)


def _dist_rhs(t, y, params, num_psites: int):
    A, B, C, D = params[0], params[1], params[2], params[3]
    s = params[4 : 4 + num_psites]
    d = params[4 + num_psites : 4 + 2 * num_psites]
    R, P = y[0], y[1]
    X = y[2 : 2 + num_psites]
    dR = A - B * R
    dP = C * R - (D + jnp.sum(s)) * P + jnp.sum(X)
    dX = s * P - (1.0 + d) * X
    return jnp.concatenate([jnp.asarray([dR, dP], dtype=y.dtype), dX])


def _succ_rhs(t, y, params, num_psites: int):
    A, B, C, D = params[0], params[1], params[2], params[3]
    s = params[4 : 4 + num_psites]
    d = params[4 + num_psites : 4 + 2 * num_psites]
    R, P = y[0], y[1]
    X = y[2 : 2 + num_psites]
    dR = A - B * R
    dP = C * R - (D + s[0]) * P + X[0] if num_psites else C * R - D * P
    vals = []
    for i in range(num_psites):
        upstream = P if i == 0 else X[i - 1]
        downstream_return = X[i + 1] if i < num_psites - 1 else 0.0
        next_loss = s[i + 1] * X[i] if i < num_psites - 1 else 0.0
        vals.append(s[i] * upstream - (1.0 + d[i]) * X[i] - next_loss + downstream_return)
    return jnp.concatenate([jnp.asarray([dR, dP], dtype=y.dtype), jnp.asarray(vals, dtype=y.dtype)])


def _rand_rhs(t, y, params, num_psites: int):
    A, B, C, D = params[0], params[1], params[2], params[3]
    n = int(num_psites)
    m = (1 << n) - 1
    s = params[4 : 4 + n]
    ddeg = params[4 + n : 4 + n + m]
    R, P = y[0], y[1]
    X = y[2 : 2 + m]
    dR = A - B * R
    dP = C * R - D * P
    dX = jnp.zeros((m,), dtype=y.dtype)

    # P -> monophosphorylated states.
    for j in range(n):
        idx = (1 << j) - 1
        rate = s[j] * P
        dX = dX.at[idx].add(rate)
        dP = dP - rate

    # Transitions among phosphorylated subsets.  Dephosphorylation follows the
    # legacy numba model: unit-rate return to the lower subset/P, with Ddeg used
    # as state-specific degradation rather than as the dephosphorylation rate.
    for state in range(1, m + 1):
        base = state - 1
        xi = X[base]
        for j in range(n):
            bit = 1 << j
            if state & bit:
                lower = state & ~bit
                rate = xi
                if lower == 0:
                    dP = dP + rate
                else:
                    dX = dX.at[lower - 1].add(rate)
                dX = dX.at[base].add(-rate)
            else:
                target = (state | bit) - 1
                rate = s[j] * xi
                dX = dX.at[target].add(rate)
                dX = dX.at[base].add(-rate)
        dX = dX.at[base].add(-ddeg[base] * xi)

    return jnp.concatenate([jnp.asarray([dR, dP], dtype=y.dtype), dX])


def make_local_model_rhs(model_name: str | None, num_psites: int):
    """Return a Diffrax-compatible RHS matching the selected local ODE model."""
    model = _canonical_model_name(model_name)
    n = int(num_psites)
    if model in {"protwise", "distmod"}:
        return partial(_dist_rhs, num_psites=n)
    if model == "succmod":
        return partial(_succ_rhs, num_psites=n)
    if model == "randmod":
        return partial(_rand_rhs, num_psites=n)
    raise ValueError(f"Unsupported local ODE model: {model_name!r}")


def solve_protwise_ode(params, init_cond, num_psites, t, model_name: str | None = None):
    """Solve the selected local ODE model with the centralized Diffrax backend."""
    from config.constants import ODE_MODEL  # defer to avoid circular imports at module load

    model = _canonical_model_name(model_name or ODE_MODEL)
    expected_params = get_num_params(model, num_psites)
    params_arr = np.asarray(params, dtype=np.float64).reshape(-1)
    if params_arr.size != expected_params:
        raise ValueError(f"{model} expects {expected_params} parameters for {num_psites} sites, got {params_arr.size}.")
    rhs = make_local_model_rhs(model, num_psites)
    sol = np.asarray(
        solve_diffrax(
            np.asarray(init_cond, dtype=np.float64),
            np.asarray(t, dtype=np.float64),
            params=params_arr,
            rhs=rhs,
            config=DiffraxSolverConfig(),
        ),
        dtype=np.float64,
    )
    sol = np.clip(sol, 0, None)
    if NORMALIZE_MODEL_OUTPUT:
        norm_init = np.asarray(init_cond, dtype=sol.dtype)
        norm_init = np.where(norm_init == 0, 1.0, norm_init)
        sol = sol / norm_init[np.newaxis, :]
    r_fitted = sol[5:, 0].T if sol.shape[0] > 5 else sol[:, 0].T
    pr_fitted = sol[:, 1].T if sol.shape[1] > 1 else np.asarray([], dtype=sol.dtype)
    p_fitted = sol[:, 2:2 + num_psites].T if num_psites else np.asarray([], dtype=sol.dtype)
    return sol, np.concatenate((np.ravel(r_fitted), np.ravel(pr_fitted), np.ravel(p_fitted)))
