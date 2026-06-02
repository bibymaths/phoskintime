"""Diffrax-backed local phosphorylation model solver helpers."""
from __future__ import annotations

from functools import partial
import numpy as np
import jax.numpy as jnp

from networkmodel.jax_backend import solve_diffrax, DiffraxSolverConfig
from config.constants import NORMALIZE_MODEL_OUTPUT, get_num_params
from config.helpers import randmod_subset_masks


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


def _rand_rhs(t, y, params, num_psites: int, subset_masks: tuple[int, ...], mask_to_index: dict[int, int]):
    A, B, C, D = params[0], params[1], params[2], params[3]
    n = int(num_psites)
    m = len(subset_masks)
    s = params[4 : 4 + n]
    ddeg = params[4 + n : 4 + n + m]
    R, P = y[0], y[1]
    X = y[2 : 2 + m]
    dR = A - B * R
    dP = C * R - D * P
    dX = jnp.zeros((m,), dtype=y.dtype)

    # Randmod state order is canonical combination order (P1, P2, P3, P12, ...),
    # not raw bitmask order. All state and Ddeg indexing goes through mask_to_index.
    for j in range(n):
        idx = mask_to_index[1 << j]
        rate = s[j] * P
        dX = dX.at[idx].add(rate)
        dP = dP - rate

    # Transitions among phosphorylated subsets. Dephosphorylation follows the
    # legacy numba model: unit-rate return to the lower subset/P, with Ddeg used
    # as state-specific degradation rather than as the dephosphorylation rate.
    for base, state_mask in enumerate(subset_masks):
        xi = X[base]
        for j in range(n):
            bit = 1 << j
            if state_mask & bit:
                lower = state_mask & ~bit
                rate = xi
                if lower == 0:
                    dP = dP + rate
                else:
                    dX = dX.at[mask_to_index[lower]].add(rate)
                dX = dX.at[base].add(-rate)
            else:
                target = state_mask | bit
                rate = s[j] * xi
                dX = dX.at[mask_to_index[target]].add(rate)
                dX = dX.at[base].add(-rate)
        dX = dX.at[base].add(-ddeg[base] * xi)

    return jnp.concatenate([jnp.asarray([dR, dP], dtype=y.dtype), dX])



def aggregate_randmod_site_phospho(sol, num_psites: int):
    """Return randmod site-level phospho curves from subset-level states.

    The JAX objective compares site-level phospho observations, so public
    solve_ode output must also sum every subset state into each site it contains.
    """
    arr = np.asarray(sol)
    n = int(num_psites)
    if n <= 0:
        return np.zeros((0, arr.shape[0]), dtype=arr.dtype)
    subset_masks = randmod_subset_masks(n)
    m = len(subset_masks)
    subset_states = arr[:, 2 : 2 + m]
    ph_site = np.zeros((n, arr.shape[0]), dtype=arr.dtype)
    for site_idx in range(n):
        containing = [idx for idx, mask in enumerate(subset_masks) if mask & (1 << site_idx)]
        if containing:
            ph_site[site_idx, :] = np.sum(subset_states[:, containing], axis=1)
    return ph_site

def make_local_model_rhs(model_name: str | None, num_psites: int):
    """Return a Diffrax-compatible RHS matching the selected local ODE model."""
    model = _canonical_model_name(model_name)
    n = int(num_psites)
    if model in {"protwise", "distmod"}:
        return partial(_dist_rhs, num_psites=n)
    if model == "succmod":
        return partial(_succ_rhs, num_psites=n)
    if model == "randmod":
        subset_masks = randmod_subset_masks(n)
        mask_to_index = {mask: idx for idx, mask in enumerate(subset_masks)}
        if n == 3:
            assert subset_masks == (1, 2, 4, 3, 5, 6, 7)
        return partial(_rand_rhs, num_psites=n, subset_masks=subset_masks, mask_to_index=mask_to_index)
    raise ValueError(f"Unsupported local ODE model: {model_name!r}")


def solve_protwise_ode(params, init_cond, num_psites, t, model_name: str | None = None):
    """Solve the selected local ODE model with the centralized Diffrax backend."""
    from config.constants import ODE_MODEL  # defer to avoid circular imports at module load

    # model_name=None intentionally means "use global ODE_MODEL". Mechanism-specific
    # wrappers must pass their own name so direct module calls cannot pick the wrong model.
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
    sol_raw = np.clip(sol, 0, None)

    if NORMALIZE_MODEL_OUTPUT:
        norm_init = np.asarray(init_cond, dtype=sol_raw.dtype)
        norm_init = np.where(norm_init == 0, 1.0, norm_init)
        sol_out = sol_raw / norm_init[np.newaxis, :]
    else:
        sol_out = sol_raw.copy()

    if model == "randmod" and num_psites:
        # Sensitivity/plotting callers read solution[:, 2:2+num_psites] as
        # site-level phospho. Randmod stores subset states internally, so expose a
        # site-level view in those columns while keeping the flattened fit matched
        # to the JAX objective. Normalize after aggregation when requested.
        p_fitted = aggregate_randmod_site_phospho(sol_raw, num_psites)
        if NORMALIZE_MODEL_OUTPUT:
            init_vals = p_fitted[:, 0:1]
            p_fitted = np.where(init_vals != 0, p_fitted / init_vals, p_fitted)
        sol_out[:, 2:2 + num_psites] = p_fitted.T
    else:
        p_fitted = sol_out[:, 2:2 + num_psites].T if num_psites else np.asarray([], dtype=sol_out.dtype)

    r_fitted = sol_out[5:, 0].T if sol_out.shape[0] > 5 else sol_out[:, 0].T
    pr_fitted = sol_out[:, 1].T if sol_out.shape[1] > 1 else np.asarray([], dtype=sol_out.dtype)
    return sol_out, np.concatenate((np.ravel(r_fitted), np.ravel(pr_fitted), np.ravel(p_fitted)))
