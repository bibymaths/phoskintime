"""Diffrax-backed local phosphorylation model solver helpers."""
from __future__ import annotations

from functools import lru_cache, partial
import numpy as np
import jax.numpy as jnp

from networkmodel.backend import solve_diffrax, DiffraxSolverConfig
from config.constants import get_num_params
from config.helpers import randmod_subset_masks


def _canonical_model_name(model_name: str | None) -> str:
    model = str(model_name or "protwise").strip().lower()
    aliases = {"dist": "distmod", "distributive": "distmod", "succ": "succmod", "successive": "succmod",
               "random": "randmod"}
    return aliases.get(model, model)


# Dephosphorylation rate k = 1.0 /min is a fixed constant in all three local models (the "1.0" below), not a fitted
# parameter: it is not identifiable from the current data (KNOWN_ISSUES P9; fitted k 1.04-1.27, no improvement).
def _dist_rhs(t, y, params, num_psites: int):
    A, B, C, D = params[0], params[1], params[2], params[3]
    s = params[4: 4 + num_psites]
    d = params[4 + num_psites: 4 + 2 * num_psites]
    R, P = y[0], y[1]
    X = y[2: 2 + num_psites]
    dR = A - B * R
    dP = C * R - (D + jnp.sum(s)) * P + jnp.sum(X)
    dX = s * P - (1.0 + d) * X
    return jnp.concatenate([jnp.asarray([dR, dP], dtype=y.dtype), dX])


def _succ_rhs(t, y, params, num_psites: int):
    A, B, C, D = params[0], params[1], params[2], params[3]
    s = params[4: 4 + num_psites]
    d = params[4 + num_psites: 4 + 2 * num_psites]
    R, P = y[0], y[1]
    X = y[2: 2 + num_psites]
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
    s = params[4: 4 + n]
    ddeg = params[4 + n: 4 + n + m]
    R, P = y[0], y[1]
    X = y[2: 2 + m]
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
    subset_states = arr[:, 2: 2 + m]
    ph_site = np.zeros((n, arr.shape[0]), dtype=arr.dtype)
    for site_idx in range(n):
        containing = [idx for idx, mask in enumerate(subset_masks) if mask & (1 << site_idx)]
        if containing:
            ph_site[site_idx, :] = np.sum(subset_states[:, containing], axis=1)
    return ph_site


def make_local_model_rhs(model_name: str | None, num_psites: int):
    """Return a Diffrax-compatible RHS matching the selected local ODE model.

    The same function object is returned for the same (model, number of sites), so
    Diffrax's jit cache, which is keyed on the vector field, is reused across solves
    instead of re-tracing and recompiling on every call.
    """
    return _make_local_model_rhs_cached(_canonical_model_name(model_name), int(num_psites))


@lru_cache(maxsize=None)
def _make_local_model_rhs_cached(model: str, n: int):
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
    raise ValueError(f"Unsupported local ODE model: {model!r}")


def rna_time_grid(time_points, n_rna: int | None = None, rna_times=None):
    """Union solve grid for protwise (P1, any grid): protein/phospho are read at ``time_points``, RNA at the RNA
    observation times; one rule for fit, final fit, export and sensitivity.

    RNA times: ``rna_times`` if given; otherwise the configured RNA grid when ``n_rna`` is None or equals its
    length; otherwise ``time_points`` when ``n_rna`` equals their number (RNA measured on the protein grid);
    anything else is an error. ``n_rna = 0`` means no RNA. The initial time stays ``time_points[0]``; RNA times
    before it are an error. Returns ``(grid, t_idx, rna_idx)`` with ``grid[t_idx] == time_points`` and
    ``grid[rna_idx] == rna_times``.
    """
    from config.constants import TIME_POINTS_RNA  # defer to avoid circular imports at module load

    t = np.asarray(time_points, dtype=np.float64).reshape(-1)
    cfg_r = np.asarray(TIME_POINTS_RNA, dtype=np.float64).reshape(-1)
    if rna_times is not None:
        r = np.asarray(rna_times, dtype=np.float64).reshape(-1)
        if n_rna not in (None, 0) and r.size != n_rna:
            raise ValueError(f"{n_rna} RNA observations but {r.size} RNA times {r.tolist()}.")
    elif n_rna == 0:
        r = cfg_r[:0]
    elif n_rna is None or n_rna == cfg_r.size:
        r = cfg_r
    elif n_rna == t.size:
        r = t
    else:
        raise ValueError(f"{n_rna} RNA observations match neither the configured RNA grid ({cfg_r.size} times) nor "
                         f"the time grid ({t.size} times); pass rna_times explicitly.")
    if r.size and t.size and r.min() < t[0]:
        raise ValueError(f"RNA times {r.tolist()} start before the initial time {t[0]}.")
    grid = np.union1d(t, r)
    return grid, np.searchsorted(grid, t), np.searchsorted(grid, r)


def split_flat_fit(flat, num_psites: int, n_t: int, n_rna: int | None = None):
    """Split the flat protwise fit vector ``[RNA (n_rna) | protein (n_t) | phospho (num_psites x n_t)]``
    returned by :func:`solve_protwise_ode` (P2: sizes from the grids, not hard-coded 9/14/23).
    ``n_rna`` defaults to the configured RNA grid.
    """
    from config.constants import TIME_POINTS_RNA  # defer to avoid circular imports at module load

    flat = np.asarray(flat).reshape(-1)
    n_r = len(TIME_POINTS_RNA) if n_rna is None else int(n_rna)
    n_t, num_psites = int(n_t), int(num_psites)
    expected = n_r + n_t + num_psites * n_t
    if flat.size != expected:
        raise ValueError(f"fit vector has {flat.size} values; expected {n_r} RNA + {n_t} protein + "
                         f"{num_psites}x{n_t} phospho = {expected}.")
    return flat[:n_r], flat[n_r:n_r + n_t], flat[n_r + n_t:].reshape(num_psites, n_t)


def solve_protwise_ode(params, init_cond, num_psites, t, model_name: str | None = None, rna_times=None):
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
    grid, t_idx, rna_idx = rna_time_grid(t, None, rna_times)
    sol_grid = np.asarray(
        solve_diffrax(
            np.asarray(init_cond, dtype=np.float64),
            grid,
            params=params_arr,
            rhs=rhs,
            config=DiffraxSolverConfig(),
        ),
        dtype=np.float64,
    )
    sol_raw = np.clip(sol_grid[t_idx], 0, None)
    r_raw = np.clip(sol_grid[rna_idx, 0], 0, None)

    sol_out = sol_raw.copy()
    r_fitted = r_raw.copy()

    if model == "randmod" and num_psites:
        # Sensitivity/plotting callers read solution[:, 2:2+num_psites] as
        # site-level phospho. Randmod stores subset states internally, so expose a
        # site-level view in those columns while keeping the flattened fit matched
        # to the JAX objective.
        p_fitted = aggregate_randmod_site_phospho(sol_raw, num_psites)
        sol_out[:, 2:2 + num_psites] = p_fitted.T
    else:
        p_fitted = sol_out[:, 2:2 + num_psites].T if num_psites else np.asarray([], dtype=sol_out.dtype)

    pr_fitted = sol_out[:, 1].T if sol_out.shape[1] > 1 else np.asarray([], dtype=sol_out.dtype)
    return sol_out, np.concatenate((np.ravel(r_fitted), np.ravel(pr_fitted), np.ravel(p_fitted)))
