"""JAXopt/Diffrax single-objective parameter estimation for protwise models."""
from __future__ import annotations

import logging
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from config.constants import ODE_MODEL, USE_REGULARIZATION, get_num_params, get_param_names
from config.helpers import randmod_subset_masks
from networkmodel.backend import optimize_scalar_objective, solve_diffrax, DiffraxSolverConfig
from protwise.models.diffrax_solver import make_local_model_rhs

logger = logging.getLogger(__name__)

# Keep this deliberately small: the objective is already normalized by data scale, so
# this only discourages extreme rates without making small/flat dynamics attractive.
REGULARIZATION_WEIGHT = 1e-6
_MIN_PHYS_PARAM = 1e-8
_UNBOUNDED_UPPER_CAP = 1e6


def _canonical_model_name(model_name: str | None) -> str:
    model = str(model_name or ODE_MODEL).strip().lower()
    aliases = {"dist": "distmod", "distributive": "distmod", "succ": "succmod", "successive": "succmod", "random": "randmod"}
    return aliases.get(model, model)


def to_opt_space(params_phys, model_name: str | None = None):
    """Map physical kinetic parameters to optimizer coordinates.

    ProjectedGradient already enforces box constraints, so the optimizer space is
    intentionally identical to physical space.  This avoids the old mixed
    softplus/log/bounds parameterization where bounds were specified in physical
    units but optimization happened in a transformed space for only some models.
    """
    _canonical_model_name(model_name)  # validate/normalize for future extensions
    return np.asarray(params_phys, dtype=np.float64)


def from_opt_space(theta_opt, model_name: str | None = None):
    """Map optimizer coordinates to physical kinetic parameters."""
    _canonical_model_name(model_name)
    return jnp.asarray(theta_opt, dtype=jnp.float64)


def _as_bound_pair(value, name: str) -> tuple[float, float]:
    try:
        lo, hi = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bound for {name!r} must be a (lower, upper) pair; got {value!r}.") from exc
    lo_f, hi_f = float(lo), float(hi)
    if not np.isfinite(lo_f):
        lo_f = 0.0
    if hi_f < lo_f:
        raise ValueError(f"Bound for {name!r} must satisfy upper >= lower; got {(lo_f, hi_f)!r}.")
    return lo_f, hi_f


def _bounds_for_name(name: str, bounds: dict) -> tuple[float, float]:
    if name in bounds:
        return _as_bound_pair(bounds[name], name)
    if name.startswith("S") and "S(i)" in bounds:
        return _as_bound_pair(bounds["S(i)"], name)
    if name.startswith("D") and name != "D" and "D(i)" in bounds:
        return _as_bound_pair(bounds["D(i)"], name)
    raise ValueError(f"No bounds supplied for parameter {name!r}.")


def _normalize_bounds(bounds, model_name: str, num_psites: int):
    """Return physical lower/upper arrays in get_param_names() order.

    This replaces np.resize-based padding/repetition.  Resizing silently mapped,
    for example, a site bound onto a base kinetic parameter when the requested
    parameter count differed from the six CLI bound entries.
    """
    n_params = get_num_params(model_name, num_psites)
    if bounds is None:
        return np.full(n_params, _MIN_PHYS_PARAM, dtype=np.float64), np.full(n_params, 10.0, dtype=np.float64)

    if isinstance(bounds, dict):
        names = get_param_names(num_psites, model_name)
        pairs = [_bounds_for_name(name, bounds) for name in names]
        lower = np.asarray([p[0] for p in pairs], dtype=np.float64)
        upper = np.asarray([p[1] for p in pairs], dtype=np.float64)
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must be None, a dict, or a (lower, upper) pair.")
        lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
        upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
        if lower.size == 1:
            lower = np.full(n_params, float(lower[0]), dtype=np.float64)
        if upper.size == 1:
            upper = np.full(n_params, float(upper[0]), dtype=np.float64)
        if lower.size != n_params or upper.size != n_params:
            raise ValueError(
                f"Bounds length mismatch for {model_name}: lower={lower.size}, upper={upper.size}, expected {n_params}."
            )

    # Preserve explicit zero-fixed boxes such as (0, 0); only truly negative
    # lower bounds are lifted into the positive kinetic-parameter domain.
    lower = np.where(lower < 0.0, _MIN_PHYS_PARAM, lower)
    if np.any(np.isnan(upper) | (upper == -np.inf)):
        raise ValueError("Upper bounds must not be NaN or -inf.")
    if np.any(upper == np.inf):
        # Treat +inf as intentionally unbounded above, represented by a large
        # numeric cap for stable finite initialization/projection in JAXopt.
        upper = np.where(upper == np.inf, _UNBOUNDED_UPPER_CAP, upper)
    if np.any(upper < lower):
        raise ValueError("Parameter bounds must satisfy upper >= lower after normalization.")
    return lower, upper


def _loss_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    dyn = float(np.ptp(finite))
    var = float(np.var(finite))
    mag = float(np.mean(np.abs(finite)))
    return max(var, 0.25 * dyn * dyn, mag * mag, 1.0, 1e-8)


def aggregate_randmod_phospho(sol, num_psites):
    """Aggregate randmod subset states into site-level phospho predictions.

    Observations are site-level, while randmod states are subset-level. For example,
    site 1 signal is P1 + P12 + P13 + P123, so multi-site states must contribute
    to every site they contain.
    """
    m = (1 << int(num_psites)) - 1
    subset_states = sol[:, 2 : 2 + m]
    subset_masks = randmod_subset_masks(num_psites)
    if num_psites == 3:
        assert subset_masks == (1, 2, 4, 3, 5, 6, 7)
    membership = jnp.asarray(
        [[1.0 if mask & (1 << site_idx) else 0.0 for mask in subset_masks] for site_idx in range(num_psites)],
        dtype=sol.dtype,
    )
    return membership @ subset_states.T


def _split_predictions(sol, num_psites, n_rna_times, model_name: str | None = None):
    r = sol[:, 0]
    pr = sol[:, 1]
    model = _canonical_model_name(model_name)
    if num_psites and model == "randmod":
        ph = aggregate_randmod_phospho(sol, num_psites)
    else:
        ph = sol[:, 2:2 + num_psites].T if num_psites else jnp.zeros((0, sol.shape[0]), dtype=sol.dtype)
    r_fit = r[-n_rna_times:] if n_rna_times else jnp.asarray([], dtype=sol.dtype)
    return r_fit, pr, ph


def protwise_objective(theta, target, init_cond, num_psites, time_points, mode_weights, model_name: str | None = None):
    model = _canonical_model_name(model_name)
    params = from_opt_space(theta, model)
    rhs = make_local_model_rhs(model, num_psites)
    sol = solve_diffrax(
        jnp.asarray(init_cond, dtype=jnp.float64),
        jnp.asarray(time_points, dtype=jnp.float64),
        params=params,
        rhs=rhs,
        config=DiffraxSolverConfig(),
    )
    r_fit, pr_fit, ph_fit = _split_predictions(sol, num_psites, int(mode_weights["n_rna"]), model)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    if mode_weights["fit_mrna"]:
        total += jnp.mean((r_fit.reshape(-1) - target["mrna"]) ** 2) / mode_weights["scale_mrna"]
    if mode_weights["fit_protein"]:
        total += jnp.mean((pr_fit.reshape(-1) - target["protein"]) ** 2) / mode_weights["scale_protein"]
    if mode_weights["fit_phospho"]:
        total += jnp.mean((ph_fit.reshape(-1) - target["phospho"]) ** 2) / mode_weights["scale_phospho"]
    if USE_REGULARIZATION:
        total += REGULARIZATION_WEIGHT * jnp.mean(params * params)
    return jnp.asarray(total, dtype=jnp.float64)


def normest(gene, pr_data, p_data, r_data, init_cond, num_psites, time_points, bounds, bootstraps=0):
    """Estimate local-model parameters with a deterministic JAXopt projected-gradient path."""
    model = _canonical_model_name(ODE_MODEL)
    expected_state = 2 + ((1 << int(num_psites)) - 1 if model == "randmod" else int(num_psites))
    if len(init_cond) != expected_state:
        raise ValueError(f"[{gene}] Initial condition length {len(init_cond)} does not match {model} state dimension {expected_state}.")

    pr = np.asarray(pr_data, dtype=np.float64).reshape(-1)
    ph = np.asarray(p_data, dtype=np.float64).reshape(-1)
    mrna = np.asarray(r_data, dtype=np.float64).reshape(-1)
    mode = {
        "fit_mrna": mrna.size > 0,
        "fit_protein": pr.size > 0,
        "fit_phospho": ph.size > 0,
        "n_rna": mrna.size,
        "scale_mrna": _loss_scale(mrna),
        "scale_protein": _loss_scale(pr),
        "scale_phospho": _loss_scale(ph),
    }
    active = [name for name, flag in (("mrna", mode["fit_mrna"]), ("protein", mode["fit_protein"]), ("phospho", mode["fit_phospho"])) if flag]
    if not active:
        raise ValueError(f"[{gene}] No protwise data layers available for fitting.")
    logger.info("[%s] Detected protwise data mode: %s", gene, "+".join(active))
    logger.info("[%s] Selected optimizer backend: jaxopt.ProjectedGradient", gene)
    logger.info("[%s] Selected solver backend: diffrax.Kvaerno4", gene)

    n_params = get_num_params(model, num_psites)
    lower_phys, upper_phys = _normalize_bounds(bounds, model, num_psites)
    theta0_phys = np.clip(0.5 * (lower_phys + upper_phys), lower_phys, upper_phys)
    theta0 = to_opt_space(theta0_phys, model)
    lower_opt = to_opt_space(lower_phys, model)
    upper_opt = to_opt_space(upper_phys, model)

    target = {"mrna": jnp.asarray(mrna), "protein": jnp.asarray(pr), "phospho": jnp.asarray(ph)}

    def objective(x):
        return protwise_objective(x, target, init_cond, num_psites, time_points, mode, model)

    initial_value = float(objective(theta0))
    scaled_initial_value = float(objective(np.clip(theta0_phys * 2.0, lower_phys, upper_phys)))
    logger.info("[%s] Initial scalar objective value: %.8g", gene, initial_value)
    logger.info("[%s] Objective at doubled/clipped initial physical parameters: %.8g", gene, scaled_initial_value)

    best, state, value = optimize_scalar_objective(objective, theta0, lower_opt, upper_opt, maxiter=200, tol=1e-7, logger_obj=logger)
    final_params = np.asarray(from_opt_space(best, model), dtype=np.float64)
    final_params = np.clip(final_params, lower_phys, upper_phys)
    at_lower = np.mean(final_params <= (lower_phys + 1e-7 * np.maximum(1.0, np.abs(lower_phys))))
    logger.info("[%s] Final scalar objective value: %.8g (initial %.8g); fraction at lower bounds=%.3f", gene, value, initial_value, at_lower)
    logger.info("[%s] Optimizer iterations: %s", gene, getattr(state, "iter_num", "unknown"))

    rhs = make_local_model_rhs(model, num_psites)
    sol = np.asarray(solve_diffrax(np.asarray(init_cond, dtype=np.float64), np.asarray(time_points, dtype=np.float64), params=final_params, rhs=rhs, config=DiffraxSolverConfig()), dtype=np.float64)
    r_fit = sol[-mrna.size:, 0].reshape(-1) if mrna.size else np.asarray([], dtype=np.float64)
    pr_fit = sol[:, 1].reshape(-1)
    ph_fit = np.asarray(aggregate_randmod_phospho(jnp.asarray(sol), num_psites) if model == "randmod" and num_psites else sol[:, 2:2 + num_psites].T, dtype=np.float64).reshape(-1) if num_psites else np.asarray([], dtype=np.float64)
    seq_model_fit = np.concatenate([r_fit if mode["fit_mrna"] else np.asarray([]), pr_fit if mode["fit_protein"] else np.asarray([]), ph_fit if mode["fit_phospho"] else np.asarray([])])
    target_fit = np.concatenate([mrna if mode["fit_mrna"] else np.asarray([]), pr if mode["fit_protein"] else np.asarray([]), ph if mode["fit_phospho"] else np.asarray([])])
    errors = seq_model_fit - target_fit if target_fit.size == seq_model_fit.size else np.asarray([float(value)])
    estimated_params = np.vstack([final_params])
    model_fits = [(sol, seq_model_fit)]
    reg_term = float(REGULARIZATION_WEIGHT * np.mean(final_params * final_params)) if USE_REGULARIZATION else 0.0
    if final_params.size != n_params:
        raise ValueError(f"[{gene}] Estimated {final_params.size} parameters but {model} requires {n_params}.")
    return estimated_params, model_fits, errors, reg_term
