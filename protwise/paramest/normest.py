"""JAXopt/Diffrax single-objective parameter estimation for protwise models."""
from __future__ import annotations

import logging
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from config.config import score_fit
from config.constants import get_param_names, ODE_MODEL, USE_REGULARIZATION
from networkmodel.jax_backend import optimize_scalar_objective, solve_diffrax, DiffraxSolverConfig

logger = logging.getLogger(__name__)


def _split_predictions(sol, num_psites, n_rna_times):
    r = sol[:, 0]
    pr = sol[:, 1]
    ph = sol[:, 2:2 + num_psites].T if num_psites else jnp.zeros((0, sol.shape[0]), dtype=sol.dtype)
    r_fit = r[-n_rna_times:] if n_rna_times else jnp.asarray([], dtype=sol.dtype)
    return r_fit, pr, ph


def protwise_objective(theta, target, init_cond, num_psites, time_points, mode_weights):
    params = jax.nn.softplus(theta) if ODE_MODEL == "randmod" else jnp.clip(theta, 0.0, jnp.inf)
    sol = solve_diffrax(jnp.asarray(init_cond, dtype=jnp.float64), jnp.asarray(time_points, dtype=jnp.float64), params=params, config=DiffraxSolverConfig())
    r_fit, pr_fit, ph_fit = _split_predictions(sol, num_psites, int(mode_weights["n_rna"]))
    total = jnp.asarray(0.0, dtype=jnp.float64)
    if mode_weights["fit_mrna"]:
        total += jnp.mean((r_fit.reshape(-1) - target["mrna"]) ** 2)
    if mode_weights["fit_protein"]:
        total += jnp.mean((pr_fit.reshape(-1) - target["protein"]) ** 2)
    if mode_weights["fit_phospho"]:
        total += jnp.mean((ph_fit.reshape(-1) - target["phospho"]) ** 2)
    if USE_REGULARIZATION:
        total += 1e-4 * jnp.mean(params * params)
    return jnp.asarray(total, dtype=jnp.float64)


def normest(gene, pr_data, p_data, r_data, init_cond, num_psites, time_points, bounds, bootstraps=0):
    """Estimate protwise parameters with a deterministic JAXopt projected-gradient path."""
    pr = np.asarray(pr_data, dtype=np.float64).reshape(-1)
    ph = np.asarray(p_data, dtype=np.float64).reshape(-1)
    mrna = np.asarray(r_data, dtype=np.float64).reshape(-1)
    mode = {
        "fit_mrna": mrna.size > 0,
        "fit_protein": pr.size > 0,
        "fit_phospho": ph.size > 0,
        "n_rna": mrna.size,
    }
    active = [name for name, flag in (("mrna", mode["fit_mrna"]), ("protein", mode["fit_protein"]), ("phospho", mode["fit_phospho"])) if flag]
    if not active:
        raise ValueError(f"[{gene}] No protwise data layers available for fitting.")
    logger.info("[%s] Detected protwise data mode: %s", gene, "+".join(active))
    logger.info("[%s] Selected optimizer backend: jaxopt.ProjectedGradient", gene)
    logger.info("[%s] Selected solver backend: diffrax.Kvaerno4", gene)

    n_params = 4 + 2 * int(num_psites)
    if bounds is None:
        lower = np.zeros(n_params, dtype=np.float64)
        upper = np.full(n_params, 10.0, dtype=np.float64)
    else:
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        if lower.size != n_params:
            lower = np.resize(lower, n_params)
        if upper.size != n_params:
            upper = np.resize(upper, n_params)
    theta0 = np.clip((lower + upper) / 2.0, lower, upper)
    if ODE_MODEL == "randmod":
        theta0 = np.log(np.maximum(theta0, 1e-8))
        lower_opt = np.log(np.maximum(lower, 1e-8))
        upper_opt = np.log(np.maximum(upper, 1e-8))
    else:
        lower_opt, upper_opt = lower, upper

    target = {"mrna": jnp.asarray(mrna), "protein": jnp.asarray(pr), "phospho": jnp.asarray(ph)}

    def objective(x):
        return protwise_objective(x, target, init_cond, num_psites, time_points, mode)

    best, state, value = optimize_scalar_objective(objective, theta0, lower_opt, upper_opt, maxiter=50, tol=1e-6, logger_obj=logger)
    final_params = np.exp(best) if ODE_MODEL == "randmod" else np.clip(best, 0.0, None)
    sol = np.asarray(solve_diffrax(np.asarray(init_cond, dtype=np.float64), np.asarray(time_points, dtype=np.float64), params=final_params), dtype=np.float64)
    r_fit = sol[-mrna.size:, 0].reshape(-1) if mrna.size else np.asarray([], dtype=np.float64)
    pr_fit = sol[:, 1].reshape(-1)
    ph_fit = sol[:, 2:2 + num_psites].T.reshape(-1) if num_psites else np.asarray([], dtype=np.float64)
    seq_model_fit = np.concatenate([r_fit if mode["fit_mrna"] else np.asarray([]), pr_fit if mode["fit_protein"] else np.asarray([]), ph_fit if mode["fit_phospho"] else np.asarray([])])
    target_fit = np.concatenate([mrna if mode["fit_mrna"] else np.asarray([]), pr if mode["fit_protein"] else np.asarray([]), ph if mode["fit_phospho"] else np.asarray([])])
    errors = seq_model_fit - target_fit if target_fit.size == seq_model_fit.size else np.asarray([float(value)])
    estimated_params = np.vstack([final_params])
    model_fits = [(sol, seq_model_fit)]
    reg_term = float(value)
    logger.info("[%s] Final scalar objective value: %.8g", gene, value)
    return estimated_params, model_fits, errors, reg_term
