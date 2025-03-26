
import numpy as np
from config.config import score_fit
from config.constants import get_param_names
from config.logging_config import setup_logger
from models.ode_model import solve_ode
from models.weights import early_emphasis, get_weight_options
from scipy.optimize import curve_fit

logger = setup_logger(__name__)


def prepare_model_func(num_psites, init_cond, bounds, fixed_params,
                       time_points, use_regularization=True, lambda_reg=1e-3):
    num_total_params = 4 + 2 * num_psites

    lower_bounds_full = (
        [bounds["A"][0], bounds["B"][0], bounds["C"][0], bounds["D"][0]] +
        [bounds["Ssite"][0]] * num_psites +
        [bounds["Dsite"][0]] * num_psites
    )
    upper_bounds_full = (
        [bounds["A"][1], bounds["B"][1], bounds["C"][1], bounds["D"][1]] +
        [bounds["Ssite"][1]] * num_psites +
        [bounds["Dsite"][1]] * num_psites
    )

    fixed_values = {}
    free_indices = []
    param_names = get_param_names(num_psites)

    for i, name in enumerate(param_names):
        val = fixed_params.get(name)
        if val is not None:
            fixed_values[i] = val
        else:
            free_indices.append(i)

    def model_func(t, *p_free):
        p_full = np.zeros(num_total_params)
        free_iter = iter(p_free)
        for i in range(num_total_params):
            p_full[i] = fixed_values[i] if i in fixed_values else next(free_iter)
        _, P_fitted = solve_ode(p_full, init_cond, num_psites, np.atleast_1d(t))
        y_model = P_fitted.flatten()
        if use_regularization:
            reg = np.sqrt(lambda_reg) * np.array(p_free)
            return np.concatenate([y_model, reg])
        return y_model

    free_bounds = ([lower_bounds_full[i] for i in free_indices],
                   [upper_bounds_full[i] for i in free_indices])
    logger.info(f"Model Building Complete")
    return model_func, free_indices, free_bounds, fixed_values, num_total_params

def fit_parameters(time_points, P_data, model_func, p0_free,
                   bounds, weight_options,
                   use_regularization=True):
    scores, popts = {}, {}
    target = P_data.flatten()

    if use_regularization:
        target = np.concatenate([target, np.zeros(len(p0_free))])

    for key, sigma in weight_options.items():
        try:
            popt, _ = curve_fit(model_func, time_points, target,
                                p0=p0_free, bounds=bounds,
                                sigma=sigma, absolute_sigma=True,
                                maxfev=20000)
        except Exception as e:
            logger.warning(f"Fit failed with {key}: {e}")
            popt = p0_free

        popts[key] = popt
        prediction = model_func(time_points, *popt)
        score = score_fit(target, prediction, popt)
        scores[key] = score
        logger.debug(f"[{key}] Score: {score:.4f}")

    best_key = min(scores, key=scores.get)
    best_score = scores[best_key]
    logger.info(f"Best Fit Weight: {best_key} with Score: {best_score:.4f}")
    return popts[best_key], best_key, scores

def sequential_estimation(P_data, time_points, init_cond, bounds,
                          fixed_params, num_psites, gene,
                          use_regularization=True, lambda_reg=1e-3):
    est_params, model_fits, error_vals = [], [], []

    model_func, free_indices, free_bounds, fixed_values, num_total_params = (
        prepare_model_func(num_psites, init_cond, bounds, fixed_params,
                           time_points, use_regularization, lambda_reg)
    )

    p0_free = np.array([(lb + ub) / 2 for lb, ub in zip(*free_bounds)])

    for i in range(1, len(time_points) + 1):
        t_now = time_points[:i]
        y_now = P_data[:, :i] if P_data.ndim > 1 else P_data[:i].reshape(1, -1)
        y_flat = y_now.flatten()

        if use_regularization:
            target_fit = np.concatenate([y_flat, np.zeros(len(p0_free))])
        else:
            target_fit = y_flat

        try:
            popt_init, _ = curve_fit(model_func, t_now, target_fit, p0=p0_free, bounds=free_bounds, maxfev=20000)
        except Exception as e:
            logger.warning(f"Initial fit failed at time index {i} for gene {gene}: {e}")
            popt_init = p0_free

        early_emphasis_weights = early_emphasis(y_now, t_now, num_psites)
        weights = get_weight_options(y_flat, t_now, num_psites,
                                     use_regularization, len(p0_free), early_emphasis_weights)

        best_fit, weight_key, _ = fit_parameters(t_now, y_now, model_func, popt_init,
                                                 free_bounds, weights,
                                                 use_regularization)
        p_full = np.zeros(num_total_params)
        free_iter = iter(best_fit)
        for j in range(num_total_params):
            p_full[j] = fixed_values[j] if j in fixed_values else next(free_iter)
        est_params.append(p_full)
        sol, P_fit = solve_ode(p_full, init_cond, num_psites, t_now)
        model_fits.append((sol, P_fit))
        error_vals.append(np.sum((y_flat - P_fit.flatten()) ** 2))
        p0_free = best_fit

        logger.info(f"[{gene}] Time Index {i}: Best Weight = {weight_key}")

    return est_params, model_fits, error_vals
