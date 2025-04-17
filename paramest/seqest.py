import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations
from typing import cast, Tuple

from config.config import score_fit
from config.constants import LAMBDA_REG, USE_REGULARIZATION, ODE_MODEL
from config.logconf import setup_logger
from models import solve_ode
from models.weights import early_emphasis, get_weight_options
from config.constants import get_param_names

logger = setup_logger()

def prepare_model_func(num_psites, init_cond, bounds, fixed_params,
                       use_regularization=True, lambda_reg=1e-3):
    """
    Prepare the model function for sequential parameter estimation.

    This function builds the model function based on the specified
    ODE model type and the number of phosphorylation sites. It also
    sets up the bounds for the free parameters and handles fixed
    parameters. The model function is used for curve fitting to
    estimate the parameters of the ODE model.

    :param num_psites:
    :param init_cond:
    :param bounds:
    :param fixed_params:
    :param use_regularization:
    :param lambda_reg:
    :return: model_func, free_indices, free_bounds, fixed_values, num_total_params
    """
    if ODE_MODEL == 'randmod':
        # Build lower and upper bounds from config.
        lower_bounds_full = [
            bounds["A"][0], bounds["B"][0], bounds["C"][0], bounds["D"][0]
        ]
        upper_bounds_full = [
            bounds["A"][1], bounds["B"][1], bounds["C"][1], bounds["D"][1]
        ]
        # For phosphorylation parameters: use Ssite bounds.
        lower_bounds_full += [bounds["Ssite"][0]] * num_psites
        upper_bounds_full += [bounds["Ssite"][1]] * num_psites
        # For dephosphorylation parameters: for each combination, use Dsite bounds.
        for i in range(1, num_psites + 1):
            for _ in combinations(range(1, num_psites + 1), i):
                lower_bounds_full.append(bounds["Dsite"][0])
                upper_bounds_full.append(bounds["Dsite"][1])
        num_total_params = len(lower_bounds_full)
    else:
        # Existing approach for distributive or successive models.
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
        for p in range(num_total_params):
            p_full[p] = fixed_values[i] if i in fixed_values else next(free_iter)
        _, p_fitted = solve_ode(p_full, init_cond, num_psites, np.atleast_1d(t))
        y_model = p_fitted.flatten()
        if use_regularization:
            reg = np.sqrt(lambda_reg) * np.array(p_free)
            return np.concatenate([y_model, reg])
        return y_model

    free_bounds = ([lower_bounds_full[i] for i in free_indices],
                   [upper_bounds_full[i] for i in free_indices])
    # logger.info(f"Model Built")
    return model_func, free_indices, free_bounds, fixed_values, num_total_params

def fit_parameters(time_points, p_data, model_func, p0_free,
                   bounds, weight_options,
                   use_regularization=True):
    """
    Fit the parameters of the model using curve fitting with different
    weighting options. This function iterates over the provided
    weight options, performs the curve fitting, and evaluates the
    goodness of fit using a scoring function. The best fitting
    parameters and their corresponding score are returned.

    :param time_points:
    :param p_data:
    :param model_func:
    :param p0_free:
    :param bounds:
    :param weight_options:
    :param use_regularization:
    :return: best_fit_params, best_weight_key, scores
    """
    scores, popts = {}, {}
    target = p_data.flatten()

    if use_regularization:
        target = np.concatenate([target, np.zeros(len(p0_free))])

    for key, sigma in weight_options.items():
        try:
            result = cast(Tuple[np.ndarray, np.ndarray],
                          curve_fit(model_func, time_points, target,
                          p0=p0_free, bounds=bounds,
                          sigma=sigma, absolute_sigma=True,
                          maxfev=20000))
            popt, _ = result
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
    logger.info(f"Score: {best_score:.2f}")
    return popts[best_key], best_key, scores

def sequential_estimation(p_data, time_points, init_cond, bounds,
                          fixed_params, num_psites, gene,
                          use_regularization=USE_REGULARIZATION, lambda_reg=LAMBDA_REG):
    """
    Perform sequential parameter estimation for a given gene using
    the specified model function. This function iteratively fits the
    model to the data at each time point, updating the initial guess
    for the parameters based on the previous fit. The estimated
    parameters, model fits, and error values are returned.

    :param p_data:
    :param time_points:
    :param init_cond:
    :param bounds:
    :param fixed_params:
    :param num_psites:
    :param gene:
    :param use_regularization:
    :param lambda_reg:
    :return:
    """
    est_params, model_fits, error_vals = [], [], []

    model_func, free_indices, free_bounds, fixed_values, num_total_params = (
        prepare_model_func(num_psites, init_cond, bounds, fixed_params,
                           use_regularization, lambda_reg)
    )

    p0_free = np.array([(lb + ub) / 2 for lb, ub in zip(*free_bounds)])

    for i in range(1, len(time_points) + 1):
        t_now = time_points[:i]
        y_now = p_data[:, :i] if p_data.ndim > 1 else p_data[:i].reshape(1, -1)
        y_flat = y_now.flatten()

        if use_regularization:
            target_fit = np.concatenate([y_flat, np.zeros(len(p0_free))])
        else:
            target_fit = y_flat

        try:
            # Attempt to get a good initial estimate using curve_fit.
            result = cast(Tuple[np.ndarray, np.ndarray],
                          curve_fit(model_func, t_now, target_fit,
                          p0=p0_free, bounds=free_bounds,
                          maxfev=20000))
            popt_init, _ = result
        except Exception as e:
            logger.warning(f"Initial fit failed at time index {i} for gene {gene}: {e}")
            popt_init = p0_free

        # Get weights for the model fitting.
        early_emphasis_weights = early_emphasis(y_now, t_now, num_psites)
        weights = get_weight_options(y_flat, t_now, num_psites,
                                     use_regularization, len(p0_free), early_emphasis_weights)

        # Perform the fit with the best weights.
        best_fit, weight_key, _ = fit_parameters(t_now, y_now, model_func, popt_init,
                                                 free_bounds, weights,
                                                 use_regularization)
        p_full = np.zeros(num_total_params)
        # Fill in the fixed parameters and free parameters.
        free_iter = iter(best_fit)
        # Iterate over the total parameters and assign values.
        for j in range(num_total_params):
            # If the parameter is fixed, use the fixed value.
            p_full[j] = fixed_values[j] if j in fixed_values else next(free_iter)
        # Append the estimated parameters and model fit.
        est_params.append(p_full)
        # Solve the ODE with the estimated parameters.
        sol, p_fit = solve_ode(p_full, init_cond, num_psites, t_now)
        # Flatten the model fit for error calculation.
        model_fits.append((sol, p_fit))
        # Calculate the mean square error value.
        error_vals.append(np.sum(np.abs(y_flat - p_fit.flatten()) ** 2))
        # Update the initial guess for the next iteration.
        p0_free = best_fit

        logger.info(f"[{gene}] Time Index {i} Best Weight = {weight_key}")

    return est_params, model_fits, error_vals

