import pandas as pd
import numpy as np
from typing import cast, Tuple
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

from config.config import score_fit
from config.constants import LAMBDA_REG, USE_REGULARIZATION, ODE_MODEL
from config.logconf import setup_logger
from models import solve_ode
from models.weights import early_emphasis, get_weight_options
from plotting import Plotter
from config.constants import get_param_names

logger = setup_logger()

def prepare_model_func(num_psites, init_cond, bounds, fixed_params,
                       use_regularization=USE_REGULARIZATION, lambda_reg=LAMBDA_REG):
    """
    Prepares the model function for curve fitting.
    Adaptive Estimation is used to estimate the parameters of the model.

    :param num_psites:
    :param init_cond:
    :param bounds:
    :param fixed_params:
    :param use_regularization:
    :param lambda_reg:
    :return: model function, free indices, free bounds, fixed values, total params
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
        for r in range(1, num_psites + 1):
            for _ in combinations(range(1, num_psites + 1), r):
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
    # Create a dictionary of fixed parameters.
    fixed_values = {}
    # Create a dictionary of fixed values.
    free_indices = []
    # Create a list of free indices.
    param_names = get_param_names(num_psites)
    for p, name in enumerate(param_names):
        val = fixed_params.get(name)
        if val is not None:
            fixed_values[p] = val
        else:
            free_indices.append(p)

    def model_func(t, *p_free):
        """
        Model function for curve fitting.

        :param t:
        :param p_free:
        :return: model
        """
        p_full = np.zeros(num_total_params)
        # Fill in the fixed values.
        free_iter = iter(p_free)
        # Fill in the free values.
        for i in range(num_total_params):
            # If the index is in the fixed values, use the fixed value.
            p_full[i] = fixed_values[i] if i in fixed_values else next(free_iter)

        _, p_fitted = solve_ode(p_full, init_cond, num_psites, np.atleast_1d(t))
        y_model = p_fitted.flatten()
        # If using regularization, add the regularization term.
        if use_regularization:
            # Add the regularization term.
            reg = np.sqrt(lambda_reg) * np.array(p_free)
            # Add the regularization term to the model.
            return np.concatenate([y_model, reg])
        # Return the model.
        return y_model

    # Create the free bounds.
    free_bounds = ([lower_bounds_full[i] for i in free_indices],
                   [upper_bounds_full[i] for i in free_indices])

    # logger.info("Model built")
    return model_func, free_indices, free_bounds, fixed_values, num_total_params


def adaptive_estimation(p_data, init_cond, num_psites, time_points, t,
                        bounds, fixed_params, gene, bootstraps=0, extra_fixed=None,
                        use_regularization=USE_REGULARIZATION, lambda_reg=LAMBDA_REG):
    """
    Adaptive estimation of parameters using curve fitting.
    This function estimates the parameters of the model using curve fitting.
    Parameter will be fixed at certain time points.

    :param p_data:
    :param init_cond:
    :param num_psites:
    :param time_points:
    :param t:
    :param bounds:
    :param fixed_params:
    :param gene:
    :param bootstraps:
    :param extra_fixed:
    :param use_regularization:
    :param lambda_reg:
    :return: estimated parameters
    """
    # Get the index of the time points that are less than or equal to t.
    t_idx = np.where(time_points <= t)[0]
    t_target = time_points[t_idx]
    p_interp = np.zeros((num_psites, len(t_target)))

    # Interpolate the data to get the values at the target time points.
    for i in range(num_psites):
        # Get the values at the target time points.
        y_vals = p_data.iloc[i, 2:].values if isinstance(p_data, pd.DataFrame) else p_data[i, 2:]
        # Interpolate the values at the target time points.
        interp_func = PchipInterpolator(time_points, y_vals)
        # Get the values at the target time points.
        p_interp[i, :] = interp_func(t_target)

    full_fixed = fixed_params.copy()
    if extra_fixed:
        full_fixed.update(extra_fixed)

    # Get model function and bounds.
    model_func, free_idx, free_bounds, fixed_vals, total_params = prepare_model_func(
        num_psites, init_cond, bounds,
        full_fixed, use_regularization, lambda_reg
    )

    # Initial values for the free parameters.
    p0 = np.array([(l + u) / 2 for l, u in zip(*free_bounds)])
    target = p_interp.flatten()

    # Make the fitting vector the same size as the model.
    target_fit = np.concatenate([target, np.zeros(len(p0))]) if use_regularization else target
    default_sigma = 1 / np.maximum(np.abs(target_fit), 1e-5)

    try:
        # Perform initial fit to get starting values.
        result = cast(Tuple[np.ndarray, np.ndarray],
                      curve_fit(model_func, t_target, target_fit, p0=p0,
                                bounds=free_bounds, sigma=default_sigma,
                                absolute_sigma=True, maxfev=20000))
        popt_init, _ = result
    except Exception as e:
        logger.warning(f"[{gene}] T={t} init fit failed: {e}")
        popt_init = p0

    # Get weight options for the fit.
    early_weights = early_emphasis(p_interp, t_target, num_psites)
    weight_options = get_weight_options(target, t_target, num_psites, use_regularization, len(p0), early_weights)

    # Make a dictionary of the weights, and parameters.
    scores, popts = {}, {}
    for wname, sigma in weight_options.items():
        try:
            # Perform the fit with the weights.
            result = cast(Tuple[np.ndarray, np.ndarray],
                          curve_fit(model_func, t_target, target_fit, p0=popt_init,
                                    bounds=free_bounds, sigma=sigma,
                                    absolute_sigma=True, maxfev=20000))
            popt_weight, _ = result
        except Exception as e:
            logger.warning(f"[{gene}] T={t} fit failed for {wname}: {e}")
            popt_weight = popt_init
        popts[wname] = popt_weight
        pred = model_func(t_target, *popt_weight)
        # Calculate the score for the fit.
        scores[wname] = score_fit(target_fit, pred, popt_weight)

    # Get the best weight and score.
    best_weight = min(scores, key=scores.get)
    best_score = scores[best_weight]
    # Get the best parameters.
    popt_best = popts[best_weight]
    # Display the best weight and score for time points.
    logger.info(f"[{gene}] Time = {t} Best Weight = {best_weight} Score: {best_score:.2f}")

    if bootstraps > 0:
        logger.info(f"[{gene}] Bootstrapping {bootstraps}x at T = {t}")
        # Perform bootstrapping to get the final parameters.
        est_list = []

        for _ in range(bootstraps):
            # Generate bootstrap samples.
            noise = np.random.normal(0, 0.05, size=target_fit.shape)
            noisy_target = target_fit * (1 + noise)
            try:
                # Perform the fit with the bootstrap samples.
                result = cast(Tuple[np.ndarray, np.ndarray],
                              curve_fit(model_func, t_target, noisy_target, p0=popt_best,
                                        bounds=free_bounds, sigma=weight_options[best_weight],
                                        absolute_sigma=True, maxfev=20000))
                popt_weight_bootstrap, _ = result
            except (RuntimeError, ValueError) as e:
                logger.warning(f"[{gene}] Bootstrap fit failed: {e}")
                popt_weight_bootstrap = popt_best
            est_list.append(popt_weight_bootstrap)

        # Get the best parameters from bootstrap samples.
        popt_best = np.mean(est_list, axis=0)

    p_full = np.zeros(total_params)
    free_iter = iter(popt_best)

    # Fill in the fixed values.
    for i in range(total_params):
        # If the index is in the fixed values, use the fixed value.
        p_full[i] = fixed_vals[i] if i in fixed_vals else next(free_iter)

    return p_full


def estimate_profiles(gene, p_data, init_cond, num_psites, time_points, desired_times,
                      bounds, fixed_params, bootstraps, time_fixed_dict):
    """
    Estimate adaptive profiles for a given gene at specified time points.
    This function estimates the parameters of the model using adaptive estimation.
    The parameters are estimated at the specified time points and stored in a DataFrame.
    The function also plots the profiles.

    :param gene:
    :param p_data:
    :param init_cond:
    :param num_psites:
    :param time_points:
    :param desired_times:
    :param bounds:
    :param fixed_params:
    :param bootstraps:
    :param time_fixed_dict:
    :return: dataframe, profiles
    """
    profiles = {}
    for T in desired_times:
        # logger.info(f"[{gene}] Estimating Adaptive Profile at T = {T}")
        extra_fixed = time_fixed_dict.get(str(T), {})
        profiled_params = adaptive_estimation(p_data, init_cond, num_psites, time_points, T,
                                              bounds, fixed_params, gene, bootstraps, extra_fixed)
        profiles[T] = profiled_params

    param_names = get_param_names(num_psites)
    data = {"Time": list(profiles.keys())}
    for i, name in enumerate(param_names):
        data[name] = [profiles[T][i] for T in desired_times]
    df = pd.DataFrame(data)
    plotter = Plotter(gene)
    plotter.plot_profiles(df)
    return df, profiles
