
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

from config.config import OUTPUT_DIR
from plotting.plotting import plot_profiles
from config.config import score_fit
from config.constants import get_param_names, LAMBDA_REG, USE_REGULARIZATION
from config.logging_config import setup_logger
from models.ode_model import solve_ode
from models.weights import early_emphasis, get_weight_options


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

def profile_estimation(P_data, init_cond, num_psites, time_points, T, bounds, fixed_params, gene,
                       num_bootstraps=0, extra_fixed=None):
    valid_idx = np.where(time_points <= T)[0]
    t_target = time_points[valid_idx]
    P_interp = np.zeros((num_psites, len(t_target)))

    for i in range(num_psites):
        y_vals = P_data.iloc[i, 2:].values if isinstance(P_data, pd.DataFrame) else P_data[i, 2:]
        interp_func = PchipInterpolator(time_points, y_vals)
        P_interp[i, :] = interp_func(t_target)

    local_fixed = fixed_params.copy()
    if extra_fixed:
        local_fixed.update(extra_fixed)

    model_func, free_indices, free_bounds, fixed_values, num_total_params = (
        prepare_model_func(num_psites, init_cond, bounds, local_fixed, t_target,
                           use_regularization=USE_REGULARIZATION, lambda_reg=LAMBDA_REG)
    )

    p0_free = np.array([(lb + ub) / 2 for lb, ub in zip(*free_bounds)])
    target = P_interp.flatten()
    if USE_REGULARIZATION:
        target_fit = np.concatenate([target, np.zeros(len(p0_free))])
    else:
        target_fit = target

    default_sigma = 1 / np.maximum(np.abs(target_fit), 1e-5)

    try:
        popt_free_initial, _ = curve_fit(model_func, t_target, target_fit, p0=p0_free, maxfev=20000,
                                         bounds=free_bounds,
                                         sigma=default_sigma, absolute_sigma=True)
    except Exception as e:
        logger.warning(f"Initial estimation failed for gene {gene} at T={T}: {e}")
        popt_free_initial = p0_free

    early_weights = early_emphasis(P_interp, t_target, num_psites)
    weight_options = get_weight_options(target, t_target, num_psites, USE_REGULARIZATION, len(p0_free), early_weights)

    scores, popts = {}, {}
    for weight_name, sigma in weight_options.items():
        try:
            popt, _ = curve_fit(model_func, t_target, target_fit, p0=p0_free, maxfev=20000,
                                bounds=free_bounds,
                                sigma=sigma, absolute_sigma=True)
        except Exception as e:
            logger.warning(f"Profiled Estimation failed with {weight_name} scheme: {e}")
            popt = popt_free_initial
        popts[weight_name] = popt
        pred = model_func(t_target, *popt)
        scores[weight_name] = score_fit(target_fit, pred, popt)

    best_weight = min(scores, key=scores.get)
    popt_best = popts[best_weight]
    logger.info(f"[{gene}] T={T} Profiled Estimation Best Weight = {best_weight}")

    if num_bootstraps > 0:
        boot_estimates = []
        for b in range(num_bootstraps):
            noise = np.random.normal(0, 0.05, size=target_fit.shape)
            boot_target = target_fit * (1 + noise)
            try:
                popt, _ = curve_fit(model_func, t_target, boot_target, p0=popt_best, maxfev=20000,
                                    bounds=free_bounds,
                                    sigma=weight_options[best_weight], absolute_sigma=True)
            except Exception as e:
                logger.warning(f"Bootstrap {b} failed for gene {gene} at T={T}: {e}")
                popt = popt_best
            boot_estimates.append(popt)
        popt_best = np.mean(boot_estimates, axis=0)

        logger.info(f"[{gene}] Profiled Bootstrapped ({num_bootstraps} times) Estimation Best Weight = {best_weight}")

    p_full = np.zeros(num_total_params)
    free_iter = iter(popt_best)
    for i in range(num_total_params):
        p_full[i] = fixed_values[i] if i in fixed_values else next(free_iter)

    return p_full


def save_profiles(gene, P_data, init_cond, num_psites, time_points, desired_times, bounds, fixed_params,
                  bootstraps, time_fixed, out_dir=OUTPUT_DIR):
    if init_cond is None or num_psites is None:
        if isinstance(P_data, pd.DataFrame):
            gene_data = P_data[P_data['Gene'] == gene]
            num_psites = gene_data.shape[0]
            P_data_array = gene_data.iloc[:, 2:].values
        else:
            raise ValueError("P_data must be a DataFrame when init_cond and num_psites are None")
    else:
        P_data_array = P_data

    profiles = {}
    for T in desired_times:
        logger.info(f"Profiled Estimation for {gene} at T = {T} min")
        extra_fixed = time_fixed.get(str(T))
        est = profile_estimation(P_data_array, init_cond, num_psites, time_points, T,
                                 bounds, fixed_params, gene, num_bootstraps=bootstraps,
                                 extra_fixed=extra_fixed)
        profiles[T] = est

    full_names = get_param_names(num_psites)
    data = {"Time (min)": list(profiles.keys())}
    for i, name in enumerate(full_names):
        data[name] = [profiles[T][i] for T in desired_times]

    df = pd.DataFrame(data)
    filename = os.path.join(out_dir, f"{gene}_profiled.xlsx")
    df.to_excel(filename, index=False)
    plot_profiles(gene, df, out_dir)
    logger.info(f"\nParameter Profiles for {gene} are saved at defined times to {filename}")