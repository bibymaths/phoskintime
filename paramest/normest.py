from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from itertools import combinations
from typing import cast, Tuple

from config.config import score_fit
from config.constants import get_param_names, USE_REGULARIZATION, ODE_MODEL, ALPHA_CI, OUT_DIR, \
    USE_CUSTOM_WEIGHTS
from config.logconf import setup_logger
from models import solve_ode
from models.weights import early_emphasis, get_weight_options, get_protein_weights
from plotting import Plotter
from .identifiability import confidence_intervals

logger = setup_logger()

def worker_find_lambda(
    lam: float,
    gene: str,
    target: np.ndarray,
    p0: np.ndarray,
    time_points: np.ndarray,
    free_bounds: Tuple[np.ndarray, np.ndarray],
    init_cond: np.ndarray,
    num_psites: int,
    p_data: np.ndarray
) -> Tuple[float, float, str]:
    """
    Worker function for a single lambda value.
    """
    def model_func(tpts, *params):
        param_vec = np.exp(np.asarray(params)) if ODE_MODEL == 'randmod' else np.asarray(params)
        _, p_fitted = solve_ode(param_vec, init_cond, num_psites, np.atleast_1d(tpts))
        y_model = p_fitted.flatten()
        reg = lam/len(params) * np.square(params)
        return np.concatenate([y_model, reg])

    tf = np.concatenate([target, np.zeros(len(p0))])
    early_weights = early_emphasis(p_data, time_points, num_psites)
    ms_gauss_weights = get_protein_weights(gene)

    weight_options = get_weight_options(
        target, time_points, num_psites,
        use_regularization=True,
        reg_len=len(p0),
        early_weights=early_weights,
        ms_gauss_weights=ms_gauss_weights
    )

    best_score = float("inf")
    best_weight_key = None

    for weight_key, sigma in weight_options.items():
        try:
            result = cast(Tuple[np.ndarray, np.ndarray], curve_fit(
                model_func,
                time_points,
                tf,
                p0=p0,
                bounds=free_bounds,
                sigma=sigma,
                x_scale='jac',
                absolute_sigma=not USE_CUSTOM_WEIGHTS,
                maxfev=20000
            ))

            popt_try, _ = result

            _, pred = solve_ode(
                np.exp(popt_try) if ODE_MODEL == 'randmod' else popt_try,
                init_cond,
                num_psites,
                time_points
            )

            score = score_fit(gene, popt_try, f"{weight_key}_lambda_{lam}", target, pred)

            if score < best_score:
                best_score = score
                best_weight_key = weight_key

        except Exception as e:
            logger.warning(f"[{gene}] Fit failed for {weight_key}: {e}")

    if best_weight_key:
        logger.info(f"[{gene}] λ = {lam:.3f} |  "
                    f"Best Weight: '{' '.join(w.capitalize() for w in best_weight_key.split('_'))}' |  "
                    f"Score = {best_score:.2f}")
    else:
        logger.warning(f"[{gene}] All fits failed for lambda = {lam:.2f}")

    return lam, best_score, best_weight_key

def find_best_lambda(
    gene: str,
    target: np.ndarray,
    p0: np.ndarray,
    time_points: np.ndarray,
    free_bounds: Tuple[np.ndarray, np.ndarray],
    init_cond: np.ndarray,
    num_psites: int,
    p_data: np.ndarray,
    lambdas = np.linspace(1e-3, 1, 10),
    max_workers: int = os.cpu_count(),
) -> Tuple[float, str]:
    """
    Finds best lambda_reg to use in model_func.
    """

    best_lambda = None
    best_score = np.inf

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                worker_find_lambda,
                lam, gene, target, p0, time_points, free_bounds,
                init_cond, num_psites, p_data
            ): lam for lam in lambdas
        }
        for future in as_completed(futures):
            lam, score, weight = future.result()
            if score < best_score:
                best_score = score
                best_lambda = lam
                best_score_weight = weight

    return best_lambda, best_score_weight

def normest(gene, p_data, r_data, init_cond, num_psites, time_points, bounds,
            bootstraps, use_regularization=USE_REGULARIZATION):
    """
    Perform normal parameter estimation using all provided time points at once.
    Uses the provided bounds and supports bootstrapping if specified.

    Parameters:
      - p_data: Measurement data (DataFrame or numpy array). Assumes data starts at column index 2.
      - r_data: mRNA data (DataFrame or numpy array). Assumes data starts at column index 1.
      - init_cond: Initial condition for the ODE solver.
      - num_psites: Number of phosphorylation sites.
      - time_points: Array of time points to use.
      - bounds: Dictionary of parameter bounds.
      - bootstraps: Number of bootstrapping iterations.
      - use_regularization: Flag to apply Tikhonov regularization.
      - lambda_reg: Regularization strength.

    Returns:
      - est_params: List with the full estimated parameter vector.
      - model_fits: List with the ODE solution and model predictions.
      - error_vals: List with the squared error (data vs. model prediction).
    """
    est_params, model_fits, error_vals = [], [], []

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
        # If using log scale, transform bounds (ensure lower bounds > 0)
        eps = 1e-8  # small epsilon to avoid log(0)
        lower_bounds_full = [np.log(max(b, eps)) for b in lower_bounds_full]
        upper_bounds_full = [np.log(b) for b in upper_bounds_full]
    else:
        # Existing approach for distributive or successive models.
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


    free_bounds = (lower_bounds_full, upper_bounds_full)

    # Set initial guess for all parameters (midpoint of bounds).
    # p0 = np.array([(l + u) / 2 for l, u in zip(*free_bounds)])

    np.random.seed(42)
    p0 = np.array([np.random.uniform(low=l, high=u) for l, u in zip(*free_bounds)])

    # Build the target vector from the measured data.
    target = np.concatenate([r_data.flatten(), p_data.flatten()])
    target_fit = np.concatenate([target, np.zeros(len(p0))]) if use_regularization else target

    default_sigma = 1 / np.maximum(np.abs(target_fit), 1e-5)

    logger.info(f"[{gene}] Finding best regularization term λ...")

    lambda_reg, lambda_weight = find_best_lambda(gene, target, p0, time_points, free_bounds, init_cond, num_psites, p_data)

    logger.info(f"[{gene}] Using λ = {lambda_reg/len(p0)}")

    def model_func(tpts, *params):
        """
        Define the model function for curve fitting.

        :param tpts:
        :param params:
        :return: model predictions
        """
        if ODE_MODEL == 'randmod':
            param_vec = np.exp(np.asarray(params))
        else:
            param_vec = np.asarray(params)
        _, p_fitted = solve_ode(param_vec, init_cond, num_psites, np.atleast_1d(tpts))
        y_model = p_fitted.flatten()
        if use_regularization:
            reg =  lambda_reg/len(param_vec) * np.square(params)
            return np.concatenate([y_model, reg])
        return y_model

    try:
        # Attempt to get a good initial estimate using curve_fit.
        result = cast(Tuple[np.ndarray, np.ndarray],
                      curve_fit(model_func, time_points, target_fit, x_scale='jac',
                      p0=p0, bounds=free_bounds, sigma=default_sigma,
                      absolute_sigma=not USE_CUSTOM_WEIGHTS, maxfev=20000))
        popt_init, _ = result
    except Exception as e:
        logger.warning(f"[{gene}] Normal initial estimation failed: {e}")
        popt_init = p0

    # Get weights for the model fitting.
    early_weights = early_emphasis(p_data, time_points, num_psites)
    ms_gauss_weights = get_protein_weights(gene)
    weight_options = get_weight_options(target, time_points, num_psites,
                                        use_regularization, len(p0), early_weights, ms_gauss_weights)

    # Use only the best weight returned from find_best_lambda
    sigma = weight_options[lambda_weight]
    wname = lambda_weight

    scores, popts, pcovs = {}, {}, {}
    try:
        result = cast(Tuple[np.ndarray, np.ndarray],
                      curve_fit(model_func, time_points, target_fit, p0=popt_init,
                                bounds=free_bounds, sigma=sigma, x_scale='jac',
                                absolute_sigma=not USE_CUSTOM_WEIGHTS, maxfev=20000))
        popt, pcov = result
    except Exception as e:
        logger.warning(f"[{gene}] Final fit failed for {wname}: {e}")
        popt = popt_init
        pcov = None

    popts[wname] = popt
    pcovs[wname] = pcov
    _, pred = solve_ode(np.exp(popt) if ODE_MODEL == 'randmod' else popt,
                        init_cond, num_psites, time_points)

    # Calculate the score for the fit.
    scores[wname] = score_fit(gene, np.exp(popt) if ODE_MODEL == 'randmod' else popt, wname, target, pred)

    # Select the best weight based on the score.
    best_weight = min(scores, key=scores.get)
    best_score = scores[best_weight]

    # Get the best parameters and covariance matrix.
    popt_best = popts[best_weight]
    pcov_best = pcovs[best_weight]
    logger.info(f"[{gene}] Using '{' '.join(w.capitalize() for w in best_weight.split('_'))}' as weights")
    logger.info(f"[{gene}] Fit Score: {best_score:.2f}")

    # Get confidence intervals for the best parameters.
    ci_results = confidence_intervals(
        gene,
        np.exp(popt_best) if ODE_MODEL == 'randmod' else popt_best,
        pcov_best,
        target_fit,
        model_func(time_points, *popt_best),
        alpha_val=ALPHA_CI
    )

    # Bootstrapping
    boot_estimates = []
    boot_covariances = []
    if bootstraps > 0:
        logger.info(f"[{gene}] Performing bootstrapping with {bootstraps} iterations")
        for _ in range(bootstraps):
            noise = np.random.normal(0, 0.05, size=target_fit.shape)
            noisy_target = target_fit * (1 + noise)
            try:
                # Attempt to fit the model using the noisy target.
                result = cast(Tuple[np.ndarray, np.ndarray],
                              curve_fit(model_func, time_points, noisy_target,
                                        p0=popt_best, bounds=free_bounds, sigma=default_sigma,
                                        absolute_sigma=not USE_CUSTOM_WEIGHTS, maxfev=20000))
                popt_bs, pcov_bs = result
            except Exception as e:
                logger.warning(f"Bootstrapping iteration failed: {e}")
                popt_bs = popt_best
                pcov_bs = None
            boot_estimates.append(popt_bs)
            boot_covariances.append(pcov_bs)

        # Convert boot_estimates to an array and compute the mean parameter estimates.
        popt_best = np.mean(boot_estimates, axis=0)

        # Process bootstrap covariance matrices:
        # Only include iterations where pcov_bs is not None.
        valid_covs = [cov for cov in boot_covariances if cov is not None]
        if valid_covs:
            # Compute an average covariance matrix from the valid ones.
            pcov_best = np.mean(valid_covs, axis=0)
        else:
            pcov_best = None

        # Compute confidence intervals.
        ci_results = confidence_intervals(
            gene,
            np.exp(popt_best) if ODE_MODEL == 'randmod' else popt_best,
            pcov_best,
            target_fit,
            model_func(time_points, *popt_best),
            alpha_val=ALPHA_CI
        )

    # Save the confidence intervals.
    param_names = get_param_names(num_psites)
    ci_df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': ci_results['beta_hat'],
        'Std_Error': ci_results['se_lin'],
        'p_value': ci_results['pval'],
        'Lower_95CI': ci_results['lwr_ci'],
        'Upper_95CI': ci_results['upr_ci']
    })
    ci_df.to_csv(f"{OUT_DIR}/{gene}_confidence_intervals.csv", index=False)

    plotter = Plotter(gene, OUT_DIR)

    # Plot the estimated parameters with confidence intervals.
    plotter.plot_params_bar(ci_results, get_param_names(num_psites))

    # Since all parameters are free, param_final is simply the best-fit vector.
    # If parameters were estimated in log-space, convert them back.
    if ODE_MODEL == 'randmod':
        param_final = np.exp(popt_best)
    else:
        param_final = popt_best
    est_params.append(param_final)
    sol, p_fit = solve_ode(param_final, init_cond, num_psites, time_points)
    model_fits.append((sol, p_fit))
    error_vals.append(np.sum(np.abs(p_fit.flatten()-target) ** 2)/target.size)
    return est_params, model_fits, error_vals