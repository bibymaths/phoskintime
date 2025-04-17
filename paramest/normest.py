import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations
from typing import cast, Tuple

from config.config import score_fit
from config.constants import LAMBDA_REG, USE_REGULARIZATION, ODE_MODEL, ALPHA_CI
from config.logconf import setup_logger
from models import solve_ode
from models.weights import early_emphasis, get_weight_options
from .identifiability import confidence_intervals

logger = setup_logger()

def normest(gene, p_data, init_cond, num_psites, time_points, bounds,
            bootstraps, use_regularization=USE_REGULARIZATION, lambda_reg=LAMBDA_REG):
    """
    Perform normal parameter estimation using all provided time points at once.
    Uses the provided bounds (ignores fixed_params so that all parameters are estimated)
    and supports bootstrapping if specified.

    Parameters:
      - p_data: Measurement data (DataFrame or numpy array). Assumes data starts at column index 2.
      - init_cond: Initial condition for the ODE solver.
      - num_psites: Number of phosphorylation sites.
      - time_points: Array of time points to use.
      - bounds: Dictionary of parameter bounds.
      - fixed_params: (Ignored in normest) Provided for interface consistency.
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

    def model_func(tpts, *params):
        """
        Define the model function for curve fitting.

        :param tpts:
        :param params:
        :return: model predictions
        """
        if ODE_MODEL == 'randmod':
            param_vec = np.exp(np.array(params))
        else:
            param_vec = np.array(params)
        _, p_fitted = solve_ode(param_vec, init_cond, num_psites, np.atleast_1d(tpts))
        y_model = p_fitted.flatten()
        if use_regularization:
            reg = np.sqrt(lambda_reg) * np.array(params)
            return np.concatenate([y_model, reg])
        return y_model

    free_bounds = (lower_bounds_full, upper_bounds_full)

    # Set initial guess for all parameters (midpoint of bounds).
    p0 = np.array([(l + u) / 2 for l, u in zip(*free_bounds)])

    # Build the target vector from the measured data.
    target = p_data.flatten()
    target_fit = np.concatenate([target, np.zeros(len(p0))]) if use_regularization else target

    default_sigma = 1 / np.maximum(np.abs(target_fit), 1e-5)

    try:
        # Attempt to get a good initial estimate using curve_fit.
        result = cast(Tuple[np.ndarray, np.ndarray],
                      curve_fit(model_func, time_points, target_fit, x_scale='jac',
                      p0=p0, bounds=free_bounds, sigma=default_sigma,
                      absolute_sigma=True, maxfev=20000))
        popt_init, _ = result
    except Exception as e:
        logger.warning(f"[{gene}] Normal initial estimation failed: {e}")
        popt_init = p0

    # Get weights for the model fitting.
    early_weights = early_emphasis(p_data, time_points, num_psites)
    weight_options = get_weight_options(target, time_points, num_psites,
                                        use_regularization, len(p0), early_weights)

    scores, popts, pcovs = {}, {}, {}
    for wname, sigma in weight_options.items():
        try:
            # Attempt to fit the model using the specified weights.
            result = cast(Tuple[np.ndarray, np.ndarray],
                          curve_fit(model_func, time_points, target_fit, p0=popt_init,
                          bounds=free_bounds, sigma=sigma, x_scale='jac',
                          absolute_sigma=True, maxfev=20000))
            popt, pcov = result
        except Exception as e:
            logger.warning(f"[{gene}] Fit failed for {wname}: {e}")
            popt = popt_init
            pcov = None
        popts[wname] = popt
        pcovs[wname] = pcov
        pred = model_func(time_points, *popt)
        # Calculate the score for the fit.
        scores[wname] = score_fit(target_fit, pred, popt)

    # Select the best weight based on the score.
    best_weight = min(scores, key=scores.get)
    best_score = scores[best_weight]
    # Get the best parameters and covariance matrix.
    popt_best = popts[best_weight]
    pcov_best = pcovs[best_weight]

    logger.info(f"[{gene}] Best weight: {best_weight} with score: {best_score:.4f}")

    # Get confidence intervals for the best parameters.
    ci_results = confidence_intervals(
        np.exp(popt_best) if ODE_MODEL == 'randmod' else popt_best,
        pcov_best,
        target,
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
                                        absolute_sigma=True, maxfev=20000))
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
        # Compute confidence intervals for the bootstrapped estimates.
        ci_results = confidence_intervals(
            np.exp(popt_best) if ODE_MODEL == 'randmod' else popt_best,
            pcov_best,
            target,
            alpha_val=ALPHA_CI
        )
    # Since all parameters are free, param_final is simply the best-fit vector.
    # If parameters were estimated in log-space, convert them back.
    if ODE_MODEL == 'randmod':
        param_final = np.exp(popt_best)
    else:
        param_final = popt_best
    est_params.append(param_final)
    sol, p_fit = solve_ode(param_final, init_cond, num_psites, time_points)
    model_fits.append((sol, p_fit))
    error_vals.append(np.sum(np.abs(target - p_fit.flatten()) ** 2))
    return est_params, model_fits, error_vals #, ci_results