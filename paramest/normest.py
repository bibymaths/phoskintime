
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from itertools import combinations
from typing import cast, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
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
        p_data: np.ndarray,
        pr_data: np.ndarray
) -> Tuple[float, float, str]:
    """
    Worker function for a single lambda value.

    Args:
        lam: Regularization parameter.
        gene: Gene name.
        target: Target data.
        p0: Initial parameter guess.
        time_points: Time points for the model fitting.
        free_bounds: Parameter bounds for the optimization.
        init_cond: Initial conditions for the ODE solver.
        num_psites: Number of phosphorylation sites.
        p_data: Measurement data for protein-phospho.
        pr_data: Reference data for protein.

    Returns:
        Tuple containing the lambda value, score, and weight key.
    """

    def model_func(tpts, *params):
        param_vec = np.exp(np.asarray(params)) if ODE_MODEL == 'randmod' else np.asarray(params)
        _, p_fitted = solve_ode(param_vec, init_cond, num_psites, np.atleast_1d(tpts))
        y_model = p_fitted.flatten()
        reg = lam / len(params) * np.square(params)
        return np.concatenate([y_model, reg])

    tf = np.concatenate([target, np.zeros(len(p0))])

    early_weights = early_emphasis(pr_data, p_data, time_points, num_psites)

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

        result = cast(Tuple[np.ndarray, np.ndarray], cast(object, curve_fit(
            model_func,
            time_points,
            tf,
            p0=p0,
            bounds=free_bounds,
            sigma=sigma,
            x_scale='jac',
            absolute_sigma=not USE_CUSTOM_WEIGHTS,
            maxfev=20000
        )))

        popt_try, _ = result

        _, pred = solve_ode(
            np.exp(popt_try) if ODE_MODEL == 'randmod' else popt_try,
            init_cond,
            num_psites,
            time_points
        )

        score = score_fit(np.exp(popt_try) if ODE_MODEL == 'randmod'
        else popt_try, target, pred)

        if score < best_score:
            best_score = score
            best_weight_key = weight_key

    if best_weight_key:
        logger.info(f"[{gene}]\t\t| "
                    f"λ = {lam / len(p0) * np.sum(np.square(p0)):6.2f} | "
                    f"Weight: {(' '.join(w.capitalize() for w in best_weight_key.split('_'))):20} | "
                    f"Score = {best_score:6.2f}")
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
        pr_data: np.ndarray,
        lambdas=np.logspace(-2, 0, 10),
        max_workers: int = 4,
        per_lambda_timeout: float = 1800.0 # 30 minutes for each lambda worker
) -> Tuple[float, str]:
    """
    Finds best lambda_reg to use in model_func.
    """

    best_lambda = None
    best_score = np.inf
    best_score_weight = None

    ctx = mp.get_context("spawn") # avoids fork deadlocks with MKL/OpenMP

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(worker_find_lambda, lam, gene, target, p0, time_points, free_bounds,
                            init_cond, num_psites, p_data, pr_data): lam
            for lam in lambdas
        }

        for future in as_completed(futures, timeout=per_lambda_timeout):
            lam = futures[future]
            try:
                lam, score, weight = future.result(timeout=per_lambda_timeout)
            except TimeoutError:
                logger.warning(f"[{gene}] lambda worker timed out (lam={lam}). Skipping.")
                continue
            except Exception as e:
                logger.warning(f"[{gene}] lambda worker failed (lam={lam}): {e}")
                continue

            if score < best_score:
                best_score = score
                best_lambda = lam
                best_score_weight = weight

    return best_lambda, best_score_weight

def _curve_fit_multistart(
        gene: str,
        model_func,
        time_points: np.ndarray,
        target_fit: np.ndarray,
        base_p0: np.ndarray,
        free_bounds: Tuple[np.ndarray, np.ndarray],
        sigma: np.ndarray | None,
        init_cond: np.ndarray,
        num_psites: int,
        target: np.ndarray,
        n_starts: int = 24,
        jitter_frac: float = 0.10,
        maxfev: int = 20000,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray | None, float]:
    """
    Perform multi-start curve fitting to find the best parameter estimates.

    This function attempts curve fitting from multiple initial parameter guesses to avoid
    local minima. It generates candidate starting points using jittering around the base guess
    and stratified uniform sampling, then evaluates each fit, and returns the best result.

    Args:
        gene: Gene name for logging purposes.
        model_func: Model function to fit (callable with signature model_func(tpts, *params)).
        time_points: Time points for the model fitting.
        target_fit: Target data to fit (may include regularization terms).
        base_p0: Base initial parameter guess.
        free_bounds: Tuple of (lower_bounds, upper_bounds) for parameters.
        sigma: Weights/uncertainties for fitting (None for uniform weights).
        init_cond: Initial conditions for the ODE solver.
        num_psites: Number of phosphorylation sites.
        target: Target data without regularization terms (for scoring).
        n_starts: Number of multi-start attempts (default: 24).
        jitter_frac: Fraction of parameter range to use for jittering (default: 0.10).
        maxfev: Maximum number of function evaluations per fit (default: 20000).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple containing:
            - popt_best: Best-fit parameters found across all starts.
            - pcov_best: Covariance matrix for the best fit (or None if unavailable).
            - best_score: Score of the best fit.

    Raises:
        ValueError: If bounds are not finite.
        RuntimeError: If all multi-start attempts fail.
    """

    lb, ub = free_bounds
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # Defensive: bounds must be finite for sampling.
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError("free_bounds must be finite for multistart sampling.")

    # Reproducible per gene (stable across runs)
    # Use a small gene hash to diversify seeds without non-determinism.
    gene_hash = (sum(ord(c) for c in str(gene)) % 1000003)
    rng = np.random.default_rng(int(seed + gene_hash))

    # Build candidate p0 list
    p0_list: list[np.ndarray] = []

    base = np.asarray(base_p0, dtype=float).copy()
    base = np.clip(base, lb, ub)
    p0_list.append(base)

    # Jitter around base p0 (in-place but clipped)
    # jitter scale relative to (ub-lb); this works in both linear and log-parameter space.
    span = (ub - lb)
    span[span <= 0] = 1.0  # avoid zero-span issues
    for _ in range(max(0, n_starts // 3)):
        noise = rng.normal(0.0, 1.0, size=base.shape[0])
        cand = base + (jitter_frac * span) * noise
        cand = np.clip(cand, lb, ub)
        p0_list.append(cand)

    # “LHS-like” stratified uniform sampling (cheap, no extra dependency)
    # For each dim, sample within n bins and permute -> reduces clustering vs pure uniform.
    remaining = max(0, n_starts - len(p0_list))
    if remaining > 0:
        d = base.shape[0]
        bins = remaining
        U = np.empty((bins, d), dtype=float)

        for j in range(d):
            # stratified samples in [0,1)
            u = (np.arange(bins) + rng.random(bins)) / float(bins)
            rng.shuffle(u)
            U[:, j] = u

        # scale into [lb, ub]
        cands = lb + U * (ub - lb)
        for k in range(cands.shape[0]):
            p0_list.append(cands[k])

    # Evaluate each start and choose best by your score_fit on ODE prediction
    best_score = float("inf")
    popt_best = base
    pcov_best = None

    n_ok = 0
    n_fail = 0

    for s_idx, p0_try in enumerate(p0_list):
        try:
            result = cast(
                Tuple[np.ndarray, np.ndarray],
                cast(object, curve_fit(
                    model_func,
                    time_points,
                    target_fit,
                    p0=p0_try,
                    bounds=free_bounds,
                    sigma=sigma,
                    x_scale="jac",
                    absolute_sigma=not USE_CUSTOM_WEIGHTS,
                    maxfev=maxfev
                ))
            )
            popt_try, pcov_try = result

            # Score based on true ODE prediction
            _, pred = solve_ode(
                np.exp(popt_try) if ODE_MODEL == "randmod" else popt_try,
                init_cond,
                num_psites,
                time_points
            )

            score = score_fit(
                np.exp(popt_try) if ODE_MODEL == "randmod" else popt_try,
                target,
                pred
            )

            n_ok += 1
            if score < best_score:
                best_score = float(score)
                popt_best = popt_try
                pcov_best = pcov_try

        except Exception as e:
            n_fail += 1
            # DEBUG-level behaviour; do not spam logs
            # but still provide a hint if all starts fail.
            continue

    if n_ok == 0:
        raise RuntimeError(f"[{gene}] multistart curve_fit: all starts failed (n={len(p0_list)}).")

    logger.info(
        f"[{gene}]\t\tMultistart curve_fit: "
        f"starts={len(p0_list)} ok={n_ok} fail={n_fail} best_score={best_score:6.2f}"
    )

    return popt_best, pcov_best, best_score

def normest(gene, pr_data, p_data, r_data, init_cond, num_psites, time_points, bounds,
            bootstraps, use_regularization=USE_REGULARIZATION):
    """
    Function to estimate parameters for a given gene using ODE models.

    Args:
        gene: Gene name.
        pr_data: Protein data.
        p_data: Phosphorylation data.
        r_data: Reference data.
        init_cond: Initial conditions for the ODE solver.
        num_psites: Number of phosphorylation sites.
        time_points: Time points for the model fitting.
        bounds: Parameter bounds for the optimization.
        bootstraps: Number of bootstrap iterations.
        use_regularization: Whether to use regularization in the fitting process.

    Returns:
        Tuple containing estimated parameters, model fits, error values, and regularization term.
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
        # For phosphorylation parameters: use S(i) bounds.
        lower_bounds_full += [bounds["S(i)"][0]] * num_psites
        upper_bounds_full += [bounds["S(i)"][1]] * num_psites
        # For dephosphorylation parameters: for each combination, use D(i) bounds.
        for i in range(1, num_psites + 1):
            for _ in combinations(range(1, num_psites + 1), i):
                lower_bounds_full.append(bounds["D(i)"][0])
                upper_bounds_full.append(bounds["D(i)"][1])
        # If using log scale, transform bounds (ensure lower bounds > 0)
        eps = 1e-8  # small epsilon to avoid log(0)
        lower_bounds_full = [np.log(max(b, eps)) for b in lower_bounds_full]
        upper_bounds_full = [np.log(b) for b in upper_bounds_full]
    else:
        # Existing approach for distributive or successive models.
        lower_bounds_full = (
                [bounds["A"][0], bounds["B"][0], bounds["C"][0], bounds["D"][0]] +
                [bounds["S(i)"][0]] * num_psites +
                [bounds["D(i)"][0]] * num_psites
        )
        upper_bounds_full = (
                [bounds["A"][1], bounds["B"][1], bounds["C"][1], bounds["D"][1]] +
                [bounds["S(i)"][1]] * num_psites +
                [bounds["D(i)"][1]] * num_psites
        )

    free_bounds = (lower_bounds_full, upper_bounds_full)

    # Set seed for reproducibility.
    np.random.seed(42)

    # Generate a random initial guess for the parameters.
    p0 = np.array([np.random.uniform(low=l, high=u) for l, u in zip(*free_bounds)])

    # Build the target vector from the measured data.
    target = np.concatenate([r_data.flatten(), pr_data.flatten(), p_data.flatten()])
    target_fit = np.concatenate([target, np.zeros(len(p0))]) if use_regularization else target

    logger.info(f"[{gene}]      Finding best regularization term λ...")

    lambda_reg, lambda_weight = find_best_lambda(gene, target, p0, time_points, free_bounds, init_cond, num_psites,
                                                 p_data, pr_data, max_workers=10)

    logger.info("           --------------------------------")
    logger.info(f"[{gene}]      Using λ = {lambda_reg / len(p0) * np.sum(np.square(p0)): .4f}")

    def model_func(tpts, *params):
        """
        Define the model function for curve fitting.

        Args:
            tpts: Time points.
            params: Parameters for the model.

        Returns:
            y_model: Model predictions.
        """
        if ODE_MODEL == 'randmod':
            param_vec = np.exp(np.asarray(params))
        else:
            param_vec = np.asarray(params)
        _, p_fitted = solve_ode(param_vec, init_cond, num_psites, np.atleast_1d(tpts))
        y_model = p_fitted.flatten()
        if use_regularization:
            reg = lambda_reg / len(param_vec) * np.square(params)
            return np.concatenate([y_model, reg])
        return y_model

    # Get weights for the model fitting.
    early_weights = early_emphasis(pr_data, p_data, time_points, num_psites)
    ms_gauss_weights = get_protein_weights(gene)
    weight_options = get_weight_options(target, time_points, num_psites,
                                        use_regularization, len(p0), early_weights, ms_gauss_weights)

    # Use only the best weight returned from find_best_lambda
    sigma = weight_options[lambda_weight]
    wname = lambda_weight

    scores, popts, pcovs = {}, {}, {}
    try:
        popt, pcov, best_ms_score = _curve_fit_multistart(
            gene=gene,
            model_func=model_func,
            time_points=time_points,
            target_fit=target_fit,
            base_p0=p0,
            free_bounds=free_bounds,
            sigma=sigma,
            init_cond=init_cond,
            num_psites=num_psites,
            target=target,
            n_starts=48,  # increase if needed (e.g., 48 for hard genes)
            jitter_frac=0.10,  # 10% of bound span
            maxfev=20000,
            seed=42
        )
    except Exception as e:
        logger.warning(f"[{gene}] Final multistart fit failed for {wname}: {e}")
        popt = p0
        pcov = None

    popts[wname] = popt
    pcovs[wname] = pcov

    _, pred = solve_ode(np.exp(popt) if ODE_MODEL == 'randmod' else popt,
                        init_cond, num_psites, time_points)

    # Calculate the score for the fit.
    scores[wname] = score_fit(np.exp(popt) if ODE_MODEL == 'randmod' else popt, target, pred)

    # Select the best weight based on the score.
    best_weight = min(scores, key=scores.get)

    # Get the best parameters and covariance matrix.
    popt_best = popts[best_weight]
    pcov_best = pcovs[best_weight]
    logger.info(f"[{gene}]      Using '{' '.join(w.capitalize() for w in best_weight.split('_'))}' as weights")
    logger.info(f"[{gene}]      Fit Score: {scores[wname]:.2f}")
    logger.info("           --------------------------------")

    # Get confidence intervals for the best parameters.
    ci_results = confidence_intervals(
        gene,
        np.exp(popt_best) if ODE_MODEL == 'randmod' else popt_best,
        pcov_best,
        target_fit,
        model_func(time_points, *popt_best),
        alpha_val=ALPHA_CI
    )

    # Bootstrapping with gaussian noise added to the target data.
    boot_estimates = []
    boot_covariances = []
    if bootstraps > 0:
        logger.info("           --------------------------------")
        logger.info(f"[{gene}]      Performing bootstrapping with {bootstraps} iterations")
        logger.info("           --------------------------------")
        for _ in range(bootstraps):
            noise = np.random.normal(0, 0.05, size=target_fit.shape)
            noisy_target = target_fit * (1 + noise)
            try:
                # Attempt to fit the model using the noisy target.
                result = cast(Tuple[np.ndarray, np.ndarray],
                              cast(object, curve_fit(model_func, time_points, noisy_target,
                                                     p0=popt_best, bounds=free_bounds, sigma=sigma,
                                                     absolute_sigma=not USE_CUSTOM_WEIGHTS, maxfev=20000)))
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
    error_vals.append(np.sum(np.abs(p_fit.flatten() - target) ** 2) / target.size) 
    # average-per-parameter L2 penalty
    regularization_term = lambda_reg / len(param_final) * np.sum(np.square(param_final))

    return est_params, model_fits, error_vals, regularization_term
