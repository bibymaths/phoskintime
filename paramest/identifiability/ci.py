import numpy as np
import scipy.stats as stats

from config.constants import USE_CUSTOM_WEIGHTS
from config.logconf import setup_logger

logger = setup_logger()


def confidence_intervals(gene, popt, pcov, target, model, alpha_val=0.05):
    """
    Computes the confidence intervals for parameter estimates using Wald Intervals approach.

    Args:
        gene (str): Gene name.
        popt (np.ndarray): Optimized parameter estimates.
        pcov (np.ndarray): Covariance matrix of the optimized parameters.
        target (np.ndarray): Target data.
        model (np.ndarray): Model predictions.
        alpha_val (float, optional): Significance level for confidence intervals. Defaults to 0.05.

    Returns:
        dict: A dictionary containing the confidence intervals and other statistics.
    """
    if pcov is None:
        msg = "No covariance matrix available; cannot compute confidence intervals using linearization."
        logger.info(msg)
        return None

    # Best-fit parameter estimates.
    beta_hat = popt

    # Degrees of freedom: number of observations minus number of parameters, minimum 1 to avoid division by zero.
    df_lin = max(target.size - beta_hat.size, 1)

    # Compute scaled residuals (error per data point).
    residuals = (target - model) / target.size

    # Residual sum of squares (RSS) from scaled residuals.
    rss = np.sum(residuals ** 2)

    # Mean squared error (MSE) to rescale covariance matrix.
    mse = rss / df_lin

    # Standard errors from scaled covariance matrix.
    if USE_CUSTOM_WEIGHTS:
        se_lin = np.sqrt(np.diag(pcov))
    else:
        se_lin = np.sqrt(np.diag(pcov * mse))

    # t-statistics for each parameter estimate.
    t_stat = beta_hat / se_lin

    # Two-tailed p-values from t-statistics.
    pval = stats.t.sf(np.abs(t_stat), df_lin) * 2

    # Critical t-value for the desired confidence level.
    qt_lin = stats.t.ppf(1 - alpha_val / 2, df_lin)

    # Lower bound of confidence interval, clipped at zero.
    lwr_ci = np.maximum(beta_hat - qt_lin * se_lin, 0)

    # Upper bound of confidence interval.
    upr_ci = beta_hat + qt_lin * se_lin

    logger.info(f"[{gene}]      Parameter\t\t Estimate\t  SE\t\t Pr(>|t|)\t\t   95% CI")
    logger.info(f"            -----------\t\t --------\t  --\t\t --------\t\t   ------")
    for i, (b, se, p, lwr, upr) in enumerate(zip(beta_hat, se_lin, pval, lwr_ci, upr_ci)):
        logger.info(f"[{gene}]      Rate{i}:\t\t {b:.4f}\t\t {se:.3f}\t\t {p:.3e}\t\t ({lwr:.4f} - {upr:.4f})")

    results = {
        'beta_hat': beta_hat,
        'se_lin': se_lin,
        'df_lin': df_lin,
        't_stat': t_stat,
        'pval': pval,
        'qt_lin': qt_lin,
        'lwr_ci': lwr_ci,
        'upr_ci': upr_ci
    }
    return results
