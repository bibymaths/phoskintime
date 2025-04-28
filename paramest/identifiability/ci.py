import numpy as np
import scipy.stats as stats
from config.logconf import setup_logger

logger = setup_logger()

def confidence_intervals(gene, popt, pcov, target, model, alpha_val=0.05):
    """
    Computes the confidence intervals for parameter estimates using a linearization approach.

    Parameters:
      - popt: 1D numpy array of best-fit parameter estimates.
      - pcov: Square covariance matrix (numpy array) corresponding to popt.
      - target: 1D numpy array of observed data (used to compute degrees of freedom).
      - alpha_val: Significance level (default 0.05 for a 95% confidence interval).

    Returns:
      A dictionary with the following keys:
        'beta_hat': Best-fit parameter estimates.
        'se_lin': Standard errors (sqrt of diagonal of pcov).
        'df_lin': Degrees of freedom (n_obs - n_params).
        't_stat': t-statistics for each parameter.
        'pval': Two-sided p-values for each parameter.
        'qt_lin': t critical value for the given alpha and degrees of freedom.
        'lwr_ci': Lower 95% confidence intervals.
        'upr_ci': Upper 95% confidence intervals.
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

    # Log the summary.
    header = "Parameter\t Estimate\t SE\t\t Pr(>|t|)\t\t 95% CI"
    logger.info(f"[{gene}] Confidence Intervals:")
    logger.info(header)
    for i, (b, se, p, lwr, upr) in enumerate(zip(beta_hat, se_lin, pval, lwr_ci, upr_ci)):
        logger.info(f"Rate{i}:\t\t {b:.2f}\t\t {se:.2f}\t\t {p:.1e}\t\t ({lwr:.2f} - {upr:.2f})")

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
