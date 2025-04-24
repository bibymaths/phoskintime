import numpy as np
import scipy.stats as stats
from config.logconf import setup_logger

logger = setup_logger()

def confidence_intervals(popt, pcov, target, alpha_val=0.05):
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

    beta_hat = popt
    se_lin = np.sqrt(np.diag(pcov))
    # Degrees of freedom: number of observations (target should be 1D) minus number of parameters.
    df_lin = target.size - beta_hat.size
    # Compute t-statistics for each parameter.
    t_stat = beta_hat / se_lin
    # Two-sided p-values.
    pval = stats.t.sf(np.abs(t_stat), df_lin) * 2
    # t critical value for a two-tailed confidence interval.
    qt_lin = stats.t.ppf(1 - alpha_val / 2, df_lin)
    # Calculate confidence intervals.
    lwr_ci = np.maximum(beta_hat - qt_lin * se_lin, 0)
    upr_ci = beta_hat + qt_lin * se_lin

    # Log the summary.
    header = "Parameter\tEstimate\tStd. Error\tPr(>|t|)\t 95% CI"
    logger.info("Confidence Intervals:")
    logger.info(header)
    for i, (b, se, p, lwr, upr) in enumerate(zip(beta_hat, se_lin, pval, lwr_ci, upr_ci)):
        logger.info(f"Rate{i}:\t {b:.2f}\t {se:.2f}\t {p:.1e}\t ({lwr:.2f} - {upr:.2f})")

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
