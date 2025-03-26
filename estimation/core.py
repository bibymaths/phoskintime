
import os

import numpy as np
import pandas as pd

from phoskintime.config.constants import get_param_names, generate_labels, OUT_DIR
from phoskintime.config.logging_config import setup_logger
from phoskintime.estimation.estimation import sequential_estimation
from phoskintime.models.ode_model import solve_ode

from phoskintime.plotting.plotting import (plot_parallel, plot_tsne, plot_pca, pca_components,
                                           plot_param_series, plot_model_fit, plot_A_S)

from numba import njit
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = setup_logger(__name__)

# ----------------------------------
# Parameter + State Initialization
# ----------------------------------
def initial_condition(num_psites: int) -> list:
    A, B, C, D = 1, 1, 1, 1
    S_rates = np.ones(num_psites)
    D_rates = np.ones(num_psites)

    def steady_state_equations(y):
        R, P, *P_sites = y
        dR_dt = A - B * R
        dP_dt = C * R - (D + np.sum(S_rates)) * P + np.sum(P_sites)
        dP_sites_dt = [S_rates[i] * P - (1 + D_rates[i]) * P_sites[i] for i in range(num_psites)]
        return [dR_dt, dP_dt] + dP_sites_dt

    y0_guess = np.ones(num_psites + 2)
    bounds_local = [(1e-6, None)] * (num_psites + 2)
    result = minimize(lambda y: 0, y0_guess, method='SLSQP', bounds=bounds_local,
                      constraints={'type': 'eq', 'fun': steady_state_equations})
    logger.info("Steady-State conditions calculated")
    if result.success:
        return result.x.tolist()
    else:
        raise ValueError("Failed to find steady-state conditions")

# ----------------------------------
# Early-Weighted Scheme
# ----------------------------------
@njit
def early_emphasis(P_data, time_points, num_psites):
    if P_data.ndim == 1:
        P_data = P_data.reshape(1, P_data.size)

    n_times = len(time_points)
    custom_weights = np.ones((num_psites, n_times))
    time_diffs = np.empty(n_times)
    time_diffs[0] = 0.0
    for j in range(1, n_times):
        time_diffs[j] = time_points[j] - time_points[j - 1]

    for i in range(num_psites):
        limit = min(5, n_times)
        for j in range(1, limit):
            data_based_weight = 1.0 / (abs(P_data[i, j]) + 1e-5)
            time_based_weight = 1.0 / (time_diffs[j] + 1e-5)
            custom_weights[i, j] = data_based_weight * time_based_weight
        for j in range(5, n_times):
            custom_weights[i, j] = 1.0

    return custom_weights.ravel()

# ----------------------------------
# Process Protein
# ----------------------------------
def process_gene(gene, measurement_data, time_points, bounds, fixed_params,
                 desired_times=None, time_fixed=None, bootstraps=0, out_dir=OUT_DIR):
    gene_data = measurement_data[measurement_data['Gene'] == gene]
    num_psites = gene_data.shape[0]
    psite_values = gene_data['Psite'].values
    init_cond = initial_condition(num_psites)
    P_data = gene_data.iloc[:, 2:].values

    # Estimate parameters sequentially
    estimated_params, model_fits, errors = sequential_estimation(
        P_data, time_points, init_cond, bounds, fixed_params, num_psites, gene
    )

    # Extract final model fit at each step
    seq_model_fit = np.zeros((num_psites, len(time_points)))
    for i, (_, P_fitted) in enumerate(model_fits):
        seq_model_fit[:, i] = P_fitted[:, -1]

    # Generate outputs
    labels = generate_labels(num_psites)
    final_params = estimated_params[-1]
    sol_full, _ = solve_ode(final_params, init_cond, num_psites, time_points)
    param_names = get_param_names(num_psites)

    plot_parallel(sol_full, labels, gene, out_dir)
    plot_tsne(sol_full, gene, perplexity=5, out_dir=out_dir)
    plot_pca(sol_full, gene, components=3, out_dir=out_dir)
    pca_components(sol_full, gene, target_variance=0.99, out_dir=out_dir)
    plot_param_series(gene, estimated_params, param_names, time_points, out_dir)
    plot_model_fit(gene, seq_model_fit, P_data, sol_full, num_psites, psite_values, time_points, out_dir)
    plot_A_S(gene, estimated_params, num_psites, time_points, out_dir)

    logger.info("Parameter Estimation Finished")
    logger.info(f"Plots Saved: {out_dir}")

    df_params = pd.DataFrame(estimated_params, columns=get_param_names(num_psites))
    df_params.insert(0, "Time", time_points[:len(estimated_params)])
    df_params.to_excel(os.path.join(out_dir, f"{gene}_parameters.xlsx"), index=False)

    gene_psite_dict_local = {'Protein': gene}
    for i, name in enumerate(get_param_names(num_psites)):
        gene_psite_dict_local[name] = [final_params[i]]

    final_data_flat = P_data.flatten()
    final_pred = seq_model_fit.flatten()
    mse = mean_squared_error(final_data_flat, final_pred)
    mae = mean_absolute_error(final_data_flat, final_pred)
    logger.info(f"{gene} → MSE: {mse:.4f}, MAE: {mae:.4f}")

    return {
        "gene": gene,
        "estimated_params": estimated_params,
        "seq_model_fit": seq_model_fit,
        "errors": errors,
        "gene_psite_data": gene_psite_dict_local,
    }

def process_gene_wrapper(gene, measurement_data, time_points, bounds, fixed_params,
                         desired_times, time_fixed, bootstraps, out_dir=OUT_DIR):
    return process_gene(
        gene=gene,
        measurement_data=measurement_data,
        time_points=time_points,
        bounds=bounds,
        fixed_params=fixed_params,
        desired_times=desired_times,
        time_fixed=time_fixed,
        bootstraps=bootstraps,
        out_dir=out_dir
    )