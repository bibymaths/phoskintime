import numpy as np
from kinopt.local.objfn import estimated_series

def extract_parameters(P_initial, gene_kinase_counts, total_alpha, unique_kinases, K_index, optimized_params):
    """
    Extracts the alpha and beta parameters from the optimized parameters.

    :param P_initial:
    :param gene_kinase_counts:
    :param total_alpha:
    :param unique_kinases:
    :param K_index:
    :param optimized_params:
    :return: Alpha and beta values as dictionaries
    """
    alpha_values = {}
    alpha_start = 0
    for key, count in zip(P_initial.keys(), gene_kinase_counts):
        kinases = P_initial[key]['Kinases']
        alpha_values[key] = dict(zip(kinases, optimized_params[alpha_start:alpha_start+count]))
        alpha_start += count
    beta_values = {}
    beta_start = total_alpha
    for kinase in unique_kinases:
        for (psite, _) in K_index[kinase]:
            beta_values[(kinase, psite)] = optimized_params[beta_start]
            beta_start += 1
    return alpha_values, beta_values

def compute_metrics(optimized_params, P_init_dense, t_max, gene_alpha_starts, gene_kinase_counts,
                    gene_kinase_idx, total_alpha, kinase_beta_starts, kinase_beta_counts,
                    K_data, K_indices, K_indptr):
    """
    Computes the estimated series and various metrics based on the optimized parameters.

    :param optimized_params:
    :param P_init_dense:
    :param t_max:
    :param gene_alpha_starts:
    :param gene_kinase_counts:
    :param gene_kinase_idx:
    :param total_alpha:
    :param kinase_beta_starts:
    :param kinase_beta_counts:
    :param K_data:
    :param K_indices:
    :param K_indptr:
    :return: Estimated series, residuals, MSE, RMSE, MAE, MAPE, R-squared
    """
    P_est = estimated_series(optimized_params, t_max, P_init_dense.shape[0],
                                 gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
                                 total_alpha, kinase_beta_starts, kinase_beta_counts,
                                 K_data, K_indices, K_indptr)
    residuals = P_init_dense - P_est
    mse = np.sum(residuals**2) / P_init_dense.size
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / (P_init_dense + 1e-12))) * 100
    r_squared = 1 - (np.sum(residuals**2) / np.sum((P_init_dense - np.mean(P_init_dense))**2))
    return P_est, residuals, mse, rmse, mae, mape, r_squared