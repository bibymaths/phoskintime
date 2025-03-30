import numpy as np
from abopt.local.objfn import estimated_series

def extract_parameters(P_initial, gene_kinase_counts, total_alpha, unique_kinases, K_index, optimized_params):
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
    P_est = estimated_series(optimized_params, t_max, P_init_dense.shape[0],
                                 gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
                                 total_alpha, kinase_beta_starts, kinase_beta_counts,
                                 K_data, K_indices, K_indptr)
    residuals = P_est - P_init_dense
    mse = np.sum(residuals**2) / P_init_dense.size
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / (P_init_dense + 1e-12))) * 100
    r_squared = 1 - (np.sum(residuals**2) / np.sum((P_init_dense - np.mean(P_init_dense))**2))
    return P_est, residuals, mse, rmse, mae, mape, r_squared