import numpy as np
from numba import njit

@njit
def _objective(params, P_init, t_max, n,
              gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
              total_alpha, kinase_beta_starts, kinase_beta_counts,
              K_data, K_indices, K_indptr,
              time_weights, loss_flag):
    n_gene = P_init.shape[0]
    n_kinase = kinase_beta_starts.shape[0]
    M = np.zeros((n_kinase, t_max))
    for k in range(n_kinase):
        start = kinase_beta_starts[k]
        count = kinase_beta_counts[k]
        for r in range(count):
            beta_val = params[total_alpha + start + r]
            global_row = start + r
            row_start = K_indptr[global_row]
            row_end = K_indptr[global_row + 1]
            for idx in range(row_start, row_end):
                col = K_indices[idx]
                M[k, col] += beta_val * K_data[idx]
    pred = np.zeros((n_gene, t_max))
    for i in range(n_gene):
        start_alpha = gene_alpha_starts[i]
        count = gene_kinase_counts[i]
        for j in range(count):
            alpha_val = params[start_alpha + j]
            kinase_idx = gene_kinase_idx[start_alpha + j]
            for t in range(t_max):
                pred[i, t] += alpha_val * M[kinase_idx, t]
    loss_val = 0.0
    total_weight = 0.0
    for i in range(n_gene):
        for t in range(t_max):
            diff = pred[i, t] - P_init[i, t]
            if loss_flag == 0:
                loss_val += diff * diff
            elif loss_flag == 1:
                loss_val += time_weights[t] * diff * diff
                total_weight += time_weights[t]
            elif loss_flag == 2:
                loss_val += 2.0 * (np.sqrt(1.0 + 0.5 * diff * diff) - 1.0)
            elif loss_flag == 3:
                loss_val += np.log(1.0 + 0.5 * diff * diff)
            elif loss_flag == 4:
                loss_val += np.arctan(diff * diff)
    if loss_flag == 1:
        return loss_val / total_weight
    else:
        return loss_val / n

@njit
def _estimated_series(params, t_max, n, gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
                         total_alpha, kinase_beta_starts, kinase_beta_counts,
                         K_data, K_indices, K_indptr):
    n_gene = n
    n_kinase = kinase_beta_starts.shape[0]
    M = np.zeros((n_kinase, t_max))
    for k in range(n_kinase):
        start = kinase_beta_starts[k]
        count = kinase_beta_counts[k]
        for r in range(count):
            beta_val = params[total_alpha + start + r]
            global_row = start + r
            row_start = K_indptr[global_row]
            row_end = K_indptr[global_row + 1]
            for idx in range(row_start, row_end):
                col = K_indices[idx]
                M[k, col] += beta_val * K_data[idx]
    pred = np.zeros((n_gene, t_max))
    for i in range(n_gene):
        start_alpha = gene_alpha_starts[i]
        count = gene_kinase_counts[i]
        for j in range(count):
            alpha_val = params[start_alpha + j]
            kinase_idx = gene_kinase_idx[start_alpha + j]
            for t in range(t_max):
                pred[i, t] += alpha_val * M[kinase_idx, t]
    return pred

def _objective_wrapper(params, P_init_dense, t_max, gene_alpha_starts, gene_kinase_counts,
                      gene_kinase_idx, total_alpha, kinase_beta_starts, kinase_beta_counts,
                      K_data, K_indices, K_indptr, time_weights, loss_type):
    mapping = {"base": 0, "weighted": 1, "softl1": 2, "cauchy": 3, "arctan": 4}
    flag = mapping.get(loss_type, 0)
    return _objective(params, P_init_dense, t_max, P_init_dense.shape[0],
                     gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
                     total_alpha, kinase_beta_starts, kinase_beta_counts,
                     K_data, K_indices, K_indptr, time_weights, flag)
