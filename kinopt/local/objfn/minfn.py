import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _objective(params, P_init, t_max, n,
              gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
              total_alpha, kinase_beta_starts, kinase_beta_counts,
              K_data, K_indices, K_indptr,
              time_weights, loss_flag):
    """
    Objective function for optimization.
    This function computes the loss value based on the predicted and actual values.
    The loss function can be selected based on the loss_flag parameter.

    The loss function can be one of the following:
    0: Base loss (L2 norm)
    1: Weighted loss (L2 norm with time weights)
    2: Soft L1 loss
    3: Cauchy loss
    4: Arctan loss

    :param params:
    :param P_init:
    :param t_max:
    :param n:
    :param gene_alpha_starts:
    :param gene_kinase_counts:
    :param gene_kinase_idx:
    :param total_alpha:
    :param kinase_beta_starts:
    :param kinase_beta_counts:
    :param K_data:
    :param K_indices:
    :param K_indptr:
    :param time_weights:
    :param loss_flag:
    :return: loss value
    """
    # Initialize the number of genes and kinases
    n_gene = P_init.shape[0]
    n_kinase = kinase_beta_starts.shape[0]

    # Initialize M matrix - kinase x time
    M = np.zeros((n_kinase, t_max))
    for k in prange(n_kinase):
        # For each kinase, get the starting index and count
        start = kinase_beta_starts[k]
        count = kinase_beta_counts[k]
        # For each beta value, compute the contribution to M
        for r in range(count):
            # Get the beta value and compute the global row index
            beta_val = params[total_alpha + start + r]
            global_row = start + r
            # Get the start and end indices for the current row in the sparse matrix
            row_start = K_indptr[global_row]
            row_end = K_indptr[global_row + 1]
            # For each non-zero entry in the sparse matrix, update M
            for idx in range(row_start, row_end):
                col = K_indices[idx]
                # Update M based on the beta value and the sparse matrix data
                M[k, col] += beta_val * K_data[idx]
    # Initialize the predicted matrix
    pred = np.zeros((n_gene, t_max))
    for i in range(n_gene):
        # For each gene, get the starting index and count
        start_alpha = gene_alpha_starts[i]
        count = gene_kinase_counts[i]
        # For each alpha value, compute the contribution to the predicted matrix
        for j in range(count):
            # Get the alpha value and kinase index
            alpha_val = params[start_alpha + j]
            kinase_idx = gene_kinase_idx[start_alpha + j]
            # For each time point, update the predicted matrix
            for t in range(t_max):
                # Update the predicted value based on the kinase activity
                pred[i, t] += alpha_val * M[kinase_idx, t]
    # Compute the loss value based on the selected loss function
    loss_val = 0.0
    total_weight = 0.0
    # Loop through each gene and time point to compute the loss
    for i in range(n_gene):
        # For each gene, get the starting index and count
        for t in range(t_max):
            # Compute the difference between the actual and predicted values
            diff = P_init[i, t] - pred[i, t]
            # Apply the selected loss function
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
        # Normalize the loss value by the total weight
        return loss_val / total_weight
    else:
        # Normalize the loss value by the number of genes
        return loss_val / n

@njit(parallel=True)
def _estimated_series(params, t_max, n, gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
                         total_alpha, kinase_beta_starts, kinase_beta_counts,
                         K_data, K_indices, K_indptr):
    """
    Compute the estimated series based on the parameters and the sparse matrix representation of the data.
    This function computes the predicted values for each gene at each time point based on the parameters
    and the sparse matrix.

    The predicted values are computed by first calculating the kinase activity matrix M and then using it
    to compute the predicted values for each gene.

    :param params:
    :param t_max:
    :param n:
    :param gene_alpha_starts:
    :param gene_kinase_counts:
    :param gene_kinase_idx:
    :param total_alpha:
    :param kinase_beta_starts:
    :param kinase_beta_counts:
    :param K_data:
    :param K_indices:
    :param K_indptr:
    :return: predicted values for each gene at each time point
    """
    # Initialize the number of genes and kinases
    n_gene = n
    n_kinase = kinase_beta_starts.shape[0]
    # Initialize M matrix - kinase x time
    M = np.zeros((n_kinase, t_max))
    # Loop through each kinase and update the M matrix based on the beta values
    for k in prange(n_kinase):
        # For each kinase, get the starting index and count
        start = kinase_beta_starts[k]
        count = kinase_beta_counts[k]
        # For each beta value, compute the contribution to M
        for r in range(count):
            # Get the beta value and compute the global row index
            beta_val = params[total_alpha + start + r]
            global_row = start + r
            # Get the start and end indices for the current row in the sparse matrix
            row_start = K_indptr[global_row]
            row_end = K_indptr[global_row + 1]
            # For each non-zero entry in the sparse matrix, update M
            for idx in range(row_start, row_end):
                # Update M based on the beta value and the sparse matrix data
                col = K_indices[idx]
                # Update the M matrix with the contribution from the sparse matrix
                M[k, col] += beta_val * K_data[idx]
    # Initialize the predicted matrix
    pred = np.zeros((n_gene, t_max))
    # Loop through each gene and compute the predicted values
    for i in range(n_gene):
        # For each gene, get the starting index and count
        start_alpha = gene_alpha_starts[i]
        # Get the count of kinases for the current gene
        count = gene_kinase_counts[i]
        # For each alpha value, compute the contribution to the predicted matrix
        for j in range(count):
            # Get the alpha value and kinase index
            alpha_val = params[start_alpha + j]
            kinase_idx = gene_kinase_idx[start_alpha + j]
            # For each time point, update the predicted matrix
            for t in range(t_max):
                # Update the predicted value based on the kinase activity
                pred[i, t] += alpha_val * M[kinase_idx, t]
    # Return the predicted values for each gene at each time point
    return pred

def _objective_wrapper(params, P_init_dense, t_max, gene_alpha_starts, gene_kinase_counts,
                      gene_kinase_idx, total_alpha, kinase_beta_starts, kinase_beta_counts,
                      K_data, K_indices, K_indptr, time_weights, loss_type):
    """
    Wrapper function for the objective function.
    This function calls the _objective function with the appropriate parameters
    and loss function based on the loss_type parameter.
    The loss function can be one of the following:
    0: Base loss (L2 norm)
    1: Weighted loss (L2 norm with time weights)
    2: Soft L1 loss
    3: Cauchy loss
    4: Arctan loss

    :param params:
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
    :param time_weights:
    :param loss_type:
    :return: objective value
    """
    mapping = {"base": 0, "weighted": 1, "softl1": 2, "cauchy": 3, "arctan": 4}
    flag = mapping.get(loss_type, 0)
    return _objective(params, P_init_dense, t_max, P_init_dense.shape[0],
                     gene_alpha_starts, gene_kinase_counts, gene_kinase_idx,
                     total_alpha, kinase_beta_starts, kinase_beta_counts,
                     K_data, K_indices, K_indptr, time_weights, flag)