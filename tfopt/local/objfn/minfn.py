import numpy as np
from numba import prange, njit

# -------------------------------
# Objective function for TFOpt
# -------------------------------
@njit(fastmath=True, parallel=True, nogil=True)
def objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
               beta_start_indices, num_psites, loss_type, lam1=1e-3, lam2=1e-6):
    """
    Computes a loss value using one of several loss functions.

    Parameters:
      x                  : Decision vector.
      expression_matrix  : (n_genes x T_use) measured gene expression values.
      regulators         : (n_genes x n_reg) indices of TF regulators for each gene.
      tf_protein_matrix  : (n_TF x T_use) TF protein time series.
      psite_tensor       : (n_TF x n_psite_max x T_use) matrix of PSite signals (padded with zeros).
      n_reg              : Maximum number of regulators per gene.
      T_use              : Number of time points used.
      n_genes, n_TF     : Number of genes and TF respectively.
      beta_start_indices : Integer array giving the starting index (in the β–segment) for each TF.
      num_psites         : Integer array with the actual number of PSites for each TF.
      loss_type          : Integer indicating the loss type (0: MSE, 1: MAE, 2: soft L1, 3: Cauchy, 4: Arctan, 5: Elastic Net, 6: Tikhonov).
      lam1, lam2         : Regularization parameters (used for loss_type 5 and 6).

    Returns:
      The computed loss (a scalar).
    """
    total_loss = 0.0
    n_alpha = n_genes * n_reg
    for i in prange(n_genes):
        R_meas = expression_matrix[i, :T_use]
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1:  # No valid TF for this regulator
                continue
            a = x[i * n_reg + r]
            protein = tf_protein_matrix[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]  # actual length of beta vector for TF
            beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        # For each time point, add loss according to loss_type.
        for t in range(T_use):
            e = R_meas[t] - R_pred[t]
            if loss_type == 0:  # MSE
                total_loss += e * e
            elif loss_type == 1:  # MAE
                total_loss += abs(e)
            elif loss_type == 2:  # Soft L1 (pseudo-Huber)
                total_loss += 2.0 * (np.sqrt(1.0 + e * e) - 1.0)
            elif loss_type == 3:  # Cauchy
                total_loss += np.log(1.0 + e * e)
            elif loss_type == 4:  # Arctan
                total_loss += np.arctan(e * e)
            else:
                total_loss += e * e
    loss = total_loss / (n_genes * T_use)

    # For elastic net (loss_type 5), add L1 and L2 penalties on the beta portion.
    if loss_type == 5:
        l1 = 0.0
        l2 = 0.0
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            l1 += abs(v)
            l2 += v * v
        loss += lam1 * l1 + lam2 * l2

    # For Tikhonov (loss_type 6), add L2 penalty on the beta portion.
    if loss_type == 6:
        l2 = 0.0
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            l2 += v * v
        loss += lam1 * l2

    return loss

def compute_predictions(x, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices,
                        num_psites):
    n_alpha = n_genes * n_reg
    predictions = np.zeros((n_genes, T_use))
    for i in range(n_genes):
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1:
                continue
            a = x[i * n_reg + r]
            protein = tf_protein_matrix[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]
            beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        predictions[i, :] = R_pred
    return predictions

def objective_wrapper(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                      beta_start_indices, num_psites, loss_type):
    return objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                      beta_start_indices, num_psites, loss_type)