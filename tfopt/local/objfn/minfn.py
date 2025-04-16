import numpy as np
from numba import prange, njit

# -------------------------------
# Objective function for TFOpt
# -------------------------------
@njit(cache=False, fastmath=False, parallel=True, nogil=False)
def objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
               beta_start_indices, num_psites, loss_type, lam1=1e-6, lam2=1e-6):
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
    nT = n_genes * T_use
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

            # VECTORIZED RESIDUAL - use when mRNAs are > 500
        #     diff = R_meas - R_pred
        #     if loss_type == 0:  # MSE
        #         total_loss += np.dot(diff, diff)
        #     elif loss_type == 1:  # MAE
        #         total_loss += np.sum(np.abs(diff))
        #     elif loss_type == 2:  # Soft L1 (pseudo-Huber)
        #         total_loss += 2.0 * np.sum(np.sqrt(1.0 + diff * diff) - 1.0)
        #     elif loss_type == 3:  # Cauchy
        #         total_loss += np.sum(np.log(1.0 + diff * diff))
        #     elif loss_type == 4:  # Arctan
        #         total_loss += np.sum(np.arctan(diff * diff))
        #     else:
        #         total_loss += np.dot(diff, diff)
        #
        # loss = total_loss / nT
        #
        # # For elastic net penalty (loss_type 5) using vectorized operations.
        # if loss_type == 5:
        #     beta = x[n_alpha:]
        #     loss += lam1 * np.sum(np.abs(beta)) + lam2 * np.dot(beta, beta)
        #
        # # For Tikhonov regularization (loss_type 6).
        # if loss_type == 6:
        #     beta = x[n_alpha:]
        #     loss += lam1 * np.dot(beta, beta)

            # Residuals computed timepoint-by-timepoint
            for t in range(T_use):
                diff = R_meas[t] - R_pred[t]
                if loss_type == 0:  # MSE
                    total_loss += diff * diff
                elif loss_type == 1:  # MAE
                    total_loss += np.abs(diff)
                elif loss_type == 2:  # Soft L1
                    total_loss += 2.0 * (np.sqrt(1.0 + diff * diff) - 1.0)
                elif loss_type == 3:  # Cauchy
                    total_loss += np.log(1.0 + diff * diff)
                elif loss_type == 4:  # Arctan
                    total_loss += np.arctan(diff * diff)
                else:  # default to MSE
                    total_loss += diff * diff

        loss = total_loss / nT

        # Regularization penalties
        if loss_type == 5:
            beta = x[n_alpha:]
            loss += lam1 * np.sum(np.abs(beta)) + lam2 * np.dot(beta, beta)
        elif loss_type == 6:
            beta = x[n_alpha:]
            loss += lam1 * np.dot(beta, beta)

        return loss


# @njit(cache=False, fastmath=False, parallel=True, nogil=False)
# def objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
#                      beta_start_indices, num_psites, loss_type, lam1=1e-6, lam2=1e-6):
#     """
#     Computes loss per TF and returns a single scalar loss value.
#
#     For each gene and each valid regulator (TF), the function computes the TF's predicted
#     contribution to the gene expression and then compares it (point-by-point over time)
#     to the measured expression. The loss for each TF is accumulated over all gene-time
#     instances where that TF appears. The per-TF losses are then averaged (only over TFs
#     that contributed) and regularization penalties are applied.
#
#     Parameters:
#       x                  : Decision vector.
#       expression_matrix  : (n_genes x T_use) measured gene expression values.
#       regulators         : (n_genes x n_reg) indices of TF regulators for each gene.
#       tf_protein_matrix  : (n_TF x T_use) TF protein time series.
#       psite_tensor       : (n_TF x n_psite_max x T_use) matrix of PSite signals.
#       n_reg              : Maximum number of regulators per gene.
#       T_use              : Number of time points used.
#       n_genes            : Number of genes.
#       beta_start_indices : Array (length n_TF) giving the starting index (in the β–segment)
#                            for each TF.
#       num_psites         : Array (length n_TF) with the actual number of PSites for each TF.
#       loss_type          : Indicator for the loss metric (0: MSE, 1: MAE, 2: Soft L1, 3: Cauchy,
#                            4: Arctan, 5: Elastic Net, 6: Tikhonov).
#       lam1, lam2         : Regularization parameters (used for loss_type 5 and 6).
#
#     Returns:
#       total_loss         : A single scalar loss value (float).
#     """
#     n_alpha = n_genes * n_reg
#     n_tf = beta_start_indices.shape[0]  # Number of TFs
#     loss_per_tf = np.zeros(n_tf)
#     count_per_tf = np.zeros(n_tf)  # Count of timepoint contributions per TF
#
#     # Loop over genes (parallelized over genes)
#     for i in prange(n_genes):
#         R_meas = expression_matrix[i, :T_use]
#         for r in range(n_reg):
#             tf_idx = regulators[i, r]
#             if tf_idx == -1:
#                 continue  # Skip if no valid TF
#
#             # Extract parameters for the current regulator
#             a = x[i * n_reg + r]
#             protein = tf_protein_matrix[tf_idx, :T_use]
#             beta_start = beta_start_indices[tf_idx]
#             length = 1 + num_psites[tf_idx]  # Actual length of beta vector for this TF
#             beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
#
#             # Compute the TF effect: beta[0]*protein + sum over PSites.
#             tf_effect = beta_vec[0] * protein
#             for k in range(num_psites[tf_idx]):
#                 tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
#
#             # TF's predicted contribution to gene expression:
#             predicted_tf = a * tf_effect
#
#             # Compute loss for each time point based solely on the TF's contribution.
#             for t in range(T_use):
#                 diff = R_meas[t] - predicted_tf[t]
#                 if loss_type == 0:  # MSE
#                     loss_per_tf[tf_idx] += diff * diff
#                 elif loss_type == 1:  # MAE
#                     loss_per_tf[tf_idx] += np.abs(diff)
#                 elif loss_type == 2:  # Soft L1 (pseudo-Huber)
#                     loss_per_tf[tf_idx] += 2.0 * (np.sqrt(1.0 + diff * diff) - 1.0)
#                 elif loss_type == 3:  # Cauchy
#                     loss_per_tf[tf_idx] += np.log(1.0 + diff * diff)
#                 elif loss_type == 4:  # Arctan
#                     loss_per_tf[tf_idx] += np.arctan(diff * diff)
#                 else:  # Default to MSE if unspecified
#                     loss_per_tf[tf_idx] += diff * diff
#                 count_per_tf[tf_idx] += 1
#
#     # Average the loss per TF by the number of contributing timepoints.
#     for tf in range(n_tf):
#         if count_per_tf[tf] > 0:
#             loss_per_tf[tf] /= count_per_tf[tf]
#
#     # Apply regularization penalties per TF (if using loss types 5 or 6).
#     if loss_type == 5:
#         # Elastic Net penalty: combining L1 and L2 on each TF's beta vector.
#         for tf in range(n_tf):
#             beta_start = beta_start_indices[tf]
#             length = 1 + num_psites[tf]
#             beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
#             loss_per_tf[tf] += lam1 * np.sum(np.abs(beta_vec)) + lam2 * np.dot(beta_vec, beta_vec)
#     elif loss_type == 6:
#         # Tikhonov regularization penalty (L2)
#         for tf in range(n_tf):
#             beta_start = beta_start_indices[tf]
#             length = 1 + num_psites[tf]
#             beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
#             loss_per_tf[tf] += lam1 * np.dot(beta_vec, beta_vec)
#
#     # Reduce the per-TF losses to a single scalar by averaging over the TFs that contributed.
#     total_loss = 0.0
#     valid_tf_count = 0
#     for tf in range(n_tf):
#         if count_per_tf[tf] > 0:
#             total_loss += loss_per_tf[tf]
#             valid_tf_count += 1
#     if valid_tf_count > 0:
#         total_loss /= valid_tf_count
#
#     return total_loss

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
