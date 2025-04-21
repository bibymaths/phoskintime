import numpy as np
from numba import prange, njit

from tfopt.local.config.constants import VECTORIZED_LOSS_FUNCTION


@njit(cache=False, fastmath=False, parallel=True, nogil=False)
def objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
               beta_start_indices, num_psites, loss_type, lam1=1e-6, lam2=1e-6):
    """
    Originally implemented by Julius Normann.

    This version has been modified and optimized
    for consistency & speed in submodules by Abhinav Mishra.

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
    # Initialize loss
    total_loss = 0.0
    # Initialize the number of genes and time points
    n_alpha = n_genes * n_reg
    nT = n_genes * T_use
    # Initialize the number of regulators
    for i in prange(n_genes):
        # Compute the predicted expression for each gene
        R_meas = expression_matrix[i, :T_use]
        # Initialize the predicted expression
        R_pred = np.zeros(T_use)
        # Loop over the regulators for each gene
        for r in range(n_reg):
            # Get the index of the transcription factor (TF) for this regulator
            tf_idx = regulators[i, r]
            if tf_idx == -1:  # No valid TF for this regulator
                continue
            # Get the TF protein time series
            a = x[i * n_reg + r]
            protein = tf_protein_matrix[tf_idx, :T_use]
            # Get the starting index for the beta vector for this TF
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]  # actual length of beta vector for TF
            # Extract the beta vector for this TF
            beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
            # Compute the effect on the TF from the missing phosphorylation sites
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                # Compute the effect of each post-translational modification
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            # Compute the predicted expression
            R_pred += a * tf_effect
        # Ensure non-negative predictions
        np.clip(R_pred, 0.0, None, out=R_pred)

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

def compute_predictions(x, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices,
                        num_psites):
    """
    Computes the predicted expression matrix based on the decision vector x.
    This function uses the regulators, TF protein matrix, and post-translational modification tensor to generate
    predictions for each gene at each time point.
    Parameters:
      x                  : Decision vector.
      regulators         : (n_genes x n_reg) indices of TF regulators for each gene.
      tf_protein_matrix  : (n_TF x T_use) TF protein time series.
      psite_tensor       : (n_TF x n_psite_max x T_use) matrix of PSite signals (padded with zeros).
      n_reg              : Maximum number of regulators per gene.
      T_use              : Number of time points used.
      n_genes, n_TF     : Number of genes and TF respectively.
      beta_start_indices : Integer array giving the starting index (in the β–segment) for each TF.
      num_psites         : Integer array with the actual number of PSites for each TF.
    Returns:
        predictions        : (n_genes x T_use) matrix of predicted gene expression values.
    """
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
    """
    Wrapper function for the objective function.
    This function is used to call the objective function with the appropriate parameters.
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
    Returns:
        loss               : The computed loss (a scalar).
    """
    return objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                      beta_start_indices, num_psites, loss_type)