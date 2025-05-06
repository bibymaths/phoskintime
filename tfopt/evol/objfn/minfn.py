import numpy as np
from numba import njit, prange
from pymoo.core.problem import Problem

from tfopt.evol.config.constants import VECTORIZED_LOSS_FUNCTION


# -------------------------------
# Multi-Objective Problem Definition
# -------------------------------
class TFOptimizationMultiObjectiveProblem(Problem):
    """
    Originally implemented by Julius Normann.

    This version has been modified and optimized
    for consistency & speed in submodules by Abhinav Mishra.

    Multi-objective optimization problem for TF optimization.
    This class defines a multi-objective optimization problem for the
    transcription factor (TF) optimization problem. It inherits from the
    `Problem` class in the pymoo library. The problem is defined with three
    objectives: f1 (error), f2 (alpha violation), and f3 (beta violation).
    """

    def __init__(self, n_var, n_mRNA, n_TF, n_reg, n_psite_max, n_alpha,
                 mRNA_mat, regulators, protein_mat, psite_tensor, T_use,
                 beta_start_indices, num_psites, no_psite_tf, xl=None, xu=None,
                 **kwargs):
        """
        Initialize the multi-objective optimization problem.

        Args:
            n_var (int): Number of decision variables.
            n_mRNA (int): Number of mRNAs.
            n_TF (int): Number of transcription factors.
            n_reg (int): Number of regulators.
            n_psite_max (int): Maximum number of phosphorylation sites.
            n_alpha (int): Number of alpha parameters.
            mRNA_mat (np.ndarray): Matrix of mRNA measurements.
            regulators (np.ndarray): Matrix of regulators for each mRNA.
            protein_mat (np.ndarray): Matrix of TF protein levels.
            psite_tensor (np.ndarray): Tensor of phosphorylation sites.
            T_use (int): Number of time points to use.
            beta_start_indices (list): List of starting indices for beta parameters.
            num_psites (list): List of number of phosphorylation sites for each TF.
            no_psite_tf (list): List indicating if a TF has no phosphorylation site.
            xl (np.ndarray, optional): Lower bounds for decision variables. Defaults to None.
            xu (np.ndarray, optional): Upper bounds for decision variables. Defaults to None.
        """
        super().__init__(n_var=n_var, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.n_mRNA = n_mRNA
        self.n_TF = n_TF
        self.n_reg = n_reg
        self.n_psite_max = n_psite_max
        self.n_alpha = n_alpha
        self.mRNA_mat = mRNA_mat
        self.regulators = regulators
        self.protein_mat = protein_mat
        self.psite_tensor = psite_tensor
        self.T_use = T_use
        self.beta_start_indices = beta_start_indices
        self.num_psites = num_psites
        self.no_psite_tf = no_psite_tf
        self.loss_type = kwargs.get("loss_type", 0)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the objectives for the given decision variables.

        Args:
            X (np.ndarray): Decision variable matrix.
            out (dict): Dictionary to store the results.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        n_pop = X.shape[0]
        F = np.empty((n_pop, 3))
        n_alpha = self.n_alpha
        for i in range(n_pop):
            xi = X[i]
            f1 = objective_(xi, self.mRNA_mat, self.regulators, self.protein_mat,
                            self.psite_tensor, self.n_reg, self.T_use, self.n_mRNA, self.beta_start_indices,
                            self.num_psites, self.loss_type)
            f2 = 0.0
            for m in range(self.n_mRNA):
                s = 0.0
                for r in range(self.n_reg):
                    s += xi[m * self.n_reg + r]
                f2 += (s - 1.0) ** 2
            f3 = 0.0
            for tf in range(self.n_TF):
                start = n_alpha + self.beta_start_indices[tf]
                length = 1 + self.num_psites[tf]
                beta_vec = xi[start: start + length]
                f3 += (np.sum(beta_vec) - 1.0) ** 2
                if self.no_psite_tf[tf]:
                    for q in range(1, length):
                        f3 += beta_vec[q] ** 2
            # Three objectives:
            # f1 (error)
            F[i, 0] = f1
            # f2 (alpha violation)
            F[i, 1] = f2
            # f3 (beta violation)
            F[i, 2] = f3
        out["F"] = F

@njit(cache=True, fastmath=False, parallel=True, nogil=False)
def objective_(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA,
               beta_start_indices, num_psites, loss_type, lam1=1e-3, lam2=1e-3):
    """
    Computes a loss value for transcription factor optimization using evolutionary algorithms.

    Args:
        x (np.ndarray): Optimization variables.
        mRNA_mat (np.ndarray): Matrix of mRNA measurements.
        regulators (np.ndarray): Matrix of regulators for each mRNA.
        protein_mat (np.ndarray): Matrix of TF protein levels.
        psite_tensor (np.ndarray): Tensor of phosphorylation sites.
        n_reg (int): Number of regulators.
        T_use (int): Number of time points to use.
        n_mRNA (int): Number of mRNAs.
        beta_start_indices (list): List of starting indices for beta parameters.
        num_psites (list): List of number of phosphorylation sites for each TF.
        loss_type (int): Type of loss function to use.
        lam1 (float, optional): L1 penalty coefficient. Defaults to 1e-3.
        lam2 (float, optional): L2 penalty coefficient. Defaults to 1e-3.

    Returns:
        float: Computed loss value.
    """
    # Initialize loss to zero.
    total_loss = 0.0
    # Compute the loss for each mRNA.
    n_alpha = n_mRNA * n_reg
    nT = n_mRNA * T_use
    for i in prange(n_mRNA):
        # Get the measured mRNA values and initialize the predicted mRNA values.
        R_meas = mRNA_mat[i, :T_use]
        R_pred = np.zeros(T_use)
        # For each regulator, compute the predicted mRNA values.
        for r in range(n_reg):
            # Get the index of the TF for this regulator.
            tf_idx = regulators[i, r]
            if tf_idx == -1:  # No valid TF for this regulator
                continue
            # Get the TF activity, protein levels, and beta vector.
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            # Get the length of the beta vector for this TF.
            length = 1 + num_psites[tf_idx]  # actual length of beta vector for TF
            beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
            # Compute the predicted mRNA values.
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                # Add the effect of each phosphorylation site.
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            # Compute the predicted mRNA values.
            R_pred += a * tf_effect
        # Ensure R_pred is non-negative
        np.clip(R_pred, 0.0, None, out=R_pred)

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
                # Default to MSE if unknown.
                total_loss += e * e

    loss = total_loss / nT

    # For elastic net (loss_type 5), add L1 and L2 penalties on the beta portion.
    if loss_type == 5:
        l1 = 0.0
        l2 = 0.0
        # Compute over beta parameters only.
        for i in range(n_alpha, x.shape[0]):
            # Get the beta vector for this TF.
            v = x[i]
            # Add L1 and L2 penalties.
            l1 += abs(v)
            l2 += v * v
        # Compute the penalties.
        loss += lam1 * l1 + lam2 * l2

    # For Tikhonov (loss_type 6), add L2 penalty on the beta portion.
    if loss_type == 6:
        l2 = 0.0
        # Compute over beta parameters only.
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            # Add L2 penalty.
            l2 += v * v
        # Compute the penalty.
        loss += lam1 * l2

    return loss
