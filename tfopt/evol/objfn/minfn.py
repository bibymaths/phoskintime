import numpy as np
from numba import njit, prange
from pymoo.core.problem import Problem


# -------------------------------
# Numba helpers
# -------------------------------

@njit(cache=True, nogil=True)
def _alpha_violation_sq(x, n_mRNA, n_reg):
    # sum_r alpha_{i,r} == 1 for each mRNA i
    v = 0.0
    for i in range(n_mRNA):
        s = 0.0
        base = i * n_reg
        for r in range(n_reg):
            s += x[base + r]
        d = s - 1.0
        v += d * d
    return v


@njit(cache=True, nogil=True)
def _beta_violation_sq(x, n_TF, n_alpha, beta_starts, beta_lens, no_psite_tf):
    # sum beta_vec == 1 for each TF
    # if TF has no psite: penalize beta_vec[1:] toward 0
    v = 0.0
    for tf in range(n_TF):
        start = n_alpha + beta_starts[tf]
        length = beta_lens[tf]

        s = 0.0
        for j in range(length):
            s += x[start + j]

        d = s - 1.0
        v += d * d

        if no_psite_tf[tf]:
            # penalize non-protein components (beta[1:]) if TF has no psite
            for j in range(1, length):
                bj = x[start + j]
                v += bj * bj
    return v


@njit(cache=True, fastmath=False, nogil=True)
def _loss_single(
    x,
    mRNA_mat, regulators, protein_mat, psite_tensor,
    n_reg, T_use, n_mRNA, n_alpha,
    beta_starts, num_psites, loss_type,
    lam1, lam2
):
    """
    Single-individual loss. This mirrors your original objective_ but:
      - avoids slicing beta_vec (Numba-friendly, faster)
      - keeps correctness identical
    """
    total_loss = 0.0
    nT = n_mRNA * T_use

    for i in range(n_mRNA):
        # build prediction for this mRNA i
        # NOTE: allocate per-i vector
        R_pred = np.zeros(T_use, dtype=np.float64)

        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1:
                continue

            a = x[i * n_reg + r]

            # beta vector layout: [beta0 (protein), beta1..betaK (psites)]
            b_start = n_alpha + beta_starts[tf_idx]
            k_ps = num_psites[tf_idx]
            length = 1 + k_ps

            # protein effect
            b0 = x[b_start]
            for t in range(T_use):
                R_pred[t] += a * (b0 * protein_mat[tf_idx, t])

            # psite effects
            for k in range(k_ps):
                bk = x[b_start + 1 + k]
                if bk == 0.0:
                    continue
                for t in range(T_use):
                    R_pred[t] += a * (bk * psite_tensor[tf_idx, k, t])

        # clip non-negative
        for t in range(T_use):
            if R_pred[t] < 0.0:
                R_pred[t] = 0.0

        # accumulate loss
        for t in range(T_use):
            e = mRNA_mat[i, t] - R_pred[t]

            if loss_type == 0:      # MSE
                total_loss += e * e
            elif loss_type == 1:    # MAE
                total_loss += abs(e)
            elif loss_type == 2:    # soft L1
                total_loss += 2.0 * (np.sqrt(1.0 + e * e) - 1.0)
            elif loss_type == 3:    # Cauchy
                total_loss += np.log(1.0 + e * e)
            elif loss_type == 4:    # Arctan
                total_loss += np.arctan(e * e)
            else:
                total_loss += e * e

    loss = total_loss / float(nT)

    # Elastic net (loss_type 5): penalties on beta portion only
    if loss_type == 5:
        l1 = 0.0
        l2 = 0.0
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            l1 += abs(v)
            l2 += v * v
        loss += lam1 * l1 + lam2 * l2

    # Tikhonov (loss_type 6): L2 penalty on beta portion only
    if loss_type == 6:
        l2 = 0.0
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            l2 += v * v
        loss += lam1 * l2

    return loss


@njit(cache=True, fastmath=False, nogil=True)
def _evaluate_population(
    X,
    mRNA_mat, regulators, protein_mat, psite_tensor,
    n_reg, T_use, n_mRNA, n_TF, n_alpha,
    beta_starts, num_psites, no_psite_tf,
    loss_type, lam1, lam2
):
    """
    Batched evaluation: returns F (n_pop, 3).
    """
    n_pop = X.shape[0]
    F = np.empty((n_pop, 3), dtype=np.float64)

    # Precompute beta length per TF once (Numba-friendly)
    beta_lens = np.empty(n_TF, dtype=np.int32)
    for tf in range(n_TF):
        beta_lens[tf] = 1 + num_psites[tf]

    for p in range(n_pop):
        x = X[p]

        f1 = _loss_single(
            x,
            mRNA_mat, regulators, protein_mat, psite_tensor,
            n_reg, T_use, n_mRNA, n_alpha,
            beta_starts, num_psites, loss_type,
            lam1, lam2
        )

        f2 = _alpha_violation_sq(x, n_mRNA, n_reg)
        f3 = _beta_violation_sq(x, n_TF, n_alpha, beta_starts, beta_lens, no_psite_tf)

        F[p, 0] = f1
        F[p, 1] = f2
        F[p, 2] = f3

    return F

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

        # Store scalars
        self.n_mRNA = int(n_mRNA)
        self.n_TF = int(n_TF)
        self.n_reg = int(n_reg)
        self.n_psite_max = int(n_psite_max)
        self.n_alpha = int(n_alpha)
        self.T_use = int(T_use)

        # Ensure arrays are NumPy arrays with stable dtypes (required for Numba)
        self.mRNA_mat = np.asarray(mRNA_mat, dtype=np.float64)
        self.regulators = np.asarray(regulators, dtype=np.int32)
        self.protein_mat = np.asarray(protein_mat, dtype=np.float64)
        self.psite_tensor = np.asarray(psite_tensor, dtype=np.float64)

        self.beta_start_indices = np.asarray(beta_start_indices, dtype=np.int32)
        self.num_psites = np.asarray(num_psites, dtype=np.int32)
        self.no_psite_tf = np.asarray(no_psite_tf, dtype=np.bool_)

        # Loss config
        self.loss_type = int(kwargs.get("loss_type", 0))
        self.lam1 = float(kwargs.get("lam1", 1e-3))
        self.lam2 = float(kwargs.get("lam2", 1e-3))

        # warm up compilation once
        # Create a tiny dummy X with correct shape.
        dummy = np.zeros((1, self.n_var), dtype=np.float64)
        _ = _evaluate_population(
            dummy,
            self.mRNA_mat, self.regulators, self.protein_mat, self.psite_tensor,
            self.n_reg, self.T_use, self.n_mRNA, self.n_TF, self.n_alpha,
            self.beta_start_indices, self.num_psites, self.no_psite_tf,
            self.loss_type, self.lam1, self.lam2
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the objectives for the given decision variables.

        Args:
            X (np.ndarray): Decision variable matrix.
            out (dict): Dictionary to store the results.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        X = np.asarray(X, dtype=np.float64)
        F = _evaluate_population(
            X,
            self.mRNA_mat, self.regulators, self.protein_mat, self.psite_tensor,
            self.n_reg, self.T_use, self.n_mRNA, self.n_TF, self.n_alpha,
            self.beta_start_indices, self.num_psites, self.no_psite_tf,
            self.loss_type, self.lam1, self.lam2
        )
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
