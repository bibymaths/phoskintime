from __future__ import annotations

import os
from multiprocessing.pool import ThreadPool
from typing import Optional

import numpy as np
from numba import njit
from pymoo.core.problem import Problem


@njit(cache=True, nogil=True)
def _alpha_violation_sq(x: np.ndarray, n_mRNA: int, n_reg: int) -> float:
    """
    Computes the squared violation of the alpha constraint for a set of mRNA and regulatory factors.

    The function ensures that the sum of alpha values for each mRNA equals 1. If the constraint is
    violated, the function calculates the squared deviation from 1 for each mRNA and returns the
    total squared violation.

    Args:
        x (np.ndarray): A 1D array representing the flattened set of alpha values. Each mRNA has
            `n_reg` associated alpha values, stored consecutively in the array.
        n_mRNA (int): The number of mRNA molecules being analyzed.
        n_reg (int): The number of regulatory factors associated with each mRNA.

    Returns:
        float: The total squared violation of the alpha constraint for all mRNA molecules.
    """
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
def _beta_violation_sq(
    x: np.ndarray,
    n_TF: int,
    n_alpha: int,
    beta_starts: np.ndarray,
    beta_lens: np.ndarray,
    no_psite_tf: np.ndarray
) -> float:
    """
    Computes the squared violation of beta constraint conditions for a given set of transcription factors (TFs).

    The function evaluates the sum of squared differences between the total β values associated with each TF and their expected
    value of 1. Furthermore, it penalizes non-protein components (β[1:]) for TFs that have no binding site (psite).

    Args:
        x (np.ndarray): The input parameter vector which contains all involved α and β values.
        n_TF (int): The total number of transcription factors.
        n_alpha (int): The number of α values, which denotes the offset in the input vector for β values.
        beta_starts (np.ndarray): Array indicating start indices of β values for each TF in the input vector.
        beta_lens (np.ndarray): Array containing the lengths (number of β terms) for each TF.
        no_psite_tf (np.ndarray): A boolean array where each entry indicates whether a TF has no binding site (True if no psite).

    Returns:
        float: The accumulated squared violation, including penalties for TFs with no binding site.
    """
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
def _loss_single_inplace(
    x: np.ndarray,
    mRNA_mat: np.ndarray,
    regulators: np.ndarray,
    protein_mat: np.ndarray,
    psite_tensor: np.ndarray,
    n_reg: int,
    T_use: int,
    n_mRNA: int,
    n_alpha: int,
    beta_starts: np.ndarray,
    num_psites: np.ndarray,
    loss_type: int,
    lam1: float,
    lam2: float,
    R_pred: np.ndarray,  # scratch length T_use
) -> float:
    """
    Calculates the loss for a single mRNA sequence and updates the result in-place
    using various loss functions and optional penalties.

    This function computes the regulation hypothesis for a single mRNA using the
    values of regulatory proteins, sites, and their coefficients, compares it to
    the observed mRNA data, and accumulates the loss based on the specified loss
    type. It also supports elastic net and Tikhonov penalties for regularization.

    Args:
        x (np.ndarray): Coefficients vector. The first `n_alpha` values correspond
            to alpha coefficients, and the remaining portion corresponds to beta
            coefficients.
        mRNA_mat (np.ndarray): Matrix containing observed mRNA data with shape
            (n_mRNA, T_use).
        regulators (np.ndarray): Matrix specifying regulatory relationships. Each
            row defines the indices of the regulatory proteins for the
            corresponding mRNA. Non-regulators are marked as -1.
        protein_mat (np.ndarray): Protein expression matrix, indexed by protein
            and time point, with shape (n_proteins, T_use).
        psite_tensor (np.ndarray): Tensor containing phosphorylation site effects
            with shape (n_proteins, max_psites, T_use).
        n_reg (int): The number of regulators per mRNA.
        T_use (int): The number of time points to consider for calculations.
        n_mRNA (int): The total number of mRNA sequences in the system.
        n_alpha (int): The count of alpha coefficients. Alpha coefficients are
            used for regulatory interactions.
        beta_starts (np.ndarray): Vector of starting indices for each regulator's
            beta coefficients within `x`.
        num_psites (np.ndarray): Array defining the number of phosphorylation sites
            for each regulator protein.
        loss_type (int): Indicator of the loss function to use. Supported values:
            - 0: Mean Squared Error (MSE)
            - 1: Mean Absolute Error (MAE)
            - 2: Soft L1 loss
            - 3: Cauchy loss
            - 4: Arctan loss
            - 5: Elastic net (includes L1 and L2 penalties on beta coefficients)
            - 6: Tikhonov (includes L2 penalty on beta coefficients)
        lam1 (float): Elastic net L1 penalty coefficient. Used only when
            `loss_type` is 5.
        lam2 (float): Elastic net or Tikhonov L2 penalty coefficient. Used only
            when `loss_type` is 5 or 6.
        R_pred (np.ndarray): Temporary scratch array of length `T_use` to store the
            predicted mRNA values at each time point.

    Returns:
        float: The computed loss value normalized by the total number of data
        points (n_mRNA * T_use).
    """
    total_loss = 0.0
    nT = n_mRNA * T_use

    for i in range(n_mRNA):
        # zero scratch
        for t in range(T_use):
            R_pred[t] = 0.0

        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1:
                continue

            a = x[i * n_reg + r]

            # beta vector layout: [beta0 (protein), beta1..betaK (psites)]
            b_start = n_alpha + beta_starts[tf_idx]
            k_ps = num_psites[tf_idx]

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
def _evaluate_population_slice(
    X: np.ndarray,
    F: np.ndarray,
    p0: int,
    p1: int,
    mRNA_mat: np.ndarray,
    regulators: np.ndarray,
    protein_mat: np.ndarray,
    psite_tensor: np.ndarray,
    n_reg: int,
    T_use: int,
    n_mRNA: int,
    n_TF: int,
    n_alpha: int,
    beta_starts: np.ndarray,
    num_psites: np.ndarray,
    no_psite_tf: np.ndarray,
    loss_type: int,
    lam1: float,
    lam2: float
) -> None:
    """
    Evaluates a slice of the population to compute various loss components.

    This function calculates loss values for a specified slice of the population
    using parameters, matrices, and tensor representations of mRNA, regulatory
    data, protein data, and p-sites. The computed loss components are stored in
    the designated output matrix for further processing or evaluation.

    Args:
        X (np.ndarray): Population matrix where each row represents a set of
            model parameters for an individual population member.
        F (np.ndarray): Output matrix to store computed values for different
            loss components.
        p0 (int): Starting index of the population slice to evaluate.
        p1 (int): Ending index (exclusive) of the population slice to evaluate.
        mRNA_mat (np.ndarray): Matrix representing mRNA levels across time points.
        regulators (np.ndarray): Matrix representing regulatory factors across
            time points and targets.
        protein_mat (np.ndarray): Matrix representing protein levels for
            transcription factors across time points.
        psite_tensor (np.ndarray): Tensor containing p-site data for the protein
            interactions.
        n_reg (int): Number of regulatory factors considered.
        T_use (int): Number of time points used in the analysis.
        n_mRNA (int): Number of mRNA molecules considered in the analysis.
        n_TF (int): Number of transcription factors considered.
        n_alpha (int): Number of alpha parameters for mRNA-regulator interaction.
        beta_starts (np.ndarray): Array storing starting indices for beta
            parameters for each transcription factor.
        num_psites (np.ndarray): Array storing the number of p-sites for each
            transcription factor.
        no_psite_tf (np.ndarray): Array indicating transcription factors without
            p-sites.
        loss_type (int): Integer flag determining the type of loss function
            applied.
        lam1 (float): Regularization coefficient for penalty term one.
        lam2 (float): Regularization coefficient for penalty term two.
    """
    # precompute beta lengths
    beta_lens = np.empty(n_TF, dtype=np.int32)
    for tf in range(n_TF):
        beta_lens[tf] = 1 + num_psites[tf]

    # per-slice scratch for loss
    R_pred = np.empty(T_use, dtype=np.float64)

    for p in range(p0, p1):
        x = X[p]

        f1 = _loss_single_inplace(
            x,
            mRNA_mat, regulators, protein_mat, psite_tensor,
            n_reg, T_use, n_mRNA, n_alpha,
            beta_starts, num_psites, loss_type,
            lam1, lam2,
            R_pred
        )

        f2 = _alpha_violation_sq(x, n_mRNA, n_reg)
        f3 = _beta_violation_sq(x, n_TF, n_alpha, beta_starts, beta_lens, no_psite_tf)

        F[p, 0] = f1
        F[p, 1] = f2
        F[p, 2] = f3


def _as_contig(a: np.ndarray, dtype) -> np.ndarray:
    """
    Converts the input array into a contiguous array of the specified data type.

    This function ensures that the given array is converted into a contiguous
    memory layout with the specified data type. If the input is not an array,
    it is first converted into one before being made contiguous.

    Args:
        a (np.ndarray): The input array to be converted. If not already an array,
            it will be cast to an array.
        dtype: The desired data type for the output array.

    Returns:
        np.ndarray: A contiguous memory layout array with the specified data type.
    """
    return np.ascontiguousarray(np.asarray(a, dtype=dtype))


class TFOptimizationMultiObjectiveProblem(Problem):
    """
    Represents a multi-objective optimization problem specific to transcription factor (TF) and
    mRNA synthesis dynamics.

    This class is an extension of the `Problem` class and is designed to model complex biological
    processes by incorporating various dynamic parameters like regulators, protein matrices,
    psite tensors, and associated configurations. It supports parallel evaluation for
    multi-thread usage, optimizing performance for large populations.

    Attributes:
        n_mRNA (int): Number of mRNA species in the system.
        n_TF (int): Number of transcription factor species in the system.
        n_reg (int): Number of regulators.
        n_psite_max (int): Maximum number of potential p-sites.
        n_alpha (int): Number of alpha parameters used in modeling.
        T_use (int): Number of time units or steps to use in the evaluation.

        mRNA_mat (np.ndarray): A matrix representing the mRNA dynamics.
        regulators (np.ndarray): Array of regulator IDs associated with mRNA and TF interactions.
        protein_mat (np.ndarray): A matrix representing the protein synthesis rates or patterns.
        psite_tensor (np.ndarray): A tensor indicating probabilistic binding sites of proteins.

        beta_start_indices (np.ndarray): Array indicating the starting indices of beta coefficients.
        num_psites (np.ndarray): Array indicating the number of p-sites per transcription factor.
        no_psite_tf (np.ndarray): Boolean array indicating TFs with zero p-sites.

        loss_type (int): Configurable loss function type for the optimization. Defaults to 0.
        lam1 (float): First regularization parameter for loss calculation. Defaults to 1e-3.
        lam2 (float): Second regularization parameter for loss calculation. Defaults to 1e-3.

        max_threads (int): Maximum number of threads to use for evaluation. 0 indicates
            automatic thread selection based on system capacity.
    """

    def __init__(
        self,
        n_var: int,
        n_mRNA: int,
        n_TF: int,
        n_reg: int,
        n_psite_max: int,
        n_alpha: int,
        mRNA_mat: np.ndarray,
        regulators: np.ndarray,
        protein_mat: np.ndarray,
        psite_tensor: np.ndarray,
        T_use: int,
        beta_start_indices: np.ndarray,
        num_psites: np.ndarray,
        no_psite_tf: np.ndarray,
        xl: Optional[np.ndarray] = None,
        xu: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initializes the class with various parameters required for computational evaluation.

        Args:
            n_var (int): The number of variables.
            n_mRNA (int): The number of mRNA molecules.
            n_TF (int): The number of transcription factors (TFs).
            n_reg (int): The number of regulators.
            n_psite_max (int): The maximum number of p-sites.
            n_alpha (int): The number of alpha coefficients.
            mRNA_mat (np.ndarray): Matrix representing the mRNA data.
            regulators (np.ndarray): Array representing the regulator mappings.
            protein_mat (np.ndarray): Matrix representing the protein data.
            psite_tensor (np.ndarray): Tensor representing the p-site data.
            T_use (int): The time step or usage parameter.
            beta_start_indices (np.ndarray): Array of start indices for beta calculations.
            num_psites (np.ndarray): Array representing the number of p-sites.
            no_psite_tf (np.ndarray): Boolean array indicating TFs with no associated p-sites.
            xl (Optional[np.ndarray]): Optional lower-bound array for the variables.
            xu (Optional[np.ndarray]): Optional upper-bound array for the variables.
            **kwargs: Additional optional arguments such as "loss_type", "lam1", "lam2", and "threads".
        """
        super().__init__(n_var=int(n_var), n_obj=3, n_constr=0, xl=xl, xu=xu)

        # scalars
        self.n_mRNA = int(n_mRNA)
        self.n_TF = int(n_TF)
        self.n_reg = int(n_reg)
        self.n_psite_max = int(n_psite_max)
        self.n_alpha = int(n_alpha)
        self.T_use = int(T_use)

        # arrays (stable dtypes, contiguous)
        self.mRNA_mat = _as_contig(mRNA_mat, np.float64)
        self.regulators = _as_contig(regulators, np.int32)
        self.protein_mat = _as_contig(protein_mat, np.float64)
        self.psite_tensor = _as_contig(psite_tensor, np.float64)

        self.beta_start_indices = _as_contig(beta_start_indices, np.int32)
        self.num_psites = _as_contig(num_psites, np.int32)
        self.no_psite_tf = _as_contig(no_psite_tf, np.bool_)

        # loss config
        self.loss_type = int(kwargs.get("loss_type", 0))
        self.lam1 = float(kwargs.get("lam1", 1e-3))
        self.lam2 = float(kwargs.get("lam2", 1e-3))

        # thread config (optional override)
        # choose min(pop, cpu) at runtime; but allow forcing via kwargs.
        self.max_threads = int(kwargs.get("threads", 0))  # 0 => auto

        # Warm-up compile (single slice) to avoid compilation inside threadpool
        dummy_X = np.zeros((1, self.n_var), dtype=np.float64)
        dummy_F = np.zeros((1, 3), dtype=np.float64)
        _evaluate_population_slice(
            dummy_X, dummy_F, 0, 1,
            self.mRNA_mat, self.regulators, self.protein_mat, self.psite_tensor,
            self.n_reg, self.T_use, self.n_mRNA, self.n_TF, self.n_alpha,
            self.beta_start_indices, self.num_psites, self.no_psite_tf,
            self.loss_type, self.lam1, self.lam2
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates a population of input candidates by computing fitness scores in a multi-objective
        optimization context. The function supports multi-threaded execution to efficiently process
        the input data.

        Args:
            X: Input array containing candidate solutions to be evaluated, where each row
                corresponds to an individual candidate and columns represent respective features.
            out: A dictionary-like container to store the output of computed fitness scores.
                The resulting scores are stored in a key "F" as an ndarray of shape (n_pop, 3).
            *args: Additional positional arguments that are ignored in the current implementation.
            **kwargs: Additional keyword arguments that are ignored in the current implementation.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_pop = int(X.shape[0])
        F = np.empty((n_pop, 3), dtype=np.float64)

        if n_pop == 0:
            out["F"] = F
            return

        # threads
        if self.max_threads > 0:
            n_threads = min(self.max_threads, n_pop)
        else:
            n_threads = min(os.cpu_count() or 1, n_pop)

        # If only one thread, run directly (lower overhead)
        if n_threads <= 1:
            _evaluate_population_slice(
                X, F, 0, n_pop,
                self.mRNA_mat, self.regulators, self.protein_mat, self.psite_tensor,
                self.n_reg, self.T_use, self.n_mRNA, self.n_TF, self.n_alpha,
                self.beta_start_indices, self.num_psites, self.no_psite_tf,
                self.loss_type, self.lam1, self.lam2
            )
            out["F"] = F
            return

        # chunk slices
        chunk = (n_pop + n_threads - 1) // n_threads
        jobs = []
        for t in range(n_threads):
            p0 = t * chunk
            p1 = min(n_pop, (t + 1) * chunk)
            if p0 < p1:
                jobs.append((p0, p1))

        # ThreadPool: safe because kernels run with nogil=True
        pool = ThreadPool(n_threads)
        try:
            # Use starmap with a top-level callable to avoid pickling issues.
            # (ThreadPool does not pickle like ProcessPool, but keep it clean.)
            for (p0, p1) in jobs:
                pool.apply_async(
                    _evaluate_population_slice,
                    (
                        X, F, p0, p1,
                        self.mRNA_mat, self.regulators, self.protein_mat, self.psite_tensor,
                        self.n_reg, self.T_use, self.n_mRNA, self.n_TF, self.n_alpha,
                        self.beta_start_indices, self.num_psites, self.no_psite_tf,
                        self.loss_type, self.lam1, self.lam2
                    )
                )
            pool.close()
            pool.join()
        finally:
            # ensure resources released even if an exception bubbles up
            try:
                pool.terminate()
            except Exception:
                pass

        out["F"] = F

# ----------------------
# DEPRECATED CODE BELOW
# ----------------------
# @njit(cache=True, fastmath=False, parallel=True, nogil=False)
# def objective_(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA,
#                beta_start_indices, num_psites, loss_type, lam1=1e-3, lam2=1e-3):
#     """
#     Computes a loss value for transcription factor optimization using evolutionary algorithms.
#
#     Args:
#         x (np.ndarray): Optimization variables.
#         mRNA_mat (np.ndarray): Matrix of mRNA measurements.
#         regulators (np.ndarray): Matrix of regulators for each mRNA.
#         protein_mat (np.ndarray): Matrix of TF protein levels.
#         psite_tensor (np.ndarray): Tensor of phosphorylation sites.
#         n_reg (int): Number of regulators.
#         T_use (int): Number of time points to use.
#         n_mRNA (int): Number of mRNAs.
#         beta_start_indices (list): List of starting indices for beta parameters.
#         num_psites (list): List of number of phosphorylation sites for each TF.
#         loss_type (int): Type of loss function to use.
#         lam1 (float, optional): L1 penalty coefficient. Defaults to 1e-3.
#         lam2 (float, optional): L2 penalty coefficient. Defaults to 1e-3.
#
#     Returns:
#         float: Computed loss value.
#     """
#     # Initialize loss to zero.
#     total_loss = 0.0
#     # Compute the loss for each mRNA.
#     n_alpha = n_mRNA * n_reg
#     nT = n_mRNA * T_use
#     for i in prange(n_mRNA):
#         # Get the measured mRNA values and initialize the predicted mRNA values.
#         R_meas = mRNA_mat[i, :T_use]
#         R_pred = np.zeros(T_use)
#         # For each regulator, compute the predicted mRNA values.
#         for r in range(n_reg):
#             # Get the index of the TF for this regulator.
#             tf_idx = regulators[i, r]
#             if tf_idx == -1:  # No valid TF for this regulator
#                 continue
#             # Get the TF activity, protein levels, and beta vector.
#             a = x[i * n_reg + r]
#             protein = protein_mat[tf_idx, :T_use]
#             beta_start = beta_start_indices[tf_idx]
#             # Get the length of the beta vector for this TF.
#             length = 1 + num_psites[tf_idx]  # actual length of beta vector for TF
#             beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
#             # Compute the predicted mRNA values.
#             tf_effect = beta_vec[0] * protein
#             for k in range(num_psites[tf_idx]):
#                 # Add the effect of each phosphorylation site.
#                 tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
#             # Compute the predicted mRNA values.
#             R_pred += a * tf_effect
#         # Ensure R_pred is non-negative
#         np.clip(R_pred, 0.0, None, out=R_pred)
#
#         # For each time point, add loss according to loss_type.
#         for t in range(T_use):
#             e = R_meas[t] - R_pred[t]
#             if loss_type == 0:  # MSE
#                 total_loss += e * e
#             elif loss_type == 1:  # MAE
#                 total_loss += abs(e)
#             elif loss_type == 2:  # Soft L1 (pseudo-Huber)
#                 total_loss += 2.0 * (np.sqrt(1.0 + e * e) - 1.0)
#             elif loss_type == 3:  # Cauchy
#                 total_loss += np.log(1.0 + e * e)
#             elif loss_type == 4:  # Arctan
#                 total_loss += np.arctan(e * e)
#             else:
#                 # Default to MSE if unknown.
#                 total_loss += e * e
#
#     loss = total_loss / nT
#
#     # For elastic net (loss_type 5), add L1 and L2 penalties on the beta portion.
#     if loss_type == 5:
#         l1 = 0.0
#         l2 = 0.0
#         # Compute over beta parameters only.
#         for i in range(n_alpha, x.shape[0]):
#             # Get the beta vector for this TF.
#             v = x[i]
#             # Add L1 and L2 penalties.
#             l1 += abs(v)
#             l2 += v * v
#         # Compute the penalties.
#         loss += lam1 * l1 + lam2 * l2
#
#     # For Tikhonov (loss_type 6), add L2 penalty on the beta portion.
#     if loss_type == 6:
#         l2 = 0.0
#         # Compute over beta parameters only.
#         for i in range(n_alpha, x.shape[0]):
#             v = x[i]
#             # Add L2 penalty.
#             l2 += v * v
#         # Compute the penalty.
#         loss += lam1 * l2
#
#     return loss
