import numpy as np
from numba import njit
from pymoo.core.problem import ElementwiseProblem
from kinopt.evol.config import include_regularization, lb, ub, loss_type
from kinopt.evol.optcon import n, P_initial_array

@njit(cache=True, fastmath=False)
def _predict_matrix_numba(params, i_idx, a_idx, b_idx, k_row_idx, K_array, i_max, t_max):
    """
    Predicts the phosphorylation matrix using the edge list representation.

    Args:
        params (np.ndarray): Parameter vector containing alpha and beta values.
        i_idx (np.ndarray): Array of gene-psite indices.
        a_idx (np.ndarray): Array of alpha parameter indices.
        b_idx (np.ndarray): Array of beta parameter indices.
        k_row_idx (np.ndarray): Array of kinase row indices in K_array.
        K_array (np.ndarray): Array of kinase-psite time-series data.
        i_max (int): Number of gene-psite combinations.
        t_max (int): Number of time points.

    Returns:
        np.ndarray: Predicted phosphorylation matrix (i_max x t_max) with negative values clipped to zero.
    """

    P = np.zeros((i_max, t_max), dtype=np.float64)
    n_edges = i_idx.shape[0]

    for e in range(n_edges):
        i = i_idx[e]
        a = params[a_idx[e]]
        b = params[b_idx[e]]

        # k_time_series is K_array[k_row_idx[e], :]
        for t in range(t_max):
            P[i, t] += a * b * K_array[k_row_idx[e], t]

    # clip negatives
    for i in range(i_max):
        for t in range(t_max):
            if P[i, t] < 0.0:
                P[i, t] = 0.0

    return P


@njit(cache=True)
def _alpha_violation_numba(params, alpha_starts, alpha_counts, eps=0.0):
    """
    Calculates the sum-to-one constraint violation for alpha parameters.

    Args:
        params (np.ndarray): Parameter vector containing alpha and beta values.
        alpha_starts (np.ndarray): Starting indices for each alpha block.
        alpha_counts (np.ndarray): Number of alpha parameters in each block.
        eps (float, optional): Tolerance for constraint violation. Defaults to 0.0.

    Returns:
        float: Total alpha constraint violation.
    """
    v = 0.0
    for i in range(alpha_counts.shape[0]):
        s = 0.0
        start = alpha_starts[i]
        cnt = alpha_counts[i]
        for j in range(cnt):
            s += params[start + j]
        d = abs(s - 1.0)
        if d > eps:
            v += d
    return v


@njit(cache=True)
def _beta_violation_numba(params, beta_starts, beta_counts, eps=0.0):
    """
    Calculates the sum-to-one constraint violation for beta parameters.

    Args:
        params (np.ndarray): Parameter vector containing alpha and beta values.
        beta_starts (np.ndarray): Starting indices for each beta block.
        beta_counts (np.ndarray): Number of beta parameters in each block.
        eps (float, optional): Tolerance for constraint violation. Defaults to 0.0.

    Returns:
        float: Total beta constraint violation.
    """
    v = 0.0
    for k in range(beta_counts.shape[0]):
        s = 0.0
        start = beta_starts[k]
        cnt = beta_counts[k]
        for j in range(cnt):
            s += params[start + j]
        d = abs(s - 1.0)
        if d > eps:
            v += d
    return v


@njit(cache=True)
def _mse_sse_numba(P_obs, P_pred, n):
    """
    Calculates the mean squared error (MSE) between observed and predicted matrices.

    Args:
        P_obs (np.ndarray): Observed phosphorylation matrix.
        P_pred (np.ndarray): Predicted phosphorylation matrix.
        n (float): Normalization factor (typically the number of observations).

    Returns:
        float: Mean squared error.
    """
    err = 0.0
    i_max, t_max = P_obs.shape
    for i in range(i_max):
        for t in range(t_max):
            r = P_obs[i, t] - P_pred[i, t]
            err += r * r
    return err / n


class PhosphorylationOptimizationProblem(ElementwiseProblem):
    """
    Multi-objective optimization:
      F[0] = main loss (error)
      F[1] = alpha sum-to-1 violations (aggregated)
      F[2] = beta  sum-to-1 violations (aggregated)

    Args:
        P_initial (dict): Dictionary with keys as (gene, psite) and values containing 'Kinases' and 'TimeSeries'.
        P_initial_array (np.ndarray): Array of observed gene-psite data.
        K_index (dict): Dictionary mapping each kinase to a list of (psite, time_series) tuples.
        K_array (np.ndarray): Array of kinase-psite time-series data.
        gene_psite_counts (list): List of integers indicating the number of kinases associated with each gene-psite.
        beta_counts (dict): Dictionary indicating how many beta values correspond to each kinase-psite combination.
    """

    def __init__(self, P_initial, P_initial_array, K_index, K_array, gene_psite_counts, beta_counts, **kwargs):
        self.P_initial = P_initial
        self.P_initial_array = P_initial_array
        self.K_index = K_index
        self.K_array = K_array
        self.gene_psite_counts = gene_psite_counts
        self.beta_counts = beta_counts

        self.num_alpha = int(np.sum(np.asarray(gene_psite_counts, dtype=np.int64)))
        self._kinase_keys_in_kindex = list(K_index.keys())

        beta_keys = list(beta_counts.keys())
        if len(beta_keys) > 0 and all((k in K_index) for k in beta_keys):
            kinase_ids = beta_keys  # uses beta_counts insertion order
        else:
            kinase_ids = self._kinase_keys_in_kindex  # K_index insertion order

        # Build beta_counts_arr aligned to kinase_ids
        beta_counts_arr = np.zeros(len(kinase_ids), dtype=np.int64)
        for pos, kinase in enumerate(kinase_ids):
            if kinase in beta_counts:
                beta_counts_arr[pos] = int(beta_counts[kinase])
            else:
                # fallback: beta_counts might be keyed by integer pos
                beta_counts_arr[pos] = int(beta_counts.get(pos, len(K_index.get(kinase, []))))

        self._kinase_ids = kinase_ids
        self._beta_counts_arr = beta_counts_arr
        self.num_beta = int(np.sum(beta_counts_arr))

        # Define pymoo problem
        super().__init__(
            n_var=self.num_alpha + self.num_beta,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.concatenate([np.zeros(self.num_alpha), np.full(self.num_beta, lb)]),
            xu=np.concatenate([np.ones(self.num_alpha), np.full(self.num_beta, ub)]),
            **kwargs
        )

        # ----------------------------
        # Precompute alpha block starts
        # ----------------------------
        alpha_counts = np.asarray(self.gene_psite_counts, dtype=np.int64)
        alpha_starts = np.zeros(alpha_counts.shape[0], dtype=np.int64)
        s = 0
        for i in range(alpha_counts.shape[0]):
            alpha_starts[i] = s
            s += int(alpha_counts[i])

        self._alpha_counts = alpha_counts
        self._alpha_starts = alpha_starts

        # ----------------------------
        # Precompute beta block starts (beta params start at num_alpha)
        # ----------------------------
        beta_starts = np.zeros(self._beta_counts_arr.shape[0], dtype=np.int64)
        s = self.num_alpha
        for k in range(self._beta_counts_arr.shape[0]):
            beta_starts[k] = s
            s += int(self._beta_counts_arr[k])

        self._beta_starts = beta_starts

        # ----------------------------
        # Build edge list arrays for fast prediction
        # ----------------------------
        # Map kinase -> beta block position
        kinase_pos = {kid: pos for pos, kid in enumerate(self._kinase_ids)}

        i_list = []
        a_list = []
        b_list = []
        krow_list = []

        alpha_cursor = 0
        for i, ((gene, psite), data) in enumerate(self.P_initial.items()):
            kinases = data["Kinases"]
            for j, kinase in enumerate(kinases):
                # alpha parameter index in params vector
                a_param = alpha_cursor + j

                kpos = kinase_pos.get(kinase, None)
                if kpos is None:
                    continue

                kinase_psites = self.K_index.get(kinase, None)
                if kinase_psites is None:
                    continue

                for local_p, (k_psite, k_row_idx) in enumerate(kinase_psites):
                    if not isinstance(k_row_idx, (int, np.integer)):
                        raise TypeError(
                            "K_index must store integer row indices into K_array for Numba.\n"
                            "Expected: K_index[kinase] = [(psite, k_row_idx:int), ...]\n"
                            "Got a non-integer k_row_idx. Convert K_index upstream."
                        )

                    b_param = int(self._beta_starts[kpos]) + local_p

                    i_list.append(i)
                    a_list.append(a_param)
                    b_list.append(b_param)
                    krow_list.append(int(k_row_idx))

            alpha_cursor += len(kinases)

        self._i_idx = np.asarray(i_list, dtype=np.int64)
        self._a_idx = np.asarray(a_list, dtype=np.int64)
        self._b_idx = np.asarray(b_list, dtype=np.int64)
        self._k_row_idx = np.asarray(krow_list, dtype=np.int64)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the multi-objective function for the given parameter vector.

        Args:
            x (np.ndarray): Parameter vector to evaluate.
            out (dict): Output dictionary to store objective values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Main objective (loss)
        err = self.objective_function(x)

        # Constraint violations (Numba)
        alpha_v = _alpha_violation_numba(x, self._alpha_starts, self._alpha_counts)
        beta_v = _beta_violation_numba(x, self._beta_starts, self._beta_counts_arr)

        out["F"] = [float(err), float(alpha_v), float(beta_v)]

    def objective_function(self, params):
        """
        Computes the main objective function (loss) for the given parameters.

        Args:
            params (np.ndarray): Parameter vector containing alpha and beta values.

        Returns:
            float: Computed loss value based on the selected loss type (base, autocorrelation, huber, or mape).
        """
        i_max, t_max = self.P_initial_array.shape

        # Fast prediction (Numba)
        P_pred = _predict_matrix_numba(
            params,
            self._i_idx, self._a_idx, self._b_idx, self._k_row_idx,
            self.K_array,
            int(i_max), int(t_max)
        )

        # Loss selection:
        if loss_type == "base":
            base = _mse_sse_numba(self.P_initial_array, P_pred, float(n))
            if include_regularization:
                base = base + float(np.sum(np.abs(params)) + np.sum(params * params))
            return float(base)

        residuals = self.P_initial_array - P_pred

        if loss_type == "autocorrelation":
            return float(np.sum([
                np.corrcoef(residuals[i, :-1], residuals[i, 1:])[0, 1] ** 2
                for i in range(i_max)
            ]) + (np.sum(np.abs(params)) + np.sum(params * params) if include_regularization else 0.0))

        if loss_type == "huber":
            hub = float(np.mean(np.where(
                np.abs(residuals) <= 1.0,
                0.5 * residuals ** 2,
                1.0 * (np.abs(residuals) - 0.5)
            )))
            if include_regularization:
                hub += float(np.sum(np.abs(params)) + np.sum(params * params))
            return hub

        if loss_type == "mape":
            m = float(np.mean(np.abs(residuals / (self.P_initial_array + 1e-12))) * 100.0)
            if include_regularization:
                m += float(np.sum(np.abs(params)) + np.sum(params * params))
            return m

        # Fallback
        return float(_mse_sse_numba(self.P_initial_array, P_pred, float(n)))

def _estimated_series(params, P_initial, K_index, K_array, gene_psite_counts, beta_counts):
    """
    Calculates the estimated time series for each gene-psite based on the optimized parameters.

    Args:
        params (np.ndarray): Optimized parameter vector containing alphas and betas.
        P_initial (dict): Dictionary with keys as (gene, psite) and values containing 'Kinases' and 'TimeSeries'.
        K_index (dict): Dictionary mapping each kinase to a list of (psite, time_series) tuples.
        K_array (np.ndarray): Array of kinase-psite time-series data.
        gene_psite_counts (list): List of integers indicating the number of kinases associated with each gene-psite.
        beta_counts (dict): Dictionary indicating how many beta values correspond to each kinase-psite combination.

    Returns:
        np.ndarray: Estimated time series matrix (i_max x t_max) for all gene-psite combinations.
    """
    # num_alpha is determined solely by gene_psite_counts
    num_alpha = int(np.sum(np.asarray(gene_psite_counts, dtype=np.int64)))

    # Determine output shape
    if P_initial_array is not None:
        i_max, t_max = P_initial_array.shape
    else:
        # fallback - no pass P_initial_array
        i_max = len(P_initial)
        t_max = int(K_array.shape[1])

    P_est = np.zeros((i_max, t_max), dtype=np.float64)

    # Walk alpha blocks (per gene-psite), and walk betas globally in K_index key order
    alpha_cursor = 0

    # Build a deterministic beta layout: concatenate all kinase-psite betas in K_index insertion order
    # beta_cursor counts how many beta scalars have been consumed so far.
    # Each kinase consumes len(K_index[kinase]) betas.
    beta_block_starts = {}
    beta_cursor = 0
    for kinase in K_index.keys():
        beta_block_starts[kinase] = beta_cursor
        beta_cursor += len(K_index[kinase])

    # Sanity (optional): ensure params has enough betas
    if num_alpha + beta_cursor > len(params):
        raise ValueError(
            f"params too short: need num_alpha({num_alpha}) + num_beta({beta_cursor}) = {num_alpha + beta_cursor}, "
            f"got {len(params)}."
        )

    # Build estimates
    for i, ((gene, psite), data) in enumerate(P_initial.items()):
        kinases = data["Kinases"]
        pred = np.zeros(t_max, dtype=np.float64)

        for j, kinase in enumerate(kinases):
            if kinase not in K_index:
                continue

            a = float(params[alpha_cursor + j])
            kb = beta_block_starts[kinase]
            kinase_psites = K_index[kinase]  # [(psite_label, k_row_idx), ...]

            for local_p, (k_psite, k_row_idx) in enumerate(kinase_psites):
                b = float(params[num_alpha + kb + local_p])
                pred += a * b * K_array[int(k_row_idx), :]

        # clip negative values
        pred[pred < 0.0] = 0.0
        P_est[i, :] = pred

        alpha_cursor += len(kinases)

    return P_est

def _residuals(P_initial_array, P_estimated):
    """
    Calculates the residuals (difference between observed and estimated values).

    Args:
        P_initial_array (np.ndarray): Observed gene-psite data.
        P_estimated (np.ndarray): Estimated gene-psite data from the model.

    Returns:
        np.ndarray: Residuals matrix (same shape as P_initial_array).
    """
    return np.asarray(P_initial_array, dtype=np.float64) - np.asarray(P_estimated, dtype=np.float64)
