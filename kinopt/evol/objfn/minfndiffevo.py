import numpy as np
from pymoo.core.problem import ElementwiseProblem

from kinopt.evol.config import include_regularization, lb, ub, loss_type
from kinopt.evol.optcon import n

from numba import njit


# -----------------------------
# 1) Loss type -> integer id
# -----------------------------
_LOSS_MAP = {
    "base": 0,
    "autocorrelation": 1,
    "huber": 2,
    "mape": 3,
}


# -----------------------------
# 2) Packing helpers (Python)
# -----------------------------
def _pack_problem_for_numba(P_initial, K_index, K_array, P_initial_array):
    """
    Convert P_initial + K_index into numeric arrays for Numba kernels.

    Outputs:
      gp_offsets: int32, shape (n_gp+1,)
      gp_kinase_ids: int32, shape (num_alpha,)
         alpha variables are aligned 1:1 with gp_kinase_ids positions
      k_offsets: int32, shape (n_k+1,)
      k_psite_rows: int32, shape (num_beta,)
         beta variables are aligned 1:1 with k_psite_rows positions
      n_gp, n_k, num_alpha, num_beta, t_max
    """
    # stable order
    kinase_names = list(K_index.keys())
    kinase_to_id = {k: i for i, k in enumerate(kinase_names)}
    n_k = len(kinase_names)

    # Pack kinase->psite rows
    k_offsets = [0]
    k_psite_rows = []
    for k in kinase_names:
        entries = K_index[k]  # [(psite_label, k_row_idx), ...]
        for (_, row_idx) in entries:
            k_psite_rows.append(int(row_idx))
        k_offsets.append(len(k_psite_rows))

    k_offsets = np.asarray(k_offsets, dtype=np.int32)
    k_psite_rows = np.asarray(k_psite_rows, dtype=np.int32)

    # Pack gene-psite -> kinase ids (alpha ordering)
    gp_offsets = [0]
    gp_kinase_ids = []

    # IMPORTANT: we assume P_initial iteration order matches P_initial_array row order.
    # If you ever re-order either, you must pack consistently.
    for (gene, psite), data in P_initial.items():
        kinases = data["Kinases"]
        for kin in kinases:
            if kin in kinase_to_id:
                gp_kinase_ids.append(kinase_to_id[kin])
            else:
                # if your preprocessing guarantees consistency, this should not happen
                # but skipping here is safer than crashing at runtime
                continue
        gp_offsets.append(len(gp_kinase_ids))

    gp_offsets = np.asarray(gp_offsets, dtype=np.int32)
    gp_kinase_ids = np.asarray(gp_kinase_ids, dtype=np.int32)

    n_gp = len(gp_offsets) - 1
    num_alpha = int(gp_offsets[-1])
    num_beta = int(k_offsets[-1])

    # Basic shape sanity
    i_max, t_max = P_initial_array.shape
    if n_gp != i_max:
        raise ValueError(
            f"Packing mismatch: n_gp={n_gp} but P_initial_array has i_max={i_max}. "
            "Ensure P_initial iteration order matches rows in P_initial_array."
        )
    if K_array.shape[1] != t_max:
        raise ValueError(
            f"K_array t_max={K_array.shape[1]} but P_initial_array t_max={t_max}."
        )

    return gp_offsets, gp_kinase_ids, k_offsets, k_psite_rows, n_gp, n_k, num_alpha, num_beta, t_max


# -----------------------------
# 3) Numba kernels
# -----------------------------
@njit(cache=True)
def _corr_sq_lag1(x):
    # Pearson correlation between x[:-1] and x[1:], squared.
    # If variance is zero, return 0.
    n = x.size
    if n < 3:
        return 0.0
    n1 = n - 1

    m0 = 0.0
    m1 = 0.0
    for i in range(n1):
        m0 += x[i]
        m1 += x[i + 1]
    m0 /= n1
    m1 /= n1

    cov = 0.0
    v0 = 0.0
    v1 = 0.0
    for i in range(n1):
        a = x[i] - m0
        b = x[i + 1] - m1
        cov += a * b
        v0 += a * a
        v1 += b * b

    if v0 <= 0.0 or v1 <= 0.0:
        return 0.0
    r = cov / np.sqrt(v0 * v1)
    return r * r


@njit(cache=True)
def _compute_pred_matrix(params, P_obs, gp_offsets, gp_kinase_ids,
                         k_offsets, k_psite_rows, K_array):
    """
    Fast prediction using two-stage accumulation:
      1) kinase_signal[k, t] = sum_p beta(k,p) * K_array[row(p), t]
      2) pred[i, t] = sum_{kin in gp(i)} alpha(i,kin) * kinase_signal[kin, t]
    """
    i_max, t_max = P_obs.shape
    n_k = k_offsets.size - 1
    num_alpha = gp_offsets[-1]

    # kinase_signal
    kinase_signal = np.zeros((n_k, t_max), dtype=np.float64)

    # build kinase_signal from beta segment
    # beta is aligned with k_psite_rows, in the same order as k_offsets blocks
    for k in range(n_k):
        p0 = k_offsets[k]
        p1 = k_offsets[k + 1]
        for ppos in range(p0, p1):
            row = k_psite_rows[ppos]
            b = params[num_alpha + ppos]
            # accumulate vector
            for t in range(t_max):
                kinase_signal[k, t] += b * K_array[row, t]

    # build pred
    pred = np.zeros((i_max, t_max), dtype=np.float64)
    for i in range(i_max):
        a0 = gp_offsets[i]
        a1 = gp_offsets[i + 1]
        for apos in range(a0, a1):
            k = gp_kinase_ids[apos]
            a = params[apos]
            for t in range(t_max):
                pred[i, t] += a * kinase_signal[k, t]

    # clip negatives
    for i in range(i_max):
        for t in range(t_max):
            if pred[i, t] < 0.0:
                pred[i, t] = 0.0

    return pred


@njit(cache=True)
def _loss_from_residuals(residuals, P_obs, params, loss_id, include_reg, n_scalar):
    i_max, t_max = residuals.shape

    if loss_id == 0:  # base (MSE)
        sse = 0.0
        for i in range(i_max):
            for t in range(t_max):
                r = residuals[i, t]
                sse += r * r
        val = sse / n_scalar

        if include_reg:
            l1 = 0.0
            l2 = 0.0
            for j in range(params.size):
                pj = params[j]
                l1 += abs(pj)
                l2 += pj * pj
            val = val + l1 + l2

        return val

    elif loss_id == 1:  # autocorrelation
        acc = 0.0
        for i in range(i_max):
            acc += _corr_sq_lag1(residuals[i, :])

        if include_reg:
            l1 = 0.0
            l2 = 0.0
            for j in range(params.size):
                pj = params[j]
                l1 += abs(pj)
                l2 += pj * pj
            acc = acc + l1 + l2

        return acc

    elif loss_id == 2:  # huber (delta=1.0)
        delta = 1.0
        total = 0.0
        cnt = i_max * t_max
        for i in range(i_max):
            for t in range(t_max):
                r = residuals[i, t]
                ar = abs(r)
                if ar <= delta:
                    total += 0.5 * r * r
                else:
                    total += delta * (ar - 0.5 * delta)
        val = total / cnt

        if include_reg:
            l1 = 0.0
            l2 = 0.0
            for j in range(params.size):
                pj = params[j]
                l1 += abs(pj)
                l2 += pj * pj
            val = val + l1 + l2

        return val

    elif loss_id == 3:  # mape
        total = 0.0
        cnt = i_max * t_max
        for i in range(i_max):
            for t in range(t_max):
                denom = P_obs[i, t] + 1e-12
                total += abs(residuals[i, t] / denom)
        val = (total / cnt) * 100.0

        if include_reg:
            l1 = 0.0
            l2 = 0.0
            for j in range(params.size):
                pj = params[j]
                l1 += abs(pj)
                l2 += pj * pj
            val = val + l1 + l2

        return val

    else:
        # fallback: behave like base
        sse = 0.0
        for i in range(i_max):
            for t in range(t_max):
                r = residuals[i, t]
                sse += r * r
        return sse / n_scalar


@njit(cache=True)
def _evaluate_loss_and_constraints(params, P_obs, gp_offsets, gp_kinase_ids,
                                  k_offsets, k_psite_rows, K_array,
                                  eps_eq, loss_id, include_reg, n_scalar):
    pred = _compute_pred_matrix(params, P_obs, gp_offsets, gp_kinase_ids,
                                k_offsets, k_psite_rows, K_array)

    residuals = P_obs - pred
    loss = _loss_from_residuals(residuals, P_obs, params, loss_id, include_reg, n_scalar)

    # constraints g(x) <= 0, each equality becomes 2 inequalities
    n_gp = gp_offsets.size - 1
    n_k = k_offsets.size - 1
    g = np.empty(2 * (n_gp + n_k), dtype=np.float64)
    gi = 0

    # alpha groups (one per gene-psite)
    for i in range(n_gp):
        a0 = gp_offsets[i]
        a1 = gp_offsets[i + 1]
        s = 0.0
        for j in range(a0, a1):
            s += params[j]
        g[gi] = (s - 1.0) - eps_eq
        gi += 1
        g[gi] = (1.0 - s) - eps_eq
        gi += 1

    # beta groups (one per kinase)
    num_alpha = gp_offsets[-1]
    for k in range(n_k):
        b0 = k_offsets[k]
        b1 = k_offsets[k + 1]
        s = 0.0
        for j in range(b0, b1):
            s += params[num_alpha + j]
        g[gi] = (s - 1.0) - eps_eq
        gi += 1
        g[gi] = (1.0 - s) - eps_eq
        gi += 1

    return loss, g


@njit(cache=True)
def _estimated_series_jit(params, P_obs, gp_offsets, gp_kinase_ids,
                          k_offsets, k_psite_rows, K_array):
    return _compute_pred_matrix(params, P_obs, gp_offsets, gp_kinase_ids,
                                k_offsets, k_psite_rows, K_array)


@njit(cache=True)
def _residuals_jit(P_obs, P_est):
    return P_obs - P_est


# -----------------------------
# 4) The refactored class
# -----------------------------
class PhosphorylationOptimizationProblem(ElementwiseProblem):
    """
    Single-objective constrained optimization (Numba-accelerated).

    Objective:
      - minimize loss

    Constraints g(x) <= 0:
      - for each alpha group: |sum(alpha_group) - 1| <= eps
      - for each kinase beta group: |sum(beta_group) - 1| <= eps
    """

    def __init__(self, P_initial, P_initial_array, K_index, K_array,
                 gene_psite_counts, beta_counts, eps_eq=1e-10, **kwargs):

        self.P_initial = P_initial
        self.P_initial_array = np.asarray(P_initial_array, dtype=np.float64)
        self.K_index = K_index
        self.K_array = np.asarray(K_array, dtype=np.float64)

        # Pack for numba
        (self.gp_offsets,
         self.gp_kinase_ids,
         self.k_offsets,
         self.k_psite_rows,
         n_gp, n_k, num_alpha, num_beta, _tmax) = _pack_problem_for_numba(
            P_initial=self.P_initial,
            K_index=self.K_index,
            K_array=self.K_array,
            P_initial_array=self.P_initial_array
        )

        self.num_alpha = int(num_alpha)
        self.num_beta = int(num_beta)
        self.eps_eq = float(eps_eq)

        # consistency checks against provided counts (optional but useful)
        if int(sum(gene_psite_counts)) != self.num_alpha:
            raise ValueError(
                f"gene_psite_counts sum={int(sum(gene_psite_counts))} does not match packed num_alpha={self.num_alpha}. "
                "Your preprocessing must ensure only kinases present in K_index are counted per gene-psite."
            )
        if int(sum(beta_counts.values())) != self.num_beta:
            raise ValueError(
                f"beta_counts sum={int(sum(beta_counts.values()))} does not match packed num_beta={self.num_beta}. "
                "beta_counts must represent per-kinase psite counts in the same order used to build K_index."
            )

        # each equality -> 2 inequalities
        n_alpha_groups = n_gp
        n_beta_groups = n_k
        n_ieq = 2 * (n_alpha_groups + n_beta_groups)

        xl = np.concatenate([np.zeros(self.num_alpha), np.full(self.num_beta, lb)], axis=0)
        xu = np.concatenate([np.ones(self.num_alpha),  np.full(self.num_beta, ub)], axis=0)

        super().__init__(
            n_var=self.num_alpha + self.num_beta,
            n_obj=1,
            n_ieq_constr=n_ieq,
            xl=xl,
            xu=xu,
            **kwargs
        )

        # freeze config into numba-friendly scalars
        self.loss_id = int(_LOSS_MAP.get(loss_type, 0))
        self.include_reg = bool(include_regularization)
        self.n_scalar = float(n)

    def _evaluate(self, x, out, *args, **kwargs):
        loss, g = _evaluate_loss_and_constraints(
            np.asarray(x, dtype=np.float64),
            self.P_initial_array,
            self.gp_offsets, self.gp_kinase_ids,
            self.k_offsets, self.k_psite_rows,
            self.K_array,
            self.eps_eq,
            self.loss_id,
            self.include_reg,
            self.n_scalar
        )
        out["F"] = [float(loss)]
        out["G"] = g

    # Optional: keep these as methods for compatibility
    def estimated_series(self, params):
        return _estimated_series_jit(
            np.asarray(params, dtype=np.float64),
            self.P_initial_array,
            self.gp_offsets, self.gp_kinase_ids,
            self.k_offsets, self.k_psite_rows,
            self.K_array
        )

    def residuals(self, params):
        est = self.estimated_series(params)
        return _residuals_jit(self.P_initial_array, est)


# -----------------------------
# 5) Backward-compatible helpers
# -----------------------------
def _estimated_series(params, P_initial, K_index, K_array, gene_psite_counts, beta_counts, P_initial_array):
    # Use the same packing + jit kernel, without changing your external call sites too much.
    P_obs = np.asarray(P_initial_array, dtype=np.float64)
    K_arr = np.asarray(K_array, dtype=np.float64)

    gp_offsets, gp_kinase_ids, k_offsets, k_psite_rows, *_ = _pack_problem_for_numba(
        P_initial=P_initial, K_index=K_index, K_array=K_arr, P_initial_array=P_obs
    )
    return _estimated_series_jit(
        np.asarray(params, dtype=np.float64),
        P_obs,
        gp_offsets, gp_kinase_ids,
        k_offsets, k_psite_rows,
        K_arr
    )


def _residuals(P_initial_array, P_estimated):
    return np.asarray(P_initial_array, dtype=np.float64) - np.asarray(P_estimated, dtype=np.float64)
