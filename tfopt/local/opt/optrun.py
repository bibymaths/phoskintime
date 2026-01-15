from __future__ import annotations
from scipy.optimize import minimize
from tfopt.local.objfn.minfn import objective_wrapper
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
from joblib import Parallel, delayed


def run_optimizer(x0, bounds, lin_cons, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use,
                  n_genes, beta_start_indices, num_psites, loss_type):
    """
    Runs the optimization algorithm to minimize the objective function.

    Args:
      x0                  : Initial guess for the optimization variables.
      bounds              : Bounds for the optimization variables.
      lin_cons            : Linear constraints for the optimization problem.
      expression_matrix   : (n_genes x T_use) measured gene expression values.
      regulators          : (n_genes x n_reg) indices of TF regulators for each gene.
      tf_protein_matrix   : (n_TF x T_use) TF protein time series.
      psite_tensor        : (n_TF x n_psite_max x T_use) matrix of PSite signals (padded with zeros).
      n_reg               : Maximum number of regulators per gene.
      T_use               : Number of time points used.
      n_genes, n_TF     : Number of genes and TF respectively.
      beta_start_indices  : Integer array giving the starting index (in the β–segment) for each TF.
      num_psites          : Integer array with the actual number of PSites for each TF.
      loss_type           : Type of loss function to use.
    Returns:
        result             : Result of the optimization process, including the optimized parameters and objective value.
    """
    m = "SLSQP"  # or trust-constr or SLSQP
    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        args=(expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices,
              num_psites, loss_type),
        method=m,
        bounds=bounds,
        constraints=lin_cons,
        options={"disp": True, "maxiter": 10000} if m == "SLSQP" else {"disp": True, "maxiter": 10000, "verbose": 3}
    )
    return result


# --- Result ranking helpers -------------------------------------------------

def _get_constraint_violation(res: Any) -> float:
    """
    Try to infer a scalar constraint violation from common SciPy result fields.
    If nothing is available, return 0.0 (treat as feasible).
    """
    # Common patterns you might have, depending on solver / your wrapper:
    # res.constr_violation, res.maxcv, res.violation, res.constr, etc.
    for attr in ("constr_violation", "maxcv", "violation"):
        if hasattr(res, attr) and getattr(res, attr) is not None:
            v = getattr(res, attr)
            try:
                return float(v)
            except Exception:
                pass

    # If your run_optimizer returns something like res.constraints or res.constr:
    for attr in ("constraints", "constr"):
        if hasattr(res, attr) and getattr(res, attr) is not None:
            v = getattr(res, attr)
            try:
                v = np.asarray(v, dtype=float)
                return float(np.max(np.abs(v))) if v.size else 0.0
            except Exception:
                pass

    return 0.0


def _result_sort_key(res: Any) -> Tuple[int, float, float, int]:
    """
    Sort order:
      1) feasible first (constraint violation <= 1e-8)
      2) lower objective
      3) lower constraint violation (for infeasible)
      4) success first
    """
    cv = _get_constraint_violation(res)
    feasible = 1 if cv <= 1e-8 else 0
    success = 1 if getattr(res, "success", False) else 0
    fun = float(getattr(res, "fun", np.inf))
    # Sort with negatives where we want "True first"
    return (-feasible, fun, cv, -success)


# --- x0 generation ----------------------------------------------------------

def _clip_to_bounds(x: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(x, lb), ub)


def generate_multistart_x0(
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    n_starts: int,
    seed: int = 0,
    jitter_frac: float = 0.05,
    p_random: float = 0.3,
) -> List[np.ndarray]:
    """
    Build a diverse list of starting points:
      - mostly: jitter around baseline x0
      - some: fully random within bounds

    jitter_frac is relative to (ub - lb).
    p_random is fraction of starts that are random-in-bounds.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    span = np.maximum(ub - lb, 1e-12)

    starts: List[np.ndarray] = []
    for i in range(n_starts):
        if rng.random() < p_random:
            x = lb + rng.random(size=x0.size) * span
        else:
            noise = rng.normal(loc=0.0, scale=jitter_frac, size=x0.size) * span
            x = x0 + noise
        starts.append(_clip_to_bounds(x, bounds))

    # Ensure the original x0 is included (often valuable)
    starts[0] = _clip_to_bounds(x0.copy(), bounds)
    return starts


# --- Parallel multi-start runner -------------------------------------------

@dataclass(frozen=True)
class MultiStartConfig:
    n_starts: int = 32
    n_jobs: int = -1           # all cores
    seed: int = 0
    jitter_frac: float = 0.05
    p_random: float = 0.3
    backend: str = "loky"      # process-based; safer for heavy NumPy
    prefer: str = "processes"  # explicit


def _run_single_start(
    start_id: int,
    x0_i: np.ndarray,
    bounds,
    lin_cons,
    expression_matrix,
    regulators,
    tf_protein_matrix,
    psite_tensor,
    n_reg,
    T_use,
    n_genes,
    beta_start_indices,
    num_psites,
    loss_type,
    run_optimizer_func,
    base_seed: int,
):
    """
    Top-level function for joblib pickling safety.
    """
    # Make per-start seed deterministic (useful if run_optimizer uses randomness)
    np.random.seed(base_seed + start_id)

    res = run_optimizer_func(
        x0_i, bounds, lin_cons,
        expression_matrix, regulators, tf_protein_matrix, psite_tensor,
        n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type
    )

    # Attach metadata without changing solver output structure too much
    try:
        res.start_id = start_id
        res.start_x0 = x0_i
        res.seed = base_seed + start_id
    except Exception:
        pass

    return res


def run_optimizer_multistart(
    x0: np.ndarray,
    bounds,
    lin_cons,
    expression_matrix,
    regulators,
    tf_protein_matrix,
    psite_tensor,
    n_reg,
    T_use,
    n_genes,
    beta_start_indices,
    num_psites,
    loss_type,
    run_optimizer_func,
    cfg: Optional[MultiStartConfig] = None,
    polish: bool = True,
):
    """
    Multi-start wrapper around your existing run_optimizer.
    Keeps the same downstream expectations: returns a single "best" result object.

    polish=True runs one final local optimization starting from the best x.
    """
    if cfg is None:
        cfg = MultiStartConfig()

    x0_list = generate_multistart_x0(
        x0=x0,
        bounds=bounds,
        n_starts=cfg.n_starts,
        seed=cfg.seed,
        jitter_frac=cfg.jitter_frac,
        p_random=cfg.p_random,
    )

    results = Parallel(n_jobs=cfg.n_jobs, backend=cfg.backend, prefer=cfg.prefer)(
        delayed(_run_single_start)(
            i, x0_list[i],
            bounds, lin_cons,
            expression_matrix, regulators, tf_protein_matrix, psite_tensor,
            n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type,
            run_optimizer_func,
            cfg.seed * 10_000,  # base seed namespace
        )
        for i in range(cfg.n_starts)
    )

    # Sophisticated selection: feasible first, then best objective, etc.
    results_sorted = sorted(results, key=_result_sort_key)
    best = results_sorted[0]

    if polish and hasattr(best, "x") and best.x is not None:
        # One more run initialized at the best solution can tighten the optimum
        best2 = run_optimizer_func(
            np.asarray(best.x, dtype=float), bounds, lin_cons,
            expression_matrix, regulators, tf_protein_matrix, psite_tensor,
            n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type
        )
        # Keep whichever is better by the same rule
        best = sorted([best, best2], key=_result_sort_key)[0]

    return best, results_sorted
