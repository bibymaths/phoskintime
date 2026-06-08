from __future__ import annotations
from tfopt.local.objfn.minfn import objective_wrapper
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
from joblib import Parallel, delayed
from tfopt.local.config.logconf import setup_logger

logger = setup_logger()


def _bounds_arrays(bounds, n):
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    return np.asarray([b[0] for b in bounds], dtype=float), np.asarray([b[1] for b in bounds], dtype=float)


def _project_to_feasible(x, bounds, lin_cons):
    x = np.asarray(x, dtype=float).copy()
    lb, ub = _bounds_arrays(bounds, x.size)
    x = np.clip(x, lb, ub)
    for cons in lin_cons or []:
        if not hasattr(cons, "A"):
            continue
        A = np.asarray(cons.A, dtype=float)
        lows = np.asarray(cons.lb, dtype=float).reshape(-1)
        ups = np.asarray(cons.ub, dtype=float).reshape(-1)
        if lows.size == 1:
            lows = np.full(A.shape[0], float(lows[0]))
        if ups.size == 1:
            ups = np.full(A.shape[0], float(ups[0]))
        for row, lo, hi in zip(A, lows, ups):
            nz = np.flatnonzero(np.abs(row) > 1e-12)
            if nz.size and np.isclose(lo, hi):
                current = float(row @ x)
                x[nz] += (float(lo) - current) / float(row[nz].sum())
    return np.clip(x, lb, ub)


def _finite_difference_grad(obj, x):
    grad = np.zeros_like(x, dtype=float)
    fx = float(obj(x))
    eps = 1e-6 * np.maximum(1.0, np.abs(x))
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps[i]
        xm = x.copy(); xm[i] -= eps[i]
        grad[i] = (float(obj(xp)) - float(obj(xm))) / (2.0 * eps[i])
    return fx, grad


def _linear_constraint_violation(x, lin_cons):
    maxcv = 0.0
    for cons in lin_cons or []:
        if not hasattr(cons, "A"):
            continue
        vals = np.asarray(cons.A, dtype=float) @ x
        lows = np.asarray(cons.lb, dtype=float).reshape(-1)
        ups = np.asarray(cons.ub, dtype=float).reshape(-1)
        if lows.size == 1:
            lows = np.full(vals.shape, float(lows[0]))
        if ups.size == 1:
            ups = np.full(vals.shape, float(ups[0]))
        maxcv = max(maxcv, float(np.max(np.maximum(lows - vals, vals - ups))))
    return max(0.0, maxcv)

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
    def obj(v):
        return float(objective_wrapper(
            v,
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
        ))

    x = _project_to_feasible(np.asarray(x0, dtype=float), bounds, lin_cons)
    best_x = x.copy()
    best_fun = obj(best_x)
    step = 0.2
    nfev = 1
    for nit in range(1, 81):
        _, grad = _finite_difference_grad(obj, x)
        nfev += 2 * x.size
        cand = _project_to_feasible(x - step * grad, bounds, lin_cons)
        cand_fun = obj(cand)
        nfev += 1
        if np.isfinite(cand_fun) and cand_fun <= best_fun:
            x = cand
            best_x = cand.copy()
            best_fun = cand_fun
            step = min(step * 1.2, 1.0)
        else:
            step *= 0.5
        if step < 1e-8 or float(np.linalg.norm(grad)) < 1e-8:
            break
    cv = _linear_constraint_violation(best_x, lin_cons)
    return SimpleNamespace(
        x=best_x,
        fun=best_fun,
        success=bool(np.isfinite(best_fun) and cv <= 1e-6),
        message="projected finite-difference optimizer completed",
        nit=nit,
        nfev=nfev,
        constr_violation=cv,
        maxcv=cv,
    )


def _get_constraint_violation(res: Any) -> float:
    """
    Extracts the constraint violation metric from the result object, if available.

    This function attempts to retrieve a constraint violation value from the provided
    result object `res`. It checks various common attributes that are typically used 
    to store such information in optimization solvers or frameworks. If no valid 
    constraint violation value can be found, the function returns a default of 0.0.

    Args:
        res: The result object from an optimization solver. It may contain various
            attributes representing the constraint violation level.

    Returns:
        float: The constraint violation value if successfully extracted; otherwise, 
            returns 0.0.
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
    Generates a sorting key for optimization results based on feasibility and quality.

    This function creates a tuple that can be used to sort optimization results
    with the following priority order:
      1) Feasible solutions first (constraint violation <= 1e-8)
      2) Lower objective function value
      3) Lower constraint violation (for infeasible solutions)
      4) Successful optimization runs first

    Args:
        res: The result object from an optimization solver containing attributes
            such as 'fun' (objective value), 'success' (optimization status), and
            constraint violation information.

    Returns:
        Tuple[int, float, float, int]: A tuple of four values used for sorting:
            - Negative of feasibility flag (feasible=1, infeasible=0)
            - Objective function value
            - Constraint violation value
            - Negative of success flag (success=1, failure=0)
    """
    cv = _get_constraint_violation(res)
    feasible = 1 if cv <= 1e-8 else 0
    success = 1 if getattr(res, "success", False) else 0
    fun = float(getattr(res, "fun", np.inf))
    # Sort with negatives where we want "True first"
    return (-feasible, fun, cv, -success)

def _clip_to_bounds(x: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    Clips the values of an array to be within specified bounds.

    This function ensures that all elements of the input array `x` are constrained
    to lie within the lower and upper bounds specified in the `bounds` sequence.

    Args:
        x: A numpy array containing values to be clipped.
        bounds: A sequence of tuples where each tuple contains (lower_bound, upper_bound)
            for the corresponding element in `x`.

    Returns:
        np.ndarray: A numpy array with the same shape as `x`, where all values are
            clipped to be within their respective bounds.
    """
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
    Generates multiple starting points for multi-start optimization.

    Builds a diverse list of starting points:
      - mostly: jitter around baseline x0
      - some: fully random within bounds

    jitter_frac is relative to (ub - lb).
    p_random is fraction of starts that are random-in-bounds.

    Returns:
        List[np.ndarray]: A list of starting points for optimization.
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
    Executes a single optimization start for a specific set of parameters and attaches metadata
    to the result. This function ensures deterministic behavior by seeding the random number
    generator and runs an optimizer function to minimize the loss.

    Args:
        start_id (int): The unique identifier for the current optimization start.
        x0_i (np.ndarray): Initial guess for the optimization variables.
        bounds: Bounds for the optimization variables.
        lin_cons: Linear constraints for the optimization problem.
        expression_matrix: Gene expression data used as input for the optimizer.
        regulators: Regulatory matrix or data structure for modeling regulations.
        tf_protein_matrix: Matrix representing transcription factor proteins.
        psite_tensor: Tensor containing phosphorylation site-related data.
        n_reg: The number of regulatory parameters in the model.
        T_use: Temporal parameter or index used in the optimization process.
        n_genes: Total number of genes considered in the analysis.
        beta_start_indices: Starting indices for beta parameters in the optimization.
        num_psites: Number of phosphorylation sites considered in the model.
        loss_type: Loss function type used by the optimizer.
        run_optimizer_func: The optimization function to execute with the given inputs.
        base_seed (int): Base seed for ensuring deterministic behavior across runs.

    Returns:
        Optimization result from `run_optimizer_func`, enhanced with metadata such as
        `start_id`, `start_x0`, and the deterministic seed used during the run.
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

        logger.info(f"Start {start_id}: fun={getattr(res, 'fun', np.nan):.4f}, success={getattr(res, 'success', False)}, "
                    f"cv={_get_constraint_violation(res):.2e}")

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
    Executes a multistart optimization loop with parallelization and optional polishing
    to find the best solution across multiple starting points. The function leverages
    a parallel approach for running multiple optimizations, selects the best result
    based on predefined sorting criteria, and optionally refines it.

    Args:
        x0 (np.ndarray): Initial guess for the optimization variables.
        bounds: Bounds for the optimization variables, typically a sequence of (min, max) pairs.
        lin_cons: Linear constraints for the optimizer, defined as per specific optimizer requirements.
        expression_matrix: Input gene expression data utilized in the optimization process.
        regulators: Regulatory inputs or factors influencing the optimization process.
        tf_protein_matrix: Matrix representing transcription factor proteins relevant to the process.
        psite_tensor: Tensor containing phosphorylation site data used in the computations.
        n_reg: Number of regulators involved in the optimization.
        T_use: Specific configuration parameter determining time or iteration usage.
        n_genes: Number of genes considered within the problem scope.
        beta_start_indices: Indices indicating the start positions of beta parameters in the optimization.
        num_psites: Total number of phosphorylation sites accounted for in optimization.
        loss_type: Type of loss function used for evaluating optimization performance.
        run_optimizer_func: Optimization function to be executed for each starting point.
        cfg (Optional[MultiStartConfig]): Configuration object specifying multistart
            parameters such as number of starts, parallelization settings, and randomness.
        polish (bool): Indicates whether to perform a final optimization
            run initialized at the best solution. Defaults to True.

    Returns:
        Tuple: A tuple containing:
            - The best result as determined by sorting criteria.
            - A list of sorted optimization results from all starting points.
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

    # Sophisticated selection: first sort by feasibility, then by objective value, then by constraint violation
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
