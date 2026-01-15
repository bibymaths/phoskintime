from scipy.optimize import minimize
import time
import numpy as np
from dataclasses import dataclass
from joblib import Parallel, delayed
from kinopt.local.config.logconf import setup_logger

logger = setup_logger()

def run_optimization(obj_fun, params_initial, opt_method, bounds, constraints):
    """
    Run optimization using the specified method.

    Args:
        obj_fun: Objective function to minimize.
        params_initial: Initial parameters for the optimization.
        opt_method: Optimization method to use (e.g., 'SLSQP', 'trust-constr').
        bounds: Bounds for the parameters.
        constraints: Constraints for the optimization.

    Returns:
        result: Result of the optimization.
        optimized_params: Optimized parameters.
    """
    result = minimize(obj_fun, params_initial, method=opt_method,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 20000, 'verbose': 3} if opt_method == "trust-constr" else {'maxiter': 20000})
    return result, result.x


@dataclass
class StartOutcome:
    start_id: int
    seed: int
    result: object
    optimized_params: np.ndarray
    fun: float
    success: bool
    constr_violation: float
    runtime_s: float


def _get_attr(x, name, default=None):
    # supports both dict-like and attribute-like results
    if hasattr(x, name):
        return getattr(x, name)
    if isinstance(x, dict) and name in x:
        return x[name]
    return default


def _extract_fun(result):
    fun = _get_attr(result, "fun", None)
    if fun is None:
        # some optimizers use "fval" or similar
        fun = _get_attr(result, "fval", np.inf)
    return float(fun)


def _extract_success(result):
    s = _get_attr(result, "success", None)
    if s is None:
        # be conservative: treat missing as False
        return False
    return bool(s)


def _extract_constr_violation(result):
    # Try common names; if not available, assume 0 (or treat as unknown).
    # Scipy trust-constr uses "constr_violation" in some versions.
    cv = _get_attr(result, "constr_violation", None)
    if cv is None:
        cv = _get_attr(result, "maxcv", None)
    if cv is None:
        # if optimizer doesn’t expose it, we cannot know; default to 0
        return 0.0
    return float(cv)


def _sample_initial(params_initial, bounds, rng, strategy="jitter", jitter_scale=0.15):
    """
    Produce a new start point within bounds with minimal assumptions.

    strategy:
      - "jitter": multiplicative jitter around current init, then clip to bounds
      - "uniform": full random in bounds
      - "hybrid": mostly jitter, sometimes uniform
    """
    p0 = np.asarray(params_initial, dtype=float).copy()

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    if strategy == "uniform":
        p = rng.uniform(lb, ub)
        return p

    if strategy == "hybrid":
        if rng.random() < 0.25:
            return rng.uniform(lb, ub)
        # else fall through to jitter

    # default: jitter
    # If p0 can be 0, additive jitter is safer than multiplicative-only
    span = (ub - lb)
    noise = rng.normal(loc=0.0, scale=jitter_scale, size=p0.shape)
    p = p0 + noise * span
    p = np.clip(p, lb, ub)
    return p


def _run_one_start(start_id, seed, obj_fun, params_initial, opt_method, bounds, constraints,
                   init_strategy="hybrid", jitter_scale=0.15):
    rng = np.random.default_rng(seed)
    p0 = _sample_initial(params_initial, bounds, rng, strategy=init_strategy, jitter_scale=jitter_scale)

    t0 = time.time()
    result, optimized_params = run_optimization(obj_fun, p0, opt_method, bounds, constraints)
    runtime_s = time.time() - t0

    fun = _extract_fun(result)
    success = _extract_success(result)
    cv = _extract_constr_violation(result)

    logger.info(f"[Start {start_id}] fun={fun:.6g} cv={cv:.3g} success={success} runtime={runtime_s:.2f}s")

    return StartOutcome(
        start_id=start_id,
        seed=seed,
        result=result,
        optimized_params=np.asarray(optimized_params, dtype=float),
        fun=fun,
        success=success,
        constr_violation=cv,
        runtime_s=runtime_s
    )


def multistart_run_optimization(obj_fun, params_initial, opt_method, bounds, constraints,
                                n_starts=24, n_jobs=-1, base_seed=1234,
                                init_strategy="hybrid", jitter_scale=0.15, prefer_feasible=True,
                                logger=None):
    """
    Runs run_optimization multiple times in parallel and returns (best_result, best_params, outcomes).

    Selection logic (sophisticated but simple):
      1) If prefer_feasible: prefer (cv <= 0) or smallest constraint violation.
      2) Then lowest objective.
      3) Then success=True as tie-breaker.
      4) Then shortest runtime as final tie-breaker.
    """
    seeds = [base_seed + i for i in range(n_starts)]

    if logger:
        logger.info(f"[Multistart] n_starts={n_starts}, n_jobs={n_jobs}, strategy={init_strategy}")

    outcomes = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_start)(
            i, seeds[i], obj_fun, params_initial, opt_method, bounds, constraints,
            init_strategy, jitter_scale
        )
        for i in range(n_starts)
    )

    # Rank outcomes
    def key(o: StartOutcome):
        # feasibility-first: smaller constraint violation preferred
        # if prefer_feasible, hard-prioritize feasible (cv <= 0 or very small)
        feasible = (o.constr_violation <= 1e-12)
        if prefer_feasible:
            return (
                0 if feasible else 1,           # feasible first
                o.constr_violation,             # then smallest violation
                o.fun,                          # then objective
                0 if o.success else 1,          # then success
                o.runtime_s                     # then runtime
            )
        return (o.fun, 0 if o.success else 1, o.constr_violation, o.runtime_s)

    best = min(outcomes, key=key)

    if logger:
        # brief summary
        best_feas = best.constr_violation <= 1e-12
        logger.info(
            f"[Multistart] Best start={best.start_id} seed={best.seed} "
            f"fun={best.fun:.6g} cv={best.constr_violation:.3g} "
            f"success={best.success} feasible={best_feas} runtime={best.runtime_s:.2f}s"
        )

    return best.result, best.optimized_params, outcomes
