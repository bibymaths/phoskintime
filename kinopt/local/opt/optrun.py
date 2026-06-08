import time
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from joblib import Parallel, delayed
from kinopt.local.config.logconf import setup_logger

logger = setup_logger()


def _bounds_arrays(bounds, n):
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    lb = np.asarray([b[0] for b in bounds], dtype=float)
    ub = np.asarray([b[1] for b in bounds], dtype=float)
    return lb, ub


def _project_simplex_segment(x, start, count, target=1.0):
    if count <= 0:
        return
    seg = np.maximum(x[start:start + count], 0.0)
    total = float(seg.sum())
    if total <= 0:
        seg[:] = target / count
    else:
        seg *= target / total
    x[start:start + count] = seg


def _closure_ints(fun):
    vals = []
    for cell in getattr(fun, "__closure__", ()) or ():
        val = cell.cell_contents
        if isinstance(val, (int, np.integer)):
            vals.append(int(val))
    return vals


def _project_constraints(x, constraints):
    for cons in constraints or []:
        if isinstance(cons, dict) and cons.get("type") == "eq" and callable(cons.get("fun")):
            vals = _closure_ints(cons["fun"])
            if len(vals) >= 2:
                _project_simplex_segment(x, vals[0], vals[1], 1.0)
        elif hasattr(cons, "A"):
            A = np.asarray(cons.A, dtype=float)
            lb = np.asarray(cons.lb, dtype=float).reshape(-1)
            ub = np.asarray(cons.ub, dtype=float).reshape(-1)
            if lb.size == 1:
                lb = np.full(A.shape[0], float(lb[0]))
            if ub.size == 1:
                ub = np.full(A.shape[0], float(ub[0]))
            for row, lo, hi in zip(A, lb, ub):
                nz = np.flatnonzero(np.abs(row) > 1e-12)
                if nz.size and np.isclose(lo, hi):
                    current = float(row @ x)
                    x[nz] += (float(lo) - current) / float(row[nz].sum())
    return x


def _project(x, bounds, constraints):
    x = np.asarray(x, dtype=float).copy()
    lb, ub = _bounds_arrays(bounds, x.size)
    x = np.clip(x, lb, ub)
    x = _project_constraints(x, constraints)
    x = np.clip(x, lb, ub)
    return x


def _finite_difference_grad(obj_fun, x):
    grad = np.zeros_like(x, dtype=float)
    fx = float(obj_fun(x))
    eps = 1e-6 * np.maximum(1.0, np.abs(x))
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps[i]
        xm = x.copy(); xm[i] -= eps[i]
        grad[i] = (float(obj_fun(xp)) - float(obj_fun(xm))) / (2.0 * eps[i])
    return fx, grad


def _constraint_violation(x, constraints):
    maxcv = 0.0
    for cons in constraints or []:
        if isinstance(cons, dict) and callable(cons.get("fun")):
            maxcv = max(maxcv, abs(float(cons["fun"](x))))
        elif hasattr(cons, "A"):
            A = np.asarray(cons.A, dtype=float)
            vals = A @ x
            lb = np.asarray(cons.lb, dtype=float).reshape(-1)
            ub = np.asarray(cons.ub, dtype=float).reshape(-1)
            if lb.size == 1:
                lb = np.full(vals.shape, float(lb[0]))
            if ub.size == 1:
                ub = np.full(vals.shape, float(ub[0]))
            maxcv = max(maxcv, float(np.max(np.maximum(lb - vals, vals - ub))))
    return max(0.0, maxcv)


def run_optimization(obj_fun, params_initial, opt_method, bounds, constraints):
    """Run a deterministic projected finite-difference local optimization."""
    x = _project(params_initial, bounds, constraints)
    best_x = x.copy()
    best_fun = float(obj_fun(best_x))
    step = 0.2
    nfev = 1
    maxiter = 80
    for nit in range(1, maxiter + 1):
        _, grad = _finite_difference_grad(obj_fun, x)
        nfev += 2 * x.size
        cand = _project(x - step * grad, bounds, constraints)
        cand_fun = float(obj_fun(cand)); nfev += 1
        if np.isfinite(cand_fun) and cand_fun <= best_fun:
            x = cand
            best_x = cand.copy()
            best_fun = cand_fun
            step = min(step * 1.2, 1.0)
        else:
            step *= 0.5
        if step < 1e-8 or float(np.linalg.norm(grad)) < 1e-8:
            break
    cv = _constraint_violation(best_x, constraints)
    result = SimpleNamespace(
        x=best_x,
        fun=best_fun,
        success=bool(np.isfinite(best_fun) and cv <= 1e-6),
        message="projected finite-difference optimizer completed",
        nit=nit,
        nfev=nfev,
        constr_violation=cv,
        maxcv=cv,
    )
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
    if hasattr(x, name):
        return getattr(x, name)
    if isinstance(x, dict) and name in x:
        return x[name]
    return default


def _extract_fun(result):
    fun = _get_attr(result, "fun", None)
    if fun is None:
        fun = _get_attr(result, "fval", np.inf)
    return float(fun)


def _extract_success(result):
    s = _get_attr(result, "success", None)
    if s is None:
        return False
    return bool(s)


def _extract_constr_violation(result):
    cv = _get_attr(result, "constr_violation", None)
    if cv is None:
        cv = _get_attr(result, "maxcv", None)
    if cv is None:
        return 0.0
    return float(cv)


def _sample_initial(params_initial, bounds, rng, strategy="jitter", jitter_scale=0.15):
    p0 = np.asarray(params_initial, dtype=float).copy()
    lb, ub = _bounds_arrays(bounds, p0.size)
    finite_span = np.where(np.isfinite(ub - lb), ub - lb, 1.0)
    if strategy == "uniform":
        return rng.uniform(lb, ub)
    if strategy == "hybrid" and rng.random() < 0.25:
        return rng.uniform(lb, ub)
    noise = rng.normal(loc=0.0, scale=jitter_scale, size=p0.shape)
    return np.clip(p0 + noise * finite_span, lb, ub)


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
    return StartOutcome(start_id, seed, result, np.asarray(optimized_params, dtype=float), fun, success, cv, runtime_s)


def multistart_run_optimization(obj_fun, params_initial, opt_method, bounds, constraints,
                                n_starts=24, n_jobs=-1, base_seed=1234,
                                init_strategy="hybrid", jitter_scale=0.15, prefer_feasible=True,
                                logger=None):
    seeds = [base_seed + i for i in range(n_starts)]
    if logger:
        logger.info(f"[Multistart] n_starts={n_starts}, n_jobs={n_jobs}, strategy={init_strategy}")
    outcomes = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_start)(i, seeds[i], obj_fun, params_initial, opt_method, bounds, constraints, init_strategy, jitter_scale)
        for i in range(n_starts)
    )

    def key(o: StartOutcome):
        feasible = (o.constr_violation <= 1e-12)
        if prefer_feasible:
            return (0 if feasible else 1, o.constr_violation, o.fun, 0 if o.success else 1, o.runtime_s)
        return (o.fun, 0 if o.success else 1, o.constr_violation, o.runtime_s)
    outcomes_sorted = sorted(outcomes, key=key)
    best = outcomes_sorted[0]
    return best.result, best.optimized_params, outcomes_sorted
