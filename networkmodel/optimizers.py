"""Bounded optimizers for the networkmodel fit behind one protocol (KNOWN_ISSUES O01).

``minimize_bounded(objective, theta0, lower, upper, fixed_mask=None, fixed_values=None, optimizer=..., ...)``
takes a scalar JAX objective of the raw (softplus-space) parameter vector, a raw start vector, raw box bounds and an
optional fixed mask, and returns an :class:`OptimizerResult` with the optimized raw vector, status, stop reason,
iteration and evaluation counts, the final objective, the KKT residual (projected-gradient norm) and the active set.

Optimizers (``[networkmodel] optimizer``):

- ``"projected_gradient"`` (default): jaxopt.ProjectedGradient through ``backend.optimize_scalar_objective``,
  unchanged (same call, same settings). Stopping: jaxopt's error <= ``tol`` or ``maxiter`` iterations (the
  iteration limit, ``n_gen``). Evaluation counts are not exposed by jaxopt: reported as None ("not measured").
- ``"lbfgsb"``: ``scipy.optimize.minimize(method="L-BFGS-B")`` with the raw box bounds, value and gradient from a
  jitted ``jax.value_and_grad`` (float64). Only the free coordinates are optimized; fixed coordinates keep exactly
  their fixed values. Stopping: scipy's ``ftol`` (relative reduction of f), ``gtol`` (= pgtol: max |projected
  gradient component|), ``maxiter``, ``maxfun``.

The fit's numerics (objective, bounds, transform, start, fixed mask) are not changed by either optimizer.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

OPTIMIZERS = ("projected_gradient", "lbfgsb")
OPTIMIZER_LABELS = {"projected_gradient": "jaxopt.ProjectedGradient", "lbfgsb": "scipy.optimize.minimize L-BFGS-B"}

# L-BFGS-B stopping criteria (defaults; overridable by [networkmodel] lbfgsb_maxiter / lbfgsb_maxfun / lbfgsb_ftol /
# lbfgsb_pgtol / lbfgsb_maxcor / lbfgsb_maxls). maxiter is set high so that the fit stops on ftol/pgtol, not the cap.
LBFGSB_DEFAULTS = {"maxiter": 15000, "maxfun": 30000, "ftol": 1e-10, "pgtol": 1e-6, "maxcor": 10, "maxls": 20}

# Coordinates within this relative distance of a bound (|theta - b| <= ACTIVE_TOL * (1 + |b|)) are in the active set.
ACTIVE_TOL = 1e-12


@dataclass
class OptimizerResult:
    """Result of one bounded optimization (raw parameter space)."""

    theta: np.ndarray
    optimizer: str
    label: str
    status: str  # "converged" | "not_converged"
    stop_reason: str
    iterations: int
    n_fun_evals: int | None  # None: not measured (jaxopt does not expose it)
    n_grad_evals: int | None
    final_objective: float
    kkt_residual: float  # ||clip(theta - g, l, u) - theta||_inf over the free coordinates
    proj_grad_norm: float  # 2-norm of the projected gradient over the free coordinates
    active_lower: list = field(default_factory=list)
    active_upper: list = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
    wall_time_s: float = float("nan")
    state: Any = None  # the jaxopt state (projected_gradient) or scipy OptimizeResult (lbfgsb)

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    def summary(self) -> dict:
        """JSON-serialisable summary (no theta, no state)."""
        d = asdict(self)
        for k in ("theta", "state"):
            d.pop(k, None)
        d["active_set_size"] = len(self.active_lower) + len(self.active_upper)
        return d


_VG_CACHE: dict = {}


def _value_and_grad(objective):
    """jit(value_and_grad(objective)), cached per objective so repeated fits (multistart, profiles) compile once."""
    import jax

    key = id(objective)
    hit = _VG_CACHE.get(key)
    if hit is not None and hit[0] is objective:
        return hit[1]
    vg = jax.jit(jax.value_and_grad(objective))
    _VG_CACHE[key] = (objective, vg)
    return vg


def kkt_diagnostics(objective, theta, lower, upper, fixed_mask=None, grad=None):
    """KKT residual, projected-gradient norm and active set of a box-constrained point.

    Over the free coordinates: KKT residual = max |clip(theta - g, l, u) - theta|; projected gradient = g in the
    interior, min(g, 0) at the lower bound, max(g, 0) at the upper bound.
    """
    theta = np.asarray(theta, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if grad is None:
        _, grad = _value_and_grad(objective)(theta)
    g = np.asarray(grad, dtype=np.float64)
    free = np.ones(theta.size, bool) if fixed_mask is None else ~np.asarray(fixed_mask, bool)
    at_lo = free & (np.abs(theta - lower) <= ACTIVE_TOL * (1.0 + np.abs(lower)))
    at_hi = free & (np.abs(theta - upper) <= ACTIVE_TOL * (1.0 + np.abs(upper))) & ~at_lo
    pg = np.where(at_lo, np.minimum(g, 0.0), np.where(at_hi, np.maximum(g, 0.0), g))[free]
    step = (np.clip(theta - g, lower, upper) - theta)[free]
    kkt = float(np.max(np.abs(step))) if step.size else 0.0
    return {"kkt_residual": kkt, "proj_grad_norm": float(np.linalg.norm(pg)),
            "active_lower": [int(i) for i in np.flatnonzero(at_lo)],
            "active_upper": [int(i) for i in np.flatnonzero(at_hi)]}


def lbfgsb_criteria(**overrides) -> dict:
    """The L-BFGS-B stopping criteria in force: LBFGSB_DEFAULTS, then [networkmodel] lbfgsb_* config keys, then
    explicit overrides (None = keep)."""
    crit = dict(LBFGSB_DEFAULTS)
    try:
        import networkmodel.config as nmc

        crit.update({k: v for k, v in getattr(nmc, "LBFGSB_CRITERIA", {}).items() if v is not None})
    except Exception:  # pragma: no cover - config not importable in isolation
        pass
    crit.update({k: v for k, v in overrides.items() if v is not None})
    return crit


def _run_projected_gradient(objective, theta0, lower, upper, fixed_mask, fixed_values, maxiter, tol, logger_obj):
    from networkmodel.backend import optimize_scalar_objective

    t0 = time.perf_counter()
    params, state, value = optimize_scalar_objective(objective, theta0, lower, upper, maxiter=maxiter, tol=tol,
                                                     fixed_mask=fixed_mask, fixed_values=fixed_values,
                                                     logger_obj=logger_obj)
    wall = time.perf_counter() - t0
    it = int(getattr(state, "iter_num", -1))
    err = float(getattr(state, "error", np.nan))
    if np.isfinite(err) and err <= tol:
        status, reason = "converged", f"tolerance reached: jaxopt error {err:.3g} <= tol {tol:g} after {it} iterations"
    else:
        status = "not_converged"
        reason = (f"iteration limit reached: {it} of maxiter {maxiter} iterations, jaxopt error {err:.3g} > tol {tol:g}"
                  if it >= maxiter else f"stopped after {it} iterations with jaxopt error {err:.3g} (tol {tol:g})")
    return params, state, float(value), status, reason, it, None, None, wall, {
        "optimizer": "projected_gradient", "maxiter": int(maxiter), "tol": float(tol),
        "tol_meaning": "jaxopt ProjectedGradient error (norm of the projected-gradient step) <= tol",
        "stepsize": "backtracking line search (stepsize=0.0), maxls=30, FISTA acceleration"}


def _run_lbfgsb(objective, theta0, lower, upper, fixed_mask, fixed_values, crit, logger_obj):
    from scipy.optimize import minimize

    from networkmodel.backend import ensure_jax_float64, project_bounds

    ensure_jax_float64()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    base = np.asarray(project_bounds(theta0, lower, upper, fixed_mask, fixed_values), dtype=np.float64)
    free = np.ones(base.size, bool) if fixed_mask is None else ~np.asarray(fixed_mask, bool)
    vg = _value_and_grad(objective)
    counts = {"nfev": 0, "nonfinite": 0}

    def fg(z):
        x = base.copy()
        x[free] = z
        v, g = vg(x)
        v = float(v)
        counts["nfev"] += 1
        if not np.isfinite(v):
            counts["nonfinite"] += 1
        return v, np.asarray(g, dtype=np.float64)[free]

    t0 = time.perf_counter()
    res = minimize(fg, base[free], jac=True, method="L-BFGS-B", bounds=list(zip(lower[free], upper[free])),
                   options={"maxiter": int(crit["maxiter"]), "maxfun": int(crit["maxfun"]), "ftol": float(crit["ftol"]),
                            "gtol": float(crit["pgtol"]), "maxcor": int(crit["maxcor"]), "maxls": int(crit["maxls"])})
    wall = time.perf_counter() - t0
    theta = base.copy()
    theta[free] = np.clip(np.asarray(res.x, dtype=np.float64), lower[free], upper[free])
    msg = res.message.decode() if isinstance(res.message, bytes) else str(res.message)
    converged = bool(res.success) and "CONVERGENCE" in msg.upper()
    reason = f"{msg} (nit {int(res.nit)}, nfev {counts['nfev']})"
    if counts["nonfinite"]:
        reason += f"; {counts['nonfinite']} non-finite objective evaluation(s)"
    value = float(vg(theta)[0])
    return theta, res, value, "converged" if converged else "not_converged", reason, int(res.nit), \
        counts["nfev"], counts["nfev"], wall, {"optimizer": "lbfgsb", **{k: crit[k] for k in LBFGSB_DEFAULTS},
                                               "ftol_meaning": "(f_k - f_k+1)/max(|f_k|, |f_k+1|, 1) <= ftol",
                                               "pgtol_meaning": "max_i |projected gradient_i| <= pgtol",
                                               "gradient": "jax.jit(jax.value_and_grad(objective)), float64"}


def minimize_bounded(objective, theta0, lower, upper, *, fixed_mask=None, fixed_values=None,
                     optimizer: str = "projected_gradient", maxiter: int = 20000, tol: float = 1e-6,
                     lbfgsb_options: dict | None = None, logger_obj=None, diagnostics: bool = True) -> OptimizerResult:
    """Minimize ``objective`` over the raw box [lower, upper] with the selected optimizer (see module docstring).

    ``maxiter``/``tol`` are the ProjectedGradient iteration limit and tolerance; ``lbfgsb_options`` overrides the
    L-BFGS-B criteria (keys of LBFGSB_DEFAULTS). ``diagnostics=False`` skips the KKT/active-set evaluation (one extra
    value-and-gradient call) and reports NaN / empty sets.
    """
    import logging

    log = logger_obj or logging.getLogger(__name__)
    name = str(optimizer).strip().lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}.")
    if fixed_mask is not None and fixed_values is None:
        raise ValueError("fixed_values is required with fixed_mask.")
    if name == "projected_gradient":
        out = _run_projected_gradient(objective, theta0, lower, upper, fixed_mask, fixed_values, maxiter, tol, log)
    else:
        crit = lbfgsb_criteria(**(lbfgsb_options or {}))
        log.info("[Optimizer] Selected optimizer backend: %s (%s)", OPTIMIZER_LABELS[name], crit)
        out = _run_lbfgsb(objective, theta0, lower, upper, fixed_mask, fixed_values, crit, log)
    theta, state, value, status, reason, it, nfev, ngev, wall, criteria = out
    theta = np.asarray(theta, dtype=np.float64)
    if not np.isfinite(value):
        raise RuntimeError(f"{OPTIMIZER_LABELS[name]} failed: non-finite final objective {value}")
    diag = (kkt_diagnostics(objective, theta, lower, upper, fixed_mask) if diagnostics else
            {"kkt_residual": float("nan"), "proj_grad_norm": float("nan"), "active_lower": [], "active_upper": []})
    result = OptimizerResult(theta=theta, optimizer=name, label=OPTIMIZER_LABELS[name], status=status,
                             stop_reason=reason, iterations=it, n_fun_evals=nfev, n_grad_evals=ngev,
                             final_objective=value, criteria=criteria, wall_time_s=wall, state=state, **diag)
    log.info("[Optimizer] %s: status=%s; %s; objective=%.10g; KKT residual=%.3g; active set %d (lower %d, upper %d)",
             result.label, status, reason, value, result.kkt_residual, len(result.active_lower) +
             len(result.active_upper), len(result.active_lower), len(result.active_upper))
    return result
