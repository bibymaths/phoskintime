"""Single-objective JAX optimization wrapper for the PhosKinTime network model."""
from __future__ import annotations

import logging
import numpy as np
import jax.numpy as jnp

from networkmodel.jax_backend import (
    DataMode,
    detect_data_mode,
    ensure_jax_float64,
    make_simple_objective,
    optimize_scalar_objective,
    validate_loss_data,
)

logger = logging.getLogger(__name__)


def build_weight_functions(method_protein="uniform", method_rna="uniform", time_grid=None):
    """Backward-compatible weight hook; scalar objective uses provided per-row weights."""
    return {"protein": method_protein, "rna": method_rna, "time_grid": time_grid}


class GlobalODEScalarObjective:
    """JAX-compatible scalar objective replacing the legacy vector-objective problem."""

    def __init__(self, sys, slices, loss_data, defaults, lambdas, time_grid, xl, xu, fail_value=1e12,
                 data_mode: DataMode | None = None, **_):
        ensure_jax_float64()
        self.sys = sys
        self.slices = slices
        self.loss_data = loss_data
        self.defaults = defaults
        self.lambdas = lambdas or {}
        self.time_grid = np.asarray(time_grid, dtype=np.float64)
        self.xl = np.asarray(xl, dtype=np.float64)
        self.xu = np.asarray(xu, dtype=np.float64)
        theta_len = sum(int(sl.stop) - int(sl.start) for sl in slices.values())
        if self.xl.shape != self.xu.shape or self.xl.ndim != 1:
            raise ValueError(f"xl/xu must be same-length 1D vectors, got {self.xl.shape} and {self.xu.shape}")
        if theta_len != self.xl.size:
            raise ValueError(f"Slice layout length {theta_len} does not match bounds length {self.xl.size}")
        if "alpha" in slices or "beta" in slices:
            raise ValueError("alpha/beta are network construction weights and must not be optimized in theta.")
        self.n_var = len(self.xl)
        self.n_obj = 1
        self.fail_value = float(fail_value)
        self.data_mode = data_mode or detect_data_mode(loss_data=loss_data, logger_obj=logger)
        validate_loss_data(loss_data, self.data_mode)
        logger.info("[Objective] Single scalar objective initialized for mode %s", self.data_mode.data_mode)
        logger.info("[Objective] Parameter vector size: %d", self.n_var)
        # Global networkmodel scalar objective combines weighted modality MSEs on
        # fold-change observables. The networkmodel_layout flag keeps the global
        # state-to-observation mapping separate from protwise's shared-backend path.
        objective_weights = {
            "protein": self.lambdas.get("protein", 1.0),
            "rna": self.lambdas.get("rna", 1.0),
            "phospho": self.lambdas.get("phospho", 1.0),
        }
        y0 = sys.y0() if hasattr(sys, "y0") else None
        self._objective = make_simple_objective(
            loss_data,
            self.data_mode,
            self.time_grid,
            weights=objective_weights,
            defaults=self.defaults,
            prior_weight=float(self.lambdas.get("prior", 0.0)),
            networkmodel_layout=True,
            y0=y0,
            sys=sys,
            slices=slices,
        )
        self._objective_raw = make_simple_objective(
            loss_data,
            self.data_mode,
            self.time_grid,
            weights=objective_weights,
            defaults=self.defaults,
            prior_weight=float(self.lambdas.get("prior", 0.0)),
            networkmodel_layout=True,
            return_breakdown=True,
            y0=y0,
            sys=sys,
            slices=slices,
        )
        self.final_loss_breakdown = {}

    def objective(self, x):
        return self._objective(jnp.asarray(x, dtype=jnp.float64))

    def evaluate(self, x) -> float:
        val = self.objective(x)
        return float(val) if np.isfinite(float(val)) else self.fail_value

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.asarray([self.evaluate(x)], dtype=np.float64)

    def solve(self, theta0, maxiter=50, tol=1e-6):
        theta0 = np.asarray(theta0, dtype=np.float64)
        logger.info("[GlobalObjective] theta0.shape=%s xl.shape=%s xu.shape=%s", theta0.shape, self.xl.shape, self.xu.shape)
        if theta0.shape != self.xl.shape or theta0.shape != self.xu.shape:
            raise ValueError(f"theta0/xl/xu shape mismatch: {theta0.shape}, {self.xl.shape}, {self.xu.shape}")
        params, state, value = optimize_scalar_objective(self.objective, theta0, self.xl, self.xu, maxiter=maxiter,
                                                         tol=tol, logger_obj=logger)
        for name, sl in self.slices.items():
            delta = np.max(np.abs(params[sl] - theta0[sl])) if (sl.stop - sl.start) else 0.0
            logger.info("[Optimizer] Parameter group movement %s: max_abs_delta=%.8g", name, float(delta))
        raw_val, breakdown = self._objective_raw(jnp.asarray(params, dtype=jnp.float64))
        self.final_loss_breakdown = {k: float(v) for k, v in breakdown.items()}
        logger.info("[GlobalObjective] Final per-modality loss: %s", self.final_loss_breakdown)
        if "phospho" not in self.final_loss_breakdown and self.data_mode.fit_phospho:
            logger.warning("[GlobalObjective] Phospho data were detected but no phospho loss was reported.")
        if np.isfinite(float(raw_val)) and abs(float(raw_val) - float(value)) > max(1e-6, 1e-6 * abs(float(value))):
            logger.warning("[GlobalObjective] Raw objective %.8g differs from optimizer value %.8g.",
                           float(raw_val), float(value))
        return params, state, value


class GlobalODE_MOO(GlobalODEScalarObjective):
    """Compatibility alias for old imports; routes to scalar JAX objective."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "[Deprecated API] GlobalODE_MOO now constructs a single-objective JAXopt problem, not a legacy multi-layer vector-objective problem.")
        super().__init__(*args, **kwargs)
