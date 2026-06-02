"""JAX/JAXopt/Diffrax backend for PhosKinTime networkmodel and protwise fitting.

This module is intentionally independent from pandas and string identifiers inside
its differentiable functions. Data frames are converted to typed arrays before the
objective is called; JAX computations receive numeric arrays only.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
import diffrax

logger = logging.getLogger(__name__)

LAYERS = ("mrna", "protein", "phospho")
ALIASES = {"rna": "mrna", "mrna": "mrna", "protein": "protein", "prot": "protein", "phospho": "phospho", "pho": "phospho"}


@dataclass(frozen=True)
class DataMode:
    available_layers: tuple[str, ...]
    data_mode: str
    fit_mrna: bool
    fit_protein: bool
    fit_phospho: bool

    @property
    def active_loss_terms(self) -> tuple[str, ...]:
        return tuple(f"{layer}_loss" for layer in self.available_layers)

    @property
    def skipped_loss_terms(self) -> tuple[str, ...]:
        return tuple(f"{layer}_loss" for layer in LAYERS if layer not in self.available_layers)


def ensure_jax_float64() -> bool:
    jax.config.update("jax_enable_x64", True)
    return bool(jax.config.jax_enable_x64)


def detect_data_mode(*, mrna=None, protein=None, phospho=None, loss_data: Mapping | None = None, logger_obj=None) -> DataMode:
    """Detect the non-empty mRNA/protein/phospho mode after loading input data."""
    def has_frame(x) -> bool:
        if x is None:
            return False
        if hasattr(x, "empty"):
            return not bool(x.empty)
        arr = np.asarray(x)
        return arr.size > 0

    if loss_data is not None:
        mrna_on = int(loss_data.get("n_r", len(loss_data.get("obs_rna", [])))) > 0 and len(loss_data.get("obs_rna", [])) > 0
        protein_on = int(loss_data.get("n_p", len(loss_data.get("obs_prot", [])))) > 0 and len(loss_data.get("obs_prot", [])) > 0
        phospho_on = int(loss_data.get("n_ph", len(loss_data.get("obs_pho", [])))) > 0 and len(loss_data.get("obs_pho", [])) > 0
    else:
        mrna_on = has_frame(mrna)
        protein_on = has_frame(protein)
        phospho_on = has_frame(phospho)

    active = tuple(layer for layer, on in (("mrna", mrna_on), ("protein", protein_on), ("phospho", phospho_on)) if on)
    if not active:
        raise ValueError("No PhosKinTime data layers were detected. Provide at least one of mRNA, protein, or phospho data.")
    mode = DataMode(active, "+".join(active), mrna_on, protein_on, phospho_on)
    log = logger_obj or logger
    log.info("[DataMode] Detected data mode: %s", mode.data_mode)
    log.info("[DataMode] Available layers: %s", ", ".join(mode.available_layers))
    log.info("[Objective] Active loss terms: %s", ", ".join(mode.active_loss_terms))
    log.info("[Objective] Skipped loss terms: %s", ", ".join(mode.skipped_loss_terms) or "none")
    return mode


def validate_loss_data(loss_data: Mapping, mode: DataMode) -> None:
    required = {
        "protein": ("p_prot", "t_prot", "obs_prot", "w_prot"),
        "mrna": ("p_rna", "t_rna", "obs_rna", "w_rna"),
        "phospho": ("p_pho", "s_pho", "t_pho", "obs_pho", "w_pho"),
    }
    for layer in mode.available_layers:
        lengths = []
        for key in required[layer]:
            if key not in loss_data:
                raise ValueError(f"Missing required {layer} loss array '{key}'.")
            arr = np.asarray(loss_data[key])
            if arr.ndim != 1:
                raise ValueError(f"Loss array '{key}' must be one-dimensional, got shape {arr.shape}.")
            if np.any(~np.isfinite(arr.astype(float, copy=False))):
                raise ValueError(f"Loss array '{key}' contains NaN or infinite values.")
            lengths.append(arr.shape[0])
        if len(set(lengths)) != 1:
            raise ValueError(f"{layer} loss arrays have mismatched lengths: {dict(zip(required[layer], lengths))}.")


@dataclass(frozen=True)
class DiffraxSolverConfig:
    solver_name: str = "Kvaerno4"
    rtol: float = 1e-5
    atol: float = 1e-7
    max_steps: int = 20000
    root_max_steps: int = 20

    def solver(self):
        name = str(self.solver_name).lower()
        if name == "kvaerno5":
            return diffrax.Kvaerno5(
                root_finder=diffrax.VeryChord(rtol=self.rtol, atol=self.atol, kappa=0.01),
                root_find_max_steps=self.root_max_steps,
            )
        return diffrax.Kvaerno4(
            root_finder=diffrax.VeryChord(rtol=self.rtol, atol=self.atol, kappa=0.01),
            root_find_max_steps=self.root_max_steps,
        )


def _default_rhs(t, y, args):
    rates = args
    n = y.shape[0]
    base = jnp.resize(rates, (n,))
    return base - (0.05 + jnp.abs(base)) * y


def solve_diffrax(y0, t_eval, params=None, rhs=None, config: DiffraxSolverConfig | None = None):
    """Solve an ODE with Diffrax Kvaerno4/Kvaerno5 and return (time, state)."""
    ensure_jax_float64()
    cfg = config or DiffraxSolverConfig()

    if not isinstance(t_eval, jax.core.Tracer):
        ts_np = np.asarray(t_eval, dtype=np.float64)
        if ts_np.ndim != 1 or ts_np.size == 0:
            raise ValueError("t_eval must be a non-empty one-dimensional time grid.")
        if ts_np.size > 1 and np.any(np.diff(ts_np) <= 0.0):
            raise ValueError("t_eval must be strictly increasing for the Diffrax solver.")

    # Use jax.numpy throughout in the solve path; t_eval may be a tracer.
    ts = jnp.asarray(t_eval, dtype=jnp.float64)

    if ts.ndim != 1 or ts.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional time grid.")

    y0_j = jnp.asarray(y0, dtype=jnp.float64)
    if params is None:
        params = jnp.ones(max(1, y0_j.size), dtype=jnp.float64)
    args = jnp.asarray(params, dtype=jnp.float64)

    term = diffrax.ODETerm(rhs or _default_rhs)
    try:
        sol = diffrax.diffeqsolve(
            term,
            cfg.solver(),
            t0=ts[0],
            t1=ts[-1],
            dt0=jnp.maximum((ts[-1] - ts[0]) / jnp.maximum(ts.size - 1, 1), 1e-3),
            y0=y0_j,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol),
            max_steps=cfg.max_steps,
        )
    except Exception as exc:
        raise RuntimeError(f"Diffrax solver failed with {cfg.solver_name}: {exc}") from exc

    ys = jnp.asarray(sol.ys, dtype=jnp.float64)
    if ys.shape[0] != ts.shape[0]:
        raise ValueError(f"Diffrax returned invalid shape {ys.shape}; expected first dimension {ts.shape[0]}.")
    return ys

def _extract_offsets(prot_map):
    pm = jnp.asarray(prot_map, dtype=jnp.int32)
    return pm[:, 0], pm[:, 1]


def multimodal_loss_from_trajectory(Y, loss_data: Mapping, mode: DataMode, weights: Mapping[str, float] | None = None):
    weights = weights or {}
    prot_map = jnp.asarray(loss_data["prot_map"], dtype=jnp.int32)
    offsets, counts = _extract_offsets(prot_map)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    breakdown = {}

    if mode.fit_protein:
        p = jnp.asarray(loss_data["p_prot"], dtype=jnp.int32)
        t = jnp.asarray(loss_data["t_prot"], dtype=jnp.int32)
        obs = jnp.asarray(loss_data["obs_prot"], dtype=jnp.float64)
        w = jnp.asarray(loss_data["w_prot"], dtype=jnp.float64)
        pred = Y[t, offsets[p] + 1]
        loss = jnp.sum(w * (pred - obs) ** 2) / jnp.maximum(jnp.sum(w), 1.0)
        total = total + float(weights.get("protein", 1.0)) * loss
        breakdown["protein"] = loss
    if mode.fit_mrna:
        p = jnp.asarray(loss_data["p_rna"], dtype=jnp.int32)
        t = jnp.asarray(loss_data["t_rna"], dtype=jnp.int32)
        obs = jnp.asarray(loss_data["obs_rna"], dtype=jnp.float64)
        w = jnp.asarray(loss_data["w_rna"], dtype=jnp.float64)
        pred = Y[t, offsets[p]]
        loss = jnp.sum(w * (pred - obs) ** 2) / jnp.maximum(jnp.sum(w), 1.0)
        total = total + float(weights.get("mrna", weights.get("rna", 1.0))) * loss
        breakdown["mrna"] = loss
    if mode.fit_phospho:
        p = jnp.asarray(loss_data["p_pho"], dtype=jnp.int32)
        s = jnp.asarray(loss_data["s_pho"], dtype=jnp.int32)
        t = jnp.asarray(loss_data["t_pho"], dtype=jnp.int32)
        obs = jnp.asarray(loss_data["obs_pho"], dtype=jnp.float64)
        w = jnp.asarray(loss_data["w_pho"], dtype=jnp.float64)
        pred = Y[t, offsets[p] + 2 + s]
        loss = jnp.sum(w * (pred - obs) ** 2) / jnp.maximum(jnp.sum(w), 1.0)
        total = total + float(weights.get("phospho", 1.0)) * loss
        breakdown["phospho"] = loss
    return total, breakdown


def project_simplex(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    u = jnp.sort(x)[::-1]
    cssv = jnp.cumsum(u) - 1.0
    ind = jnp.arange(1, x.size + 1, dtype=jnp.float64)
    cond = u - cssv / ind > 0
    rho = jnp.sum(cond) - 1
    theta = cssv[rho] / (rho + 1.0)
    return jnp.maximum(x - theta, 0.0)


def project_alpha_blocks(alpha, block_ids):
    """Project alpha values onto one [0, 1] sum-to-one simplex per block."""
    a = jnp.asarray(alpha, dtype=jnp.float64)
    bids = np.asarray(block_ids)
    out = []
    for b in np.unique(bids):
        idxs = np.where(bids == b)[0]
        out.append((idxs, project_simplex(a[idxs])))
    res = a
    for idxs, vals in out:
        res = res.at[jnp.asarray(idxs)].set(vals)
    return res


def project_beta_blocks(beta, block_ids, lower=-4.0, upper=4.0):
    """Project beta blocks to bounded affine sum-to-one sets without forcing non-negativity.

    Beta weights are allowed to be negative, so a standard non-negative simplex is
    biologically wrong. We first shift each block to satisfy the affine sum exactly,
    then clip to [-4, 4] and redistribute any residual sum error over entries that
    still have room. This keeps negative beta values when the optimum requires them.
    """
    b = jnp.asarray(beta, dtype=jnp.float64)
    bids = np.asarray(block_ids)
    res = b
    for block in np.unique(bids):
        idxs_np = np.where(bids == block)[0]
        idxs = jnp.asarray(idxs_np)
        vals = res[idxs]
        vals = vals + (1.0 - jnp.sum(vals)) / vals.size
        vals = jnp.clip(vals, lower, upper)
        for _ in range(8):
            residual = 1.0 - jnp.sum(vals)
            room = jnp.where(residual >= 0, upper - vals, vals - lower)
            mask = room > 1e-12
            share = residual / jnp.maximum(jnp.sum(mask), 1)
            vals = jnp.where(mask, jnp.clip(vals + share, lower, upper), vals)
        res = res.at[idxs].set(vals)
    return res


def project_bounds(theta, lower, upper, fixed_mask=None, fixed_values=None):
    clipped = jnp.clip(jnp.asarray(theta, dtype=jnp.float64), jnp.asarray(lower, dtype=jnp.float64), jnp.asarray(upper, dtype=jnp.float64))
    if fixed_mask is not None:
        clipped = jnp.where(jnp.asarray(fixed_mask, dtype=bool), jnp.asarray(fixed_values, dtype=jnp.float64), clipped)
    return clipped


@dataclass
class JaxoptResult:
    X: np.ndarray
    F: np.ndarray
    objective_value: float
    params: np.ndarray
    state: object
    data_mode: DataMode
    loss_breakdown: dict
    optimizer: str = "jaxopt.ProjectedGradient"


def optimize_scalar_objective(objective_fun, theta0, lower, upper, *, maxiter=20000, tol=1e-6, fixed_mask=None, fixed_values=None, logger_obj=None):
    ensure_jax_float64()
    log = logger_obj or logger
    log.info("[Optimizer] Selected optimizer backend: jaxopt.ProjectedGradient")
    log.info("[Constraints] Kinetic parameters use bound projection; fixed parameters are restored after each projection.")
    lower_j = jnp.asarray(lower, dtype=jnp.float64)
    upper_j = jnp.asarray(upper, dtype=jnp.float64)
    fixed_mask_j = None if fixed_mask is None else jnp.asarray(fixed_mask, dtype=bool)
    fixed_values_j = None if fixed_values is None else jnp.asarray(fixed_values, dtype=jnp.float64)

    def projection(x, hyperparams):
        lo, hi = hyperparams
        return project_bounds(x, lo, hi, fixed_mask_j, fixed_values_j)

    solver = jaxopt.ProjectedGradient(fun=objective_fun, projection=projection, maxiter=int(maxiter), tol=float(tol))
    init = project_bounds(theta0, lower_j, upper_j, fixed_mask_j, fixed_values_j)
    params, state = solver.run(init, hyperparams_proj=(lower_j, upper_j))
    val = objective_fun(params)
    val_f = float(val)
    if not np.isfinite(val_f):
        log.error("[Optimizer] JAXopt failed: final scalar objective is not finite (%s).", val_f)
        raise RuntimeError("JAXopt optimization failed: final scalar objective is not finite.")
    log.info("[Optimizer] Convergence status: iterations=%s final scalar objective=%.8g", getattr(state, "iter_num", "unknown"), val_f)
    return np.asarray(params), state, val_f


def make_simple_objective(loss_data: Mapping, mode: DataMode, time_grid: Sequence[float], weights=None, defaults=None, prior_weight=0.0):
    validate_loss_data(loss_data, mode)
    t = jnp.asarray(time_grid, dtype=jnp.float64)
    prot_map = np.asarray(loss_data["prot_map"])
    state_dim = int(np.max(prot_map[:, 0] + np.maximum(prot_map[:, 1] + 2, 2))) if len(prot_map) else 2
    y0 = jnp.ones(state_dim, dtype=jnp.float64)
    defaults_j = None if defaults is None else jnp.asarray(defaults, dtype=jnp.float64)

    def objective(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        rates = jax.nn.softplus(theta)
        Y = solve_diffrax(y0, t, params=rates)
        total, _ = multimodal_loss_from_trajectory(Y, loss_data, mode, weights=weights)
        if defaults_j is not None and prior_weight:
            d = rates[: defaults_j.size] - defaults_j
            total = total + float(prior_weight) * jnp.mean(d * d)
        return jnp.asarray(total, dtype=jnp.float64)

    return objective


def warn_deprecated_backend_options(options: Mapping | object | None, logger_obj=None):
    if options is None:
        return
    log = logger_obj or logger
    get = options.get if isinstance(options, Mapping) else lambda k, d=None: getattr(options, k, d)
    optimizer = get("optimizer", get("solver", None))
    if optimizer and str(optimizer).lower() in {"pymoo", "optuna", "nsga3", "unsga3", "spea2", "de", "ga", "scipy"}:
        log.warning("[Deprecated Config] optimizer/solver=%r is accepted for compatibility and mapped to jaxopt.ProjectedGradient.", optimizer)
    for key in ("n_gen", "pop", "population_size", "use_custom_solver", "odeint", "solve_ivp"):
        val = get(key, None)
        if val is not None:
            log.warning("[Deprecated Config] %s=%r is accepted but ignored by the JAXopt/Diffrax PhosKinTime path.", key, val)
