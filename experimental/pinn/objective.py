"""PINN / NeuralODE objective construction for networkmodel."""

from __future__ import annotations

from typing import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
import numpy as np

from networkmodel.backend import (
    DataMode,
    DiffraxSolverConfig,
    make_networkmodel_rhs,
    multimodal_loss_from_trajectory,
    validate_loss_data,
    _defaults_vector_jax,
    _flatten_params_for_slices_jax,
    _unpack_theta_jax,
)
from networkmodel.pinn.config import PinnConfig
from networkmodel.pinn.pack import PinnParameterSpec


def _rebuild_model(spec: PinnParameterSpec, nn_flat):
    arrays = spec.nn_unravel(jnp.asarray(nn_flat, dtype=jnp.float64))
    return eqx.combine(arrays, spec.nn_static)


def _solve_neuralode(y0, t_eval, rhs, args, solver_cfg: DiffraxSolverConfig):
    ts = jnp.asarray(t_eval, dtype=jnp.float64)
    y0 = jnp.asarray(y0, dtype=jnp.float64)

    term = diffrax.ODETerm(rhs)

    sol = diffrax.diffeqsolve(
        term,
        solver_cfg.solver(),
        t0=ts[0],
        t1=ts[-1],
        dt0=jnp.maximum((ts[-1] - ts[0]) / jnp.maximum(ts.size - 1, 1), 1e-3),
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(
            rtol=solver_cfg.rtol,
            atol=solver_cfg.atol,
        ),
        max_steps=solver_cfg.max_steps,
    )

    Y_raw = jnp.asarray(sol.ys, dtype=jnp.float64)
    bad = ~jnp.all(jnp.isfinite(Y_raw))
    Y = jnp.nan_to_num(Y_raw, nan=1e6, posinf=1e6, neginf=-1e6)
    return Y, bad


def make_pinn_objective(
    *,
    loss_data: Mapping,
    mode: DataMode,
    time_grid: Sequence[float],
    weights=None,
    defaults=None,
    prior_weight: float = 0.0,
    networkmodel_layout: bool = False,
    return_breakdown: bool = False,
    y0=None,
    sys=None,
    slices=None,
    pinn_config: PinnConfig,
    pinn_spec: PinnParameterSpec,
):
    """Create a hybrid mechanistic-NeuralODE or pure NeuralODE scalar objective."""
    if sys is None:
        raise ValueError("PINN/NeuralODE objective requires sys.")
    if slices is None:
        raise ValueError("PINN/NeuralODE objective requires mechanistic parameter slices.")
    if pinn_spec is None:
        raise ValueError("PINN/NeuralODE objective requires pinn_spec.")
    if not pinn_config.enabled or pinn_config.mode == "off":
        raise ValueError("make_pinn_objective called while PINN mode is disabled.")

    validate_loss_data(loss_data, mode)

    t = jnp.asarray(time_grid, dtype=jnp.float64)

    prot_map = np.asarray(loss_data["prot_map"])
    if len(prot_map):
        layout = str(loss_data.get("state_layout", "standard"))
        offsets_np = prot_map[:, 0]
        counts_np = prot_map[:, 1]
        if layout == "combinatorial":
            state_width = 1 + counts_np
        else:
            state_width = 2 + counts_np
        state_dim = int(np.max(offsets_np + state_width))
    else:
        state_dim = int(pinn_spec.state_dim)

    y0 = sys.y0() if y0 is None and hasattr(sys, "y0") else y0
    y0 = jnp.ones(state_dim, dtype=jnp.float64) if y0 is None else jnp.asarray(y0, dtype=jnp.float64)

    if int(y0.shape[0]) != int(state_dim):
        raise ValueError(
            f"Initial state dimension {y0.shape[0]} does not match "
            f"loss-data state dimension {state_dim}."
        )

    is_pure_neuralode = str(pinn_config.mode).lower() == "neuralode"

    mech_rhs = None if is_pure_neuralode else make_networkmodel_rhs(sys, slices)
    defaults_j = None if is_pure_neuralode else _defaults_vector_jax(defaults, slices)
    solver_cfg = DiffraxSolverConfig()

    def rhs(ti, yi, args):
        base_theta, nn_model = args

        yi = jnp.asarray(yi, dtype=jnp.float64)
        ti = jnp.asarray(ti, dtype=jnp.float64)

        ti_scaled = ti / float(pinn_config.t_scale)
        yi_scaled = yi / float(pinn_config.y_scale)

        neural = nn_model(ti_scaled, yi_scaled)

        if pinn_config.mode == "hybrid":
            mechanistic = mech_rhs(ti, yi, base_theta)
            return mechanistic + neural

        if pinn_config.mode == "neuralode":
            return neural

        raise ValueError(f"Unsupported pinn mode: {pinn_config.mode!r}")

    def objective(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)

        base_theta = (
            jnp.empty((0,), dtype=jnp.float64)
            if is_pure_neuralode
            else theta[: pinn_spec.base_size]
        )
        nn_flat = theta[pinn_spec.nn_slice]
        nn_model = _rebuild_model(pinn_spec, nn_flat)

        Y, bad = _solve_neuralode(
            y0=y0,
            t_eval=t,
            rhs=rhs,
            args=(base_theta, nn_model),
            solver_cfg=solver_cfg,
        )

        bad_traj_penalty = jnp.where(bad, 1e12, 0.0)

        if Y.shape[1] != state_dim:
            raise ValueError(
                f"Solved trajectory state width {Y.shape[1]} does not match expected {state_dim}."
            )

        total, breakdown = multimodal_loss_from_trajectory(
            Y,
            loss_data,
            mode,
            weights=weights,
            networkmodel_layout=networkmodel_layout,
        )

        total = total + bad_traj_penalty

        if (not is_pure_neuralode) and defaults_j is not None and prior_weight:
            physical_params = _unpack_theta_jax(base_theta, slices)
            rates = _flatten_params_for_slices_jax(physical_params, slices)
            d = rates[: defaults_j.size] - defaults_j
            prior = float(prior_weight) * jnp.mean(d * d)
            total = total + prior
            breakdown["prior"] = prior

        if pinn_config.l2_regularization:
            nn_l2 = float(pinn_config.l2_regularization) * jnp.mean(nn_flat * nn_flat)
            total = total + nn_l2
            breakdown["pinn_l2"] = nn_l2

        if return_breakdown:
            breakdown["scalar_total"] = jnp.asarray(total, dtype=jnp.float64)
            return jnp.asarray(total, dtype=jnp.float64), breakdown

        return jnp.asarray(total, dtype=jnp.float64)

    return objective