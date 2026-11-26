"""Flat-parameter packing helpers for appending Equinox neural parameters to theta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from networkmodel.pinn.config import PinnConfig
from networkmodel.pinn.networks import make_neural_rhs


@dataclass(frozen=True)
class PinnParameterSpec:
    """Metadata needed to reconstruct the Equinox neural RHS from flat theta."""

    base_size: int
    nn_slice: slice
    state_dim: int
    nn_static: Any
    nn_unravel: Callable
    parameter_names: tuple[str, ...]

    @property
    def n_neural_params(self) -> int:
        return int(self.nn_slice.stop - self.nn_slice.start)


def _state_dim_from_system(sys) -> int:
    y0 = sys.y0() if hasattr(sys, "y0") else None
    if y0 is None:
        raise ValueError("PINN mode requires System.y0() to determine state dimension.")
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.ndim != 1 or y0.size == 0:
        raise ValueError(f"System.y0() must return a non-empty 1D state vector, got {y0.shape}.")
    return int(y0.size)


def build_pinn_parameter_spec(
    *,
    sys,
    base_theta_size: int,
    config: PinnConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PinnParameterSpec]:
    """Create initial neural parameters and bounds to append to theta0/xl/xu."""
    if not config.enabled or config.mode == "off":
        empty = np.empty(0, dtype=np.float64)
        spec = PinnParameterSpec(
            base_size=int(base_theta_size),
            nn_slice=slice(int(base_theta_size), int(base_theta_size)),
            state_dim=_state_dim_from_system(sys),
            nn_static=None,
            nn_unravel=lambda x: x,
            parameter_names=(),
        )
        return empty, empty, empty, spec

    state_dim = _state_dim_from_system(sys)
    key = jax.random.PRNGKey(int(config.seed))

    model = make_neural_rhs(
        state_dim=state_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        activation=config.activation,
        output_scale=config.output_scale,
        key=key,
    )

    arrays, static = eqx.partition(model, eqx.is_array)
    flat, unravel = ravel_pytree(arrays)

    flat = jnp.asarray(flat, dtype=jnp.float64)
    nn0 = np.asarray(flat, dtype=np.float64)
    xl = np.full(nn0.shape, -float(config.weight_bound), dtype=np.float64)
    xu = np.full(nn0.shape, float(config.weight_bound), dtype=np.float64)

    start = int(base_theta_size)
    stop = start + int(nn0.size)

    spec = PinnParameterSpec(
        base_size=start,
        nn_slice=slice(start, stop),
        state_dim=state_dim,
        nn_static=static,
        nn_unravel=unravel,
        parameter_names=tuple(f"pinn_nn[{i}]" for i in range(nn0.size)),
    )

    return nn0, xl, xu, spec


def extend_theta_with_pinn(
    theta0,
    xl,
    xu,
    sys,
    config: PinnConfig,
):
    """Append neural parameters to the existing mechanistic optimizer vector."""
    theta0 = np.asarray(theta0, dtype=np.float64)
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)

    if theta0.shape != xl.shape or theta0.shape != xu.shape:
        raise ValueError(
            f"theta0/xl/xu shape mismatch before PINN extension: "
            f"{theta0.shape}, {xl.shape}, {xu.shape}"
        )

    nn0, nn_xl, nn_xu, spec = build_pinn_parameter_spec(
        sys=sys,
        base_theta_size=int(theta0.size),
        config=config,
    )

    if not config.enabled or config.mode == "off":
        return theta0, xl, xu, spec

    theta0_ext = np.concatenate([theta0, nn0]).astype(np.float64, copy=False)
    xl_ext = np.concatenate([xl, nn_xl]).astype(np.float64, copy=False)
    xu_ext = np.concatenate([xu, nn_xu]).astype(np.float64, copy=False)

    return theta0_ext, xl_ext, xu_ext, spec