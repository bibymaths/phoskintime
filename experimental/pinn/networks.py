"""Equinox neural RHS modules for networkmodel NeuralODE objectives."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _activation(name: str):
    name = str(name).strip().lower()
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return jax.nn.gelu
    if name in {"swish", "silu"}:
        return jax.nn.silu
    if name == "softplus":
        return jax.nn.softplus
    return jnp.tanh


class NeuralRHS(eqx.Module):
    """Small MLP representing a neural ODE correction or full neural RHS."""

    mlp: eqx.nn.MLP
    output_scale: float = eqx.field(static=True)

    def __call__(self, t, y):
        t = jnp.asarray(t, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        x = jnp.concatenate([jnp.ravel(t)[0:1], y], axis=0)
        return self.output_scale * self.mlp(x)


def make_neural_rhs(
    *,
    state_dim: int,
    hidden_size: int,
    depth: int,
    activation: str,
    output_scale: float,
    key,
) -> NeuralRHS:
    """Construct a float64 Equinox MLP for neural RHS modeling."""
    model = NeuralRHS(
        mlp=eqx.nn.MLP(
            in_size=int(state_dim) + 1,
            out_size=int(state_dim),
            width_size=int(hidden_size),
            depth=int(depth),
            activation=_activation(activation),
            final_activation=lambda x: x,
            key=key,
        ),
        output_scale=float(output_scale),
    )

    arrays, static = eqx.partition(model, eqx.is_array)
    arrays = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=jnp.float64),
        arrays,
    )
    return eqx.combine(arrays, static)