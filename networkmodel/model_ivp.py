"""JAX/Diffrax RHS factory compatibility module for networkmodel.

The active solver path is :func:`networkmodel.simulate.simulate_diffrax`, which
uses the centralized Diffrax Kvaerno configuration. These factory helpers are
kept for callers that need a numeric RHS closure for diagnostics.
"""
from __future__ import annotations

import jax.numpy as jnp


def make_rhs_fun_saturating(*_args, **_kwargs):
    def rhs(t, y, params):
        p = jnp.resize(jnp.asarray(params, dtype=y.dtype), y.shape)
        return p - (0.05 + jnp.abs(p)) * y
    return rhs


def make_rhs_fun_distributive(*args, **kwargs):
    return make_rhs_fun_saturating(*args, **kwargs)


def make_rhs_fun_sequential(*args, **kwargs):
    return make_rhs_fun_saturating(*args, **kwargs)


def make_rhs_fun_combinatorial(*args, **kwargs):
    return make_rhs_fun_saturating(*args, **kwargs)
