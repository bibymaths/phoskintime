"""Protein-wise local-model wrapper.

The protwise mechanism shares the distributive site-level RHS but must identify
itself explicitly when called directly for mechanism comparisons.
"""
from __future__ import annotations


def solve_ode(params, init_cond, num_psites, t, **kwargs):
    """Solve the protwise mechanism with the centralized Diffrax backend."""
    from protwise.models.diffrax_solver import solve_protwise_ode

    # Pass an explicit model name so direct calls to this module are not affected
    # by the global config.constants.ODE_MODEL selected for a different mechanism.
    return solve_protwise_ode(params, init_cond, num_psites, t, model_name="protwise", **kwargs)
