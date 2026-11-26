"""PINN / NeuralODE solver dispatch.

This module keeps optimizer-mode decisions out of runner.py.
"""

from __future__ import annotations

from typing import Any


def should_use_pinn_solver(problem: Any) -> bool:
    """Return True when the problem should bypass the default JAXopt solver."""
    cfg = getattr(problem, "pinn_config", None)
    spec = getattr(problem, "pinn_spec", None)

    if cfg is None or spec is None:
        return False

    return (
        bool(getattr(cfg, "enabled", False))
        and str(getattr(cfg, "mode", "off")).lower() == "neuralode"
        and getattr(spec, "n_neural_params", 0) > 0
    )


def solve_pinn_problem(
    *,
    problem: Any,
    theta0,
    maxiter: int,
    logger=None,
):
    """Solve a PINN/NeuralODE problem using the appropriate PINN optimizer.

    Currently:
      - neuralode -> Optax
      - hybrid    -> not handled here; keep using existing JAXopt path
    """
    cfg = getattr(problem, "pinn_config", None)
    spec = getattr(problem, "pinn_spec", None)

    if cfg is None or spec is None:
        raise ValueError("solve_pinn_problem requires problem.pinn_config and problem.pinn_spec.")

    mode = str(getattr(cfg, "mode", "off")).lower()

    if mode == "neuralode":
        from networkmodel.pinn.neuralsolver import solve_neuralode_optax

        return solve_neuralode_optax(
            problem=problem,
            theta0=theta0,
            xl=problem.xl,
            xu=problem.xu,
            pinn_spec=spec,
            maxiter=maxiter,
            learning_rate=float(getattr(cfg, "optax_learning_rate", 1e-3)),
            clip_norm=float(getattr(cfg, "optax_clip_norm", 1.0)),
            weight_decay=float(getattr(cfg, "optax_weight_decay", 1e-6)),
            optimizer=str(getattr(cfg, "optax_optimizer", "adamw")),
            logger=logger,
        )

    raise ValueError(f"No PINN-specific solver registered for pinn_mode={mode!r}.")