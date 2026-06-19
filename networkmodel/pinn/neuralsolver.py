from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _make_optimizer(
    *,
    learning_rate: float,
    clip_norm: float,
    weight_decay: float,
    optimizer: str,
):
    optimizer = str(optimizer).strip().lower()

    transforms = []

    if clip_norm and clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(float(clip_norm)))

    if optimizer == "adamw":
        transforms.append(
            optax.adamw(
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
            )
        )
    elif optimizer == "adam":
        transforms.append(optax.adam(float(learning_rate)))
    elif optimizer == "amsgrad":
        transforms.append(optax.amsgrad(float(learning_rate)))
    else:
        raise ValueError(
            f"Unsupported Optax optimizer {optimizer!r}. "
            "Supported: 'adam', 'adamw', 'amsgrad'."
        )

    return optax.chain(*transforms)


def solve_neuralode_optax(
    *,
    problem,
    theta0,
    xl,
    xu,
    pinn_spec,
    maxiter: int,
    learning_rate: float = 1e-3,
    clip_norm: float = 1.0,
    weight_decay: float = 1e-6,
    optimizer: str = "adamw",
    logger=None,
):
    """Train pure NeuralODE mode with Optax.

    Only the neural parameters are optimized. Mechanistic parameters are kept
    fixed at theta0 and are included only so downstream networkmodel export code
    can continue to consume a full theta vector.
    """
    if pinn_spec is None or pinn_spec.n_neural_params <= 0:
        raise ValueError("Optax NeuralODE solver requires a non-empty pinn_spec.")

    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)

    nn_slice = pinn_spec.nn_slice
    nn0 = theta0_j[nn_slice]
    nn_lower = xl_j[nn_slice]
    nn_upper = xu_j[nn_slice]

    # Full theta template: mechanistic parameters stay fixed.
    def assemble_theta(nn_params):
        return theta0_j.at[nn_slice].set(nn_params)

    def safe_objective_from_nn(nn_params):
        theta = assemble_theta(nn_params)
        value = problem.objective(theta)
        return jnp.nan_to_num(
            value,
            nan=jnp.asarray(1e12, dtype=jnp.float64),
            posinf=jnp.asarray(1e12, dtype=jnp.float64),
            neginf=jnp.asarray(1e12, dtype=jnp.float64),
        )

    opt = _make_optimizer(
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        weight_decay=weight_decay,
        optimizer=optimizer,
    )

    opt_state = opt.init(nn0)

    @jax.jit
    def step(nn_params, opt_state):
        value, grads = jax.value_and_grad(safe_objective_from_nn)(nn_params)

        grads = jnp.nan_to_num(
            grads,
            nan=jnp.asarray(0.0, dtype=jnp.float64),
            posinf=jnp.asarray(0.0, dtype=jnp.float64),
            neginf=jnp.asarray(0.0, dtype=jnp.float64),
        )

        updates, opt_state = opt.update(grads, opt_state, nn_params)
        nn_next = optax.apply_updates(nn_params, updates)

        # Project back into PINN bounds.
        nn_next = jnp.clip(nn_next, nn_lower, nn_upper)

        return nn_next, opt_state, value

    nn = nn0
    best_nn = nn0
    best_value = np.inf

    maxiter = int(maxiter)
    log_every = max(1, maxiter // 20)

    if logger is not None:
        logger.info(
            "[Optax-NeuralODE] Starting optimizer=%s lr=%g clip_norm=%g weight_decay=%g nn_params=%d",
            optimizer,
            learning_rate,
            clip_norm,
            weight_decay,
            int(nn0.size),
        )

    for i in range(maxiter):
        nn, opt_state, value = step(nn, opt_state)
        value_f = float(np.asarray(value))

        if np.isfinite(value_f) and value_f < best_value:
            best_value = value_f
            best_nn = nn

        if logger is not None and (i == 0 or (i + 1) % log_every == 0 or i + 1 == maxiter):
            logger.info("[Optax-NeuralODE] iter=%d/%d objective=%.6g best=%.6g", i + 1, maxiter, value_f, best_value)

    best_theta = assemble_theta(best_nn)
    best_theta_np = np.asarray(best_theta, dtype=np.float64)
    best_f = float(best_value)

    # Repopulate final loss breakdown for downstream reporting.
    try:
        objective_raw = getattr(problem, "objective_raw", getattr(problem, "_objective_raw", None))
        if objective_raw is not None:
            _, breakdown = objective_raw(best_theta_np)
            problem.final_loss_breakdown = {k: float(v) for k, v in breakdown.items()}
    except Exception as exc:
        if logger is not None:
            logger.warning("[Optax-NeuralODE] Could not repopulate loss breakdown: %s", exc)

    state = SimpleNamespace(
        optimizer="optax",
        optax_optimizer=optimizer,
        iter_num=maxiter,
        value=best_f,
        learning_rate=float(learning_rate),
        clip_norm=float(clip_norm),
        weight_decay=float(weight_decay),
        neural_slice=(int(nn_slice.start), int(nn_slice.stop)),
        n_neural_params=int(pinn_spec.n_neural_params),
    )

    return best_theta_np, state, best_f