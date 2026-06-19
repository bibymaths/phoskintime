"""Configuration helpers for optional networkmodel PINN / NeuralODE objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PinnMode = Literal["off", "hybrid", "neuralode"]


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _mode(x: Any) -> PinnMode:
    raw = str(x or "off").strip().lower().replace("-", "_")
    aliases = {
        "false": "off",
        "none": "off",
        "disabled": "off",
        "mechanistic": "off",
        "mechanistic_only": "off",
        "hybrid_neuralode": "hybrid",
        "mechanistic_neuralode": "hybrid",
        "mechanistic_plus_neuralode": "hybrid",
        "hybrid": "hybrid",
        "neuralode": "neuralode",
        "neural_ode": "neuralode",
        "pure_neuralode": "neuralode",
        "pure": "neuralode",
    }
    out = aliases.get(raw, raw)
    if out not in {"off", "hybrid", "neuralode"}:
        raise ValueError(
            "Invalid pinn_mode. Expected one of: "
            "'off', 'hybrid', 'neuralode'. "
            f"Got {x!r}."
        )
    return out  # type: ignore[return-value]


@dataclass(frozen=True)
class PinnConfig:
    """Runtime configuration for optional hybrid/PINN NeuralODE modeling."""

    enabled: bool = False
    mode: PinnMode = "off"

    # Equinox MLP architecture.
    hidden_size: int = 32
    depth: int = 2
    activation: str = "tanh"

    # Neural RHS scaling and parameter bounds.
    output_scale: float = 1e-2
    weight_bound: float = 0.25
    l2_regularization: float = 1e-6

    # Input normalization.
    t_scale: float = 1.0
    y_scale: float = 1.0

    # Reproducibility.
    seed: int = 0

    @staticmethod
    def from_args(args: Any) -> "PinnConfig":
        enabled = _as_bool(getattr(args, "enable_pinn", False))
        mode = _mode(getattr(args, "pinn_mode", "off"))

        if enabled and mode == "off":
            mode = "hybrid"

        hidden_size = int(getattr(args, "pinn_hidden_size", 32))
        depth = int(getattr(args, "pinn_depth", 2))
        output_scale = float(getattr(args, "pinn_output_scale", 1e-2))
        weight_bound = float(getattr(args, "pinn_weight_bound", 0.25))
        l2 = float(getattr(args, "pinn_l2_regularization", 1e-6))
        t_scale = float(getattr(args, "pinn_t_scale", 1.0))
        y_scale = float(getattr(args, "pinn_y_scale", 1.0))

        if hidden_size <= 0:
            raise ValueError("pinn_hidden_size must be positive.")
        if depth < 0:
            raise ValueError("pinn_depth must be non-negative.")
        if output_scale < 0:
            raise ValueError("pinn_output_scale must be non-negative.")
        if weight_bound <= 0:
            raise ValueError("pinn_weight_bound must be positive.")
        if l2 < 0:
            raise ValueError("pinn_l2_regularization must be non-negative.")
        if t_scale <= 0:
            raise ValueError("pinn_t_scale must be positive.")
        if y_scale <= 0:
            raise ValueError("pinn_y_scale must be positive.")

        return PinnConfig(
            enabled=bool(enabled),
            mode=mode,
            hidden_size=hidden_size,
            depth=depth,
            activation=str(getattr(args, "pinn_activation", "tanh")),
            output_scale=output_scale,
            weight_bound=weight_bound,
            l2_regularization=l2,
            t_scale=t_scale,
            y_scale=y_scale,
            seed=int(getattr(args, "pinn_seed", getattr(args, "seed", 0))),
        )