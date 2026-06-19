"""NeuralODE / hybrid mechanistic-NeuralODE extensions for networkmodel."""

from networkmodel.pinn.config import PinnConfig
from networkmodel.pinn.pack import PinnParameterSpec, extend_theta_with_pinn
from networkmodel.pinn.objective import make_pinn_objective
from networkmodel.pinn.solver import should_use_pinn_solver, solve_pinn_problem

__all__ = [
    "PinnConfig",
    "PinnParameterSpec",
    "extend_theta_with_pinn",
    "make_pinn_objective",
    "solve_pinn_problem",
    "should_use_pinn_solver"
]