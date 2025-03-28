
import importlib
from config.constants import ODE_MODEL

try:
    model_module = importlib.import_module(f'models.{ODE_MODEL}')
except ModuleNotFoundError as e:
    raise ImportError(f"Cannot import model module 'models.{ODE_MODEL}'") from e

solve_ode = model_module.solve_ode
