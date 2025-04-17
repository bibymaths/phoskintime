
import importlib
from config.constants import ODE_MODEL

# Import the ODE model module dynamically based on the ODE_MODEL constant
try:
    model_module = importlib.import_module(f'models.{ODE_MODEL}')
except ModuleNotFoundError as e:
    raise ImportError(f"Cannot import model module 'models.{ODE_MODEL}'") from e

# Import the functions from the dynamically loaded module to the current namespace
# Solve the ODE using the imported model
solve_ode = model_module.solve_ode
