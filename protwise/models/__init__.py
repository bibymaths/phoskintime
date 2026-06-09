import importlib
from config.constants import ODE_MODEL, _normalize_model_name

# Import the ODE model module dynamically based on the ODE_MODEL constant
try:
    model_module = importlib.import_module(f'protwise.models.{_normalize_model_name(ODE_MODEL)}')
except ModuleNotFoundError as e:
    raise ImportError(f"Cannot import model module 'protwise.models.{_normalize_model_name(ODE_MODEL)}'") from e

# Import the functions from the dynamically loaded module to the current namespace
# Solve the ODE using the imported model
solve_ode = model_module.solve_ode
