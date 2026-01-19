"""
Configurations for PhoskinTime Global model
"""
from phoskintime_global.utils import load_config_toml

cfg = load_config_toml("phoskintime_global/config.toml")

def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "on"}
    return bool(x)

NORMALIZE_FC_STEADY = _as_bool(cfg.normalize_fc_steady)
USE_INITIAL_CONDITION_FROM_DATA = _as_bool(cfg.use_initial_condition_from_data)

TIME_POINTS_PROTEIN = cfg.time_points_prot
TIME_POINTS_RNA = cfg.time_points_rna
TIME_POINTS_PHOSPHO = cfg.time_points_phospho

BOUNDS_CONFIG = cfg.bounds_config

MODEL = 0 if cfg.model == "distributive" else (1 if cfg.model == "sequential" else 2)
USE_CUSTOM_SOLVER = cfg.use_custom_solver
ODE_ABS_TOL = cfg.ode_abs_tol
ODE_REL_TOL = cfg.ode_rel_tol
ODE_MAX_STEPS = cfg.ode_max_steps

LOSS_MODE = cfg.loss_mode
MAX_ITERATIONS = cfg.maximum_iterations
POPULATION_SIZE = cfg.population_size
SEED = cfg.seed

REGULARIZATION_RNA = cfg.regularization_rna
REGULARIZATION_LAMBDA = cfg.regularization_lambda
REGULARIZATION_PHOSPHO = cfg.regularization_phospho
REGULARIZATION_PROTEIN = cfg.regularization_protein

RESULTS_DIR = cfg.results_dir