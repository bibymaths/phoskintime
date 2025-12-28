"""
Configurations for PhoskinTime Global model
"""
from phoskintime_global.utils import load_config_toml

cfg = load_config_toml("config.toml")

TIME_POINTS_PROTEIN = cfg.time_points_prot
TIME_POINTS_RNA = cfg.time_points_rna
TIME_POINTS_PHOSPHO = cfg.time_points_phospho
BOUNDS_CONFIG = cfg.bounds_config
MODEL = 0 if cfg.model == "distributive" else (1 if cfg.model == "sequential" else 2)
ODE_ABS_TOL = cfg.ode_abs_tol
ODE_REL_TOL = cfg.ode_rel_tol
ODE_MAX_STEPS = cfg.ode_max_steps