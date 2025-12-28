"""
Configurations for PhoskinTime Global model
"""
from phoskintime_global.utils import load_config_toml

cfg = load_config_toml("../config.toml")

TIME_POINTS_PROTEIN = cfg.time_points_prot
TIME_POINTS_RNA = cfg.time_points_rna
TIME_POINTS_PHOSPHO = cfg.time_points_phospho
BOUNDS_CONFIG = cfg.bounds_config
MODEL = cfg.model