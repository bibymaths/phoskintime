"""
Configurations for PhoskinTime Global model
"""
import os
from global_model.utils import load_config_toml

# Load configuration relative to CWD (Project Root)
if os.path.exists("config.toml"):
    cfg = load_config_toml("config.toml")
else:
    # Fallback or useful error if running from wrong dir
    raise FileNotFoundError("config.toml not found in current directory.")


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "on"}
    return bool(x)


# --- Input Files ---
KINASE_NET_FILE = cfg.kinase_net
TF_NET_FILE = cfg.tf_net
MS_DATA_FILE = cfg.ms_data
RNA_DATA_FILE = cfg.rna_data
PHOSPHO_DATA_FILE = cfg.phospho_data
KINOPT_RESULTS_FILE = cfg.kinopt_results
TFOPT_RESULTS_FILE = cfg.tfopt_results

# --- Data Processing ---
NORMALIZE_FC_STEADY = _as_bool(cfg.normalize_fc_steady)
USE_INITIAL_CONDITION_FROM_DATA = _as_bool(cfg.use_initial_condition_from_data)

TIME_POINTS_PROTEIN = cfg.time_points_prot
TIME_POINTS_RNA = cfg.time_points_rna
TIME_POINTS_PHOSPHO = cfg.time_points_phospho

# --- Model & Solver ---
BOUNDS_CONFIG = cfg.bounds_config

# Map string model name to integer ID
# 0: Distributive, 1: Sequential, 2: Combinatorial, 4: Saturating
MODEL = 0 if cfg.model == "distributive" else (
    1 if cfg.model == "sequential" else (2 if cfg.model == "combinatorial" else 4)
)

USE_CUSTOM_SOLVER = _as_bool(cfg.use_custom_solver)
ODE_ABS_TOL = cfg.ode_abs_tol
ODE_REL_TOL = cfg.ode_rel_tol
ODE_MAX_STEPS = cfg.ode_max_steps

# --- Optimization Settings ---
LOSS_MODE = cfg.loss_mode
MAX_ITERATIONS = cfg.maximum_iterations
POPULATION_SIZE = cfg.population_size
SEED = cfg.seed
CORES = cfg.cores
REFINE = _as_bool(cfg.refine)
NUM_REFINE = cfg.num_refine

# --- Regularization ---
REGULARIZATION_RNA = cfg.regularization_rna
REGULARIZATION_LAMBDA = cfg.regularization_lambda
REGULARIZATION_PHOSPHO = cfg.regularization_phospho
REGULARIZATION_PROTEIN = cfg.regularization_protein

# --- Output ---
RESULTS_DIR = cfg.results_dir

# --- App metadata ---
APP_NAME = getattr(cfg, "app_name", "Phoskintime-Global")
VERSION = getattr(cfg, "version", "0.1.0")
PARENT_PACKAGE = getattr(cfg, "parent_package", "phoskintime")
CITATION = getattr(cfg, "citation", "")
DOI = getattr(cfg, "doi", "")
GITHUB_URL = getattr(cfg, "github_url", "")
DOCS_URL = getattr(cfg, "docs_url", "")

# --- Hyperparameter scan ---
HYPERPARAM_SCAN = getattr(cfg, "hyperparam_scan", False)

# --- Optimizer selection ---
OPTIMIZER = getattr(cfg, "optimizer", "pymoo")  # "optuna" or "pymoo"

# --- Optuna settings ---
STUDY_NAME = getattr(cfg, "study_name", "")
SAMPLER = getattr(cfg, "sampler", "TPESampler")
PRUNER = getattr(cfg, "pruner", "MedianPruner")
N_TRIALS = getattr(cfg, "n_trials", 0)

# --- Data scaling / weighting ---
SCALING_METHOD = getattr(cfg, "scaling_method", "none")
WEIGHTING_METHOD_PROTEIN = getattr(cfg, "weighting_method_protein", "uniform")
WEIGHTING_METHOD_RNA = getattr(cfg, "weighting_method_rna", "uniform")
WEIGHTING_METHOD_PHOSPHO = getattr(cfg, "weighting_method_phospho", "uniform")

# --- Sensitivity analysis ---
SENSITIVITY_ANALYSIS = _as_bool(getattr(cfg, "sensitivity_analysis", False))
SENSITIVITY_PERTURBATION = getattr(cfg, "sensitivity_perturbation", 0.2)
SENSITIVITY_TRAJECTORIES = getattr(cfg, "sensitivity_trajectories", 1000)
SENSITIVITY_LEVELS = getattr(cfg, "sensitivity_levels", 400)
SENSITIVITY_TOP_CURVES = getattr(cfg, "sensitivity_top_curves", 50)
SENSITIVITY_METRIC = getattr(cfg, "sensitivity_metric", "total_signal")

# --- Models list ---
AVAILABLE_MODELS = getattr(cfg, "available_models", ())
