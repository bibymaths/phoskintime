"""
Configurations for PhoskinTime Global model.

This module serves as the central configuration hub for the application. It loads
settings from a `config.toml` file located in the project root and exports them
as global constants. These constants control every aspect of the simulation pipeline,
including:

1.  **I/O Paths**: Locations of interaction networks and experimental data.
2.  **Model Topology**: Selection of the kinetic model structure (Distributive vs. Sequential vs. Combinatorial).
3.  **Solver Settings**: Tolerances and step sizes for the ODE integrator.
4.  **Optimization**: Hyperparameters for the parameter estimation loop (Evolutionary Strategy).
5.  **Regularization**: Penalty terms to prevent overfitting.
6.  **Metadata**: Versioning and citation information.
"""
import os
from config_loader import load_config_toml

# Load configuration relative to CWD (Project Root)
if os.path.exists("config.toml"):
    cfg = load_config_toml("config.toml")
else:
    # Fallback or useful error if running from wrong dir
    raise FileNotFoundError("config.toml not found in current directory.")


def _as_bool(x):
    """
    Safely converts various input types into a boolean.
    Handles strings ('true', 'yes', '1'), integers, and native booleans.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "on"}
    return bool(x)


# --- Input Files ---
# Paths to the network structure (CSVs) and experimental data (CSVs)
KINASE_NET_FILE = cfg.kinase_net
TF_NET_FILE = cfg.tf_net
MS_DATA_FILE = cfg.ms_data
RNA_DATA_FILE = cfg.rna_data
PHOSPHO_DATA_FILE = cfg.phospho_data

# Results from previous local optimization steps (if using a hierarchical approach)
KINOPT_RESULTS_FILE = cfg.kinopt_results
TFOPT_RESULTS_FILE = cfg.tfopt_results

# --- Data Processing ---
# Normalize fold-change data to steady-state baselines
NORMALIZE_FC_STEADY = _as_bool(cfg.normalize_fc_steady)
# If True, the simulation starts with state values derived from data at t=0
USE_INITIAL_CONDITION_FROM_DATA = _as_bool(cfg.use_initial_condition_from_data)

# Time grids for different data modalities
TIME_POINTS_PROTEIN = cfg.time_points_prot
TIME_POINTS_RNA = cfg.time_points_rna
TIME_POINTS_PHOSPHO = cfg.time_points_phospho

# --- Model & Solver ---
# Configuration for parameter boundaries (lower/upper limits)
BOUNDS_CONFIG = cfg.bounds_config

# Map string model name to integer ID for internal logic
# 0: Distributive (Independent binding)
# 1: Sequential (Ordered binding)
# 2: Combinatorial (All permutations)
# 4: Saturating (Michaelis-Menten approximation)
MODEL = 0 if cfg.model == "distributive" else (
    1 if cfg.model == "sequential" else (2 if cfg.model == "combinatorial" else 4)
)

# Solver precision and custom backend selection
USE_CUSTOM_SOLVER = _as_bool(cfg.use_custom_solver)
ODE_ABS_TOL = cfg.ode_abs_tol
ODE_REL_TOL = cfg.ode_rel_tol
ODE_MAX_STEPS = cfg.ode_max_steps

# --- Optimization Settings ---
LOSS_MODE = cfg.loss_mode
MAX_ITERATIONS = cfg.maximum_iterations
POPULATION_SIZE = cfg.population_size  # Number of particles/individuals in the optimizer
SEED = cfg.seed
CORES = cfg.cores  # Multiprocessing core count
REFINE = _as_bool(cfg.refine)  # Whether to run a local optimization polish after global search
NUM_REFINE = cfg.num_refine  # Number of iterations for the refinement step

# --- Regularization ---
# Penalties to enforce biological constraints (e.g., sparsity, smooth trajectories)
REGULARIZATION_RNA = cfg.regularization_rna
REGULARIZATION_LAMBDA = cfg.regularization_lambda
REGULARIZATION_PHOSPHO = cfg.regularization_phospho
REGULARIZATION_PROTEIN = cfg.regularization_protein

# --- Output ---
RESULTS_DIR = cfg.results_dir

# --- App metadata ---
# Information for logging, headers, and reproducibility
APP_NAME = getattr(cfg, "app_name", "Phoskintime-Global")
VERSION = getattr(cfg, "version", "0.1.0")
PARENT_PACKAGE = getattr(cfg, "parent_package", "phoskintime")
CITATION = getattr(cfg, "citation", "")
DOI = getattr(cfg, "doi", "")
GITHUB_URL = getattr(cfg, "github_url", "")
DOCS_URL = getattr(cfg, "docs_url", "")

# --- Hyperparameter scan ---
# Flag to enable grid/random search over model hyperparameters (e.g., penalties)
HYPERPARAM_SCAN = getattr(cfg, "hyperparam_scan", False)

# --- Optimizer selection ---
# Supports different backends (e.g., 'pymoo' for genetic algos, 'optuna' for TPE)
OPTIMIZER = getattr(cfg, "optimizer", "pymoo")  # "optuna" or "pymoo"

# --- Optuna settings ---
# Specific settings if OPTIMIZER == 'optuna'
STUDY_NAME = getattr(cfg, "study_name", "")
SAMPLER = getattr(cfg, "sampler", "TPESampler")
PRUNER = getattr(cfg, "pruner", "MedianPruner")
N_TRIALS = getattr(cfg, "n_trials", 0)

# --- Data scaling / weighting ---
# Methods to balance the influence of different datasets (RNA vs Protein vs Phospho)
SCALING_METHOD = getattr(cfg, "scaling_method", "none")
WEIGHTING_METHOD_PROTEIN = getattr(cfg, "weighting_method_protein", "uniform")
WEIGHTING_METHOD_RNA = getattr(cfg, "weighting_method_rna", "uniform")
WEIGHTING_METHOD_PHOSPHO = getattr(cfg, "weighting_method_phospho", "uniform")

# --- Sensitivity analysis ---
# Post-optimization analysis to determine which parameters affect the output most
SENSITIVITY_ANALYSIS = _as_bool(getattr(cfg, "sensitivity_analysis", False))
SENSITIVITY_PERTURBATION = getattr(cfg, "sensitivity_perturbation", 0.2)
SENSITIVITY_TRAJECTORIES = getattr(cfg, "sensitivity_trajectories", 1000)
SENSITIVITY_LEVELS = getattr(cfg, "sensitivity_levels", 400)
SENSITIVITY_TOP_CURVES = getattr(cfg, "sensitivity_top_curves", 50)
SENSITIVITY_METRIC = getattr(cfg, "sensitivity_metric", "total_signal")

# --- Models list ---
AVAILABLE_MODELS = getattr(cfg, "available_models", ())
