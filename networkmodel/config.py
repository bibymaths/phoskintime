"""Load networkmodel settings from config.toml and expose typed module constants for paths, time grids, model selection, solver controls, optimization controls, regularization weights, inference options, sensitivity options, and metadata; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules."""
import os
from config_loader import load_config_toml

CONFIG_ENV_VAR = "PHOSKINTIME_NETWORKMODEL_CONFIG"
CONFIG_PATH = os.environ.get(CONFIG_ENV_VAR, "config.toml")

# cfg: PhosKinConfig loaded from the selected TOML. networkmodel.runner sets
# PHOSKINTIME_NETWORKMODEL_CONFIG after parsing --conf and before importing
# modules that rely on this compatibility constants surface.
if os.path.exists(CONFIG_PATH):
    cfg = load_config_toml(CONFIG_PATH)
else:
    raise FileNotFoundError(f"networkmodel config not found: {CONFIG_PATH}")


def _as_bool(x):
    """Convert a value to a boolean"""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "on"}
    return bool(x)


# KINASE_NET_FILE: str | Path; default cfg.kinase_net from networkmodel.kinase_net; controls kinase-substrate input path.
KINASE_NET_FILE = cfg.kinase_net
# TF_NET_FILE: str | Path; default cfg.tf_net from networkmodel.tf_net; controls transcription-factor network input path.
TF_NET_FILE = cfg.tf_net
# MS_DATA_FILE: str | Path; default cfg.ms_data from networkmodel.ms; controls protein mass-spectrometry input path.
MS_DATA_FILE = cfg.ms_data
# RNA_DATA_FILE: str | Path; default cfg.rna_data from networkmodel.rna; controls RNA input path.
RNA_DATA_FILE = cfg.rna_data
# PHOSPHO_DATA_FILE: str | Path | None; default cfg.phospho_data from networkmodel.phospho; controls phospho input path.
PHOSPHO_DATA_FILE = cfg.phospho_data

# KINOPT_RESULTS_FILE: str | Path; default cfg.kinopt_results from networkmodel.kinopt; controls kinase prior-result input path.
KINOPT_RESULTS_FILE = cfg.kinopt_results
# TFOPT_RESULTS_FILE: str | Path; default cfg.tfopt_results from networkmodel.tfopt; controls transcription-factor prior-result input path.
TFOPT_RESULTS_FILE = cfg.tfopt_results

# NORMALIZE_FC_STEADY: bool; default cfg.normalize_fc_steady; controls baseline fold-change normalization.
NORMALIZE_FC_STEADY = _as_bool(cfg.normalize_fc_steady)
# USE_INITIAL_CONDITION_FROM_DATA: bool; default cfg.use_initial_condition_from_data; controls data-derived initial states.
USE_INITIAL_CONDITION_FROM_DATA = _as_bool(cfg.use_initial_condition_from_data)

# TIME_POINTS_PROTEIN: np.ndarray; default cfg.time_points_prot; controls protein observation times in minutes.
TIME_POINTS_PROTEIN = cfg.time_points_prot
# TIME_POINTS_RNA: np.ndarray; default cfg.time_points_rna; controls RNA observation times in minutes.
TIME_POINTS_RNA = cfg.time_points_rna
# TIME_POINTS_PHOSPHO: np.ndarray; default cfg.time_points_phospho; controls phospho observation times in minutes.
TIME_POINTS_PHOSPHO = cfg.time_points_phospho

# BOUNDS_CONFIG: dict[str, tuple[float, float]]; default cfg.bounds_config; controls projected optimizer bounds by parameter group.
BOUNDS_CONFIG = cfg.bounds_config

# MODEL: int; default derived from cfg.model; controls internal topology code where 0 distributive, 1 sequential, 2 combinatorial, and 4 saturation.
MODEL = 0 if cfg.model == "distributive" else (
    1 if cfg.model == "sequential" else (2 if cfg.model == "combinatorial" else 4)
)

# ODE_ABS_TOL: float; default cfg.ode_abs_tol; controls Diffrax absolute tolerance.
ODE_ABS_TOL = cfg.ode_abs_tol
# ODE_REL_TOL: float; default cfg.ode_rel_tol; controls Diffrax relative tolerance.
ODE_REL_TOL = cfg.ode_rel_tol
# ODE_MAX_STEPS: int; default cfg.ode_max_steps; controls maximum Diffrax solver steps.
ODE_MAX_STEPS = cfg.ode_max_steps

# LOSS_MODE: int; default cfg.loss_mode; controls robust-loss selector.
LOSS_MODE = cfg.loss_mode
# MAX_ITERATIONS: int; default cfg.maximum_iterations; controls maximum ProjectedGradient iterations.
MAX_ITERATIONS = cfg.maximum_iterations
# SEED: int; default cfg.seed; controls random initialization and inference seeds.
SEED = cfg.seed
# CORES: int; default cfg.cores; controls requested worker count.
CORES = cfg.cores

# REGULARIZATION_RNA: float; default cfg.regularization_rna; controls RNA loss weight.
REGULARIZATION_RNA = cfg.regularization_rna
# REGULARIZATION_LAMBDA: float; default cfg.regularization_lambda; controls prior-adherence loss weight.
REGULARIZATION_LAMBDA = cfg.regularization_lambda
# REGULARIZATION_PHOSPHO: float; default cfg.regularization_phospho; controls phospho loss weight.
REGULARIZATION_PHOSPHO = cfg.regularization_phospho
# REGULARIZATION_PROTEIN: float; default cfg.regularization_protein; controls protein loss weight.
REGULARIZATION_PROTEIN = cfg.regularization_protein

# RESULTS_DIR: str | Path; default cfg.results_dir; controls output directory.
RESULTS_DIR = cfg.results_dir

# APP_NAME: str; default "Phoskintime-Global"; controls display metadata.
APP_NAME = getattr(cfg, "app_name", "Phoskintime-Global")
# VERSION: str; default "0.1.0"; controls display metadata.
VERSION = getattr(cfg, "version", "0.1.0")
# PARENT_PACKAGE: str; default "phoskintime"; controls display metadata.
PARENT_PACKAGE = getattr(cfg, "parent_package", "phoskintime")
# CITATION: str; default ""; controls display metadata.
CITATION = getattr(cfg, "citation", "")
# DOI: str; default ""; controls display metadata.
DOI = getattr(cfg, "doi", "")
# GITHUB_URL: str; default ""; controls display metadata.
GITHUB_URL = getattr(cfg, "github_url", "")
# DOCS_URL: str; default ""; controls display metadata.
DOCS_URL = getattr(cfg, "docs_url", "")

# HYPERPARAM_SCAN: bool; default False; controls compatibility hyperparameter scan execution.
HYPERPARAM_SCAN = _as_bool(getattr(cfg, "hyperparam_scan", False))

# N_STARTS: int; default 1; controls multistart optimization count.
N_STARTS = int(getattr(cfg, "n_starts", 1))
# PROFILE_LIKELIHOOD: bool; default False; controls profile-likelihood execution.
PROFILE_LIKELIHOOD = _as_bool(getattr(cfg, "profile_likelihood", False))
# PROFILE_INDICES: str; default ""; controls comma-separated raw parameter indices for profiling.
PROFILE_INDICES = str(getattr(cfg, "profile_indices", ""))
# PROFILE_GRID_SIZE: int; default 10; controls grid count per profiled parameter.
PROFILE_GRID_SIZE = int(getattr(cfg, "profile_grid_size", 10))
# POSTERIOR_SAMPLING: bool; default False; controls optional NumPyro posterior sampling.
POSTERIOR_SAMPLING = _as_bool(getattr(cfg, "posterior_sampling", False))
# POSTERIOR_NUM_WARMUP: int; default 20; controls NumPyro warmup draw count.
POSTERIOR_NUM_WARMUP = int(getattr(cfg, "posterior_num_warmup", 20))
# POSTERIOR_NUM_SAMPLES: int; default 30; controls NumPyro posterior draw count.
POSTERIOR_NUM_SAMPLES = int(getattr(cfg, "posterior_num_samples", 30))

# SCALING_METHOD: str; default "none"; controls raw-data scaling before loss preparation.
SCALING_METHOD = getattr(cfg, "scaling_method", "none")
# WEIGHTING_METHOD_PROTEIN: str; default "uniform"; controls protein time-point weights.
WEIGHTING_METHOD_PROTEIN = getattr(cfg, "weighting_method_protein", "uniform")
# WEIGHTING_METHOD_RNA: str; default "uniform"; controls RNA time-point weights.
WEIGHTING_METHOD_RNA = getattr(cfg, "weighting_method_rna", "uniform")
# WEIGHTING_METHOD_PHOSPHO: str; default "uniform"; controls phospho time-point weights.
WEIGHTING_METHOD_PHOSPHO = getattr(cfg, "weighting_method_phospho", "uniform")

# SENSITIVITY_ANALYSIS: bool; default False; controls sensitivity-analysis execution.
SENSITIVITY_ANALYSIS = _as_bool(getattr(cfg, "sensitivity_analysis", False))
# SENSITIVITY_PERTURBATION: float; default 0.2; controls relative parameter perturbation size.
SENSITIVITY_PERTURBATION = getattr(cfg, "sensitivity_perturbation", 0.2)
# SENSITIVITY_TRAJECTORIES: int; default 1000; controls number of sensitivity trajectories.
SENSITIVITY_TRAJECTORIES = getattr(cfg, "sensitivity_trajectories", 1000)
# SENSITIVITY_LEVELS: int; default 400; controls number of sensitivity grid levels.
SENSITIVITY_LEVELS = getattr(cfg, "sensitivity_levels", 400)
# SENSITIVITY_TOP_CURVES: int; default 50; controls number of plotted sensitivity curves.
SENSITIVITY_TOP_CURVES = getattr(cfg, "sensitivity_top_curves", 50)
# SENSITIVITY_METRIC: str; default "total_signal"; controls scalar metric for sensitivity ranking.
SENSITIVITY_METRIC = getattr(cfg, "sensitivity_metric", "total_signal")

# AVAILABLE_MODELS: tuple[str, ...]; default empty tuple; controls logged model metadata.
AVAILABLE_MODELS = getattr(cfg, "available_models", ())

# ENABLE_HYPEREDGE_PREPROCESSING: bool; default False; enables optional pure-JAX hyperedge/network preprocessing before model construction.
ENABLE_HYPEREDGE_PREPROCESSING = _as_bool(getattr(cfg, "enable_hyperedge_preprocessing", False))
# HYPEREDGE_PREPROCESSING_OUTPUT_SUBDIR: str; output subdirectory for preprocessing tables/plots under standard result folders.
HYPEREDGE_PREPROCESSING_OUTPUT_SUBDIR = str(getattr(cfg, "hyperedge_preprocessing_output_subdir", "networkpruning"))
# HYPEREDGE_DISCOVERY_THRESHOLD: float; minimum score threshold used for discovered hyperedges.
HYPEREDGE_DISCOVERY_THRESHOLD = float(getattr(cfg, "hyperedge_discovery_threshold", 0.0))
# HYPEREDGE_PRUNING_THRESHOLD: float; minimum score threshold used when pruning retained triplets.
HYPEREDGE_PRUNING_THRESHOLD = float(getattr(cfg, "hyperedge_pruning_threshold", 0.0))
# HYPEREDGE_MOTIF_DETECTION: bool; controls optional motif detection during preprocessing.
HYPEREDGE_MOTIF_DETECTION = _as_bool(getattr(cfg, "hyperedge_motif_detection", True))
# HYPEREDGE_SPARSE_TENSOR_EXPORT: bool; controls sparse tensor CSV/NPZ export.
HYPEREDGE_SPARSE_TENSOR_EXPORT = _as_bool(getattr(cfg, "hyperedge_sparse_tensor_export", True))
# HYPEREDGE_IDENTIFIABILITY_PREPROCESSING: bool; controls optional identifiability diagnostics.
HYPEREDGE_IDENTIFIABILITY_PREPROCESSING = _as_bool(getattr(cfg, "hyperedge_identifiability_preprocessing", True))
# HYPEREDGE_MAX_TRIPLETS: int | None; optional maximum retained triplets for memory safety.
HYPEREDGE_MAX_TRIPLETS = getattr(cfg, "hyperedge_max_triplets", None)
# HYPEREDGE_BATCH_SIZE: int; batch size used by preprocessing kernels/adapters.
HYPEREDGE_BATCH_SIZE = int(getattr(cfg, "hyperedge_batch_size", 65536))
# HYPEREDGE_PLOT_GENERATION: bool; controls preprocessing diagnostic plot generation.
HYPEREDGE_PLOT_GENERATION = _as_bool(getattr(cfg, "hyperedge_plot_generation", True))
# HYPEREDGE_CSV_EXPORT: bool; controls preprocessing CSV/JSON table export.
HYPEREDGE_CSV_EXPORT = _as_bool(getattr(cfg, "hyperedge_csv_export", True))

# PINN / NeuralODE options.
ENABLE_PINN = _as_bool(getattr(cfg, "enable_pinn", False))
PINN_MODE = str(getattr(cfg, "pinn_mode", "off"))
PINN_HIDDEN_SIZE = int(getattr(cfg, "pinn_hidden_size", 32))
PINN_DEPTH = int(getattr(cfg, "pinn_depth", 2))
PINN_ACTIVATION = str(getattr(cfg, "pinn_activation", "tanh"))
PINN_OUTPUT_SCALE = float(getattr(cfg, "pinn_output_scale", 1e-2))
PINN_WEIGHT_BOUND = float(getattr(cfg, "pinn_weight_bound", 0.25))
PINN_L2_REGULARIZATION = float(getattr(cfg, "pinn_l2_regularization", 1e-6))
PINN_T_SCALE = float(getattr(cfg, "pinn_t_scale", 1.0))
PINN_Y_SCALE = float(getattr(cfg, "pinn_y_scale", 1.0))
PINN_SEED = int(getattr(cfg, "pinn_seed", SEED))