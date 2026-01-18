import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
from pathlib import Path

from config.helpers import *
from config_loader import load

# -------------------------------------------------------------------------------------------------
# Load configuration (ODE section) from config.toml
# This module must not import tfopt/kinopt constants to avoid hard coupling / circular dependencies.
# -------------------------------------------------------------------------------------------------
_CFG = load("local", "ode")  # mode is optional for ODE, but keeps a consistent API
_ROOT = Path(_CFG["_root"])
_PATHS = _CFG.get("_paths", {}) or {}

# Flag to indicate if the code is in development mode.
DEV_TEST = bool(_CFG.get("dev_test", False))


########################################################################################################################
# GLOBAL CONSTANTS
########################################################################################################################

# Select the ODE model for phosphorylation kinetics: distmod | succmod | randmod
ODE_MODEL = str(_CFG.get("model", "randmod"))

# Upper bounds
_bounds = _CFG.get("bounds", {}) or {}
UB_mRNA_prod = float(_bounds.get("mRNA_prod", 20))
UB_mRNA_deg = float(_bounds.get("mRNA_deg", 20))
UB_Protein_prod = float(_bounds.get("protein_prod", 20))
UB_Protein_deg = float(_bounds.get("protein_deg", 20))
UB_Phospho_prod = float(_bounds.get("phospho_prod", 20))
UB_Phospho_deg = float(_bounds.get("phospho_deg", 20))

# Bootstraps
_boot = _CFG.get("bootstrap", {}) or {}
BOOTSTRAPS = int(_boot.get("n", 0))

# Sensitivity analysis configuration (Morris)
_sens = _CFG.get("sensitivity", {}) or {}
SENSITIVITY_ANALYSIS = bool(_sens.get("enabled", True))
PERTURBATIONS_VALUE = float(_sens.get("perturbation", 0.5))

_morris = _sens.get("morris", {}) or {}
NUM_TRAJECTORIES = int(_morris.get("num_trajectories", 1000))
PARAMETER_SPACE = int(_morris.get("num_levels", 400))

# Confidence interval level (keep as code-level default unless you add it to TOML explicitly)
ALPHA_CI = float(_CFG.get("alpha_ci", 0.95))

# Time grids
_time = _CFG.get("time", {}) or {}
TIME_POINTS = np.asarray(
    _time.get(
        "protein",
        [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    ),
    dtype=float,
)
TIME_POINTS_RNA = np.asarray(
    _time.get(
        "rna",
        [4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    ),
    dtype=float,
)

# Fit controls
_fit = _CFG.get("fit", {}) or {}
NORMALIZE_MODEL_OUTPUT = bool(_fit.get("normalize_model_output", False))
USE_CUSTOM_WEIGHTS = bool(_fit.get("use_custom_weights", False))
USE_REGULARIZATION = bool(_fit.get("use_regularization", True))

# Composite weights
_w = _fit.get("composite_weights", {}) or {}
ALPHA_WEIGHT = float(_w.get("rmse", 1.0))
BETA_WEIGHT = float(_w.get("mae", 1.0))
GAMMA_WEIGHT = float(_w.get("var", 1.0))
DELTA_WEIGHT = float(_w.get("mse", 1.0))
MU_WEIGHT = float(_w.get("l2", 1.0))


########################################################################################################################
# INTERNAL CONSTANTS
########################################################################################################################

model_names = {
    "distmod": "Distributive",
    "succmod": "Successive",
    "randmod": "Random",
}
model_type = model_names.get(ODE_MODEL, "Unknown")

Y_METRIC_DESCRIPTIONS = {
    "total_signal": "Sum of all mRNA and site values across time (captures overall signal magnitude).",
    "mean_activity": "Mean of all mRNA and site values across time (captures average activity).",
    "variance": "Variance of all mRNA and site values across time (captures temporal/spatial variability).",
    "dynamics": "Sum of squared successive differences (captures how dynamic the signal is).",
    "l2_norm": "Euclidean norm of the flattened values (captures overall magnitude in L2 sense).",
}
Y_METRIC = str(_CFG.get("y_metric", "total_signal"))


########################################################################################################################
# PATHS, DIRECTORIES, AND FILES (config-driven)
########################################################################################################################

# Global project paths (from [paths] in TOML)
DATA_DIR = _ROOT / _PATHS.get("data_dir", "data")
RESULTS_DIR = _ROOT / _PATHS.get("results_dir", "results")
LOGS_DIR = _ROOT / _PATHS.get("logs_dir", "results/logs")

# ODE-specific output naming (optional)
_out = _CFG.get("output", {}) or {}
_out_dir_name = str(_out.get("out_dir_name") or "").strip()
_out_xlsx_name = str(_out.get("out_xlsx_name") or "").strip()

# Default behavior: <results_dir>/<ModelType>_results/
OUT_DIR = RESULTS_DIR / (_out_dir_name or f"{model_type}_results")
OUT_RESULTS_DIR = OUT_DIR / (_out_xlsx_name or f"{model_type}_results.xlsx")
LOG_DIR = LOGS_DIR / f"{model_type}_logs"

# Inputs
_inputs = _CFG.get("inputs", {}) or {}

def _req_path(key: str) -> Path:
    v = _inputs.get(key)
    if not v:
        raise KeyError(f"[ode.inputs] is missing required key '{key}' in config.toml")
    return _ROOT / str(v)

# These MUST be explicit in [ode.inputs] (no tfopt imports)
INPUT_EXCEL_PROTEIN = _req_path("protein_excel")
INPUT_EXCEL_PSITE = _req_path("psite_excel")
INPUT_EXCEL_RNA = _req_path("rna_excel")

# Ensure dirs exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


########################################################################################################################
# PLOTTING STYLE CONFIGURATION
########################################################################################################################

_plot = _CFG.get("plot", {}) or {}
PERTURBATIONS_TRACE_OPACITY = float(_plot.get("perturb_trace_opacity", 0.02))

COLOR_PALETTE = [mcolors.to_hex(plt.get_cmap("tab20")(i)) for i in range(0, 20, 2)]
available_markers = [
    m for m in mmarkers.MarkerStyle.markers
    if isinstance(m, str) and m not in {".", ",", " "}
]


########################################################################################################################
# MODEL-SPECIFIC LABEL HELPERS
########################################################################################################################

if ODE_MODEL == "randmod":
    get_param_names = get_param_names_rand
    generate_labels = generate_labels_rand
else:
    get_param_names = get_param_names_ds
    generate_labels = generate_labels_ds
