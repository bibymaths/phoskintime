import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
from pathlib import Path
from config.helpers import * 
from tfopt.local.config.constants import INPUT1

# Flag to indicate if the code is in development mode.
DEV_TEST = True

########################################################################################################################
# GLOBAL CONSTANTS
# The following constants are used throughout the project.
# They define various parameters and settings for the optimization process,
# including the ODE model type, estimation strategy, time points, and more.
# These constants are used to configure the behavior of the optimization process,
# including the choice of ODE model, estimation strategy, and other settings.
########################################################################################################################
# Select the ODE model for phosphorylation kinetics.
# Options:
# 'distmod' : Distributive model (phosphorylation events occur independently).
# 'succmod' : Successive model (phosphorylation events occur in a fixed order).
# 'randmod' : Random model (phosphorylation events occur randomly).
ODE_MODEL = 'randmod'
# Upper bounds for mRNA production and degradation rates
UB_mRNA_prod = 20
UB_mRNA_deg = 20
# Upper bounds for protein production and degradation rates
UB_Protein_prod = 20
UB_Protein_deg = 20
# Upper bounds for phosphorylation sites
UB_Phospho_prod = 20
UB_Phospho_deg = 20
# Select the number of bootstrap iterations for parameter estimation.
# This parameter determines how many times the parameter estimation process
# will be repeated with Gaussian noise (mean=0, std=0.05) to simulate measurement
# variability for bootstrapping
BOOTSTRAPS = 0
# Trajectories for profiling
# The number of trajectories to be generated for the Morris method.
# This parameter is crucial for the Morris method, which requires a sufficient number of trajectories
# to accurately sample the parameter space and compute sensitivity indices.
# A higher number of trajectories can lead to more reliable results,
# but it also increases computational time
# The number of trajectories to be generated for the Morris method.
# This parameter is crucial for accurately sampling the parameter space and computing sensitivity indices.
# A higher number of trajectories (e.g., 10,000) provides more reliable results but increases computational cost.
NUM_TRAJECTORIES = 1000
# Spread of parameters (has to be even number) -> SALib.morris()
# The number of intervals to divide the parameter space for the Morris method.
# This parameter determines how finely the parameter space is sampled.
# Each parameter will be divided into this number of intervals,
# and the Morris method will sample points within these intervals.
PARAMETER_SPACE = 400
# Fractional range around each parameter value for sensitivity analysis bounds.
# Lower bound = value * (1 - PERTURBATIONS_VALUE)
# Upper bound = value * (1 + PERTURBATIONS_VALUE)
PERTURBATIONS_VALUE = 0.5  # 20% perturbation
# ALPHA_CI: Confidence level for computing confidence intervals for parameter identifiability.
# For example, an ALPHA_CI of 0.95 indicates that the model will compute 95% confidence intervals.
# This corresponds to a significance level of 1 - ALPHA_CI (i.e., 0.05) when determining the critical t-value.
ALPHA_CI = 0.95
# TIME_POINTS:
# A numpy array representing the discrete time points (in minutes) obtained from experimental MS data.
TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])
# TIME_POINTS_RNA:
# A numpy array representing the discrete time points (in minutes) for RNA data.
# These time points are used for RNA data analysis and are different from the MS data time points.
TIME_POINTS_RNA = np.array([4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])
# Whether to normalize model output to match fold change (FC) data
# ----------------------------------------------------------------
# Set to True when experimental data is provided in fold change format
# (i.e., values are already normalized relative to the baseline time point, typically t=0).
#
# When enabled, model outputs Y(t) will be divided by Y(t0) for each species:
#     FC_model(t) = Y(t) / Y(t0)
#
# This ensures the model output is in the same scale and units as the FC data.
# If False, raw FC values will be used directly
# (only valid if data is also in absolute units).
# IMPORTANT: Set to True if your time series data represents relative changes.
NORMALIZE_MODEL_OUTPUT = False
# Flag to use custom weights for parameter estimation.
# If True, the function will apply custom weights to the data points
# based on their importance or reliability.
# If False, the function will use weights from uncertainties from data.
USE_CUSTOM_WEIGHTS = False
# Controls whether sensitivity analysis is performed
# during the pipeline run.
# If True:
#   - Performs global sensitivity analysis using the Morris method.
#   - Samples the parameter space, solves the ODE system for each sample,
#     and calculates sensitivity indices (mu*, sigma).
#   - Generates multiple plots (bar, scatter, radial, CDF, pie charts)
#     to visualize parameter importance and interactions.
#
# If False:
#   - Skips sensitivity analysis entirely to save computation time.
#   - Useful during development, debugging, or when only
#     parameter estimation is needed.
SENSITIVITY_ANALYSIS = True
# Enables (True) or disables (False) Tikhonov (L2) regularization during model fitting.
# RECOMMENDED to set to True for better parameter estimation.
# Used to stabilize the solution of ill-posed problems
# by adding a penalty term to the cost function.
# This penalty term is proportional to the L2 norm of the parameters,
# which discourages large parameter values and helps prevent overfitting.
# When set to True, the regularization term is added to the cost function
# during the optimization process, which can lead to more stable and reliable parameter estimates.
# When set to False, the optimization process will not include this regularization term,
# which may result in less stable solutions, especially in cases where the data is noisy or sparse.
USE_REGULARIZATION = True
# Composite Scoring Function:
# score = alpha * RMSE + beta * MAE + gamma * Var(residual) + delta * MSE + mu * L2 norm
#
# Definitions:
#   RMSE         = Root Mean Squared Error
#   MAE          = Mean Absolute Error
#   Var(residual)= Variance of residuals
#   MSE          = Mean Squared Error
#   L2 norm      = L2 norm of parameter estimates
#
# Weights:
#   alpha = weight for RMSE
#   beta  = weight for MAE
#   gamma = weight for residual variance
#   delta = weight for MSE
#   mu    = weight for L2 norm of parameter estimates
#
# Lower score indicates a better fit.
DELTA_WEIGHT = 1.0
ALPHA_WEIGHT = 1.0
BETA_WEIGHT = 1.0
GAMMA_WEIGHT = 1.0
MU_WEIGHT = 1.0
########################################################################################################################
# PATHS, DIRECTORIES, AND FILES
# The following constants define the paths and directories used in the project.
# INTERNAL CONSTANTS
# These constants are used internally within the module and are not intended to be modified by users.
########################################################################################################################
# This global constant defines a mapping between internal ODE_MODEL identifiers
# and human-readable display names for different types of ODE models.
#
# The keys in the dictionary are the internal codes used in the configuration:
#   - "distmod" stands for the Distributive model.
#   - "succmod" stands for the Successive model.
#   - "randmod" stands for the Random model.
#
# The variable model_type is set by looking up the current ODE_MODEL value in this mapping.
# If ODE_MODEL doesn't match any key, model_type defaults to "Unknown".
model_names = {
    "distmod": "Distributive",
    "succmod": "Successive",
    "randmod": "Random"
}
model_type = model_names.get(ODE_MODEL, "Unknown")
# Choose which scalar metric to use for sensitivity (Y) calculation.
# Options:
#   'total_signal'   – Sum of all mRNA and site values across time.
#                      (Captures overall signal magnitude.)
#   'mean_activity'  – Mean of all mRNA and site values across time.
#                      (Normalizes by count; captures average activity.)
#   'variance'       – Variance of all mRNA and site values across time.
#                      (Captures temporal/spatial variability.)
#   'dynamics'       – Sum of squared successive differences of the flattened values.
#                      (Captures how “dynamic” or rapidly changing the signal is.)
#   'l2_norm'        – Euclidean norm of the flattened values.
#                      (Captures overall magnitude like total_signal but in L2 sense.)
Y_METRIC_DESCRIPTIONS = {
    'total_signal': "Sum of all mRNA and site values across time (captures overall signal magnitude).",
    'mean_activity': "Mean of all mRNA and site values across time (captures average activity).",
    'variance': "Variance of all mRNA and site values across time (captures temporal/spatial variability).",
    'dynamics': "Sum of squared successive differences (captures how dynamic the signal is).",
    'l2_norm': "Euclidean norm of the flattened values (captures overall magnitude in L2 sense)."
}
Y_METRIC = 'total_signal'
# Top-Level Directory Configuration:
# - PROJECT_ROOT: The root directory of the project, determined by moving one level up from the current file.
# - OUT_DIR: Directory to store all output results.
# - OUT_RESULTS_DIR: Full path to the Excel file where results are saved.
# - DATA_DIR: Directory containing input data files.
# - INPUT_EXCEL: Full path to the Excel file with optimization results.
# - LOG_DIR: Directory to store log files.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / f'{model_type}_results'
OUT_RESULTS_DIR = OUT_DIR / f'{model_type}_results.xlsx'
DATA_DIR = PROJECT_ROOT / 'data' 
INPUT_EXCEL_PROTEIN = INPUT1
INPUT_EXCEL_PSITE = DATA_DIR / 'kinopt_results.xlsx'
INPUT_EXCEL_RNA = DATA_DIR / 'tfopt_results.xlsx'
LOG_DIR = OUT_DIR / f'{model_type}_logs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Plotting Style Configuration
PERTURBATIONS_TRACE_OPACITY = 0.02
COLOR_PALETTE = [mcolors.to_hex(plt.get_cmap('tab20')(i)) for i in range(0, 20, 2)]
available_markers = [
    m for m in mmarkers.MarkerStyle.markers
    if isinstance(m, str) and m not in {".", ",", " "}
]

#  Functions to get parameter names and generate labels based on the ODE model
#  being used. These functions are imported from the helper module.
#  The choice of function is determined by the value of ODE_MODEL.
if ODE_MODEL == 'randmod':
    get_param_names = get_param_names_rand
    generate_labels = generate_labels_rand
else:
    get_param_names = get_param_names_ds
    generate_labels = generate_labels_ds
