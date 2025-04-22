import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
from pathlib import Path
from config.helpers import *

# Select the ODE model for phosphorylation kinetics.
# Options:
# 'distmod' : Distributive model (phosphorylation events occur independently).
# 'succmod' : Successive model (phosphorylation events occur in a fixed order).
# 'randmod' : Random model (phosphorylation events occur randomly).
ODE_MODEL = 'randmod'
# ESTIMATION_MODE: Global constant to choose the estimation strategy.
# Set to "sequential" to perform time-point-by-time-point fitting (sequential estimation),
# which produces a series of parameter estimates over time (one estimate per time point).
# Set to "normal" to perform fitting using all-time points at once (normal estimation),
# yielding a single set of parameter estimates that best describes the entire time course.
ESTIMATION_MODE = 'normal'
# ALPHA_CI: Confidence level for computing confidence intervals for parameter identifiability.
# For example, an ALPHA_CI of 0.95 indicates that the model will compute 95% confidence intervals.
# This corresponds to a significance level of 1 - ALPHA_CI (i.e., 0.05) when determining the critical t-value.
ALPHA_CI = 0.95
# Whether to normalize model output to match fold change (FC) data
# ----------------------------------------------------------------
# Set to True when experimental data is provided in fold change format
# (i.e., values are already normalized relative to the baseline time point, typically t=0).
#
# When enabled, model outputs Y(t) will be divided by Y(t0) for each species:
#     FC_model(t) = Y(t) / Y(t0)
#
# This ensures the model output is in the same scale and units as the FC data.
# If False, raw concentrations will be used directly
# (only valid if data is also in absolute units).
# IMPORTANT: Set to True if your time series data represents relative changes.
NORMALIZE_MODEL_OUTPUT = False
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
    "randmod": "Random",
    "testmod": "Test",
}
model_type = model_names.get(ODE_MODEL, "Unknown")
# TIME_POINTS:
# A numpy array representing the discrete time points (in minutes) obtained from experimental MS data.
# These time points capture the dynamics of the system, with finer resolution at early times (0.0 to 16.0 minutes)
# to account for rapid changes and broader intervals later up to 960.0 minutes.
TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0,
                        16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])
# Top-Level Plotting and Regularization Settings:
# - CONTOUR_LEVELS: Defines the number of contour levels used in density plots.
# - USE_REGULARIZATION: Enables (True) or disables (False) Tikhonov (L2) regularization during model fitting.
# - LAMBDA_REG: Specifies the regularization parameter (lambda) for L2 regularization.
CONTOUR_LEVELS = 100
USE_REGULARIZATION = False
LAMBDA_REG = 1e-4
# Composite Scoring Function:
# score = alpha * RMSE + beta * MAE + gamma * Var(residual) + delta * MSE + mu * ||theta||2
#
# Definitions:
#   RMSE         = Root Mean Squared Error
#   MAE          = Mean Absolute Error
#   Var(residual)= Variance of residuals
#   MSE          = Mean Squared Error
#   ||theta||2   = L2 norm of parameters
#
# Weights:
#   alpha = weight for RMSE
#   beta  = weight for MAE
#   gamma = weight for residual variance
#   delta = weight for MSE
#   mu    = weight for L2 regularization
#
# Lower score indicates a better fit.
DELTA_WEIGHT = 1.0
ALPHA_WEIGHT = 1.0
BETA_WEIGHT = 1.0
GAMMA_WEIGHT = 1.0
MU_REG = 1.0
# Top-Level Directory Configuration:
# - PROJECT_ROOT: The root directory of the project, determined by moving one level up from the current file.
# - OUT_DIR: Directory to store all output results.
# - OUT_RESULTS_DIR: Full path to the Excel file where results are saved.
# - DATA_DIR: Directory containing input data files.
# - INPUT_EXCEL: Full path to the Excel file with optimization results.
# - LOG_DIR: Directory to store log files.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / 'results'
OUT_RESULTS_DIR = OUT_DIR / 'results.xlsx'
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_EXCEL = DATA_DIR / 'kinopt_results.xlsx'
LOG_DIR = OUT_DIR / 'logs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Plotting Style Configuration
#   A list of hexadecimal color codes generated from the 'tab20' colormap.
#   Colors are sampled every 2 steps from the colormap (from 0 to 20) to ensure distinctness.
COLOR_PALETTE = [mcolors.to_hex(plt.get_cmap('tab20')(i)) for i in range(0, 20, 2)]

#   A list of valid marker styles (as strings) from matplotlib, excluding markers like '.', ',', and whitespace.
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
