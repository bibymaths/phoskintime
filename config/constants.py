import numpy as np
import matplotlib.cm as cm
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

# Mapping ODE_MODEL values to display names.
model_names = {
    "distmod": "Distributive",
    "succmod": "Successive",
    "randmod": "Random"
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
USE_REGULARIZATION = True
LAMBDA_REG = 1e-6

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
MU_REG = 2.0

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
INPUT_EXCEL = DATA_DIR / 'optimization_results.xlsx'
LOG_DIR = OUT_DIR / 'logs'

# Plotting Style Configuration

#   A list of hexadecimal color codes generated from the 'tab20' colormap.
#   Colors are sampled every 2 steps from the colormap (from 0 to 20) to ensure distinctness.
COLOR_PALETTE = [mcolors.to_hex(cm.tab20(i)) for i in range(0, 20, 2)]

#   A list of valid marker styles (as strings) from matplotlib, excluding markers like '.', ',', and whitespace.
available_markers = [
    m for m in mmarkers.MarkerStyle.markers
    if isinstance(m, str) and m not in {".", ",", " "}
]

if ODE_MODEL == 'randmod':
    get_param_names = get_param_names_rand
    generate_labels = generate_labels_rand
else:
    get_param_names = get_param_names_ds
    generate_labels = generate_labels_ds
