import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
from pathlib import Path

def get_param_names(num_psites: int) -> list:
    return ['A', 'B', 'C', 'D'] + [f'S{i + 1}' for i in range(num_psites)] + [f'D{i + 1}' for i in range(num_psites)]

def generate_labels(num_psites: int) -> list:
    return ["R", "P"] + [f"P{i}" for i in range(1, num_psites + 1)]

# Color palette for plots
COLOR_PALETTE = [mcolors.to_hex(cm.tab20(i)) for i in range(0, 20, 2)]

# Marker styles for different phosphorylation sites
available_markers = [
    m for m in mmarkers.MarkerStyle.markers
    if isinstance(m, str) and m not in {".", ",", " "}
]

# Number of contour levels for density plots
CONTOUR_LEVELS = 10

# Tikhonov Regularization (L2) for Model Fitting
USE_REGULARIZATION = True
LAMBDA_REG = 1e-3

# Composite Scoring Function:
# score = α * RMSE + β * MAE + γ * Var(residual) + λ * ||θ||₂
# Where:
#   RMSE         = Root Mean Squared Error between model prediction and target
#   MAE          = Mean Absolute Error
#   Var(residual)= Variance of residuals to penalize unstable fits
#   ||θ||₂       = L2 norm of estimated parameters (regularization)
#
#   (alpha)    = Weight for RMSE
#   (beta)     = Weight for MAE
#   (gamma)    = Weight for residual variance
#   (delta)    = Weight for MSE
#   (mu)       = Regularization penalty for parameter magnitude
# Lower score indicates a better fit

# Weights for composite scoring function
DELTA_WEIGHT = 1.0
ALPHA_WEIGHT = 0.8
BETA_WEIGHT = 0.5
GAMMA_WEIGHT = 0.2
MU_REG = 0.1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / 'results'
OUT_RESULTS_DIR = OUT_DIR / 'results.xlsx'
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_EXCEL = DATA_DIR / 'optimization_results.xlsx'
LOG_DIR = OUT_DIR / 'logs'

TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])