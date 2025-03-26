
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

# Regularization toggle and strength
USE_REGULARIZATION = True
LAMBDA_REG = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / 'results'