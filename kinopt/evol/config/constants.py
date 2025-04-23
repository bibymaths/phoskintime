import argparse
import numpy as np
from pathlib import Path

# Top-Level Directory Configuration:
# - PROJECT_ROOT: The root directory of the project, determined by moving one level up from the current file.
# - OUT_DIR: Directory to store all output results.
# - OUT_RESULTS_DIR: Full path to the Excel file where results are saved.
# - DATA_DIR: Directory containing input data files.
# - INPUT_EXCEL: Full path to the Excel file with optimization results.
# - LOG_DIR: Directory to store log files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'
INPUT1 = DATA_DIR / 'input1.csv'
INPUT2 = DATA_DIR / 'input2.csv'
OUT_FILE = OUT_DIR / 'kinopt_results.xlsx'
LOG_DIR = OUT_DIR / 'logs'
ODE_DATA_DIR = PROJECT_ROOT.parent / "data"
ODE_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# TIME_POINTS:
# A numpy array representing the discrete time points (in minutes) obtained from experimental MS data.
# These time points capture the dynamics of the system, with finer resolution at early times (0.0 to 16.0 minutes)
# to account for rapid changes and broader intervals later up to 960.0 minutes.
TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0,
                        16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])
def _parse_arguments():
    """
    Parses command-line arguments for setting bounds, loss type,
    missing kinase-psite estimation, scaling method.
    Returns:
        - method: Optimization method (e.g., DE, NSGA-II).
        - lower_bound: Lower bound for optimization parameters.
        - upper_bound: Upper bound for optimization parameters.
        - loss_type: Type of loss function to use.
        - include_regularization: Whether to include regularization.
        - estimate_missing_kinases: Whether to estimate missing kinase-psite values.
        - scaling_method: Method for scaling time-series data.
        - split_point: Split point for temporal scaling.
        - segment_points: Segment points for segmented scaling.
    """
    parser = argparse.ArgumentParser(description="Optimization script for gene-phosphorylation site time-series data.")

    # Bounds
    parser.add_argument("--lower_bound", type=float, default=-2, help="Lower Beta bound for optimization parameters.")
    parser.add_argument("--upper_bound", type=float, default=2, help="Upper Beta bound for optimization parameters.")

    # Loss function
    parser.add_argument("--loss_type", type=str, choices=["base", "autocorrelation", "huber", "mape", "weighted"],
                        default="base", help="Loss function to use in optimization.")
    # Regularization
    parser.add_argument("--regularization", type=str, choices=["yes", "no"], default="no",
                        help="Include L1/L2 regularization? ('yes' or 'no')")
    # Missing kinase-psite estimation
    parser.add_argument("--estimate_missing_kinases", type=str, choices=["yes", "no"], default="yes",
                        help="Estimate missing kinase-psite values? ('yes' or 'no')")
    # Scaling method
    parser.add_argument("--scaling_method", type=str,
                        choices=["min_max", "log", "temporal", "segmented", "slope", "cumulative", "none"],
                        default="None", help="Scaling method for time-series data.")
    # Split point for temporal scaling
    parser.add_argument("--split_point", type=int, default=9,
                        help="Split point for temporal scaling (only used if method is 'temporal').")
    # Segmented scaling points
    parser.add_argument("--segment_points", type=str, default="0,3,6,9,14",
                        help="Comma-separated segment points for segmented scaling.")
    parser.add_argument("--method", type=str, default="DE",
                        help="Method chosen for optimization: Differential Evolution (DE) or NSGA-II (Use DE or NSGA-II)).")
    args = parser.parse_args()
    # Convert arguments to proper types
    method = args.method
    include_regularization = args.regularization == "yes"
    estimate_missing_kinases = args.estimate_missing_kinases == "yes"
    segment_points = list(map(int, args.segment_points.split(","))) if args.scaling_method == "segmented" else None

    return method, args.lower_bound, args.upper_bound, args.loss_type, include_regularization, estimate_missing_kinases, args.scaling_method, args.split_point, segment_points

