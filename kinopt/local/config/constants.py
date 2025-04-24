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

def parse_args():
    """
    Parses command-line arguments for the optimization script.
    This function uses argparse to handle various parameters related to the optimization process.
    The parameters include bounds for the optimization, loss function types, estimation of missing kinases,
    scaling methods for time-series data, and the optimization method to be used.
    The function returns a tuple containing the parsed arguments.

    :return: A tuple containing the parsed arguments.
        - lower_bound (float): Lower bound for the optimization.
        - upper_bound (float): Upper bound for the optimization.
        - loss_type (str): Type of loss function to use.
        - estimate_missing (bool): Whether to estimate missing kinase-psite values.
        - scaling_method (str): Method for scaling time-series data.
        - split_point (int): Split point for temporal scaling.
        - segment_points (list of int): Segment points for segmented scaling.
        - method (str): Optimization method to use.
    """
    parser = argparse.ArgumentParser(
        description="PhosKinTime - SLSQP/TRUST-CONSTR Kinase Phosphorylation Optimization Problem prior to ODE Modelling."
    )

    # lower_bound and upper_bound are the bounds for the optimization.
    parser.add_argument("--lower_bound", type=float, default=-2, help="Lower Beta bound.")
    parser.add_argument("--upper_bound", type=float, default=2, help="Upper Beta bound.")

    # loss_type is the type of loss function to use.
    parser.add_argument("--loss_type", type=str,
                        choices=["base", "weighted", "softl1", "cauchy", "arctan"],
                        default="base", help="Loss function to use.")

    # estimate_missing_kinases indicates whether to estimate missing kinase-psite values.
    parser.add_argument("--estimate_missing_kinases", type=str, choices=["yes", "no"], default="yes",
                        help="Estimate missing kinase-psite values?")

    # scaling_method is the method for scaling time-series data.
    parser.add_argument("--scaling_method", type=str,
                        choices=["min_max", "log", "temporal", "segmented", "slope", "cumulative", "none"],
                        default="None", help="Scaling method for time-series data.")

    # split_point is the split point for temporal scaling.
    parser.add_argument("--split_point", type=int, default=9, help="Split point for temporal scaling.")

    # segment_points is a list of segment points for segmented scaling.
    parser.add_argument("--segment_points", type=str, default="0,3,6,9,14",
                        help="Comma-separated segment points for segmented scaling.")

    # method is the optimization method to use.
    parser.add_argument("--method", type=str, choices=["slsqp", "trust-constr"], default="slsqp",
                        help="Optimization method.")

    args = parser.parse_args()
    estimate_missing = args.estimate_missing_kinases == "yes"
    seg_points = list(map(int, args.segment_points.split(","))) if args.scaling_method == "segmented" else None

    return (args.lower_bound, args.upper_bound, args.loss_type, estimate_missing,
            args.scaling_method, args.split_point, seg_points, args.method)