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
INPUT3 = DATA_DIR / 'input3.csv'
INPUT4 = DATA_DIR / 'input4.csv'
OUT_FILE = OUT_DIR / 'tfopt_results.xlsx'
LOG_DIR = OUT_DIR / 'logs'
ODE_DATA_DIR = PROJECT_ROOT.parent / "data"
ODE_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# TIME_POINTS:
# A numpy array representing the discrete time points (in minutes) obtained from experimental Rout_Limma TF data.
# These time points capture the dynamics of the system, with finer resolution at early times (4.0 to 60.0 minutes)
# to account for rapid changes and broader intervals later up to 960.0 minutes.
TIME_POINTS = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])

# VECTORIZED_LOSS_FUNCTION:
# A boolean flag indicating whether to use a vectorized loss function.
# If set to True, the loss function will be optimized for performance using vectorized operations.
# If set to False, the loss function will use standard Python loops.
# This can significantly affect the speed and efficiency of the optimization process if you have
# mRNAs and TFs in the order of 1000s.
VECTORIZED_LOSS_FUNCTION = True

def parse_args():
    """
    Parse command line arguments for the PhosKinTime optimization problem.
    This function uses argparse to handle input parameters for the optimization process.
    The parameters include:
    - lower_bound: Lower bound for the optimization variables (default: -2).
    - upper_bound: Upper bound for the optimization variables (default: 2).
    - loss_type: Type of loss function to use (default: 0).
        Options:
        0: MSE
        1: MAE
        2: soft L1
        3: Cauchy
        4: Arctan
        5: Elastic Net
        6: Tikhonov
    - optimizer: Global Evolutionary Optimization method (default: 0).
        Options:
        0: NGSA2
        1: SMSEMOA
        2: AGEMOEA

    :returns
    - lower_bound: Lower bound for the optimization variables.
    - upper_bound: Upper bound for the optimization variables.
    - loss_type: Type of loss function to use.
    - optimizer: Global Evolutionary Optimization method.
    :rtype: tuple
    :raises argparse.ArgumentError: If an invalid argument is provided.
    :raises SystemExit: If the script is run with invalid arguments.
    """

    parser = argparse.ArgumentParser(
        description="PhosKinTime - Global Optimization mRNA-TF Optimization Problem."
    )
    # Adding command line arguments for lower and upper bounds, loss type, and optimizer
    parser.add_argument("--lower_bound", type=float, default=-2, help="Lower Beta bound.")
    parser.add_argument("--upper_bound", type=float, default=2, help="Upper Beta bound.")
    parser.add_argument("--loss_type", type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=0,
                        help="Loss function to use:  "
                             "0: MSE, 1: MAE, 2: soft L1, 3: Cauchy,"
                             "4: Arctan, 5: Elastic Net, 6: Tikhonov.")
    parser.add_argument("--optimizer", type=int, choices=[0, 1, 2], default=0,
                        help="Global Evolutionary Optimization method:  "
                             "0: NGSA2, 1: SMSEMOA , 2: AGEMOEA")
    args = parser.parse_args()
    return args.lower_bound, args.upper_bound, args.loss_type, args.optimizer