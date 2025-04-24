
import os
import json
import argparse
import numpy as np
from pathlib import Path

from config.constants import (
    ALPHA_WEIGHT,
    BETA_WEIGHT,
    GAMMA_WEIGHT,
    DELTA_WEIGHT,
    MU_REG,
    INPUT_EXCEL
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def parse_bound_pair(val):
    """
    Parse a string representing a pair of bounds (lower, upper) into a tuple of floats.
    The upper bound can be 'inf' or 'infinity' to represent infinity.
    Raises ValueError if the input is not in the correct format.
    Args:
        val (str): The string to parse, e.g., "0,3" or "0,infinity".
    Returns:
        tuple: A tuple containing the lower and upper bounds as floats.
    """
    try:
        parts = val.split(',')
        if len(parts) != 2:
            raise ValueError("Bounds must be provided as 'lower,upper'")
        lower = float(parts[0])
        upper_str = parts[1].strip().lower()
        if upper_str in ["inf", "infinity"]:
            upper = float("inf")
        else:
            upper = float(parts[1])
        return lower, upper
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid bound pair '{val}': {e}")

def parse_fix_value(val):
    """
    Parse a fixed value or a list of fixed values from a string.
    If the input is a single value, it returns that value as a float.
    If the input is a comma-separated list, it returns a list of floats.
    Raises ValueError if the input is not in the correct format.
    Args:
        val (str): The string to parse, e.g., "1.0" or "1.0,2.0".
    Returns:
        float or list: The parsed fixed value(s) as a float or a list of floats.
    """
    if val is None:
        return None
    if ',' in val:
        try:
            return [float(x) for x in val.split(',')]
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid fixed value list '{val}': {e}")
    else:
        try:
            return float(val)
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid fixed value '{val}': {e}")

def ensure_output_directory(directory):
    """
    :param directory:
    :type directory: str
    """
    os.makedirs(directory, exist_ok=True)

def parse_args():
    """
    Parse command-line arguments for the PhosKinTime script.
    This function uses argparse to define and handle the command-line options.
    It includes options for setting bounds, fixed parameters, bootstrapping,
    profile estimation, and input file paths.
    The function returns the parsed arguments as a Namespace object.
    The arguments include:
        --A-bound: Bounds for parameter A (default: "0,3")
        --B-bound: Bounds for parameter B (default: "0,3")
        --C-bound: Bounds for parameter C (default: "0,3")
        --D-bound: Bounds for parameter D (default: "0,3")
        --Ssite-bound: Bounds for Ssite (default: "0,3")
        --Dsite-bound: Bounds for Dsite (default: "0,3")
        --fix-A: Fixed value for parameter A
        --fix-B: Fixed value for parameter B
        --fix-C: Fixed value for parameter C
        --fix-D: Fixed value for parameter D
        --fix-Ssite: Fixed value for Ssite
        --fix-Dsite: Fixed value for Dsite
        --fix-t: JSON string mapping time points to fixed param values
        --bootstraps: Number of bootstrapping iterations (default: 0)
        --profile-start: Start time for profile estimation (default: None)
        --profile-end: End time for profile estimation (default: 1)
        --profile-step: Step size for profile estimation (default: 0.5)
        --input-excel: Path to the input Excel file (default: INPUT_EXCEL)
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="PhosKinTime - ODE Parameter Estimation of Phosphorylation Events in Temporal Space"
    )
    parser.add_argument("--A-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--B-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--C-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--D-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--Ssite-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--Dsite-bound", type=parse_bound_pair, default="0,20")

    parser.add_argument("--fix-A", type=float, default=None)
    parser.add_argument("--fix-B", type=float, default=None)
    parser.add_argument("--fix-C", type=float, default=None)
    parser.add_argument("--fix-D", type=float, default=None)
    parser.add_argument("--fix-Ssite", type=parse_fix_value, default=None)
    parser.add_argument("--fix-Dsite", type=parse_fix_value, default=None)

    parser.add_argument("--fix-t", type=str, default='{ '
                                                     '\"0\": {\"A\": 0.85, \"S\": 0.1},  '
                                                     '\"60\": {\"A\":0.85, \"S\": 0.2},  '
                                                     '\"inf\": {\"A\":0.85, \"S\": 0.4} '
                                                     '}',
                        help="JSON string mapping time points to fixed param values, e.g. '{\"60\": {\"A\": 1.3}}'")
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument("--profile-start", type=float, default=None)
    parser.add_argument("--profile-end", type=float, default=1)
    parser.add_argument("--profile-step", type=float, default=0.5)
    parser.add_argument("--input-excel", type=str,
                        default=INPUT_EXCEL,
                        help="Path to the input Excel file")

    return parser.parse_args()

def log_config(logger, bounds, fixed_params, time_fixed, args):
    """
    Log the configuration settings for the PhosKinTime script.
    This function logs the parameter bounds, fixed parameters,
    bootstrapping iterations, time-specific fixed parameters,
    and profile estimation settings.
    It uses the provided logger to output the information.
    :param logger:
    :param bounds:
    :param fixed_params:
    :param time_fixed:
    :param args:
    :return:
    """
    logger.info("Parameter Bounds:")
    for key, val in bounds.items():
        logger.info(f"   {key}: {val}")
    logger.info("Fixed Parameters:")
    for key, val in fixed_params.items():
        logger.info(f"   {key}: {val}")

    logger.info(f"Bootstrapping Iterations: {args.bootstraps}")

    logger.info("Time-specific Fixed Parameters:")
    if time_fixed:
        for t, p in time_fixed.items():
            logger.info(f"   Time {t} min: {p}")
    else:
        logger.info("   None")

    logger.info("Profile Estimation:")
    logger.info(f"   Start: {args.profile_start} min")
    logger.info(f"   End:   {args.profile_end} min")
    logger.info(f"   Step:  {args.profile_step} min")
    np.set_printoptions(suppress=True)

def extract_config(args):
    """
    Extract configuration settings from command-line arguments.
    This function creates a dictionary containing the parameter bounds,
    fixed parameters, bootstrapping iterations, time-specific fixed parameters,
    and profile estimation settings. It also sets the maximum number of workers
    for parallel processing.
    The function returns the configuration dictionary.
    :param args:
    :return:
    """
    bounds = {
        "A": args.A_bound,
        "B": args.B_bound,
        "C": args.C_bound,
        "D": args.D_bound,
        "Ssite": args.Ssite_bound,
        "Dsite": args.Dsite_bound
    }
    fixed_params = {
        "A": args.fix_A,
        "B": args.fix_B,
        "C": args.fix_C,
        "D": args.fix_D,
        "Ssite": args.fix_Ssite,
        "Dsite": args.fix_Dsite
    }
    time_fixed = json.loads(args.fix_t) if args.fix_t.strip() else {}

    config = {
        'bounds': bounds,
        'fixed_params': fixed_params,
        'time_fixed': time_fixed,
        'bootstraps': args.bootstraps,
        'profile_start': args.profile_start,
        'profile_end': args.profile_end,
        'profile_step': args.profile_step,
        'input_excel': args.input_excel,
        # Adjust as needed for parallel processing
        # 'max_workers': os.cpu_count(),  # Use all CPU cores
        'max_workers': 1,
    }
    return config

def score_fit(target, prediction, params,
              alpha=ALPHA_WEIGHT,
              beta=BETA_WEIGHT,
              gamma=GAMMA_WEIGHT,
              delta=DELTA_WEIGHT,
              reg_penalty=MU_REG):
    """
    Calculate the score for the fit of a model to target data.
    The score is a weighted combination of various metrics including
    mean squared error (MSE), root mean squared error (RMSE),
    mean absolute error (MAE), variance, and regularization penalty.
    The weights for each metric can be adjusted using the parameters
    alpha, beta, gamma, and delta.
    The regularization penalty is controlled by the reg_penalty parameter.
    The function returns the calculated score.
    :param target:
    :param prediction:
    :param params:
    :param alpha:
    :param beta:
    :param gamma:
    :param delta:
    :param reg_penalty:
    :return:
    """
    residual = target - prediction
    mse = np.sum(np.abs(residual) ** 2)
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    variance = np.var(residual)
    l2_norm = np.linalg.norm(params)

    score = delta * mse + alpha * rmse + beta * mae + gamma * variance + reg_penalty * l2_norm
    return score