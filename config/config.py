import os
import argparse
import numpy as np
from pathlib import Path

from config.constants import (
    ALPHA_WEIGHT,
    BETA_WEIGHT,
    GAMMA_WEIGHT,
    DELTA_WEIGHT,
    INPUT_EXCEL, DEV_TEST, MU_WEIGHT, INPUT_EXCEL_RNA, TIME_POINTS, BOOTSTRAPS, UB_mRNA_prod, UB_mRNA_deg,
    UB_Protein_prod, UB_Protein_deg, UB_Phospho_prod
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from config.logconf import setup_logger

logging = setup_logger()


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
    --A-bound, --B-bound, --C-bound, --D-bound,
    --Ssite-bound, --Dsite-bound, --bootstraps,
    --input-excel, --input-excel-rna.
    Args:
        None
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="PhosKinTime - ODE Parameter Estimation of Cell Signalling Events in Temporal Space"
    )
    parser.add_argument("--A-bound", type=parse_bound_pair, default=f"0, {UB_mRNA_prod}")
    parser.add_argument("--B-bound", type=parse_bound_pair, default=f"0, {UB_mRNA_deg}")
    parser.add_argument("--C-bound", type=parse_bound_pair, default=f"0, {UB_Protein_prod}")
    parser.add_argument("--D-bound", type=parse_bound_pair, default=f"0, {UB_Protein_deg}")
    parser.add_argument("--Ssite-bound", type=parse_bound_pair, default=f"0, {UB_Phospho_prod}")
    parser.add_argument("--Dsite-bound", type=parse_bound_pair, default=f"0, {UB_Protein_deg}")
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--input-excel", type=str,
                        default=INPUT_EXCEL,
                        help="Path to the estimated optimized phosphorylation-residue file")
    parser.add_argument("--input-excel-rna", type=str,
                        default=INPUT_EXCEL_RNA,
                        help="Path to the estimated optimized mRNA-TF file")
    return parser.parse_args()


def log_config(logger, bounds, args):
    """
    Log the configuration settings for the PhosKinTime script.
    This function logs the parameter bounds
    bootstrapping iterations.
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
        logger.info(f"      {key}      : {val}")
    logger.info(f"      Bootstrapping Iterations: {args.bootstraps}")
    logger.info("           --------------------------------")
    np.set_printoptions(suppress=True)


def extract_config(args):
    """
    Extract configuration settings from command-line arguments.
    This function creates a dictionary containing the parameter bounds, bootstrapping iterations.
    The function returns the configuration dictionary.
    :param args:
    :return:
    """
    bounds = {
        "A": args.A_bound,
        "B": args.B_bound,
        "C": args.C_bound,
        "D": args.D_bound,
        "S(i)": args.Ssite_bound,
        "D(i)": args.Dsite_bound
    }
    config = {
        'bounds': bounds,
        'bootstraps': args.bootstraps,
        'input_excel': args.input_excel,
        'input_excel_rna': args.input_excel_rna,
        'max_workers': 1 if DEV_TEST else os.cpu_count(),
    }
    return config


def score_fit(gene, params, weight, target, prediction,
              alpha=ALPHA_WEIGHT,
              beta=BETA_WEIGHT,
              gamma=GAMMA_WEIGHT,
              delta=DELTA_WEIGHT,
              mu=MU_WEIGHT):
    """
    Calculate the score for the fit of a model to target data.
    The score is a weighted combination of various metrics including
    mean squared error (MSE), root mean squared error (RMSE),
    mean absolute error (MAE), variance, and regularization penalty.
    The weights for each metric can be adjusted using the parameters
    alpha, beta, gamma, and delta.
    The regularization penalty is controlled by the reg_penalty parameter.
    The function returns the calculated score.
    Args:
        gene (str): The name of the gene.
        params (np.ndarray): The model parameters.
        weight (str): The weighting schema.
        target (np.ndarray): The target data.
        prediction (np.ndarray): The predicted data.
        alpha (float): Weight for RMSE.
        beta (float): Weight for MAE.
        gamma (float): Weight for variance.
        delta (float): Weight for MSE.
        mu (float): Regularization penalty weight.
    Returns:
        float: The calculated score.
    """
    # Format the weight
    weight_display = ' '.join(w.capitalize() for w in weight.split('_'))

    # Compute scaled absolute residuals (error per data point).
    residual = np.abs(target - prediction) / target.size

    # Compute mean squared error (MSE) from residuals.
    mse = np.sum(residual ** 2)

    # Compute root mean squared error (RMSE) from residuals.
    rmse = np.sqrt(np.mean(residual ** 2))

    # Compute mean absolute error (MAE) from residuals.
    mae = np.mean(residual)

    # Compute variance of residuals.
    variance = np.var(residual)

    # L2 norm of parameters.
    l2_norm = np.linalg.norm(params, ord=2) / len(params)

    # Calculate weighted total score combining errors
    score = delta * mse + alpha * rmse + beta * mae + gamma * variance + mu * l2_norm

    # Log all calculated metrics for each weighting schema.
    # logging.info(f"[{gene}] [{weight_display}] MSE: {mse:.2e}")
    # logging.info(f"[{gene}] [{weight_display}] RMSE: {rmse:.2e}")
    # logging.info(f"[{gene}] [{weight_display}] MAE: {mae:.2e}")
    # logging.info(f"[{gene}] [{weight_display}] Variance: {variance:.2e}")
    # logging.info(f"[{gene}] [{weight_display}] L2 Norm: {l2_norm:.2e}")
    # logging.info(f"[{gene}] [{weight_display}] Score: {score:.2e}")

    return score

def future_times(n_new: int, ratio: float = None, tp: np.ndarray = TIME_POINTS) -> np.ndarray:
    """
    Extend ttime points by n_new points, each spaced by multiplying the previous interval by ratio.
    If ratio is None, it is inferred from the last two points.
    """
    times = tp.tolist()
    if ratio is None:
        # avoid divide-by-zero if the last point is zero
        ratio = times[-1] / times[-2]
    for _ in range(n_new):
        times.append(times[-1] * ratio)
    return np.array(times)