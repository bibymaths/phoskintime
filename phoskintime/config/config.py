import argparse
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_EXCEL = DATA_DIR / 'optimization_results.xlsx'
OUTPUT_DIR = PROJECT_ROOT / 'results'

def parse_bound_pair(val):
    try:
        parts = val.split(',')
        if len(parts) != 2:
            raise ValueError("Bounds must be provided as 'lower,upper'")
        lower = float(parts[0])
        upper = float(parts[1]) if parts[1].lower() not in ["inf", "infinity"] else float("inf")
        return lower, upper
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid bound pair '{val}': {e}")

def parse_fix_value(val):
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
    os.makedirs(directory, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential ODE Parameter Estimation with Fixed/Bounded Params and Bootstrap Support"
    )
    parser.add_argument("--A-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--B-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--C-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--D-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--Ssite-bound", type=parse_bound_pair, default="0,20")
    parser.add_argument("--Dsite-bound", type=parse_bound_pair, default="0,20")

    parser.add_argument("--fix-A", type=float, default=None)
    parser.add_argument("--fix-B", type=float, default=1)
    parser.add_argument("--fix-C", type=float, default=1)
    parser.add_argument("--fix-D", type=float, default=1)
    parser.add_argument("--fix-Ssite", type=parse_fix_value, default=None)
    parser.add_argument("--fix-Dsite", type=parse_fix_value, default=1)

    parser.add_argument("--fix-t", type=str, default='',
                        help="JSON string mapping time points to fixed param values, e.g. '{\"60\": {\"A\": 1.3}}'")
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument("--profile-start", type=float, default=0)
    parser.add_argument("--profile-end", type=float, default=100)
    parser.add_argument("--profile-step", type=float, default=10)
    parser.add_argument("--input-excel", type=str,
                        default=INPUT_EXCEL,
                        help="Path to the input Excel file")

    return parser.parse_args()

def log_config(logger, bounds, fixed_params, time_fixed, args):
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
        'max_workers': os.cpu_count(),
    }
    return config


def score_fit(target, prediction, params, alpha=1.0, beta=0.5, gamma=0.2, reg_penalty=0.1):
    # Composite Scoring Function:
    #
    # score = α * RMSE + β * MAE + γ * Var(residual) + λ * ||θ||₂
    #
    # Where:
    #   RMSE         = Root Mean Squared Error between model prediction and target
    #   MAE          = Mean Absolute Error
    #   Var(residual)= Variance of residuals to penalize unstable fits
    #   ||θ||₂       = L2 norm of estimated parameters (regularization)
    #
    #   α (alpha)    = Weight for RMSE
    #   β (beta)     = Weight for MAE
    #   γ (gamma)    = Weight for residual variance
    #   λ (lambda)   = Regularization penalty for parameter magnitude
    #
    # Lower score indicates a better fit
    residual = target - prediction
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    variance = np.var(residual)
    l2_norm = np.linalg.norm(params)

    score = alpha * rmse + beta * mae + gamma * variance + reg_penalty * l2_norm
    return score