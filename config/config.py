
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
        description="PhosKinTime - ODE Parameter Estimation of Phosphorylation Events in Temporal Space"
    )
    parser.add_argument("--A-bound", type=parse_bound_pair, default="0,1e6")
    parser.add_argument("--B-bound", type=parse_bound_pair, default="0,1e6")
    parser.add_argument("--C-bound", type=parse_bound_pair, default="0,1e6")
    parser.add_argument("--D-bound", type=parse_bound_pair, default="0,1e6")
    parser.add_argument("--Ssite-bound", type=parse_bound_pair, default="0,1e6")
    parser.add_argument("--Dsite-bound", type=parse_bound_pair, default="0,1e6")

    parser.add_argument("--fix-A", type=float, default=None)
    parser.add_argument("--fix-B", type=float, default=None)
    parser.add_argument("--fix-C", type=float, default=None)
    parser.add_argument("--fix-D", type=float, default=None)
    parser.add_argument("--fix-Ssite", type=parse_fix_value, default=None)
    parser.add_argument("--fix-Dsite", type=parse_fix_value, default=None)

    parser.add_argument("--fix-t", type=str, default='{ '
                                                     '\"0\": {\"A\": 0.85, \"S\": 0.1},  '
                                                     '\"60\": {\"A\":0.85, \"S\": 0.2},  '
                                                     '\"1e6\": {\"A\":0.85, \"S\": 0.4} '
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
        'max_workers': 1,
    }
    return config

def score_fit(target, prediction, params,
              alpha=ALPHA_WEIGHT,
              beta=BETA_WEIGHT,
              gamma=GAMMA_WEIGHT,
              delta=DELTA_WEIGHT,
              reg_penalty=MU_REG):
    residual = target - prediction
    mse = np.sum(np.abs(residual) ** 2)
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    variance = np.var(residual)
    l2_norm = np.linalg.norm(params)

    score = delta * mse + alpha * rmse + beta * mae + gamma * variance + reg_penalty * l2_norm
    return score