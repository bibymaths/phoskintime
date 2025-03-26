import argparse
import json
import os

import numpy as np


def parse_bound_pair(val):
    try:
        parts = val.split(',')
        if len(parts) != 2:
            raise ValueError("Bounds must be provided as 'lower,upper'")
        lower = float(parts[0])
        upper = float(parts[1]) if parts[1].lower() not in ["inf", "infinity"] else float("inf")
        return (lower, upper)
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

    parser.add_argument("--fix-A", type=float, default=1)
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
    parser.add_argument("--out-dir", type=str, default="distributive_profiles")

    # Missing: add this
    parser.add_argument("--input-excel", type=str,
                        default="optimization_results.xlsx",
                        help="Path to the input Excel file")

    return parser.parse_args()


def print_config(bounds, fixed_params, time_fixed, args):
    separator = "=" * 70
    print("\n" + separator)
    print("Sequential Phosphorylation Modeling Configuration")
    print(separator)
    print("Parameter Bounds:")
    for key in bounds:
        print(f"   {key}: {bounds[key]}")

    print("\nFixed Parameters:")
    for key in fixed_params:
        print(f"   {key}: {fixed_params.get(key)}")

    print("\nBootstrapping Iterations:", args.bootstraps)
    print("\nTime-specific Fixed Parameters:")
    if time_fixed:
        for t, p in time_fixed.items():
            print(f"   Time {t} min: {p}")
    else:
        print("   None")

    print("\nProfile Estimation:")
    print(f"   Start: {args.profile_start} min")
    print(f"   End:   {args.profile_end} min")
    print(f"   Step:  {args.profile_step} min")
    print(separator)
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
        'out_dir': args.out_dir,
        'input_excel': args.input_excel,
        'max_workers': os.cpu_count(),
    }
    return config
