from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from config_loader import load


_CFG = load("evol", "kinopt")
_PATHS = _CFG.get("_paths", {}) or {}
PROJECT_ROOT = Path(_CFG["_root"])


DATA_DIR = PROJECT_ROOT / _PATHS.get("data_dir", "data")
OUT_DIR = PROJECT_ROOT / _PATHS.get("results_dir", "results")
LOG_DIR = PROJECT_ROOT / _PATHS.get("logs_dir", "results/logs")
ODE_DATA_DIR = PROJECT_ROOT / _PATHS.get("ode_data_dir", "data/ode")

for d in (DATA_DIR, OUT_DIR, LOG_DIR, ODE_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


INPUT1 = DATA_DIR / _CFG.get("input1", "input1.csv")
INPUT2 = DATA_DIR / _CFG.get("input2", "input2.csv")

OUT_FILE = OUT_DIR / _CFG.get("out_file", "kinopt_results.xlsx")


TIME_POINTS = np.asarray(
    _CFG.get(
        "time_points",
        [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    ),
    dtype=float,
)


def _parse_arguments():
    """
    kinopt.evol CLI.
    Defaults come from config.toml.
    """
    parser = argparse.ArgumentParser(
        description="Optimization script for gene-phosphorylation site time-series data."
    )

    parser.add_argument("--lower_bound", type=float, default=float(_CFG.get("lower_bound", -2.0)))
    parser.add_argument("--upper_bound", type=float, default=float(_CFG.get("upper_bound", 2.0)))

    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["base", "autocorrelation", "huber", "mape", "weighted"],
        default=str(_CFG.get("loss_type", "base")),
        help="Loss function to use in optimization.",
    )

    parser.add_argument(
        "--regularization",
        type=str,
        choices=["yes", "no"],
        default="yes" if bool(_CFG.get("regularization", False)) else "no",
        help="Include L1/L2 regularization?",
    )

    parser.add_argument(
        "--estimate_missing_kinases",
        type=str,
        choices=["yes", "no"],
        default="yes" if bool(_CFG.get("estimate_missing_kinases", True)) else "no",
        help="Estimate missing kinase-psite values?",
    )

    parser.add_argument(
        "--scaling_method",
        type=str,
        choices=["min_max", "log", "temporal", "segmented", "slope", "cumulative", "none"],
        default=str(_CFG.get("scaling_method", "none")),
        help="Scaling method for time-series data.",
    )

    parser.add_argument("--split_point", type=int, default=int(_CFG.get("split_point", 9)))

    default_seg = _CFG.get("segment_points", [0, 3, 6, 9, 14])
    default_seg_str = ",".join(str(int(x)) for x in default_seg)

    parser.add_argument(
        "--segment_points",
        type=str,
        default=default_seg_str,
        help="Comma-separated segment points for segmented scaling.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=str(_CFG.get("method", "DE")),
        help="DE or NSGA-II (whatever your runner expects).",
    )

    args = parser.parse_args()

    method = args.method
    include_regularization = args.regularization == "yes"
    estimate_missing_kinases = args.estimate_missing_kinases == "yes"
    segment_points = list(map(int, args.segment_points.split(","))) if args.scaling_method == "segmented" else None

    return (
        method,
        args.lower_bound,
        args.upper_bound,
        args.loss_type,
        include_regularization,
        estimate_missing_kinases,
        args.scaling_method,
        args.split_point,
        segment_points,
    )
