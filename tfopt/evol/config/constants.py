from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from config_loader import load


_CFG = load("evol", "tfopt")
_PATHS = _CFG.get("_paths", {}) or {}
PROJECT_ROOT = Path(_CFG["_root"])


DATA_DIR = PROJECT_ROOT / _PATHS.get("data_dir", "data")
OUT_DIR = PROJECT_ROOT / _PATHS.get("results_dir", "results")
LOG_DIR = PROJECT_ROOT / _PATHS.get("logs_dir", "results/logs")
ODE_DATA_DIR = PROJECT_ROOT / _PATHS.get("ode_data_dir", "data/ode")

for d in (DATA_DIR, OUT_DIR, LOG_DIR, ODE_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


INPUT1 = DATA_DIR / _CFG.get("input1", "input1.csv")
INPUT3 = DATA_DIR / _CFG.get("input3", "input3.csv")
INPUT4 = DATA_DIR / _CFG.get("input4", "input4.csv")

OUT_FILE = OUT_DIR / _CFG.get("out_file", "tfopt_results.xlsx")


TIME_POINTS = np.asarray(
    _CFG.get("time_points", [4, 8, 15, 30, 60, 120, 240, 480, 960]),
    dtype=float,
)

def parse_args():
    """
    tfopt.evol CLI: bounds, loss, optimizer selection.
    Defaults come from config.toml.
    """
    parser = argparse.ArgumentParser(
        description="PhosKinTime - Global Optimization mRNA-TF Optimization Problem."
    )
    parser.add_argument("--lower_bound", type=float, default=float(_CFG.get("lower_bound", -2.0)))
    parser.add_argument("--upper_bound", type=float, default=float(_CFG.get("upper_bound", 2.0)))
    parser.add_argument(
        "--loss_type",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=int(_CFG.get("loss_type", 0)),
        help="0:MSE 1:MAE 2:softl1 3:cauchy 4:arctan 5:elastic-net 6:tikhonov",
    )
    parser.add_argument(
        "--optimizer",
        type=int,
        choices=[0, 1, 2],
        default=int(_CFG.get("optimizer", 0)),
        help="0:NSGA2 1:SMSEMOA 2:AGEMOEA",
    )

    args = parser.parse_args()
    return args.lower_bound, args.upper_bound, args.loss_type, args.optimizer
