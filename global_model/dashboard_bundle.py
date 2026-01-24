"""
Dashboard bundle saving and loading utilities.

This module provides functions to save and load a compact, dashboard-friendly bundle
containing essential data for visualization. The bundle is designed to be lean,
including only the necessary data for visualization, avoiding pickling of custom classes
for robustness.

Variables stored in the bundle:

- args: command-line arguments
- picked_index: index of the protein picked for visualization
- frechet_scores: dictionary of Frechet distances for each protein
- lambdas: dictionary of lambda values for each protein
- solver_times: dictionary of solver times for each protein
- defaults: dictionary of default values for each protein
- slices: dictionary of slice values for each protein
- xl, xu: lower and upper bounds for each protein

Note that the bundle does not store the optimization result object (res) or the
Pymoo algorithm (sys).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_dashboard_bundle(
    output_dir: str | Path,
    *,
    args: Any,
    res: Any,
    slices: Any,
    xl: Any,
    xu: Any,
    defaults: dict,
    lambdas: dict,
    solver_times,
    df_prot,
    df_rna,
    df_pho,
    frechet_scores=None,
    picked_index: int | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)


    pareto_F = np.asarray(getattr(res, "F", None)) if res is not None else None
    pareto_X = np.asarray(getattr(res, "X", None)) if res is not None and hasattr(res, "X") else None

    bundle = {
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "picked_index": picked_index,
        "frechet_scores": frechet_scores,
        "lambdas": lambdas,
        "solver_times": solver_times,
        "defaults": defaults,
        "slices": slices,
        "xl": xl,
        "xu": xu,
        "pareto_F": pareto_F,
        "pareto_X": pareto_X,
        "df_prot_obs": df_prot,
        "df_rna_obs": df_rna,
        "df_pho_obs": df_pho,
    }

    bundle_path = out / "dashboard_bundle.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bundle_path


def load_dashboard_bundle(output_dir: str | Path) -> dict:
    p = Path(output_dir) / "dashboard_bundle.pkl"
    with p.open("rb") as f:
        return pickle.load(f)
