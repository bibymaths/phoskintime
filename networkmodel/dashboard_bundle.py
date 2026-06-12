"""Save and load compact dashboard payloads for scalar optimization runs; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules."""
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
    """Save dashboard input data to disk
    
    Args:
        output_dir: Input value used by this routine.
        args: Positional arguments forwarded to the runner.
        res: Input value used by this routine.
        slices: Input value used by this routine.
        xl: Input value used by this routine.
        xu: Input value used by this routine.
        defaults: Input value used by this routine.
        lambdas: Input value used by this routine.
        solver_times: Input value used by this routine.
        df_prot: Input value used by this routine.
        df_rna: Input value used by this routine.
        df_pho: Input value used by this routine.
        frechet_scores: Input value used by this routine.
        picked_index: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    objective_values = np.asarray(getattr(res, "F", None)) if res is not None else None
    parameter_values = np.asarray(getattr(res, "X", None)) if res is not None and hasattr(res, "X") else None
    mode_obj = getattr(res, "data_mode", None)
    data_mode = getattr(mode_obj, "data_mode", None)
    active_layers = list(getattr(mode_obj, "available_layers", []) or [])

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
        "objective_values": objective_values,
        "parameter_values": parameter_values,
        "data_mode": data_mode,
        "active_layers": active_layers,
        "pareto_F": objective_values,  # backward-compatible alias
        "pareto_X": parameter_values,  # backward-compatible alias
        "df_prot_obs": df_prot,
        "df_rna_obs": df_rna,
        "df_pho_obs": df_pho,
    }

    bundle_path = out / "artifacts" / "dashboard_bundle.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bundle_path


def load_dashboard_bundle(output_dir: str | Path) -> dict:
    """Load dashboard input data from disk
    
    Args:
        output_dir: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    p = Path(output_dir) / "artifacts" / "dashboard_bundle.pkl"
    with p.open("rb") as f:
        return pickle.load(f)
