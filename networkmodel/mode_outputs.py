"""Write mode-aware metadata, result tables, and simple scalar-run plots; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.jax_backend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from networkmodel.backend import DataMode


def write_mode_metadata(output_dir: str | Path, mode: DataMode, *, objective_value: float | None = None) -> Path:
    """Write scalar-run mode metadata
    
    Args:
        output_dir: Input value used by this routine.
        mode: Input value used by this routine.
        objective_value: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_mode": mode.data_mode,
        "available_layers": list(mode.available_layers),
        "active_loss_terms": list(mode.active_loss_terms),
        "skipped_loss_terms": list(mode.skipped_loss_terms),
        "objective_backend": "jaxopt.ProjectedGradient",
        "solver_backend": "diffrax.Kvaerno4/Kvaerno5",
    }
    if objective_value is not None:
        payload["scalar_objective"] = float(objective_value)
    path = out / "mode_metadata.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_scalar_result_tables(output_dir: str | Path, mode: DataMode, objective_values) -> dict[str, Path]:
    """Write scalar objective result tables
    
    Args:
        output_dir: Input value used by this routine.
        mode: Input value used by this routine.
        objective_values: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    values = np.asarray(objective_values, dtype=float).reshape(-1)
    df = pd.DataFrame({
        "scalar_objective": values,
        "data_mode": mode.data_mode,
        "active_layers": ",".join(mode.available_layers),
    })
    scalar_path = out / "scalar_objective.csv"
    legacy_path = out / "pareto_F.csv"
    df.to_csv(scalar_path, index=False)
    df[["scalar_objective"]].to_csv(legacy_path, index=False)
    metadata_path = write_mode_metadata(out, mode, objective_value=float(values[0]) if values.size else None)
    return {"scalar_objective": scalar_path, "legacy_objective": legacy_path, "metadata": metadata_path}


def save_mode_plots(output_dir: str | Path, mode: DataMode, predictions: Mapping[str, pd.DataFrame]) -> dict[str, Path]:
    """Save scalar-run mode plots
    
    Args:
        output_dir: Input value used by this routine.
        mode: Input value used by this routine.
        predictions: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for layer in mode.available_layers:
        df = predictions.get(layer)
        if df is None or df.empty:
            continue
        x_col = "time" if "time" in df.columns else df.columns[0]
        y_col = "pred_fc" if "pred_fc" in df.columns else df.select_dtypes(include=["number"]).columns[-1]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(df[x_col], df[y_col], marker="o")
        ax.set_title(f"{layer} prediction")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        path = out / f"{layer}_prediction.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths[layer] = path
    return paths
