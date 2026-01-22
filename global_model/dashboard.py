from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_dashboard_bundle(
        output_dir: str | Path,
        *,
        args: Any,
        idx: Any,
        sys: Any,
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
    """
    Save a compact, dashboard-friendly bundle.
    Keep it lean: enough for visualization, not necessarily all raw internals.

    Note: sys/idx are custom classes; pickling should work if code is importable.
    If you later want a more robust format, serialize only primitives + dataframes.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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
        "idx": idx,
        "sys": sys,
        "res": res,
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
