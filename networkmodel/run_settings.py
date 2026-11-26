"""Effective settings of a finished networkmodel run, read from its own metadata.json (KNOWN_ISSUES N7, item 6a).

A posterior/profile rebuild must use the settings the fit actually used (scaling, time grids, baseline times, solver
configuration), never the current config.toml.
"""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED = ("scaling_method", "time_points", "baseline_times", "solver_fit")


def load_run_settings(output_dir) -> dict:
    """effective_settings of the run in ``output_dir``; a clear error if the run did not record them."""
    path = Path(output_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: a rebuild needs the run's own metadata.json (N7).")
    eff = json.loads(path.read_text(encoding="utf-8")).get("effective_settings") or {}
    missing = [k for k in REQUIRED if k not in eff]
    if missing:
        raise ValueError(f"{path} lacks effective settings {missing}: this run predates the recording of its "
                         "settings, so it cannot be rebuilt reproducibly (N7).")
    return eff


def solver_config_from(eff: dict):
    """DiffraxSolverConfig recorded by the run."""
    from networkmodel.backend import DiffraxSolverConfig

    return DiffraxSolverConfig(**eff["solver_fit"])
