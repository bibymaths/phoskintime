"""Backward-compatibility package for old pickled dashboard bundles.

Old dashboard_bundle.pkl files may reference modules such as:
    global_model.network
    global_model.params
    global_model.simulate
These have moved under networkmodel.*.
"""

from __future__ import annotations

import importlib
import sys

_MODULE_ALIASES = {
    "global_model.network": "networkmodel.network",
    "global_model.params": "networkmodel.params",
    "global_model.simulate": "networkmodel.simulate",
    "global_model.utils": "networkmodel.utils",
    "global_model.backend": "networkmodel.backend",
    "global_model.export": "networkmodel.export",
    "global_model.optproblem": "networkmodel.OptimizationProblem",
    "global_model.OptimizationProblem": "networkmodel.OptimizationProblem",
    "global_model.dashboard_bundle": "networkmodel.dashboard_bundle",
}

for old_name, new_name in _MODULE_ALIASES.items():
    try:
        sys.modules.setdefault(old_name, importlib.import_module(new_name))
    except ModuleNotFoundError:
        pass
