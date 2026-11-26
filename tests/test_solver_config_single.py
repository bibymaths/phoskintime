"""N5 (option A): one solver config ([networkmodel.solver]: rtol = atol = 1e-8, one max_steps) for the fit, the
export, the steady-state run, sensitivity and the posterior/profile rebuild."""
import inspect

import pytest


def _expected():
    import networkmodel.config as nc

    return {"rtol": float(nc.ODE_REL_TOL), "atol": float(nc.ODE_ABS_TOL), "max_steps": int(nc.ODE_MAX_STEPS)}


_CHECK = """
import json, os
[os.environ.pop(k) for k in list(os.environ) if k.startswith("PHOSKINTIME_")]
import networkmodel.config as nc
from networkmodel.backend import networkmodel_solver_config
from networkmodel.SteadyStateAnalysis import STEADY_STATE_SOLVER_SETTINGS
from networkmodel.simulate import EXPORT_SOLVER_SETTINGS
c = networkmodel_solver_config()
print(json.dumps({"cfg": [c.solver_name, c.rtol, c.atol, c.max_steps],
                  "export": EXPORT_SOLVER_SETTINGS, "steady": STEADY_STATE_SOLVER_SETTINGS,
                  "nc": [nc.ODE_REL_TOL, nc.ODE_ABS_TOL, nc.ODE_MAX_STEPS]}))
"""


def _fresh():
    # Clean process: other tests set PHOSKINTIME_NETWORKMODEL_CONFIG to fixture configs (pre-existing leak).
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if not k.startswith("PHOSKINTIME_")}
    out = subprocess.run([sys.executable, "-c", _CHECK], cwd=Path(__file__).resolve().parents[1], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_config_values_are_1e8():
    d = _fresh()
    assert d["nc"][0] == 1e-8 and d["nc"][1] == 1e-8


def test_one_config_everywhere():
    d = _fresh()
    name, rtol, atol, max_steps = d["cfg"]
    assert [rtol, atol, max_steps] == d["nc"]
    for s in (d["export"], d["steady"]):
        assert [s["rtol"], s["atol"], s["max_steps"]] == d["nc"] and s["solver_name"] == name


def test_fit_objective_passes_the_config():
    import networkmodel.backend as be

    src = inspect.getsource(be.make_simple_objective)
    assert src.count("config=solver_cfg") == 2
