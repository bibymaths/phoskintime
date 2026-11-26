"""N5 (option A): one solver config ([networkmodel.solver]: rtol = atol = 1e-8, one max_steps) for the fit, the
export, the steady-state run, sensitivity and the posterior/profile rebuild."""
import inspect

import pytest

N5 = pytest.mark.xfail(reason="N5: fit 1e-6, export 1e-5/1e-7, steady state 1e-6/1e-8", strict=True)


def _expected():
    import networkmodel.config as nc

    return {"rtol": float(nc.ODE_REL_TOL), "atol": float(nc.ODE_ABS_TOL), "max_steps": int(nc.ODE_MAX_STEPS)}


def test_config_values_are_1e8():
    exp = _expected()
    assert exp["rtol"] == 1e-8 and exp["atol"] == 1e-8


@N5
def test_one_config_everywhere():
    from networkmodel.backend import networkmodel_solver_config
    from networkmodel.SteadyStateAnalysis import STEADY_STATE_SOLVER_SETTINGS
    from networkmodel.simulate import EXPORT_SOLVER_SETTINGS

    cfg = networkmodel_solver_config()
    exp = _expected()
    assert (cfg.rtol, cfg.atol, cfg.max_steps) == (exp["rtol"], exp["atol"], exp["max_steps"])
    for s in (EXPORT_SOLVER_SETTINGS, STEADY_STATE_SOLVER_SETTINGS):
        assert {k: s[k] for k in ("rtol", "atol", "max_steps")} == exp
        assert s["solver_name"] == cfg.solver_name


@N5
def test_fit_objective_passes_the_config():
    import networkmodel.backend as be

    src = inspect.getsource(be.make_simple_objective)
    assert src.count("config=solver_cfg") == 2
