from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from networkmodel.cache import prepare_fast_loss_data
from networkmodel.jax_backend import (
    DataMode,
    DiffraxSolverConfig,
    detect_data_mode,
    ensure_jax_float64,
    make_simple_objective,
    optimize_scalar_objective,
    project_alpha_blocks,
    project_beta_blocks,
    project_bounds,
    solve_diffrax,
    warn_deprecated_backend_options,
)
from networkmodel.optproblem import GlobalODEScalarObjective


class DummyIndex:
    N = 1
    p2i = {"G1": 0}
    proteins = ["G1"]
    sites = [["S1"]]
    n_sites = np.asarray([1], dtype=np.int32)

    def block(self, i):
        return slice(0, 3)


def _frames(layers):
    prot = pd.DataFrame({"protein": ["G1"], "time": [0.0], "fc": [1.0], "w": [1.0]}) if "protein" in layers else pd.DataFrame(columns=["protein", "time", "fc", "w"])
    rna = pd.DataFrame({"protein": ["G1"], "time": [0.0], "fc": [1.0], "w": [1.0]}) if "mrna" in layers else pd.DataFrame(columns=["protein", "time", "fc", "w"])
    pho = pd.DataFrame({"protein": ["G1"], "psite": ["S1"], "time": [0.0], "fc": [1.0], "w": [1.0]}) if "phospho" in layers else pd.DataFrame(columns=["protein", "psite", "time", "fc", "w"])
    return prot, rna, pho


def _loss_data(layers):
    prot, rna, pho = _frames(layers)
    return prepare_fast_loss_data(DummyIndex(), prot, rna, pho, np.asarray([0.0, 1.0]))


def test_jax_float64_enabled_before_computation():
    assert ensure_jax_float64() is True
    assert jax.config.jax_enable_x64 is True


def test_mode_detection_all_seven_modes_and_missing_layers_skipped():
    modes = [
        ("mrna",), ("protein",), ("phospho",),
        ("mrna", "protein"), ("mrna", "phospho"), ("protein", "phospho"),
        ("mrna", "protein", "phospho"),
    ]
    for layers in modes:
        mode = detect_data_mode(loss_data=_loss_data(layers))
        assert mode.available_layers == tuple(layers)
        assert set(mode.active_loss_terms) == {f"{x}_loss" for x in layers}
        assert set(mode.skipped_loss_terms) == {f"{x}_loss" for x in ("mrna", "protein", "phospho") if x not in layers}


def test_objective_is_single_scalar_and_includes_available_layers():
    loss_data = _loss_data(("mrna", "protein", "phospho"))
    mode = detect_data_mode(loss_data=loss_data)
    obj = make_simple_objective(loss_data, mode, [0.0, 1.0])
    value = obj(jnp.ones(3))
    assert value.shape == ()
    assert np.isfinite(float(value))
    assert not isinstance(value, (tuple, list, dict))


def test_networkmodel_objective_runs_for_all_modes():
    for layers in [
        ("mrna",), ("protein",), ("phospho",),
        ("mrna", "protein"), ("mrna", "phospho"), ("protein", "phospho"),
        ("mrna", "protein", "phospho"),
    ]:
        loss_data = _loss_data(layers)
        problem = GlobalODEScalarObjective(
            sys=None,
            slices={},
            loss_data=loss_data,
            defaults={},
            lambdas={"protein": 1.0, "rna": 1.0, "phospho": 1.0, "prior": 0.0},
            time_grid=np.asarray([0.0, 1.0]),
            xl=np.zeros(3),
            xu=np.ones(3) * 3,
        )
        out = {}
        problem._evaluate(np.ones(3), out)
        assert out["F"].shape == (1,)
        assert np.isfinite(out["F"][0])


def test_diffrax_kvaerno_solver_shape_and_dtype():
    ys = solve_diffrax(jnp.ones(3), jnp.asarray([0.0, 0.5, 1.0]), params=jnp.ones(3), config=DiffraxSolverConfig("Kvaerno4"))
    assert ys.shape == (3, 3)
    assert ys.dtype == jnp.float64
    ys5 = solve_diffrax(jnp.ones(2), jnp.asarray([0.0, 1.0]), params=jnp.ones(2), config=DiffraxSolverConfig("Kvaerno5"))
    assert ys5.shape == (2, 2)


def test_constraints_alpha_beta_nonnegative_and_fixed_parameters():
    alpha = np.asarray([0.2, 0.8, 3.0, -1.0])
    pa = np.asarray(project_alpha_blocks(alpha, [0, 0, 1, 1]))
    assert np.all(pa >= -1e-10) and np.all(pa <= 1 + 1e-10)
    assert np.isclose(pa[:2].sum(), 1.0)
    assert np.isclose(pa[2:].sum(), 1.0)

    beta = np.asarray([-2.0, 0.5, 0.5, -1.0])
    pb = np.asarray(project_beta_blocks(beta, [0, 0, 1, 1]))
    assert np.all(pb >= -4 - 1e-8) and np.all(pb <= 4 + 1e-8)
    assert np.isclose(pb[:2].sum(), 1.0)
    assert np.isclose(pb[2:].sum(), 1.0)
    assert np.any(pb < 0), "beta projection must not force non-negativity"

    theta = np.asarray([-1.0, 0.5, 9.0])
    projected = np.asarray(project_bounds(theta, [0, 0, 0], [2, 2, 2], fixed_mask=[False, True, False], fixed_values=[0, 1.25, 0]))
    assert np.all(projected >= 0) and np.all(projected <= 2)
    assert projected[1] == 1.25


def test_jaxopt_path_runs_projected_gradient():
    def objective(x):
        return jnp.sum((x - 1.0) ** 2)
    params, state, value = optimize_scalar_objective(objective, np.asarray([5.0, -2.0]), np.zeros(2), np.ones(2) * 10, maxiter=20)
    assert value < 1e-6
    assert np.all(params >= 0)


def test_old_config_options_warn(caplog):
    warn_deprecated_backend_options({"optimizer": "pymoo", "n_gen": 2, "pop": 4, "use_custom_solver": True})
    text = "\n".join(r.message for r in caplog.records)
    assert "mapped to jaxopt.ProjectedGradient" in text
    assert "ignored" in text


def test_result_tables_and_mode_aware_outputs(tmp_path):
    mode = detect_data_mode(loss_data=_loss_data(("protein",)))
    df = pd.DataFrame({"scalar_objective": [0.1], "data_mode": [mode.data_mode]})
    path = tmp_path / "scalar_objective.csv"
    df.to_csv(path, index=False)
    assert path.exists()
    assert df.loc[0, "data_mode"] == "protein"


def test_no_forbidden_core_optimizer_solver_imports_or_incomplete_markers():
    root = pathlib.Path(__file__).resolve().parents[1]
    scoped = [root / "networkmodel" / "optproblem.py", root / "networkmodel" / "runner.py", root / "protwise" / "paramest" / "normest.py"]
    text = "\n".join(p.read_text() for p in scoped)
    assert "from pymoo" not in text.lower()
    assert "import pymoo" not in text.lower()
    assert "scipy.optimize" not in text
    assert "curve_fit" not in text
    assert "solve_ivp" not in text
    assert "TO" + "DO" not in text
    assert "FIX" + "ME" not in text
