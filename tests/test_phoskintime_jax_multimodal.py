from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
import jax
import jax.numpy as jnp

from networkmodel.cache import prepare_fast_loss_data
from networkmodel.dashboard_bundle import save_dashboard_bundle
from networkmodel.backend import (
    DataMode,
    DiffraxSolverConfig,
    JaxoptResult,
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
from networkmodel.mode_outputs import save_mode_plots, write_scalar_result_tables
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


def _all_modes():
    return [
        ("mrna",), ("protein",), ("phospho",),
        ("mrna", "protein"), ("mrna", "phospho"), ("protein", "phospho"),
        ("mrna", "protein", "phospho"),
    ]


def test_jax_float64_enabled_before_computation():
    assert ensure_jax_float64() is True
    assert jax.config.jax_enable_x64 is True


@pytest.mark.parametrize("layers", _all_modes())
def test_all_modes_detect_losses_objective_optimize_outputs_and_no_fake_missing_data(tmp_path, layers):
    loss_data = _loss_data(layers)
    mode = detect_data_mode(loss_data=loss_data)
    assert mode.available_layers == tuple(layers)
    assert set(mode.active_loss_terms) == {f"{x}_loss" for x in layers}
    assert set(mode.skipped_loss_terms) == {f"{x}_loss" for x in ("mrna", "protein", "phospho") if x not in layers}

    for layer, obs_key, count_key in (("mrna", "obs_rna", "n_r"), ("protein", "obs_prot", "n_p"), ("phospho", "obs_pho", "n_ph")):
        obs = np.asarray(loss_data[obs_key])
        assert not np.isnan(obs).any()
        if layer not in layers:
            assert loss_data[count_key] == 0
            assert obs.size == 0

    obj = make_simple_objective(loss_data, mode, [0.0, 1.0])
    value = obj(jnp.ones(3))
    assert value.shape == ()
    assert np.isfinite(float(value))
    assert not isinstance(value, (tuple, list, dict))

    params, state, opt_value = optimize_scalar_objective(lambda x: jnp.sum((x - 0.25) ** 2), np.ones(3), np.zeros(3), np.ones(3), maxiter=5)
    assert np.isfinite(opt_value)
    assert np.all(params >= 0) and np.all(params <= 1)

    paths = write_scalar_result_tables(tmp_path, mode, [float(value)])
    assert paths["scalar_objective"].exists()
    assert paths["legacy_objective"].exists()
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["data_mode"] == mode.data_mode
    assert metadata["available_layers"] == list(layers)

    pred_df = pd.DataFrame({"time": [0.0, 1.0], "pred_fc": [1.0, 1.1]})
    plot_paths = save_mode_plots(tmp_path, mode, {layer: pred_df for layer in layers})
    assert set(plot_paths) == set(layers)
    for layer in ("mrna", "protein", "phospho"):
        assert (tmp_path / f"{layer}_prediction.png").exists() is (layer in layers)


def test_networkmodel_result_is_single_scalar_objective():
    loss_data = _loss_data(("mrna", "protein", "phospho"))
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
    assert problem.n_obj == 1
    assert out["F"].shape == (1,)
    assert np.isfinite(out["F"][0])


def test_diffrax_kvaerno_solver_shape_dtype_and_failure_message():
    ys = solve_diffrax(jnp.ones(3), jnp.asarray([0.0, 0.5, 1.0]), params=jnp.ones(3), config=DiffraxSolverConfig("Kvaerno4"))
    assert ys.shape == (3, 3)
    assert ys.dtype == jnp.float64
    ys5 = solve_diffrax(jnp.ones(2), jnp.asarray([0.0, 1.0]), params=jnp.ones(2), config=DiffraxSolverConfig("Kvaerno5"))
    assert ys5.shape == (2, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_diffrax(jnp.ones(2), jnp.asarray([0.0, 0.0]), params=jnp.ones(2))


def test_constraints_alpha_beta_bounded_and_fixed_parameters_after_optimization():
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

    params, state, value = optimize_scalar_objective(
        lambda x: jnp.sum((x - jnp.asarray([0.1, 1.9, 0.1])) ** 2),
        np.asarray([5.0, -2.0, 5.0]),
        np.zeros(3),
        np.ones(3) * 2,
        fixed_mask=[False, True, False],
        fixed_values=[0.0, 1.25, 0.0],
        maxiter=10,
    )
    assert np.isfinite(value)
    assert params[1] == pytest.approx(1.25)
    assert np.all(params >= 0) and np.all(params <= 2)


def test_optimizer_failure_has_clear_error(caplog):
    with pytest.raises(RuntimeError, match="final scalar objective is not finite"):
        optimize_scalar_objective(lambda x: jnp.asarray(jnp.nan), np.zeros(1), np.zeros(1), np.ones(1), maxiter=1)
    assert "JAXopt failed" in "\n".join(r.message for r in caplog.records)


def test_protwise_uses_jaxopt_diffrax_path_structurally():
    import protwise.paramest.normest as normest
    source = pathlib.Path(normest.__file__).read_text()
    assert "optimize_scalar_objective" in source
    assert "solve_diffrax" in source
    assert "curve_fit" not in source
    assert "scipy.optimize" not in source


def test_dashboard_bundle_and_dashboard_loader_accept_scalar_fields(tmp_path, monkeypatch):
    mode = DataMode(("protein",), "protein", False, True, False)
    res = JaxoptResult(X=np.asarray([[0.1, 0.2]]), F=np.asarray([[0.03]]), objective_value=0.03, params=np.asarray([0.1, 0.2]), state=None, data_mode=mode, loss_breakdown={})
    save_dashboard_bundle(tmp_path, args={"solver": "jaxopt"}, res=res, slices={}, xl=np.zeros(2), xu=np.ones(2), defaults={}, lambdas={}, solver_times=[0, 1], df_prot=pd.DataFrame(), df_rna=pd.DataFrame(), df_pho=pd.DataFrame())
    write_scalar_result_tables(tmp_path, mode, [0.03])

    fake_streamlit = types.SimpleNamespace(image=lambda *a, **k: None, video=lambda *a, **k: None, markdown=lambda *a, **k: None)
    fake_px = types.SimpleNamespace(scatter=lambda *a, **k: types.SimpleNamespace(add_trace=lambda *a, **k: None, update_layout=lambda *a, **k: None))
    fake_go = types.SimpleNamespace(Scatter=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "plotly", types.SimpleNamespace(express=fake_px, graph_objects=fake_go))
    monkeypatch.setitem(sys.modules, "plotly.express", fake_px)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)
    dashboard = importlib.reload(importlib.import_module("networkmodel.dashboard_app"))
    bundle, df_objective, *_ = dashboard._load_outputs(tmp_path)
    assert bundle["data_mode"] == "protein"
    assert bundle["active_layers"] == ["protein"]
    assert list(df_objective.columns)[:1] == ["scalar_objective"]
    fig = dashboard._fig_scalar_objective(df_objective, picked_index=0)
    assert hasattr(fig, "update_layout")


def test_old_config_options_warn(caplog):
    warn_deprecated_backend_options({"optimizer": "pymoo", "n_gen": 2, "pop": 4, "use_custom_solver": True})
    text = "\n".join(r.message for r in caplog.records)
    assert "mapped to jaxopt.ProjectedGradient" in text
    assert "ignored" in text


def test_forbidden_stack_not_in_active_networkmodel_protwise_paths():
    root = pathlib.Path(__file__).resolve().parents[1]
    active_paths = [
        root / "networkmodel" / "backend.py",
        root / "networkmodel" / "optproblem.py",
        root / "networkmodel" / "runner.py",
        root / "networkmodel" / "simulate.py",
        root / "networkmodel" / "mode_outputs.py",
        root / "protwise" / "paramest" / "normest.py",
        root / "protwise" / "models" / "diffrax_solver.py",
        root / "protwise" / "models" / "distmod.py",
        root / "protwise" / "models" / "succmod.py",
        root / "protwise" / "models" / "randmod.py",
    ]
    text = "\n".join(p.read_text() for p in active_paths)
    forbidden = ["from pymoo", "import pymoo", "scipy.optimize", "curve_fit", "from scipy.integrate", "solve_ivp(", "odeint("]
    for term in forbidden:
        assert term not in text
    assert "TO" + "DO" not in text
    assert "FIX" + "ME" not in text
