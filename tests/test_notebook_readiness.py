from __future__ import annotations

import importlib
import json
import pathlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from tests.dummy_fixtures import (
    networkmodel_dummy_frames,
    protwise_dummy_series,
)


def test_clean_imports_for_notebook_modules():
    for name in [
        "protwise.paramest.normest",
        "protwise.models.diffrax_solver",
        "networkmodel.backend",
        "networkmodel.cache",
        "networkmodel.mode_outputs",
    ]:
        assert importlib.import_module(name)


def test_protwise_dummy_solve_objective_gradient_optimization_and_plot(tmp_path):
    from protwise.models.diffrax_solver import solve_protwise_ode
    from protwise.paramest.normest import _normalize_bounds, protwise_objective
    from protwise.plotting.plotting import Plotter
    from networkmodel.backend import optimize_scalar_objective

    t, mrna, protein, phospho, init = protwise_dummy_series(num_psites=1)
    params = np.asarray([0.3, 0.2, 0.4, 0.1, 0.2, 0.15], dtype=float)
    sol, flat = solve_protwise_ode(params, init, 1, t, model_name="distmod")
    assert sol.shape == (len(t), 3)
    assert flat.size == (len(t) - 5) + len(t) + len(t)

    mode = {"fit_mrna": True, "fit_protein": True, "fit_phospho": True, "n_rna": len(mrna), "scale_mrna": 1.0,
            "scale_protein": 1.0, "scale_phospho": 1.0}
    target = {"mrna": jnp.asarray(mrna), "protein": jnp.asarray(protein), "phospho": jnp.asarray(phospho)}
    objective = lambda x: protwise_objective(x, target, init, 1, t, mode, "distmod")
    assert np.isfinite(float(objective(params)))
    grad = jax.grad(objective)(jnp.asarray(params))
    assert grad.shape == params.shape

    lower, upper = _normalize_bounds(
        {"A": (0.01, 1), "B": (0.01, 1), "C": (0.01, 1), "D": (0.01, 1), "S(i)": (0.01, 1), "D(i)": (0.01, 1)},
        "distmod", 1)
    opt_params, _, value = optimize_scalar_objective(objective, params, lower, upper, maxiter=2)
    assert opt_params.shape == params.shape
    assert np.isfinite(value)

    plotter = Plotter("dummy_prot", out_dir=str(tmp_path))
    plotter.plot_parallel(np.asarray(sol), ["R", "P", "P1"])
    assert (tmp_path / "dummy_prot_parallel_coordinates_.png").exists()


def test_networkmodel_dummy_mode_handling_loss_multistart_and_exports(tmp_path):
    from networkmodel.BayesianInference import InferenceContext, run_multistart
    from networkmodel.backend import (
        detect_data_mode,
        make_simple_objective,
        multimodal_loss_from_trajectory,
        optimize_scalar_objective,
        project_alpha_blocks,
        project_beta_blocks,
        solve_diffrax,
        validate_loss_data,
    )
    from networkmodel.cache import prepare_fast_loss_data
    from networkmodel.dashboard_bundle import load_dashboard_bundle, save_dashboard_bundle
    from networkmodel.mode_outputs import save_mode_plots, write_mode_metadata, write_scalar_result_tables

    idx, df_prot, df_rna, df_pho, time_grid = networkmodel_dummy_frames(include_rna=True, include_phospho=True)
    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, time_grid)
    mode = detect_data_mode(loss_data=loss_data)
    validate_loss_data(loss_data, mode)
    assert mode.fit_mrna and mode.fit_protein and mode.fit_phospho

    y0 = np.ones(6)
    Y = solve_diffrax(y0, time_grid, params=np.asarray([0.1, 0.2]))
    total, parts = multimodal_loss_from_trajectory(Y, loss_data, mode, networkmodel_layout=True)
    assert np.isfinite(float(total))
    assert {"protein", "mrna", "phospho"}.issubset(parts)

    objective = make_simple_objective(loss_data, mode, time_grid, y0=y0, networkmodel_layout=True)
    theta0 = np.asarray([0.1, 0.2])
    assert np.isfinite(float(objective(theta0)))
    assert jax.grad(objective)(jnp.asarray(theta0)).shape == theta0.shape
    params, _, value = optimize_scalar_objective(objective, theta0, np.zeros(2), np.ones(2), maxiter=2)
    assert np.isfinite(value)

    ctx = InferenceContext(objective, theta0, np.zeros(2), np.ones(2), mode, tmp_path, parameter_names=("k1", "k2"),
                           maxiter=2)
    ensemble = run_multistart(ctx, n_starts=2, seed=2, max_workers=1)
    assert len(ensemble["summary"]) == 2
    assert bool(ensemble["summary"].sort_values("final_objective").iloc[0]["selected_best"])

    assert np.allclose(project_alpha_blocks(np.asarray([0.2, 0.8]), [0, 0]).sum(), 1.0)
    assert np.all(project_beta_blocks(np.asarray([-3.0, 4.0]), [0, 0]) <= 4.0)

    write_mode_metadata(tmp_path, mode, objective_value=value)
    write_scalar_result_tables(tmp_path, mode, [value])
    pred_df = pd.DataFrame({"time": [0.0, 1.0], "pred_fc": [1.0, 1.1]})
    plots = save_mode_plots(tmp_path, mode, {"protein": pred_df, "mrna": pred_df, "phospho": pred_df})
    save_dashboard_bundle(tmp_path, args=SimpleNamespace(dummy=True),
                          res=SimpleNamespace(X=np.asarray([params]), F=np.asarray([[value]]), objective_value=value,
                                              params=params, state=None, data_mode=mode, loss_breakdown={}), slices={},
                          xl=np.zeros(2), xu=np.ones(2), defaults={}, lambdas={}, solver_times=time_grid,
                          df_prot=df_prot, df_rna=df_rna, df_pho=df_pho)
    bundle = load_dashboard_bundle(tmp_path)
    assert bundle["data_mode"] == mode.data_mode
    assert all(path.exists() for path in plots.values())
    assert json.loads((tmp_path / "mode_metadata.json").read_text())["data_mode"] == mode.data_mode


def test_networkmodel_missing_modality_and_validation_errors():
    from networkmodel.backend import detect_data_mode, validate_loss_data
    from networkmodel.cache import prepare_fast_loss_data

    idx, df_prot, df_rna, df_pho, time_grid = networkmodel_dummy_frames(include_rna=False, include_phospho=False)
    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, time_grid)
    mode = detect_data_mode(loss_data=loss_data)
    assert mode.data_mode == "protein"
    validate_loss_data(loss_data, mode)

    with pytest.raises(ValueError, match="Time"):
        bad = df_prot.copy()
        bad.loc[0, "time"] = 99.0
        prepare_fast_loss_data(idx, bad, df_rna, df_pho, time_grid)
    with pytest.raises(KeyError):
        prepare_fast_loss_data(idx, df_prot.drop(columns=["fc"]), df_rna, df_pho, time_grid)


def test_dependency_boundaries_for_active_notebook_paths():
    root = pathlib.Path(__file__).resolve().parents[1]
    active_paths = [
        root / "protwise" / "paramest" / "normest.py",
        root / "protwise" / "models" / "diffrax_solver.py",
        root / "networkmodel" / "backend.py",
        root / "networkmodel" / "OptimizationProblem.py",
        root / "networkmodel" / "runner.py",
    ]
    text = "\n".join(path.read_text() for path in active_paths)
    forbidden = ["from pymoo", "import pymoo", "scipy.optimize", "odeint(", "solve_ivp(", "Pareto front",
                 "pareto front"]
    for term in forbidden:
        assert term not in text
    assert "evolutionary optimization" not in text.lower()
