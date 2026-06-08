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
    TFOPT_TIME_POINTS,
    kinopt_dummy_frames,
    networkmodel_dummy_frames,
    protwise_dummy_series,
    tfopt_dummy_data,
)


def test_clean_imports_for_notebook_modules():
    for name in [
        "kinopt.local.optcon.construct",
        "kinopt.local.objfn.minfn",
        "kinopt.local.opt.optrun",
        "tfopt.local.optcon.construct",
        "tfopt.local.objfn.minfn",
        "tfopt.local.opt.optrun",
        "protwise.paramest.normest",
        "protwise.models.diffrax_solver",
        "networkmodel.backend",
        "networkmodel.cache",
        "networkmodel.mode_outputs",
    ]:
        assert importlib.import_module(name)


def test_kinopt_dummy_preprocess_objective_multistart_and_export(tmp_path):
    from kinopt.local.exporter.plotout import export_outcomes_to_csv, format_timepoints
    from kinopt.local.objfn.minfn import _estimated_series, _objective
    from kinopt.local.opt.optrun import multistart_run_optimization
    from kinopt.local.optcon.construct import (
        _build_K_data,
        _build_P_initial,
        _build_constraints,
        _compute_time_weights,
        _convert_to_sparse,
        _init_parameters,
        _precompute_mappings,
    )

    full_df, interact_df = kinopt_dummy_frames()
    P_initial, P_array = _build_P_initial(full_df, interact_df)
    K_index, K_array, beta_counts = _build_K_data(full_df, interact_df, estimate_missing=False)
    K_sparse, K_data, K_indices, K_indptr = _convert_to_sparse(K_array)
    mappings = _precompute_mappings(P_initial, K_index)
    unique_kinases, gene_counts, gene_starts, gene_kinase_idx, total_alpha, beta_counts_arr, beta_starts = mappings
    t_max, P_dense, weights = _compute_time_weights(P_array, loss_type="weighted")

    assert P_dense.shape == (2, 14)
    assert K_sparse.shape == (2, 14)
    assert unique_kinases == ["K1", "K2"]
    assert format_timepoints([0, 0.5, 1.0]) == ["0", "0.5", "1"]

    np.random.seed(0)
    params_initial, bounds = _init_parameters(total_alpha, 0.0, 1.0, beta_counts_arr)
    assert len(params_initial) == total_alpha + int(beta_counts_arr.sum())

    params = np.asarray([0.55, 0.45, 1.0, 1.0, 1.0], dtype=np.float64)
    loss = _objective(
        params,
        P_dense,
        t_max,
        P_dense.shape[0],
        gene_starts,
        gene_counts,
        gene_kinase_idx,
        total_alpha,
        beta_starts,
        beta_counts_arr,
        K_data,
        K_indices,
        K_indptr,
        weights,
        0,
    )
    estimated = _estimated_series(
        params, t_max, P_dense.shape[0], gene_starts, gene_counts, gene_kinase_idx,
        total_alpha, beta_starts, beta_counts_arr, K_data, K_indices, K_indptr,
    )
    assert np.isfinite(loss)
    assert estimated.shape == P_dense.shape

    constraints = _build_constraints("SLSQP", gene_counts, unique_kinases, total_alpha, beta_counts_arr, len(params))
    assert all(abs(c["fun"](params)) < 1e-12 for c in constraints)

    def quadratic(x):
        return float(np.sum((np.asarray(x) - 0.25) ** 2))

    best_result, best_params, outcomes = multistart_run_optimization(
        quadratic,
        np.asarray([0.8, 0.2]),
        "SLSQP",
        [(0.0, 1.0), (0.0, 1.0)],
        [],
        n_starts=2,
        n_jobs=1,
        base_seed=3,
        init_strategy="uniform",
    )
    assert len(outcomes) == 2
    assert outcomes[0].fun <= outcomes[1].fun
    assert np.allclose(best_result.x, best_params)
    csv_path = tmp_path / "kinopt_multistart.csv"
    export_outcomes_to_csv(outcomes, csv_path)
    assert pd.read_csv(csv_path)["rank"].tolist() == [1, 2]


def test_kinopt_wrong_columns_fail_fast():
    from kinopt.local.optcon.construct import _build_P_initial

    full_df, interact_df = kinopt_dummy_frames()
    with pytest.raises(KeyError):
        _build_P_initial(full_df.drop(columns=["x1"]), interact_df)


def test_tfopt_dummy_preprocess_objective_optimization_visualization_and_export(tmp_path):
    from tfopt.local.exporter.plotout import plot_estimated_vs_observed
    from tfopt.local.exporter.sheetutils import export_multistart_results, save_multistart_solutions_npz
    from tfopt.local.objfn.minfn import compute_predictions, objective_wrapper
    from tfopt.local.opt.optrun import MultiStartConfig, run_optimizer, run_optimizer_multistart
    from tfopt.local.optcon.construct import (
        build_fixed_arrays,
        build_linear_constraints,
        constraint_alpha_func,
        constraint_beta_func,
    )

    gene_ids, expression, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map = tfopt_dummy_data()
    arrays = build_fixed_arrays(gene_ids, expression, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map)
    expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, _, _, num_psites = arrays
    n_genes, n_tf = len(gene_ids), len(tf_ids)
    n_alpha = n_genes * n_reg
    beta_start_indices = np.asarray([0, 1 + num_psites[0]], dtype=np.int32)
    no_psite_tf = np.asarray([False, False])
    beta_len = int(sum(1 + n for n in num_psites))
    x0 = np.r_[np.full(n_alpha, 1.0 / n_reg), np.full(beta_len, 0.5)]
    bounds = [(0.0, 1.0)] * x0.size
    constraints = build_linear_constraints(n_genes, n_tf, n_reg, n_alpha, beta_start_indices, num_psites, no_psite_tf)

    assert regulators.shape == (2, 2)
    assert psite_tensor.shape == (2, 1, 9)
    assert np.allclose(constraint_alpha_func(x0, n_genes, n_reg), 0.0)
    assert np.allclose(constraint_beta_func(x0, n_alpha, n_tf, beta_start_indices, num_psites, no_psite_tf), 0.0)

    loss = objective_wrapper(
        x0, expression_matrix, regulators, tf_protein_matrix, psite_tensor,
        n_reg, expression_matrix.shape[1], n_genes, beta_start_indices, num_psites, 0,
    )
    preds = compute_predictions(x0, regulators, tf_protein_matrix, psite_tensor, n_reg, 9, n_genes, beta_start_indices, num_psites)
    assert np.isfinite(loss)
    assert preds.shape == expression_matrix.shape

    result = run_optimizer(
        x0, bounds, constraints, expression_matrix, regulators, tf_protein_matrix, psite_tensor,
        n_reg, 9, n_genes, beta_start_indices, num_psites, 0,
    )
    assert np.isfinite(result.fun)

    best, ranked = run_optimizer_multistart(
        x0, bounds, constraints, expression_matrix, regulators, tf_protein_matrix, psite_tensor,
        n_reg, 9, n_genes, beta_start_indices, num_psites, 0, run_optimizer,
        cfg=MultiStartConfig(n_starts=2, n_jobs=1, seed=5, backend="threading", prefer="threads"),
        polish=False,
    )
    assert ranked[0].fun <= ranked[-1].fun
    assert np.isfinite(best.fun)

    plot_estimated_vs_observed(preds, expression_matrix, gene_ids, TFOPT_TIME_POINTS, regulators, tf_protein_matrix, tf_ids, 1, save_path=tmp_path)
    assert (tmp_path / "G1_model_fit_.png").exists()
    summary = export_multistart_results(ranked)
    assert {"start_id", "fun", "success"}.issubset(summary.columns)
    npz_path = tmp_path / "tfopt_solutions.npz"
    save_multistart_solutions_npz(ranked, npz_path)
    assert npz_path.exists()


def test_tfopt_shape_mismatch_fails_fast():
    from tfopt.local.optcon.construct import build_fixed_arrays

    gene_ids, expression, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map = tfopt_dummy_data()
    bad_tf_protein = dict(tf_protein)
    bad_tf_protein["TF1"] = np.asarray([1.0, 2.0])
    with pytest.raises(ValueError):
        build_fixed_arrays(gene_ids, expression, tf_ids, bad_tf_protein, tf_psite_data, tf_psite_labels, reg_map)


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

    mode = {"fit_mrna": True, "fit_protein": True, "fit_phospho": True, "n_rna": len(mrna), "scale_mrna": 1.0, "scale_protein": 1.0, "scale_phospho": 1.0}
    target = {"mrna": jnp.asarray(mrna), "protein": jnp.asarray(protein), "phospho": jnp.asarray(phospho)}
    objective = lambda x: protwise_objective(x, target, init, 1, t, mode, "distmod")
    assert np.isfinite(float(objective(params)))
    grad = jax.grad(objective)(jnp.asarray(params))
    assert grad.shape == params.shape

    lower, upper = _normalize_bounds({"A": (0.01, 1), "B": (0.01, 1), "C": (0.01, 1), "D": (0.01, 1), "S(i)": (0.01, 1), "D(i)": (0.01, 1)}, "distmod", 1)
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

    ctx = InferenceContext(objective, theta0, np.zeros(2), np.ones(2), mode, tmp_path, parameter_names=("k1", "k2"), maxiter=2)
    ensemble = run_multistart(ctx, n_starts=2, seed=2, max_workers=1)
    assert len(ensemble["summary"]) == 2
    assert bool(ensemble["summary"].sort_values("final_objective").iloc[0]["selected_best"])

    assert np.allclose(project_alpha_blocks(np.asarray([0.2, 0.8]), [0, 0]).sum(), 1.0)
    assert np.all(project_beta_blocks(np.asarray([-3.0, 4.0]), [0, 0]) <= 4.0)

    write_mode_metadata(tmp_path, mode, objective_value=value)
    write_scalar_result_tables(tmp_path, mode, [value])
    pred_df = pd.DataFrame({"time": [0.0, 1.0], "pred_fc": [1.0, 1.1]})
    plots = save_mode_plots(tmp_path, mode, {"protein": pred_df, "mrna": pred_df, "phospho": pred_df})
    save_dashboard_bundle(tmp_path, args=SimpleNamespace(dummy=True), res=SimpleNamespace(X=np.asarray([params]), F=np.asarray([[value]]), objective_value=value, params=params, state=None, data_mode=mode, loss_breakdown={}), slices={}, xl=np.zeros(2), xu=np.ones(2), defaults={}, lambdas={}, solver_times=time_grid, df_prot=df_prot, df_rna=df_rna, df_pho=df_pho)
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
        root / "kinopt" / "local" / "objfn" / "minfn.py",
        root / "kinopt" / "local" / "opt" / "optrun.py",
        root / "kinopt" / "local" / "optcon" / "construct.py",
        root / "tfopt" / "local" / "objfn" / "minfn.py",
        root / "tfopt" / "local" / "opt" / "optrun.py",
        root / "tfopt" / "local" / "optcon" / "construct.py",
        root / "protwise" / "paramest" / "normest.py",
        root / "protwise" / "models" / "diffrax_solver.py",
        root / "networkmodel" / "backend.py",
        root / "networkmodel" / "OptimizationProblem.py",
        root / "networkmodel" / "runner.py",
    ]
    text = "\n".join(path.read_text() for path in active_paths)
    forbidden = ["from pymoo", "import pymoo", "scipy.optimize", "odeint(", "solve_ivp(", "Pareto front", "pareto front"]
    for term in forbidden:
        assert term not in text
    assert "evolutionary optimization" not in text.lower()
