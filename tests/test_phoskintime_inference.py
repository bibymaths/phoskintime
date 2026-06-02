from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from networkmodel.inference import (
    InferenceContext,
    configure_jax_parallelism,
    run_multistart,
    run_numpyro_posterior,
    run_profile_likelihood,
)
from networkmodel.jax_backend import DataMode, project_alpha_blocks, project_beta_blocks


def _ctx(tmp_path, objective=None):
    mode = DataMode(("mrna", "protein"), "mrna+protein", True, True, False)
    obj = objective or (lambda x: jnp.sum((x - jnp.asarray([0.2, 0.8, 0.4])) ** 2))
    return InferenceContext(
        objective_fun=obj,
        theta0=np.asarray([0.5, 0.5, 0.5]),
        lower=np.zeros(3),
        upper=np.ones(3),
        mode=mode,
        output_dir=tmp_path,
        parameter_names=("k_tx", "k_tl", "k_ph"),
        fixed_mask=np.asarray([False, True, False]),
        fixed_values=np.asarray([0.0, 0.5, 0.0]),
        alpha_block_ids=np.asarray([0, 0]),
        beta_block_ids=np.asarray([0, 0]),
        loss_kwargs={"time_grid": [0.0, 1.0], "regularization": 0.0},
        maxiter=12,
    )


def test_parallel_strategy_is_conservative(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    strategy = configure_jax_parallelism(max_workers=2)
    assert strategy["effective_workers"] == 2
    assert "intra_op_parallelism_threads=1" in strategy["xla_flags"]
    assert strategy["omp_num_threads"] == "1"


def test_multistart_runs_selects_best_saves_csv_plots_and_constraints(tmp_path):
    ctx = _ctx(tmp_path)
    result = run_multistart(ctx, n_starts=3, seed=7, max_workers=1)
    summary = result["summary"]
    params = result["parameters"]
    assert len(summary) == 3
    assert summary["selected_best"].sum() == 1
    assert summary["success"].all()
    assert (tmp_path / "optimization" / "multistart_summary.csv").exists()
    assert (tmp_path / "optimization" / "multistart_parameters.csv").exists()
    assert (tmp_path / "optimization" / "best_fit.csv").exists()
    assert (tmp_path / "plots" / "multistart" / "objective_distribution.png").exists()
    assert (tmp_path / "plots" / "multistart" / "ranked_objective.png").exists()
    assert (tmp_path / "plots" / "multistart" / "parameter_objective_tradeoff.png").exists()
    for _, row in params[params["success"]].iterrows():
        alpha = np.asarray([row["alpha_0"], row["alpha_1"]])
        beta = np.asarray([row["beta_0"], row["beta_1"]])
        assert np.allclose(project_alpha_blocks(alpha, [0, 0]), alpha)
        assert np.isclose(alpha.sum(), 1.0)
        assert np.all(np.asarray(project_beta_blocks(beta, [0, 0])) <= 4.0)
        assert np.isclose(beta.sum(), 1.0)
        assert row["k_tl"] == pytest.approx(0.5)


def test_multistart_failed_starts_are_saved_without_success(tmp_path):
    ctx = _ctx(tmp_path, objective=lambda x: jnp.asarray(jnp.nan))
    with pytest.raises(RuntimeError, match="All multistart optimization runs failed"):
        run_multistart(ctx, n_starts=2, seed=0, max_workers=1)
    df = pd.read_csv(tmp_path / "optimization" / "multistart_summary.csv")
    assert len(df) == 2
    assert not df["success"].any()
    assert df["failure_reason"].str.contains("final scalar objective is not finite").all()


def test_profile_likelihood_fixes_profiled_parameter_reoptimizes_and_saves(tmp_path):
    ctx = _ctx(tmp_path)
    result = run_profile_likelihood(ctx, parameter_indices=[0], grid_size=3)
    summary = result["summary"]
    assert len(summary) == 3
    assert summary["parameter_name"].unique().tolist() == ["k_tx"]
    assert summary["success"].all()
    for _, row in summary.iterrows():
        optimized = np.asarray(json.loads(row["optimized_free_parameters"]), dtype=float)
        assert optimized[0] == pytest.approx(row["grid_value"])
        assert optimized[1] == pytest.approx(0.5)
    assert (tmp_path / "profiles" / "profile_likelihood_summary.csv").exists()
    assert (tmp_path / "profiles" / "profile_likelihood_k_tx.csv").exists()
    assert (tmp_path / "plots" / "profile_likelihood" / "profile_likelihood_k_tx.png").exists()
    assert set(summary["active_loss_terms"]) == {"mrna_loss,protein_loss"}


def test_numpyro_posterior_runs_or_dependency_error_and_saves(tmp_path):
    ctx = _ctx(tmp_path)
    try:
        result = run_numpyro_posterior(ctx, num_warmup=5, num_samples=6, seed=3)
    except RuntimeError as exc:
        assert "numpyro" in str(exc).lower()
        return
    assert len(result["samples"]) == 6
    assert {"k_tx", "k_tl", "k_ph", "sigma", "data_mode"}.issubset(result["samples"].columns)
    assert {"mean", "median", "sd", "ci_05", "ci_95", "ess", "r_hat"}.issubset(result["summary"].columns)
    assert (tmp_path / "posterior" / "posterior_samples.csv").exists()
    assert (tmp_path / "posterior" / "posterior_summary.csv").exists()
    assert (tmp_path / "posterior" / "posterior_predictive.csv").exists()
    assert (tmp_path / "plots" / "posterior" / "credible_intervals.png").exists()


def test_dashboard_imports_inference_outputs(tmp_path, monkeypatch):
    (tmp_path / "optimization").mkdir()
    (tmp_path / "profiles").mkdir()
    (tmp_path / "posterior").mkdir()
    pd.DataFrame({"final_objective": [0.1]}).to_csv(tmp_path / "optimization" / "multistart_summary.csv", index=False)
    pd.DataFrame({"parameter_name": ["k"], "objective_value": [0.1]}).to_csv(tmp_path / "profiles" / "profile_likelihood_summary.csv", index=False)
    pd.DataFrame({"parameter": ["k"], "mean": [0.2]}).to_csv(tmp_path / "posterior" / "posterior_summary.csv", index=False)
    fake_streamlit = types.SimpleNamespace(image=lambda *a, **k: None, video=lambda *a, **k: None, markdown=lambda *a, **k: None)
    fake_px = types.SimpleNamespace(scatter=lambda *a, **k: types.SimpleNamespace(add_trace=lambda *a, **k: None, update_layout=lambda *a, **k: None))
    fake_go = types.SimpleNamespace(Scatter=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "plotly", types.SimpleNamespace(express=fake_px, graph_objects=fake_go))
    monkeypatch.setitem(sys.modules, "plotly.express", fake_px)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)
    dashboard = importlib.reload(importlib.import_module("networkmodel.dashboard_app"))
    outputs = dashboard._load_inference_outputs(tmp_path)
    assert outputs["multistart_summary"] is not None
    assert outputs["profile_likelihood"] is not None
    assert outputs["posterior_summary"] is not None


def test_forbidden_stack_not_in_inference_active_paths():
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = [root / "networkmodel" / "inference.py", root / "protwise" / "paramest" / "inference.py"]
    text = "\n".join(p.read_text() for p in paths)
    for term in ["from pymoo", "import pymoo", "scipy.optimize", "from scipy.integrate", "solve_ivp(", "odeint("]:
        assert term not in text
    assert "TO" + "DO" not in text
    assert "FIX" + "ME" not in text
