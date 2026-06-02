"""Inference utilities for PhosKinTime scalar JAXopt/Diffrax models.

The functions here operate on a caller-provided scalar JAX objective and numeric
parameter vectors. They are shared by networkmodel and protwise wrappers and keep
pandas/matplotlib work outside differentiated functions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from networkmodel.jax_backend import (
    DataMode,
    ensure_jax_float64,
    optimize_scalar_objective,
    project_alpha_blocks,
    project_beta_blocks,
    project_bounds,
)

logger = logging.getLogger()


@dataclass(frozen=True)
class InferenceContext:
    objective_fun: Callable
    theta0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mode: DataMode
    output_dir: str | Path
    parameter_names: tuple[str, ...] | None = None
    fixed_mask: np.ndarray | None = None
    fixed_values: np.ndarray | None = None
    alpha_block_ids: np.ndarray | None = None
    beta_block_ids: np.ndarray | None = None
    loss_kwargs: Mapping | None = None
    maxiter: int = 50
    tol: float = 1e-6

    def build_worker_kwargs(self, start_id: int, seed: int, theta_start: np.ndarray) -> dict:
        """Create an immutable per-worker copy of numeric residual/loss metadata."""
        return {
            "start_id": int(start_id),
            "seed": int(seed),
            "theta_start": np.asarray(theta_start, dtype=np.float64).copy(),
            "lower": np.asarray(self.lower, dtype=np.float64).copy(),
            "upper": np.asarray(self.upper, dtype=np.float64).copy(),
            "fixed_mask": None if self.fixed_mask is None else np.asarray(self.fixed_mask, dtype=bool).copy(),
            "fixed_values": None if self.fixed_values is None else np.asarray(self.fixed_values,
                                                                              dtype=np.float64).copy(),
            "mode_metadata": {
                "data_mode": self.mode.data_mode,
                "available_layers": list(self.mode.available_layers),
                "active_loss_terms": list(self.mode.active_loss_terms),
                "skipped_loss_terms": list(self.mode.skipped_loss_terms),
            },
            "loss_kwargs": dict(self.loss_kwargs or {}),
        }


def configure_jax_parallelism(max_workers: int | None = None, logger_obj=None) -> dict:
    """Set conservative thread env defaults before JAX work and report strategy."""
    workers = max(1, int(max_workers or 1))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    ensure_jax_float64()
    strategy = {
        "requested_workers": workers,
        "effective_workers": workers,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
    (logger_obj or logger).info("[Inference] JAX/XLA multistart strategy: %s", strategy)
    return strategy


def generate_multistart_initials(theta0, lower, upper, n_starts: int, seed: int, fixed_mask=None, fixed_values=None) -> \
list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    theta0 = np.asarray(theta0, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    starts = [np.asarray(project_bounds(theta0, lower, upper, fixed_mask, fixed_values), dtype=np.float64)]
    for _ in range(1, int(n_starts)):
        candidate = rng.uniform(lower, upper)
        starts.append(np.asarray(project_bounds(candidate, lower, upper, fixed_mask, fixed_values), dtype=np.float64))
    return starts


def _param_names(n: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [f"param_{i}" for i in range(n)]
    out = list(names)
    if len(out) < n:
        out.extend(f"param_{i}" for i in range(len(out), n))
    return out[:n]


def _run_one_start(ctx: InferenceContext, start_id: int, seed: int, theta_start: np.ndarray) -> tuple[dict, np.ndarray]:
    worker_kwargs = ctx.build_worker_kwargs(start_id, seed, theta_start)
    begin = time.perf_counter()
    try:
        params, state, value = optimize_scalar_objective(
            ctx.objective_fun,
            worker_kwargs["theta_start"],
            worker_kwargs["lower"],
            worker_kwargs["upper"],
            maxiter=ctx.maxiter,
            tol=ctx.tol,
            fixed_mask=worker_kwargs["fixed_mask"],
            fixed_values=worker_kwargs["fixed_values"],
            logger_obj=logger,
        )
        elapsed = time.perf_counter() - begin
        row = {
            "start_id": start_id,
            "seed": seed,
            "convergence_status": "converged",
            "success": True,
            "failure_reason": "",
            "final_objective": float(value),
            "active_loss_terms": ",".join(ctx.mode.active_loss_terms),
            "number_of_iterations": int(getattr(state, "iter_num", -1)),
            "solver_status": "diffrax",
            "optimizer_status": "jaxopt.ProjectedGradient",
            "runtime_seconds": elapsed,
            "data_mode": ctx.mode.data_mode,
            "available_layers": ",".join(ctx.mode.available_layers),
        }
        return row, np.asarray(params, dtype=np.float64)
    except Exception as exc:
        elapsed = time.perf_counter() - begin
        logger.warning("[Inference] Start %s failed: %s", start_id, exc)
        row = {
            "start_id": start_id,
            "seed": seed,
            "convergence_status": "failed",
            "success": False,
            "failure_reason": str(exc),
            "final_objective": np.inf,
            "active_loss_terms": ",".join(ctx.mode.active_loss_terms),
            "number_of_iterations": 0,
            "solver_status": "failed",
            "optimizer_status": "failed",
            "runtime_seconds": elapsed,
            "data_mode": ctx.mode.data_mode,
            "available_layers": ",".join(ctx.mode.available_layers),
        }
        return row, np.full_like(ctx.theta0, np.nan, dtype=np.float64)


def run_multistart(ctx: InferenceContext, *, n_starts: int = 4, seed: int = 0, max_workers: int = 1) -> dict:
    out = Path(ctx.output_dir) / "optimization"
    plot_dir = Path(ctx.output_dir) / "plots" / "multistart"
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    strategy = configure_jax_parallelism(max_workers=max_workers, logger_obj=logger)
    starts = generate_multistart_initials(ctx.theta0, ctx.lower, ctx.upper, n_starts, seed, ctx.fixed_mask,
                                          ctx.fixed_values)
    rows: list[dict] = []
    params_by_start: list[np.ndarray] = []
    if strategy["effective_workers"] > 1:
        with ThreadPoolExecutor(max_workers=strategy["effective_workers"]) as ex:
            futs = [ex.submit(_run_one_start, ctx, i, seed + i, starts[i]) for i in range(len(starts))]
            for fut in as_completed(futs):
                row, params = fut.result()
                rows.append(row)
                params_by_start.append((row["start_id"], params))
    else:
        for i, theta in enumerate(starts):
            row, params = _run_one_start(ctx, i, seed + i, theta)
            rows.append(row)
            params_by_start.append((row["start_id"], params))
    rows = sorted(rows, key=lambda r: r["start_id"])
    param_map = {sid: params for sid, params in params_by_start}
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    summary = pd.DataFrame(rows)
    success = summary[summary["success"]]
    if success.empty:
        summary.to_csv(out / "multistart_summary.csv", index=False)
        raise RuntimeError("All multistart optimization runs failed; see optimization/multistart_summary.csv.")
    best_start = int(success.sort_values("final_objective").iloc[0]["start_id"])
    summary["selected_best"] = summary["start_id"] == best_start
    param_rows = []
    for row in rows:
        values = param_map[int(row["start_id"])]
        record = {"start_id": row["start_id"], "seed": row["seed"], "success": row["success"],
                  "selected_best": row["start_id"] == best_start}
        record.update({name: values[i] for i, name in enumerate(names)})
        if ctx.alpha_block_ids is not None:
            alpha = np.asarray(project_alpha_blocks(values[: len(ctx.alpha_block_ids)], ctx.alpha_block_ids),
                               dtype=float)
            record.update({f"alpha_{i}": v for i, v in enumerate(alpha)})
        if ctx.beta_block_ids is not None:
            beta = np.asarray(project_beta_blocks(values[: len(ctx.beta_block_ids)], ctx.beta_block_ids), dtype=float)
            record.update({f"beta_{i}": v for i, v in enumerate(beta)})
        param_rows.append(record)
    param_df = pd.DataFrame(param_rows)
    best_df = param_df[param_df["selected_best"]].copy()
    summary.to_csv(out / "multistart_summary.csv", index=False)
    param_df.to_csv(out / "multistart_parameters.csv", index=False)
    best_df.to_csv(out / "best_fit.csv", index=False)
    _plot_multistart(summary, param_df, names, plot_dir)
    return {"summary": summary, "parameters": param_df, "best": best_df, "best_start_id": best_start, "output_dir": out}


def _plot_multistart(summary: pd.DataFrame, params: pd.DataFrame, names: Sequence[str], plot_dir: Path) -> None:
    finite = summary[np.isfinite(summary["final_objective"])]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(finite["final_objective"], bins=max(1, min(10, len(finite))))
    ax.set_xlabel("final scalar objective")
    ax.set_ylabel("start count")
    fig.tight_layout();
    fig.savefig(plot_dir / "objective_distribution.png", dpi=120);
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.arange(len(finite)), np.sort(finite["final_objective"].to_numpy()), marker="o")
    ax.set_xlabel("rank")
    ax.set_ylabel("final scalar objective")
    fig.tight_layout();
    fig.savefig(plot_dir / "ranked_objective.png", dpi=120);
    plt.close(fig)
    if names:
        first = names[0]
        fig, ax = plt.subplots(figsize=(4, 3))
        good = params[params["success"]]
        ax.scatter(good[first], finite.sort_values("start_id")["final_objective"].to_numpy()[: len(good)])
        ax.set_xlabel(first);
        ax.set_ylabel("final scalar objective")
        fig.tight_layout();
        fig.savefig(plot_dir / "parameter_objective_tradeoff.png", dpi=120);
        plt.close(fig)


def run_profile_likelihood(ctx: InferenceContext, *, parameter_indices: Sequence[int], grid_size: int = 5) -> dict:
    out = Path(ctx.output_dir) / "profiles"
    plot_dir = Path(ctx.output_dir) / "plots" / "profile_likelihood"
    out.mkdir(parents=True, exist_ok=True);
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    all_rows = []
    for idx in parameter_indices:
        idx = int(idx)
        values = np.linspace(ctx.lower[idx], ctx.upper[idx], int(grid_size))
        rows = []
        for gv in values:
            fixed_mask = np.zeros_like(ctx.theta0, dtype=bool) if ctx.fixed_mask is None else np.asarray(ctx.fixed_mask,
                                                                                                         dtype=bool).copy()
            fixed_values = np.asarray(ctx.theta0, dtype=np.float64).copy() if ctx.fixed_values is None else np.asarray(
                ctx.fixed_values, dtype=np.float64).copy()
            fixed_mask[idx] = True;
            fixed_values[idx] = gv
            try:
                params, state, value = optimize_scalar_objective(ctx.objective_fun, fixed_values, ctx.lower, ctx.upper,
                                                                 maxiter=ctx.maxiter, tol=ctx.tol,
                                                                 fixed_mask=fixed_mask, fixed_values=fixed_values,
                                                                 logger_obj=logger)
                row = {"parameter_name": names[idx], "parameter_index": idx, "grid_value": gv, "objective_value": value,
                       "success": True, "optimizer_status": "jaxopt.ProjectedGradient", "failure_reason": "",
                       "optimized_free_parameters": json.dumps(params.tolist()), "data_mode": ctx.mode.data_mode,
                       "active_loss_terms": ",".join(ctx.mode.active_loss_terms)}
            except Exception as exc:
                row = {"parameter_name": names[idx], "parameter_index": idx, "grid_value": gv,
                       "objective_value": np.inf, "success": False, "optimizer_status": "failed",
                       "failure_reason": str(exc), "optimized_free_parameters": "[]", "data_mode": ctx.mode.data_mode,
                       "active_loss_terms": ",".join(ctx.mode.active_loss_terms)}
            rows.append(row);
            all_rows.append(row)
        df = pd.DataFrame(rows)
        finite_min = df.loc[np.isfinite(df["objective_value"]), "objective_value"].min()
        df["delta_objective"] = df["objective_value"] - finite_min
        df.to_csv(out / f"profile_likelihood_{names[idx]}.csv", index=False)
        _plot_profile(df, plot_dir / f"profile_likelihood_{names[idx]}.png")
    summary = pd.DataFrame(all_rows)
    if not summary.empty:
        mins = summary[np.isfinite(summary["objective_value"])].groupby("parameter_name")["objective_value"].transform(
            "min")
        summary.loc[np.isfinite(summary["objective_value"]), "delta_objective"] = summary.loc[np.isfinite(
            summary["objective_value"]), "objective_value"] - mins
    summary.to_csv(out / "profile_likelihood_summary.csv", index=False)
    return {"summary": summary, "output_dir": out}


def _plot_profile(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(df["grid_value"], df["objective_value"], marker="o", label="objective")
    ax.set_xlabel("profiled parameter value")
    ax.set_ylabel("objective")
    ax2 = ax.twinx()
    ax2.plot(df["grid_value"], df["delta_objective"], color="tab:orange", marker="x", label="delta")
    ax2.set_ylabel("delta objective")
    fig.tight_layout();
    fig.savefig(path, dpi=120);
    plt.close(fig)


def run_numpyro_posterior(ctx: InferenceContext, *, num_warmup: int = 20, num_samples: int = 30, seed: int = 0) -> dict:
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise RuntimeError(
            "NumPyro posterior inference requires the optional 'numpyro' dependency. Install numpyro to run posterior analysis.") from exc
    out = Path(ctx.output_dir) / "posterior"
    plot_dir = Path(ctx.output_dir) / "plots" / "posterior"
    out.mkdir(parents=True, exist_ok=True);
    plot_dir.mkdir(parents=True, exist_ok=True)
    theta_center = jnp.asarray(ctx.theta0, dtype=jnp.float64)
    lower = jnp.asarray(ctx.lower, dtype=jnp.float64);
    upper = jnp.asarray(ctx.upper, dtype=jnp.float64)

    def model():
        theta_raw = numpyro.sample("theta_raw", dist.Normal(theta_center, 1.0).to_event(1))
        theta = jnp.clip(theta_raw, lower, upper)
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        objective = ctx.objective_fun(theta)
        numpyro.deterministic("scalar_objective", objective)
        numpyro.factor("objective_likelihood", -0.5 * objective / (sigma * sigma + 1e-6))

    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values={"theta_raw": theta_center}))
    mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples), num_chains=1, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(int(seed)))
    samples = mcmc.get_samples()
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    theta_samples = np.asarray(samples["theta_raw"])
    sample_df = pd.DataFrame(theta_samples, columns=names)
    sample_df["sigma"] = np.asarray(samples["sigma"])
    if "scalar_objective" in samples:
        sample_df["scalar_objective"] = np.asarray(samples["scalar_objective"])
    sample_df["data_mode"] = ctx.mode.data_mode
    sample_df.to_csv(out / "posterior_samples.csv", index=False)
    summary_rows = []
    for col in names + ["sigma"]:
        vals = sample_df[col].to_numpy(dtype=float)
        summary_rows.append({"parameter": col, "mean": vals.mean(), "median": np.median(vals), "sd": vals.std(ddof=0),
                             "ci_05": np.quantile(vals, 0.05), "ci_95": np.quantile(vals, 0.95),
                             "ess": float(len(vals)), "r_hat": np.nan, "data_mode": ctx.mode.data_mode})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "posterior_summary.csv", index=False)
    predictive = sample_df[
        ["scalar_objective", "data_mode"]].copy() if "scalar_objective" in sample_df else pd.DataFrame(
        {"data_mode": [ctx.mode.data_mode]})
    predictive.to_csv(out / "posterior_predictive.csv", index=False)
    _plot_posterior(sample_df, summary, plot_dir)
    return {"samples": sample_df, "summary": summary, "posterior_predictive": predictive, "output_dir": out}


def _plot_posterior(samples: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path) -> None:
    numeric = [c for c in samples.columns if pd.api.types.is_numeric_dtype(samples[c])]
    for col in numeric[:3]:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(samples[col].to_numpy())
        ax.set_title(f"trace {col}")
        fig.tight_layout();
        fig.savefig(plot_dir / f"trace_{col}.png", dpi=120);
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(samples[col].to_numpy(), bins=15)
        ax.set_title(f"posterior {col}")
        fig.tight_layout();
        fig.savefig(plot_dir / f"density_{col}.png", dpi=120);
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(summary["parameter"], summary["median"],
                yerr=[summary["median"] - summary["ci_05"], summary["ci_95"] - summary["median"]], fmt="o")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("posterior median and 90% interval")
    fig.tight_layout();
    fig.savefig(plot_dir / "credible_intervals.png", dpi=120);
    plt.close(fig)
