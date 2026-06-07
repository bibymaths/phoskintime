"""Inference utilities for PhosKinTime scalar JAXopt/Diffrax models.

The functions here operate on a caller-provided scalar JAX objective and numeric
parameter vectors. They are shared by networkmodel and protwise wrappers and keep
pandas/matplotlib work outside differentiated functions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import subprocess
from networkmodel.backend import (
    DataMode,
    ensure_jax_float64,
    optimize_scalar_objective,
    project_alpha_blocks,
    project_beta_blocks,
    project_bounds,
)

from config.config import setup_logger
from networkmodel.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


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


def configure_numpyro_parallel_chains(
        num_chains: int,
        *,
        cpu_threads_per_chain: int = 1,
        logger_obj=None,
) -> dict:
    """Configure JAX CPU devices for parallel NumPyro NUTS chains.

    Must run before JAX is imported for XLA device-count changes to fully apply.
    """
    log = logger_obj or logger
    chains = max(1, int(num_chains))
    threads = max(1, int(cpu_threads_per_chain))

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(threads))

    xla_flags = os.environ.get("XLA_FLAGS", "")

    needed = f"--xla_force_host_platform_device_count={chains}"
    if needed not in xla_flags:
        os.environ["XLA_FLAGS"] = f"{xla_flags} {needed}".strip()

    # Keep Eigen intra-op threading controlled. Parallelism should come from chains.
    if "--xla_cpu_multi_thread_eigen" not in os.environ["XLA_FLAGS"]:
        os.environ["XLA_FLAGS"] += " --xla_cpu_multi_thread_eigen=false"

    if "jax" in sys.modules:
        log.warning(
            "[Posterior] JAX was already imported before setting XLA_FLAGS. "
            "Parallel CPU device count may not change in this run. "
            "Set XLA_FLAGS before importing networkmodel for full parallel-chain speedup."
        )

    strategy = {
        "num_chains": chains,
        "cpu_threads_per_chain": threads,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    log.info("[Posterior] NumPyro parallel-chain strategy: %s", strategy)
    return strategy


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


def _posterior_profile_bounds(
        lower,
        upper,
        names: Sequence[str] | None,
        nonnegative_prefixes: tuple[str, ...] = ("A_i", "B_i", "C_i", "D_i", "Dp_i", "E_i", "tf_scale"),
        eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    lower_eff = np.asarray(lower, dtype=np.float64).copy()
    upper_eff = np.asarray(upper, dtype=np.float64).copy()

    param_names = _param_names(len(lower_eff), names)

    for i, name in enumerate(param_names):
        if name == "tf_scale" or name.startswith(nonnegative_prefixes):
            lower_eff[i] = max(lower_eff[i], 0.0)

    bad = upper_eff <= lower_eff
    if np.any(bad):
        upper_eff[bad] = lower_eff[bad] + eps

    return lower_eff, upper_eff


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
        raise RuntimeError("All multistart optimization runs failed see optimization/multistart_summary.csv.")
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
    fig.tight_layout()
    fig.savefig(plot_dir / "objective_distribution.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.arange(len(finite)), np.sort(finite["final_objective"].to_numpy()), marker="o")
    ax.set_xlabel("rank")
    ax.set_ylabel("final scalar objective")
    fig.tight_layout()
    fig.savefig(plot_dir / "ranked_objective.png", dpi=300)
    plt.close(fig)
    if names:
        first = names[0]
        fig, ax = plt.subplots(figsize=(4, 3))
        good = params[params["success"]]
        ax.scatter(good[first], finite.sort_values("start_id")["final_objective"].to_numpy()[: len(good)])
        ax.set_xlabel(first)
        ax.set_ylabel("final scalar objective")
        fig.tight_layout()
        fig.savefig(plot_dir / "parameter_objective_tradeoff.png", dpi=300)
        plt.close(fig)


def run_profile_likelihood(ctx: InferenceContext, *, parameter_indices: Sequence[int], grid_size: int = 5) -> dict:
    out = Path(ctx.output_dir) / "profiles"
    plot_dir = Path(ctx.output_dir) / "plots" / "profile_likelihood"
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    lower_eff, upper_eff = _posterior_profile_bounds(
        ctx.lower,
        ctx.upper,
        ctx.parameter_names,
    )

    all_rows = []
    for idx in parameter_indices:
        idx = int(idx)
        values = np.linspace(lower_eff[idx], upper_eff[idx], int(grid_size))
        rows = []
        for gv in values:
            fixed_mask = np.zeros_like(ctx.theta0, dtype=bool) if ctx.fixed_mask is None else np.asarray(ctx.fixed_mask,
                                                                                                         dtype=bool).copy()
            fixed_values = np.asarray(ctx.theta0, dtype=np.float64).copy() if ctx.fixed_values is None else np.asarray(
                ctx.fixed_values, dtype=np.float64).copy()
            fixed_mask[idx] = True
            fixed_values[idx] = gv
            try:
                params, state, value = optimize_scalar_objective(ctx.objective_fun, fixed_values, lower_eff[idx],
                                                                 upper_eff[idx], maxiter=ctx.maxiter, tol=ctx.tol,
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
            rows.append(row)
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
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_numpyro_posterior(
        ctx: InferenceContext,
        *,
        num_warmup: int = 20,
        num_samples: int = 30,
        seed: int = 0,
        num_chains: int = 1,
        chain_method: str = "sequential",
) -> dict:
    # configure_numpyro_parallel_chains(
    #     num_chains=num_chains,
    #     cpu_threads_per_chain=1,
    #     logger_obj=logger,
    # )

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
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    lower_np, upper_np = _posterior_profile_bounds(
        ctx.lower,
        ctx.upper,
        ctx.parameter_names,
    )

    theta_center_np = np.clip(
        np.asarray(ctx.theta0, dtype=np.float64),
        lower_np,
        upper_np,
    )

    theta_center = jnp.asarray(theta_center_np, dtype=jnp.float64)
    lower = jnp.asarray(lower_np, dtype=jnp.float64)
    upper = jnp.asarray(upper_np, dtype=jnp.float64)

    def model():
        theta = numpyro.sample(
            "theta",
            dist.TruncatedNormal(
                loc=theta_center,
                scale=jnp.ones_like(theta_center),
                low=lower,
                high=upper,
            ).to_event(1),
        )

        if ctx.fixed_mask is not None and ctx.fixed_values is not None:
            fixed_mask = jnp.asarray(ctx.fixed_mask, dtype=bool)
            fixed_values = jnp.clip(
                jnp.asarray(ctx.fixed_values, dtype=jnp.float64),
                lower,
                upper,
            )
            theta = jnp.where(fixed_mask, fixed_values, theta)

        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        objective = ctx.objective_fun(theta)

        numpyro.deterministic("scalar_objective", objective)
        numpyro.factor("objective_likelihood", -0.5 * objective / (sigma * sigma + 1e-6))

    kernel = NUTS(
        model,
        init_strategy=numpyro.infer.init_to_value(
            values=
            {
                "theta": theta_center
            }
        )
    )

    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        chain_method=chain_method,
        progress_bar=True,
    )

    mcmc.run(jax.random.PRNGKey(int(seed)))

    extra_fields = mcmc.get_extra_fields(group_by_chain=False)
    if extra_fields:
        diagnostics = {}
        for key, value in extra_fields.items():
            arr = np.asarray(value)
            if arr.ndim <= 1:
                diagnostics[key] = arr.tolist()
        with open(out / "posterior_extra_fields.json", "w") as f:
            json.dump(diagnostics, f, indent=2)

    samples = mcmc.get_samples(group_by_chain=False)

    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    theta_samples = np.asarray(samples["theta"])
    sample_df = pd.DataFrame(theta_samples, columns=names)
    sample_df["sigma"] = np.asarray(samples["sigma"])

    if "scalar_objective" in samples:
        sample_df["scalar_objective"] = np.asarray(samples["scalar_objective"])
    sample_df["data_mode"] = ctx.mode.data_mode
    sample_df.to_csv(out / "posterior_samples.csv", index=False)
    summary_rows = []

    for col in names + ["sigma"]:
        vals = sample_df[col].to_numpy(dtype=np.float64)
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


def _run_single_numpyro_chain_process(
        ctx: InferenceContext,
        *,
        chain_id: int,
        seed: int,
        num_warmup: int,
        num_samples: int,
        base_output_dir: str | Path,
) -> None:
    """Run one independent NumPyro NUTS chain in one OS process.

    Uses single-chain sequential NumPyro to avoid Diffrax closure-conversion issues
    from NumPyro's internal parallel chain mode.
    """
    chain_out = Path(base_output_dir) / "posterior_chains" / f"chain_{chain_id:03d}"
    chain_out.mkdir(parents=True, exist_ok=True)

    child_ctx = replace(ctx, output_dir=chain_out)

    try:
        run_numpyro_posterior(
            child_ctx,
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            seed=int(seed) + int(chain_id),
            num_chains=1,
            chain_method="sequential",
        )

        status = {
            "chain_id": int(chain_id),
            "seed": int(seed) + int(chain_id),
            "success": True,
            "failure_reason": "",
            "posterior_samples": str(chain_out / "posterior" / "posterior_samples.csv"),
            "posterior_summary": str(chain_out / "posterior" / "posterior_summary.csv"),
        }

    except Exception as exc:
        status = {
            "chain_id": int(chain_id),
            "seed": int(seed) + int(chain_id),
            "success": False,
            "failure_reason": str(exc),
            "posterior_samples": "",
            "posterior_summary": "",
        }

    with open(chain_out / "chain_status.json", "w") as f:
        json.dump(status, f, indent=2)


def run_numpyro_posterior_standalone_processes(
        *,
        run_config_path: str | Path,
        output_dir: str | Path,
        num_warmup: int = 20,
        num_samples: int = 30,
        seed: int = 0,
        num_processes: int = 4,
        timeout_seconds: int | None = None,
) -> dict:
    """Run independent posterior chains as clean Python subprocesses.

    This avoids passing a live JAX/Diffrax objective through multiprocessing.
    Each subprocess rebuilds the objective from disk and runs one sequential NUTS chain.
    """
    run_config_path = Path(run_config_path)
    output_dir = Path(output_dir)

    chains_root = output_dir / "posterior_chains"
    out = output_dir / "posterior"
    plot_dir = output_dir / "plots" / "posterior"

    chains_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_proc = max(1, int(num_processes))
    timeout_seconds = timeout_seconds or int(os.environ.get("POSTERIOR_CHAIN_TIMEOUT_SECONDS", "0") or 0)

    logger.info(
        "[Posterior] Launching %d standalone posterior worker subprocesses.",
        n_proc,
    )

    procs = []

    for chain_id in range(n_proc):
        chain_log = chains_root / f"chain_{chain_id:03d}" / "worker_stdout_stderr.log"
        chain_log.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "networkmodel.PosteriorWorker",
            "--run-config",
            str(run_config_path),
            "--chain-id",
            str(chain_id),
            "--seed",
            str(int(seed) + chain_id),
            "--num-warmup",
            str(int(num_warmup)),
            "--num-samples",
            str(int(num_samples)),
        ]

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

        fh = open(chain_log, "w")

        logger.info("[Posterior] Starting chain %d: %s", chain_id, " ".join(cmd))

        p = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
        )

        procs.append((chain_id, p, fh, chain_log))

    statuses = []

    for chain_id, p, fh, chain_log in procs:
        try:
            if timeout_seconds > 0:
                return_code = p.wait(timeout=timeout_seconds)
            else:
                return_code = p.wait()

        except subprocess.TimeoutExpired:
            logger.error("[Posterior] Chain %d exceeded timeout; terminating.", chain_id)
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            return_code = p.returncode if p.returncode is not None else -9

        finally:
            fh.close()

        status_path = chains_root / f"chain_{chain_id:03d}" / "chain_status.json"

        if status_path.exists():
            with open(status_path) as f:
                status = json.load(f)
        else:
            status = {
                "chain_id": chain_id,
                "pid": p.pid,
                "seed": int(seed) + chain_id,
                "success": False,
                "state": "missing_status",
                "failure_reason": "chain_status.json was not written",
                "posterior_samples": "",
                "posterior_summary": "",
            }

        status["return_code"] = int(return_code)
        status["worker_log"] = str(chain_log)
        statuses.append(status)

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out / "posterior_chain_status.csv", index=False)

    failed = status_df[(status_df["success"] != True) | (status_df["return_code"] != 0)]
    if len(failed) > 0:
        logger.warning(
            "[Posterior] %d/%d standalone posterior chains failed. "
            "See posterior/posterior_chain_status.csv.",
            len(failed),
            n_proc,
        )

    sample_dfs = []

    for status in statuses:
        if not bool(status.get("success", False)):
            continue

        sample_path = status.get("posterior_samples", "")
        if not sample_path or not Path(sample_path).exists():
            continue

        df = pd.read_csv(sample_path)
        df.insert(0, "chain", int(status["chain_id"]))
        sample_dfs.append(df)

    if not sample_dfs:
        raise RuntimeError(
            "All standalone posterior chains failed. "
            "See posterior/posterior_chain_status.csv and posterior_chains/*/worker_stdout_stderr.log."
        )

    sample_df = pd.concat(sample_dfs, ignore_index=True)
    sample_df.to_csv(out / "posterior_samples.csv", index=False)

    parameter_cols = [
        c for c in sample_df.columns
        if c not in {"chain", "sigma", "scalar_objective", "data_mode"}
           and pd.api.types.is_numeric_dtype(sample_df[c])
    ]

    summary_rows = []
    for col in parameter_cols + ["sigma"]:
        if col not in sample_df.columns:
            continue

        vals = sample_df[col].to_numpy(dtype=np.float64)

        summary_rows.append(
            {
                "parameter": col,
                "mean": vals.mean(),
                "median": np.median(vals),
                "sd": vals.std(ddof=0),
                "ci_05": np.quantile(vals, 0.05),
                "ci_95": np.quantile(vals, 0.95),
                "ess": float(len(vals)),
                "r_hat": np.nan,
                "data_mode": sample_df["data_mode"].iloc[0] if "data_mode" in sample_df else "",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "posterior_summary.csv", index=False)

    if "scalar_objective" in sample_df.columns:
        predictive = sample_df[["chain", "scalar_objective", "data_mode"]].copy()
    else:
        predictive = pd.DataFrame({"data_mode": [sample_df["data_mode"].iloc[0]] if "data_mode" in sample_df else [""]})

    predictive.to_csv(out / "posterior_predictive.csv", index=False)

    _plot_posterior(sample_df, summary, plot_dir)

    logger.info(
        "[Posterior] Standalone posterior complete. Successful chains: %d/%d. Combined samples: %d.",
        len(sample_dfs),
        n_proc,
        len(sample_df),
    )

    return {
        "samples": sample_df,
        "summary": summary,
        "posterior_predictive": predictive,
        "chain_status": status_df,
        "output_dir": out,
    }


def _plot_posterior(samples: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path) -> None:
    numeric = [
        c for c in samples.columns
        if pd.api.types.is_numeric_dtype(samples[c])
    ]

    density_dir = plot_dir / "density"
    density_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric:
        values = samples[col].to_numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(values)
        ax.set_title(f"trace {col}")
        ax.set_xlabel("sample")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(density_dir / f"trace_{col}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=15)
        ax.set_title(f"posterior {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")
        fig.tight_layout()
        fig.savefig(density_dir / f"density_{col}.png", dpi=150)
        plt.close(fig)

    n_params = len(summary)

    fig_width = max(10, 0.75 * n_params)
    fig_height = 5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = range(n_params)

    ax.errorbar(
        x,
        summary["median"],
        yerr=[
            summary["median"] - summary["ci_05"],
            summary["ci_95"] - summary["median"],
        ],
        fmt="o",
        capsize=4,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        summary["parameter"].astype(str),
        rotation=60,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_ylabel("posterior median and 90% interval")
    ax.set_xlabel("parameter")

    ax.margins(x=0.03)

    fig.subplots_adjust(
        bottom=0.35,
        left=0.08,
        right=0.98,
        top=0.92,
    )

    fig.savefig(plot_dir / "credible_intervals.png", dpi=150)
    plt.close(fig)
