"""Inference utilities for PhosKinTime scalar JAXopt/Diffrax models.

The functions here operate on a caller-provided scalar JAX objective and numeric
parameter vectors. They are shared by networkmodel and protwise wrappers and keep
pandas/matplotlib work outside differentiated functions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
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

    # Preserve raw theta-space bounds exactly except for invalid intervals. Physical
    # non-negativity constraints are already encoded by callers that construct theta
    # bounds and must not be imposed here because signed parameters can be profiled.
    _param_names(len(lower_eff), names)

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


def run_profile_likelihood_standalone_processes(
        *,
        run_config_path: str | Path,
        output_dir: str | Path,
        parameter_indices: Sequence[int],
        grid_size: int = 5,
        max_workers: int = 1,
        timeout_seconds: int | None = None,
) -> dict:
    """Run profile likelihood in standalone subprocesses.

    Each subprocess profiles one parameter over the requested grid and then exits,
    releasing JAX/XLA/LLVM memory back to the OS.
    """
    run_config_path = Path(run_config_path)
    output_dir = Path(output_dir)

    worker_root = output_dir / "profile_workers"
    out = output_dir / "profiles"
    plot_dir = output_dir / "plots" / "profile_likelihood"

    worker_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(i) for i in parameter_indices]
    n_workers = max(1, int(max_workers))
    timeout_seconds = timeout_seconds or int(os.environ.get("PROFILE_WORKER_TIMEOUT_SECONDS", "0") or 0)

    logger.info(
        "[Profile] Launching standalone profile workers for %d parameters with max_workers=%d.",
        len(indices),
        n_workers,
    )

    pending = list(indices)
    running: list[tuple[int, subprocess.Popen, object, Path]] = []
    statuses = []

    def _start_worker(param_idx: int):
        worker_dir = worker_root / f"param_{param_idx:04d}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        worker_log = worker_dir / "worker_stdout_stderr.log"

        cmd = [
            sys.executable,
            "-m",
            "networkmodel.ProfileWorker",
            "--run-config",
            str(run_config_path),
            "--parameter-index",
            str(param_idx),
            "--grid-size",
            str(int(grid_size)),
        ]

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

        fh = open(worker_log, "w")

        logger.info("[Profile] Starting parameter %d: %s", param_idx, " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
        )

        return param_idx, proc, fh, worker_log

    # Keep only max_workers active at once.
    while pending or running:
        while pending and len(running) < n_workers:
            running.append(_start_worker(pending.pop(0)))

        still_running = []

        for param_idx, proc, fh, worker_log in running:
            return_code = proc.poll()

            if return_code is None:
                still_running.append((param_idx, proc, fh, worker_log))
                continue

            fh.close()

            status_path = worker_root / f"param_{param_idx:04d}" / "profile_status.json"

            if status_path.exists():
                with open(status_path) as f:
                    status = json.load(f)
            else:
                status = {
                    "parameter_index": param_idx,
                    "success": False,
                    "state": "missing_status",
                    "failure_reason": "profile_status.json was not written",
                    "result_csv": "",
                }

            status["return_code"] = int(return_code)
            status["worker_log"] = str(worker_log)
            statuses.append(status)

        running = still_running

        if running:
            time.sleep(2.0)

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out / "profile_worker_status.csv", index=False)

    result_dfs = []

    for status in statuses:
        if not bool(status.get("success", False)):
            continue

        result_csv = status.get("result_csv", "")
        if not result_csv or not Path(result_csv).exists():
            continue

        df = pd.read_csv(result_csv)
        result_dfs.append(df)

        # Copy per-parameter CSV into canonical profiles directory.
        target = out / Path(result_csv).name
        df.to_csv(target, index=False)

        try:
            _plot_profile(
                df,
                plot_dir / Path(result_csv).name.replace(".csv", ".png"),
            )
        except Exception as exc:
            logger.warning(
                "[Profile] Could not plot profile %s: %s",
                result_csv,
                exc,
            )

    if not result_dfs:
        raise RuntimeError(
            "All standalone profile workers failed. "
            "See profiles/profile_worker_status.csv and profile_workers/*/worker_stdout_stderr.log."
        )

    summary = pd.concat(result_dfs, ignore_index=True)

    summary["delta_objective"] = np.inf
    finite_mask = np.isfinite(summary["objective_value"])

    if finite_mask.any():
        mins = (
            summary.loc[finite_mask]
            .groupby("parameter_name")["objective_value"]
            .transform("min")
        )
        summary.loc[finite_mask, "delta_objective"] = (
                summary.loc[finite_mask, "objective_value"].to_numpy() - mins.to_numpy()
        )

    summary.to_csv(out / "profile_likelihood_summary.csv", index=False)

    failed = status_df[(status_df["success"] != True) | (status_df["return_code"] != 0)]
    if len(failed) > 0:
        logger.warning(
            "[Profile] %d/%d standalone profile workers failed. "
            "See profiles/profile_worker_status.csv.",
            len(failed),
            len(status_df),
        )

    logger.info(
        "[Profile] Standalone profile likelihood complete. Successful profiles: %d/%d.",
        len(result_dfs),
        len(indices),
    )

    return {
        "summary": summary,
        "worker_status": status_df,
        "output_dir": out,
    }


def run_profile_likelihood(ctx: InferenceContext, *, parameter_indices: Sequence[int], grid_size: int = 5) -> dict:
    out = Path(ctx.output_dir) / "profiles"
    plot_dir = Path(ctx.output_dir) / "plots" / "profile_likelihood"
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    lower_eff = np.asarray(ctx.lower, dtype=np.float64).copy()
    upper_eff = np.asarray(ctx.upper, dtype=np.float64).copy()

    bad = upper_eff <= lower_eff
    if np.any(bad):
        upper_eff[bad] = lower_eff[bad] + 1e-12

    all_rows = []
    for idx in parameter_indices:
        logger.info(f"Processing parameter {idx} of {len(parameter_indices)}")
        idx = int(idx)
        values = np.linspace(lower_eff[idx], upper_eff[idx], int(grid_size))
        rows = []
        for gv in values:
            logger.info(f"Processing value {gv} of {len(values)} for parameter {idx}")
            fixed_mask = np.zeros_like(ctx.theta0, dtype=bool) if ctx.fixed_mask is None else np.asarray(ctx.fixed_mask,
                                                                                                         dtype=bool).copy()
            fixed_values = np.asarray(ctx.theta0, dtype=np.float64).copy() if ctx.fixed_values is None else np.asarray(
                ctx.fixed_values, dtype=np.float64).copy()
            fixed_mask[idx] = True
            fixed_values[idx] = gv
            try:
                params, state, value = optimize_scalar_objective(
                    ctx.objective_fun,
                    fixed_values,
                    lower_eff,
                    upper_eff,
                    maxiter=ctx.maxiter,
                    tol=ctx.tol,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                    logger_obj=logger,
                )
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

def _requires_nonnegative_posterior_support(name: str) -> bool:
    """Return True for model parameters that are physically non-negative.

    These are kinetic/rate/scale parameters. Signed regulatory parameters,
    if any, are intentionally left unconstrained.
    """
    name = str(name)

    return (
        name.startswith("c_k[")
        or name.startswith("A_i[")
        or name.startswith("B_i[")
        or name.startswith("C_i[")
        or name.startswith("D_i[")
        or name.startswith("Dp_i[")
        or name.startswith("E_i[")
        or name == "tf_scale"
    )


def _sanitize_numpyro_posterior_bounds(
    lower,
    upper,
    names,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanitize posterior bounds before constructing NumPyro priors.

    NumPyro samples directly from these bounds. Therefore, for parameters that
    are physically non-negative, the posterior lower bound must be >= 0.

    This function intentionally raises on impossible intervals rather than
    silently producing invalid posterior samples.
    """
    lower = np.asarray(lower, dtype=np.float64).copy()
    upper = np.asarray(upper, dtype=np.float64).copy()
    names = list(names)

    if len(names) != len(lower):
        raise ValueError(
            f"Parameter name count does not match bounds: "
            f"len(names)={len(names)}, len(lower)={len(lower)}"
        )

    if lower.shape != upper.shape:
        raise ValueError(
            f"Posterior lower/upper shape mismatch: "
            f"lower={lower.shape}, upper={upper.shape}"
        )

    bad_finite = ~np.isfinite(lower) | ~np.isfinite(upper)
    if np.any(bad_finite):
        bad_names = [names[i] for i in np.where(bad_finite)[0]]
        raise ValueError(
            "Posterior bounds contain non-finite values for parameters: "
            f"{bad_names}"
        )

    for i, name in enumerate(names):
        if _requires_nonnegative_posterior_support(name):
            if upper[i] <= 0.0:
                raise ValueError(
                    f"Invalid posterior bounds for non-negative parameter {name}: "
                    f"lower={lower[i]}, upper={upper[i]}. "
                    "The upper bound must be > 0. Fix calculate_bio_bounds() "
                    "or init_raw_params() upstream."
                )

            lower[i] = max(lower[i], 0.0)

            if upper[i] <= lower[i]:
                raise ValueError(
                    f"Invalid posterior interval for non-negative parameter {name}: "
                    f"lower={lower[i]}, upper={upper[i]}."
                )

    bad_interval = upper <= lower
    if np.any(bad_interval):
        bad_names = [names[i] for i in np.where(bad_interval)[0]]
        raise ValueError(
            "Posterior bounds contain invalid intervals with upper <= lower: "
            f"{bad_names}"
        )

    too_narrow = (upper - lower) < eps
    if np.any(too_narrow):
        bad_names = [names[i] for i in np.where(too_narrow)[0]]
        raise ValueError(
            "Posterior bounds are too narrow for stable NUTS initialization: "
            f"{bad_names}"
        )

    return lower, upper


def _make_interior_initial_value(theta0, lower, upper, *, eps: float = 1e-8) -> np.ndarray:
    """Clip theta0 safely inside the NumPyro Uniform support."""
    theta0 = np.asarray(theta0, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    width = upper - lower
    interior_eps = np.minimum(eps, 0.25 * width)

    return np.clip(
        theta0,
        lower + interior_eps,
        upper - interior_eps,
    )


def _assert_no_negative_nonnegative_samples(sample_df: pd.DataFrame, names: Sequence[str]) -> None:
    """Fail if NumPyro produced negative samples for non-negative parameters."""
    nonnegative_cols = [
        name for name in names
        if name in sample_df.columns and _requires_nonnegative_posterior_support(name)
    ]

    if not nonnegative_cols:
        return

    mins = sample_df[nonnegative_cols].min(axis=0)
    leaked = mins[mins < -1e-10]

    if not leaked.empty:
        raise RuntimeError(
            "Posterior produced negative samples for parameters that should be "
            f"non-negative: {leaked.to_dict()}"
        )

def run_numpyro_posterior(
        ctx: InferenceContext,
        *,
        num_warmup: int = 20,
        num_samples: int = 30,
        seed: int = 0,
        num_chains: int = 1,
        chain_method: str = "sequential",
) -> dict:
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise RuntimeError(
            "NumPyro posterior inference requires the optional 'numpyro' dependency. "
            "Install numpyro to run posterior analysis."
        ) from exc

    out = Path(ctx.output_dir) / "posterior"
    plot_dir = Path(ctx.output_dir) / "plots" / "posterior"
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    names = _param_names(len(ctx.theta0), ctx.parameter_names)

    lower_np = np.asarray(ctx.lower, dtype=np.float64).copy()
    upper_np = np.asarray(ctx.upper, dtype=np.float64).copy()

    # Critical fix:
    # NumPyro samples directly from these bounds, so physical non-negativity
    # must be enforced before constructing the Uniform prior.
    lower_np, upper_np = _sanitize_numpyro_posterior_bounds(
        lower_np,
        upper_np,
        names,
    )

    theta_center_np = _make_interior_initial_value(
        ctx.theta0,
        lower_np,
        upper_np,
    )

    theta_center = jnp.asarray(theta_center_np, dtype=jnp.float64)
    lower = jnp.asarray(lower_np, dtype=jnp.float64)
    upper = jnp.asarray(upper_np, dtype=jnp.float64)

    if ctx.fixed_mask is not None:
        fixed_mask_np = np.asarray(ctx.fixed_mask, dtype=bool)
    else:
        fixed_mask_np = None

    if ctx.fixed_values is not None:
        fixed_values_np = _make_interior_initial_value(
            ctx.fixed_values,
            lower_np,
            upper_np,
        )
    else:
        fixed_values_np = None

    def model():
        theta = numpyro.sample(
            "theta",
            dist.Uniform(lower, upper).to_event(1),
        )

        if fixed_mask_np is not None and fixed_values_np is not None:
            fixed_mask = jnp.asarray(fixed_mask_np, dtype=bool)
            fixed_values = jnp.asarray(fixed_values_np, dtype=jnp.float64)
            theta = jnp.where(fixed_mask, fixed_values, theta)

        sigma = numpyro.sample("sigma", dist.Exponential(1.0))

        objective = ctx.objective_fun(theta)

        numpyro.deterministic("scalar_objective", objective)

        numpyro.factor(
            "objective_likelihood",
            -0.5 * objective / (sigma * sigma + 1e-6),
        )

    kernel = NUTS(
        model,
        init_strategy=numpyro.infer.init_to_value(
            values={"theta": theta_center}
        ),
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

    theta_samples = np.asarray(samples["theta"], dtype=np.float64)
    sample_df = pd.DataFrame(theta_samples, columns=names)

    sample_df["sigma"] = np.asarray(samples["sigma"], dtype=np.float64)

    if "scalar_objective" in samples:
        sample_df["scalar_objective"] = np.asarray(
            samples["scalar_objective"],
            dtype=np.float64,
        )

    sample_df["data_mode"] = ctx.mode.data_mode

    # Fail loudly if anything escaped the intended physical support.
    _assert_no_negative_nonnegative_samples(sample_df, names)

    sample_df.to_csv(out / "posterior_samples.csv", index=False)

    summary_rows = []

    for col in names + ["sigma"]:
        vals = sample_df[col].to_numpy(dtype=np.float64)

        summary_rows.append(
            {
                "parameter": col,
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "sd": float(vals.std(ddof=0)),
                "ci_05": float(np.quantile(vals, 0.05)),
                "ci_95": float(np.quantile(vals, 0.95)),
                "ess": float(len(vals)),
                "r_hat": np.nan,
                "data_mode": ctx.mode.data_mode,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "posterior_summary.csv", index=False)

    if "scalar_objective" in sample_df.columns:
        predictive = sample_df[["scalar_objective", "data_mode"]].copy()
    else:
        predictive = pd.DataFrame({"data_mode": [ctx.mode.data_mode]})

    predictive.to_csv(out / "posterior_predictive.csv", index=False)

    _plot_posterior(sample_df, summary, plot_dir)

    return {
        "samples": sample_df,
        "summary": summary,
        "posterior_predictive": predictive,
        "output_dir": out,
    }

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

def _safe_plot_name(name: str, max_len: int = 140) -> str:
    """Make parameter names safe for filenames."""
    safe = (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(",", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(";", "_")
    )
    return safe[:max_len]


def _smooth_density_1d(values: np.ndarray, grid_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Small NumPy-only Gaussian KDE.

    Avoids scipy/seaborn dependency.
    Returns x_grid, density.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    if values.size == 1 or np.allclose(values, values[0]):
        center = float(values[0])
        span = max(abs(center) * 0.05, 1e-3)
        x_grid = np.linspace(center - span, center + span, grid_size)
        density = np.exp(-0.5 * ((x_grid - center) / (span / 4.0)) ** 2)
        density = density / max(float(density.max()), 1e-12)
        return x_grid, density

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin

    pad = 0.1 * span
    x_grid = np.linspace(vmin - pad, vmax + pad, grid_size)

    sd = float(np.std(values, ddof=1))
    n = values.size

    # Silverman's rule. Safe fallback for near-zero variance.
    bandwidth = 1.06 * sd * (n ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, span / 200.0, 1e-8)

    z = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).mean(axis=1)
    density = density / (bandwidth * np.sqrt(2.0 * np.pi))

    max_density = float(np.max(density))
    if max_density > 0:
        density = density / max_density

    return x_grid, density


def _plot_ridgeline_parameter_distributions(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    max_params_per_fig: int = 25,
    include_sigma: bool = False,
) -> None:
    """Plot elegant batched ridgeline posterior distributions.

    Each parameter is normalized to its own posterior range on the x-axis.
    This makes the plot readable even when parameters have very different scales.
    Individual density plots still preserve the real parameter scale.
    """
    ridge_dir = plot_dir / "ridge"
    ridge_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty or "parameter" not in summary.columns:
        logger.warning("[Posterior] Empty summary; skipping ridgeline posterior plot.")
        return

    params = [
        str(p)
        for p in summary["parameter"].astype(str).tolist()
        if str(p) in samples.columns
        and pd.api.types.is_numeric_dtype(samples[str(p)])
    ]

    if not include_sigma:
        params = [p for p in params if p != "sigma"]

    if not params:
        logger.warning("[Posterior] No numeric parameters available for ridgeline plot.")
        return

    batches = [
        params[i:i + max_params_per_fig]
        for i in range(0, len(params), max_params_per_fig)
    ]

    for batch_id, batch in enumerate(batches):
        fig_height = max(6.0, 0.38 * len(batch) + 1.5)
        fig, ax = plt.subplots(figsize=(11, fig_height))

        y_positions = np.arange(len(batch))[::-1]

        for y, param in zip(y_positions, batch):
            values = samples[param].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            vmin = float(np.min(values))
            vmax = float(np.max(values))

            if np.isclose(vmin, vmax):
                x_norm = np.linspace(0.45, 0.55, 256)
                density = np.exp(-0.5 * ((x_norm - 0.5) / 0.015) ** 2)
                density = density / max(float(density.max()), 1e-12)
            else:
                x_grid, density = _smooth_density_1d(values)
                x_norm = (x_grid - vmin) / (vmax - vmin)
                x_norm = np.clip(x_norm, 0.0, 1.0)

            ridge_height = 0.75
            y_curve = y + ridge_height * density

            ax.fill_between(
                x_norm,
                y,
                y_curve,
                alpha=0.55,
                linewidth=0.0,
            )

            ax.plot(
                x_norm,
                y_curve,
                linewidth=1.0,
            )

            q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])

            if not np.isclose(vmin, vmax):
                q05n = (q05 - vmin) / (vmax - vmin)
                q50n = (q50 - vmin) / (vmax - vmin)
                q95n = (q95 - vmin) / (vmax - vmin)

                ax.plot(
                    [q05n, q95n],
                    [y + 0.05, y + 0.05],
                    linewidth=2.0,
                )

                ax.scatter(
                    [q50n],
                    [y + 0.05],
                    s=18,
                    zorder=3,
                )

            ax.text(
                1.03,
                y + 0.05,
                f"median={q50:.3g}",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(batch, fontsize=8)

        ax.set_xlim(0.0, 1.22)
        ax.set_xlabel("Normalized posterior range per parameter")
        ax.set_title("Posterior parameter distributions")
        ax.set_frame_on(False)

        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            ridge_dir / f"posterior_ridge_{batch_id:03d}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

def _plot_posterior(samples: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path) -> None:
    density_dir = plot_dir / "density"
    density_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty or "parameter" not in summary.columns:
        logger.warning("[Posterior] Empty posterior summary; skipping posterior plots.")
        return

    plot_cols = [
        str(p) for p in summary["parameter"].astype(str).tolist()
        if str(p) in samples.columns
        and pd.api.types.is_numeric_dtype(samples[str(p)])
    ]

    for col in plot_cols:
        values = samples[col].to_numpy(dtype=np.float64)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(values)
        ax.set_title(f"trace {col}")
        ax.set_xlabel("sample")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(density_dir / f"trace_{col}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=15)
        ax.set_title(f"posterior {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")
        fig.tight_layout()
        fig.savefig(density_dir / f"density_{col}.png", dpi=300)
        plt.close(fig)

    summary_plot = summary[
        summary["parameter"].astype(str).isin(plot_cols)
    ].copy()

    if summary_plot.empty:
        logger.warning("[Posterior] No numeric posterior parameters available for interval plot.")
        return

    n_params = len(summary_plot)

    fig_width = max(10, 0.75 * n_params)
    fig_height = 5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_params)

    median = summary_plot["median"].to_numpy(dtype=np.float64)
    ci_05 = summary_plot["ci_05"].to_numpy(dtype=np.float64)
    ci_95 = summary_plot["ci_95"].to_numpy(dtype=np.float64)

    ax.errorbar(
        x,
        median,
        yerr=[
            median - ci_05,
            ci_95 - median,
        ],
        fmt="o",
        capsize=4,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        summary_plot["parameter"].astype(str),
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

    fig.savefig(plot_dir / "credible_intervals.png", dpi=300)
    plt.close(fig)

    _plot_ridgeline_parameter_distributions(
        samples=samples,
        summary=summary,
        plot_dir=plot_dir,
        max_params_per_fig=25,
        include_sigma=False,
    )