"""Standalone profile-likelihood worker for one parameter.

Run as:
    python -m networkmodel.ProfileWorker \
        --run-config results/.../posterior_payload/posterior_run_config.json \
        --parameter-index 88 \
        --grid-size 7
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import numpy as np
import pandas as pd

from networkmodel.PosteriorObjective import build_networkmodel_posterior_context
from networkmodel.BayesianInference import _param_names
from networkmodel.backend import optimize_scalar_objective
from config.config import setup_logger


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


def _cleanup_after_grid_step(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass

    gc.collect()


def run_one_parameter_profile(
    *,
    run_config_path: str | Path,
    parameter_index: int,
    grid_size: int,
) -> dict:
    run_config_path = Path(run_config_path)

    with open(run_config_path) as f:
        run_cfg = json.load(f)

    base_output_dir = Path(run_cfg["output_dir"])
    param_out = base_output_dir / "profile_workers" / f"param_{int(parameter_index):04d}"
    param_out.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir=param_out)

    status_path = param_out / "profile_status.json"
    status = {
        "parameter_index": int(parameter_index),
        "pid": os.getpid(),
        "success": False,
        "state": "started",
        "failure_reason": "",
        "result_csv": "",
    }
    _write_status(status_path, status)

    try:
        status["state"] = "rebuilding_objective"
        _write_status(status_path, status)

        ctx = build_networkmodel_posterior_context(
            run_config_path=run_config_path,
            chain_output_dir=param_out,
        )

        names = _param_names(len(ctx.theta0), ctx.parameter_names)

        idx = int(parameter_index)
        if idx < 0 or idx >= len(ctx.theta0):
            raise IndexError(
                f"parameter_index={idx} outside valid range 0..{len(ctx.theta0) - 1}"
            )

        param_name = names[idx]

        lower_eff = np.asarray(ctx.lower, dtype=np.float64).copy()
        upper_eff = np.asarray(ctx.upper, dtype=np.float64).copy()

        bad = upper_eff <= lower_eff
        if np.any(bad):
            upper_eff[bad] = lower_eff[bad] + 1e-12

        values = np.linspace(lower_eff[idx], upper_eff[idx], int(grid_size))

        rows = []

        status["state"] = "running_profile"
        status["parameter_name"] = param_name
        status["grid_size"] = int(grid_size)
        _write_status(status_path, status)

        logger.info(
            "[ProfileWorker] Profiling parameter index=%d name=%s grid_size=%d",
            idx,
            param_name,
            int(grid_size),
        )

        for grid_i, gv in enumerate(values):
            logger.info(
                "[ProfileWorker] parameter=%s index=%d grid=%d/%d value=%.8g",
                param_name,
                idx,
                grid_i + 1,
                len(values),
                float(gv),
            )

            fixed_mask = (
                np.zeros_like(ctx.theta0, dtype=bool)
                if ctx.fixed_mask is None
                else np.asarray(ctx.fixed_mask, dtype=bool).copy()
            )

            fixed_values = (
                np.asarray(ctx.theta0, dtype=np.float64).copy()
                if ctx.fixed_values is None
                else np.asarray(ctx.fixed_values, dtype=np.float64).copy()
            )

            fixed_mask[idx] = True
            fixed_values[idx] = float(gv)

            params = None
            state = None

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

                row = {
                    "parameter_name": param_name,
                    "parameter_index": idx,
                    "grid_index": grid_i,
                    "grid_value": float(gv),
                    "objective_value": float(value),
                    "success": True,
                    "optimizer_status": "jaxopt.ProjectedGradient",
                    "failure_reason": "",
                    "optimized_free_parameters": json.dumps(
                        np.asarray(params, dtype=float).tolist()
                    ),
                    "data_mode": ctx.mode.data_mode,
                    "active_loss_terms": ",".join(ctx.mode.active_loss_terms),
                }

            except Exception as exc:
                logger.exception(
                    "[ProfileWorker] Failed parameter index=%d grid=%d value=%.8g",
                    idx,
                    grid_i,
                    float(gv),
                )

                row = {
                    "parameter_name": param_name,
                    "parameter_index": idx,
                    "grid_index": grid_i,
                    "grid_value": float(gv),
                    "objective_value": np.inf,
                    "success": False,
                    "optimizer_status": "failed",
                    "failure_reason": repr(exc),
                    "optimized_free_parameters": "[]",
                    "data_mode": ctx.mode.data_mode,
                    "active_loss_terms": ",".join(ctx.mode.active_loss_terms),
                }

            rows.append(row)

            partial_df = pd.DataFrame(rows)
            finite = partial_df[np.isfinite(partial_df["objective_value"])]

            if not finite.empty:
                finite_min = finite["objective_value"].min()
                partial_df["delta_objective"] = (
                    partial_df["objective_value"] - finite_min
                )
            else:
                partial_df["delta_objective"] = np.inf

            partial_path = param_out / f"profile_likelihood_{param_name}.csv"
            partial_df.to_csv(partial_path, index=False)

            _cleanup_after_grid_step(params, state, fixed_mask, fixed_values)

        result_df = pd.DataFrame(rows)
        finite = result_df[np.isfinite(result_df["objective_value"])]

        if not finite.empty:
            finite_min = finite["objective_value"].min()
            result_df["delta_objective"] = result_df["objective_value"] - finite_min
        else:
            result_df["delta_objective"] = np.inf

        result_csv = param_out / f"profile_likelihood_{param_name}.csv"
        result_df.to_csv(result_csv, index=False)

        status.update(
            {
                "success": True,
                "state": "complete",
                "failure_reason": "",
                "result_csv": str(result_csv),
            }
        )
        _write_status(status_path, status)

        logger.info(
            "[ProfileWorker] Complete parameter index=%d name=%s result=%s",
            idx,
            param_name,
            result_csv,
        )

        return status

    except Exception as exc:
        status.update(
            {
                "success": False,
                "state": "failed",
                "failure_reason": repr(exc),
                "result_csv": "",
            }
        )
        _write_status(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--parameter-index", type=int, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    args = parser.parse_args()

    run_one_parameter_profile(
        run_config_path=args.run_config,
        parameter_index=args.parameter_index,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()