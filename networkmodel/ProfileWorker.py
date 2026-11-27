"""Standalone profile-likelihood worker for one parameter.

Run as:
    python -m networkmodel.ProfileWorker \
        --run-config results/.../posterior_payload/posterior_run_config.json \
        --parameter-index 88 \
        --grid-size 7

Data-only (DECISIONS_C1 items 1-2): run once with ``--reference`` first (writes profiles/profile_reference.json),
then once per parameter.
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
from networkmodel.optimizers import minimize_bounded
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


def _softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softplus_inv(y):
    """Raw coordinate of a physical value (inverse of softplus), stable for small and large y."""
    y = np.asarray(y, dtype=np.float64)
    return y + np.log(-np.expm1(-y))


def reference_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "profiles" / "profile_reference.json"


def reference_refit(objective, theta_hat, lower, upper, *, fixed_mask, maxiter, tol, optimizer, logger_obj=None):
    """Data-only reference optimum (DECISIONS_C1 item 1): refit from theta-hat (the fitted, penalized optimum) with
    the data-only objective. Returns theta_ref, f_ref, f_at_theta_hat, converged, status, stop_reason, kkt_residual."""
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    fv = theta_hat.copy() if fixed_mask is not None else None
    res = minimize_bounded(objective, theta_hat, np.asarray(lower, dtype=np.float64),
                           np.asarray(upper, dtype=np.float64), maxiter=maxiter, tol=tol, fixed_mask=fixed_mask,
                           fixed_values=fv, optimizer=optimizer, logger_obj=logger_obj)
    from networkmodel.BayesianInference import objective_at

    return {"theta_ref": np.asarray(res.theta, dtype=np.float64), "f_ref": float(res.final_objective),
            "f_at_theta_hat": float(objective_at(objective, theta_hat)), "converged": bool(res.converged),
            "status": res.status, "stop_reason": res.stop_reason, "kkt_residual": float(res.kkt_residual),
            "optimizer": res.label}


def run_reference(*, run_config_path: str | Path) -> dict:
    """Write <output_dir>/profiles/profile_reference.json: the data-only reference and the profile threshold."""
    from networkmodel.utils import PROFILE_RULE, profile_threshold

    run_config_path = Path(run_config_path)
    with open(run_config_path) as f:
        run_cfg = json.load(f)
    out_dir = Path(run_cfg["output_dir"])
    work = out_dir / "profile_workers" / "reference"
    work.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=work)
    ctx = build_networkmodel_posterior_context(run_config_path=run_config_path, chain_output_dir=work,
                                               prior_off=True)
    ref = reference_refit(ctx.objective_fun, ctx.theta0, ctx.lower, ctx.upper,
                          fixed_mask=None if ctx.fixed_mask is None else np.asarray(ctx.fixed_mask, bool),
                          maxiter=ctx.maxiter, tol=ctx.tol, optimizer=ctx.optimizer, logger_obj=logger)
    n_eff = int(ctx.n_eff or 0)
    thr = profile_threshold(f_ref=ref["f_ref"], n_eff=n_eff) if ref["converged"] else float("nan")
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    payload = {k: v for k, v in ref.items() if k != "theta_ref"}
    payload.update({"theta_ref": ref["theta_ref"].tolist(), "theta_hat": np.asarray(ctx.theta0, float).tolist(),
                    "parameter_names": list(names), "n_eff": n_eff, "threshold": thr, "rule": PROFILE_RULE,
                    "prior": "off (data-only objective)",
                    "note": "threshold NaN when the reference refit did not converge: every label is then "
                            "'not assessed'"})
    path = reference_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[ProfileWorker] Data-only reference: f_ref=%.10g (f at theta-hat %.10g), %s, n_eff=%d, "
                "threshold=%.6g", ref["f_ref"], ref["f_at_theta_hat"], ref["status"], n_eff, thr)
    return payload


def run_one_parameter_profile(
        *,
        run_config_path: str | Path,
        parameter_index: int,
        grid_size: int,
) -> dict:
    """Data-only profile of one parameter (DECISIONS_C1 items 1-2).

    The prior is off; the centre is the data-only reference (profile_reference.json, written by ``run_reference``).
    Grid: ``utils.profile_grid_physical`` with ceil(grid_size / 2) points per side (at least 4). Each side is walked
    outward from the reference; every refit starts from the last converged point of that side (the reference for
    the first point), with the profiled coordinate fixed at the grid value.
    """
    from networkmodel.utils import profile_grid_physical

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

        with open(reference_path(base_output_dir)) as f:
            ref = json.load(f)

        ctx = build_networkmodel_posterior_context(
            run_config_path=run_config_path,
            chain_output_dir=param_out,
            prior_off=True,
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

        theta_ref = np.asarray(ref["theta_ref"], dtype=np.float64)
        v_ref = float(_softplus(theta_ref[idx]))
        n_side = max(int(np.ceil(int(grid_size) / 2)), 4)
        below, above = profile_grid_physical(v_ref, float(_softplus(lower_eff[idx])),
                                             float(_softplus(upper_eff[idx])), n_side)

        base_mask = (
            np.zeros_like(ctx.theta0, dtype=bool)
            if ctx.fixed_mask is None
            else np.asarray(ctx.fixed_mask, dtype=bool).copy()
        )
        rows = []

        status["state"] = "running_profile"
        status["parameter_name"] = param_name
        status["grid_size"] = int(len(below) + len(above))
        _write_status(status_path, status)

        logger.info(
            "[ProfileWorker] Data-only profile of parameter index=%d name=%s centre=%.6g points below=%d above=%d",
            idx, param_name, v_ref, len(below), len(above),
        )

        grid_i = 0
        for side, pts in (("below", below), ("above", above)):
            start = theta_ref.copy()
            for gv_phys in pts:
                gv = float(np.clip(_softplus_inv(gv_phys), lower_eff[idx], upper_eff[idx]))
                logger.info("[ProfileWorker] parameter=%s side=%s grid=%d value=%.8g (raw %.8g)",
                            param_name, side, grid_i, float(gv_phys), gv)
                fixed_mask = base_mask.copy()
                fixed_mask[idx] = True
                fixed_values = start.copy()
                fixed_values[idx] = gv
                params = None
                state = None
                try:
                    res = minimize_bounded(
                        ctx.objective_fun,
                        fixed_values,
                        lower_eff,
                        upper_eff,
                        maxiter=ctx.maxiter,
                        tol=ctx.tol,
                        fixed_mask=fixed_mask,
                        fixed_values=fixed_values,
                        optimizer=ctx.optimizer,
                        logger_obj=logger,
                    )
                    params, value, state = res.theta, res.final_objective, res.state
                    row = {
                        "parameter_name": param_name,
                        "parameter_index": idx,
                        "grid_index": grid_i,
                        "side": side,
                        "grid_value": gv,
                        "grid_value_physical": float(gv_phys),
                        "objective_value": float(value),
                        "iterations": int(res.iterations),
                        "success": True,
                        "optimizer_status": res.label,
                        "convergence_status": res.status,
                        "stop_reason": res.stop_reason,
                        "failure_reason": "",
                        "optimized_free_parameters": json.dumps(np.asarray(params, dtype=float).tolist()),
                        "data_mode": ctx.mode.data_mode,
                        "active_loss_terms": ",".join(ctx.mode.active_loss_terms),
                    }
                    if res.status == "converged":
                        start = np.asarray(params, dtype=np.float64).copy()
                except Exception as exc:
                    logger.exception("[ProfileWorker] Failed parameter index=%d grid=%d value=%.8g",
                                     idx, grid_i, float(gv))
                    row = {
                        "parameter_name": param_name,
                        "parameter_index": idx,
                        "grid_index": grid_i,
                        "side": side,
                        "grid_value": gv,
                        "grid_value_physical": float(gv_phys),
                        "objective_value": np.inf,
                        "iterations": -1,
                        "success": False,
                        "optimizer_status": "failed",
                        "convergence_status": "failed",
                        "stop_reason": "",
                        "failure_reason": repr(exc),
                        "optimized_free_parameters": "[]",
                        "data_mode": ctx.mode.data_mode,
                        "active_loss_terms": ",".join(ctx.mode.active_loss_terms),
                    }
                rows.append(row)
                grid_i += 1
                partial_df = pd.DataFrame(rows)
                partial_df["delta_objective"] = partial_df["objective_value"] - float(ref["f_ref"])
                partial_df.to_csv(param_out / f"profile_likelihood_{param_name}.csv", index=False)
                _cleanup_after_grid_step(params, state, fixed_mask, fixed_values)

        result_df = pd.DataFrame(rows)
        result_df["delta_objective"] = result_df["objective_value"] - float(ref["f_ref"])

        from networkmodel.BayesianInference import annotate_profile

        result_df = annotate_profile(result_df, theta_ref_raw=float(theta_ref[idx]), f_ref=float(ref["f_ref"]),
                                     theta_hat_raw=float(np.asarray(ref["theta_hat"])[idx]),
                                     maxiter=ctx.maxiter)

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
    parser.add_argument("--parameter-index", type=int, default=-1)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--reference", action="store_true",
                        help="compute the data-only reference optimum and threshold (run before the parameters)")
    args = parser.parse_args()

    if args.reference:
        run_reference(run_config_path=args.run_config)
        return
    if args.parameter_index < 0:
        parser.error("--parameter-index is required without --reference")
    run_one_parameter_profile(
        run_config_path=args.run_config,
        parameter_index=args.parameter_index,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
