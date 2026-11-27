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


PROFILE_COLUMNS = ("parameter_name", "parameter_index", "grid_index", "side", "grid_value", "grid_value_physical",
                   "objective_value", "iterations", "success", "optimizer_status", "convergence_status", "stop_reason",
                   "failure_reason", "optimized_free_parameters", "data_mode", "active_loss_terms")


def empty_profile_frame(param_name: str, idx: int) -> pd.DataFrame:
    """Profile table with no points (empty grid), with the usual columns."""
    df = pd.DataFrame({c: pd.Series(dtype=float if c in ("grid_value", "grid_value_physical", "objective_value")
                                     else object) for c in PROFILE_COLUMNS})
    df["success"] = df["success"].astype(bool)
    return df


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


REFERENCE_STARTS_FILE = "profile_reference_starts.npy"  # payload: up to 5 best kept multistart solutions (raw)


def select_reference(records):
    """DECISIONS_C1_FINAL item 2: the best (lowest objective) usable refit; usable = converged (KKT <= 1e-6) or
    near_converged (<= 1e-5, the N26 noise level). None if no start is usable."""
    from networkmodel.utils import USABLE_STATUSES

    usable = [r for r in records if r["status"] in USABLE_STATUSES and np.isfinite(float(r["f"]))]
    return min(usable, key=lambda r: float(r["f"])) if usable else None


def run_reference(*, run_config_path: str | Path) -> dict:
    """Write <output_dir>/profiles/profile_reference.json: the data-only reference and the profile threshold.

    DECISIONS_C1_FINAL item 2: the data-only objective is refitted from theta-hat and from the best kept multistart
    solutions (payload file profile_reference_starts.npy, up to 5; duplicates of theta-hat skipped); the best usable
    refit (converged or near_converged) is the reference. No usable refit -> threshold NaN, every label
    'not assessed'."""
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
    theta_hat = np.asarray(ctx.theta0, dtype=np.float64)
    starts = [("theta_hat", theta_hat)]
    extra = Path(run_cfg["payload_dir"]) / REFERENCE_STARTS_FILE
    if extra.exists():
        for k, th in enumerate(np.atleast_2d(np.load(extra))[:5]):
            th = np.asarray(th, dtype=np.float64)
            if th.shape == theta_hat.shape and not any(np.array_equal(th, s[1]) for s in starts):
                starts.append((f"multistart_rank{k + 1}", th))
    fixed = None if ctx.fixed_mask is None else np.asarray(ctx.fixed_mask, bool)
    records, thetas = [], {}
    for label, th0 in starts:
        try:
            r = reference_refit(ctx.objective_fun, th0, ctx.lower, ctx.upper, fixed_mask=fixed, maxiter=ctx.maxiter,
                                tol=ctx.tol, optimizer=ctx.optimizer, logger_obj=logger)
            thetas[label] = r["theta_ref"]
            records.append({"start": label, "f": r["f_ref"], "status": r["status"], "kkt_residual": r["kkt_residual"],
                            "stop_reason": r["stop_reason"], "f_at_start": r["f_at_theta_hat"]})
        except Exception as exc:  # recorded, not hidden
            logger.exception("[ProfileWorker] reference refit from %s failed", label)
            records.append({"start": label, "f": float("nan"), "status": "failed", "kkt_residual": float("nan"),
                            "stop_reason": repr(exc), "f_at_start": float("nan")})
        logger.info("[ProfileWorker] reference start %s: %s", label, records[-1])
    best = select_reference(records)
    n_eff = int(ctx.n_eff or 0)
    names = _param_names(len(ctx.theta0), ctx.parameter_names)
    th_ref = thetas[best["start"]] if best is not None else theta_hat
    from networkmodel.BayesianInference import objective_at

    payload = {
        "f_ref": float(best["f"]) if best is not None else float("nan"),
        "f_at_theta_hat": float(objective_at(ctx.objective_fun, theta_hat)),
        "status": best["status"] if best is not None else "not_usable",
        "usable": best is not None, "converged": best is not None,  # 'converged' kept for older readers
        "reference_start": best["start"] if best is not None else None,
        "stop_reason": best["stop_reason"] if best is not None else "",
        "kkt_residual": float(best["kkt_residual"]) if best is not None else float("nan"),
        "starts": records, "theta_ref": np.asarray(th_ref, float).tolist(), "theta_hat": theta_hat.tolist(),
        "parameter_names": list(names), "n_eff": n_eff,
        "threshold": profile_threshold(f_ref=best["f"], n_eff=n_eff) if best is not None else float("nan"),
        "rule": PROFILE_RULE, "prior": "off (data-only objective)",
        "note": "reference = best usable (converged or near_converged) data-only refit from theta-hat and the best "
                "kept multistart solutions; threshold NaN when no start is usable: every label is then 'not "
                "assessed'"}
    path = reference_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[ProfileWorker] Data-only reference: f_ref=%.10g from %s (%s; f at theta-hat %.10g), n_eff=%d, "
                "threshold=%.6g", payload["f_ref"], payload["reference_start"], payload["status"],
                payload["f_at_theta_hat"], n_eff, payload["threshold"])
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
    outward from the reference; every refit starts from the last usable (converged or near-converged) point of that side (the reference for
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
                    if res.status in ("converged", "near_converged"):  # usable (DECISIONS_C1_FINAL 2)
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

        # A parameter whose box is a point (masked, zero data influence) or whose reference sits on both bounds has
        # no grid: an empty profile, labelled "not assessed" downstream.
        result_df = pd.DataFrame(rows) if rows else empty_profile_frame(param_name, idx)
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
