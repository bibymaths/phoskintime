"""Run perturbation-based sensitivity analysis and write sensitivity diagnostics.

This module runs Morris perturbation-based sensitivity analysis on fitted
PhosKinTime/networkmodel parameters. It perturbs fitted parameters, simulates the
model, reduces each simulated trajectory to a scalar response metric, computes
Morris indices, and writes diagnostic plots/tables.

Important behavior:
- c_k is allowed to be negative.
- Rate/scale parameters are constrained non-negative.
- The default sensitivity target is l2_norm, not raw total signal.
- Failed simulations abort the Morris analysis because dropping arbitrary rows
  invalidates the Morris design.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import morris
from SALib.analyze.morris import analyze
from tqdm import tqdm

from networkmodel.config import (
    SENSITIVITY_TRAJECTORIES,
    SENSITIVITY_LEVELS,
    SENSITIVITY_PERTURBATION,
    SENSITIVITY_TOP_CURVES,
    RESULTS_DIR,
    SEED,
    TIME_POINTS_PROTEIN,
    TIME_POINTS_RNA,
    TIME_POINTS_PHOSPHO,
)
from networkmodel.simulate import simulate_and_measure

from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


def _is_nonnegative_parameter(name: str) -> bool:
    """Return whether a flattened parameter must stay non-negative.

    c_k is intentionally not included because kinase multipliers/effects may be signed.
    """
    return (
        name.startswith("A_i_")
        or name.startswith("B_i_")
        or name.startswith("C_i_")
        or name.startswith("D_i_")
        or name.startswith("Dp_i_")
        or name.startswith("E_i_")
        or name == "tf_scale"
    )


def _as_array_like(value: Any) -> np.ndarray:
    """Convert array-like parameter values to float arrays."""
    return np.asarray(value, dtype=np.float64)


def compute_bounds(params_dict, perturbation=SENSITIVITY_PERTURBATION):
    """Compute Morris perturbation bounds for fitted parameters.

    Parameters with biological non-negativity constraints are clipped at zero.
    Signed parameters such as c_k are allowed to cross below zero.
    """
    bounds = []
    names = []

    perturbation = float(perturbation)

    for key, value in params_dict.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = _as_array_like(value).reshape(-1)

            for i, raw_v in enumerate(arr):
                name = f"{key}_{i}"
                v = float(raw_v)

                if abs(v) < 1e-6:
                    if _is_nonnegative_parameter(name):
                        lb, ub = 0.0, 0.01
                    else:
                        lb, ub = -0.01, 0.01
                else:
                    a = v * (1.0 - perturbation)
                    b = v * (1.0 + perturbation)
                    lb, ub = min(a, b), max(a, b)

                if _is_nonnegative_parameter(name):
                    lb = max(0.0, lb)

                if not np.isfinite(lb) or not np.isfinite(ub):
                    raise ValueError(f"Non-finite sensitivity bounds for {name}: [{lb}, {ub}]")

                if ub <= lb:
                    ub = lb + 1e-12

                bounds.append([float(lb), float(ub)])
                names.append(name)

        else:
            name = str(key)
            v = float(value)

            if abs(v) < 1e-6:
                if _is_nonnegative_parameter(name):
                    lb, ub = 0.0, 0.01
                else:
                    lb, ub = -0.01, 0.01
            else:
                a = v * (1.0 - perturbation)
                b = v * (1.0 + perturbation)
                lb, ub = min(a, b), max(a, b)

            if _is_nonnegative_parameter(name):
                lb = max(0.0, lb)

            if not np.isfinite(lb) or not np.isfinite(ub):
                raise ValueError(f"Non-finite sensitivity bounds for {name}: [{lb}, {ub}]")

            if ub <= lb:
                ub = lb + 1e-12

            bounds.append([float(lb), float(ub)])
            names.append(name)

    if not names:
        raise ValueError("No fitted parameters were provided for sensitivity analysis.")

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }


def _infer_original_shapes(fitted_params: dict) -> dict:
    """Record original fitted parameter shapes for flat-vector reconstruction."""
    original_shapes = {}

    for key, value in fitted_params.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            original_shapes[key] = np.shape(value)
        else:
            original_shapes[key] = ()

    return original_shapes


def _reconstruct_params(param_vector, original_shapes):
    """Reconstruct fitted parameter dictionary from a flat Morris sample vector."""
    p_out = {}
    curr = 0

    param_vector = np.asarray(param_vector, dtype=np.float64)

    for key, shape in original_shapes.items():
        if shape == ():
            p_out[key] = float(param_vector[curr])
            curr += 1
        else:
            size = int(np.prod(shape))
            arr = np.asarray(param_vector[curr: curr + size], dtype=np.float64).reshape(shape)
            p_out[key] = arr
            curr += size

    if curr != len(param_vector):
        raise ValueError(
            f"Parameter reconstruction consumed {curr} values, "
            f"but vector has {len(param_vector)} values."
        )

    return p_out


def _safe_pred_values(df, value_col="pred_fc") -> np.ndarray:
    """Extract finite prediction values from a prediction DataFrame."""
    if df is None or len(df) == 0 or value_col not in df.columns:
        return np.array([], dtype=np.float64)

    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _compute_scalar_metric(df_prot, df_rna, df_phos, metric="l2_norm"):
    """Compute scalar model response used as Morris sensitivity target.

    This is output sensitivity, not loss sensitivity. If you want sensitivity of
    model fit, pass observed data and compute a residual/RMSE objective instead.
    """
    v_p = _safe_pred_values(df_prot)
    v_r = _safe_pred_values(df_rna)
    v_ph = _safe_pred_values(df_phos)

    combined = np.concatenate([v_p, v_r, v_ph])

    if len(combined) == 0:
        return 0.0

    metric = str(metric).lower()

    if metric == "total_signal":
        return float(np.sum(combined))

    if metric == "absolute_total_signal":
        return float(np.sum(np.abs(combined)))

    if metric == "mean":
        return float(np.mean(combined))

    if metric == "absolute_mean":
        return float(np.mean(np.abs(combined)))

    if metric == "variance":
        return float(np.var(combined))

    if metric == "l2_norm":
        return float(np.linalg.norm(combined))

    if metric == "max_abs":
        return float(np.max(np.abs(combined)))

    logger.warning("[Sensitivity] Unknown metric '%s'; falling back to l2_norm.", metric)
    return float(np.linalg.norm(combined))


def _subset_prediction_frames(dfp, dfr, dfph):
    """Keep only columns needed for perturbation-cloud plotting."""
    prot_df = None
    rna_df = None
    phos_df = None

    if dfp is not None and len(dfp) > 0:
        cols = [c for c in ("protein", "time", "pred_fc") if c in dfp.columns]
        if len(cols) == 3:
            prot_df = dfp[cols].copy()

    if dfr is not None and len(dfr) > 0:
        cols = [c for c in ("protein", "time", "pred_fc") if c in dfr.columns]
        if len(cols) == 3:
            rna_df = dfr[cols].copy()

    if dfph is not None and len(dfph) > 0:
        cols = [c for c in ("protein", "psite", "time", "pred_fc") if c in dfph.columns]
        if len(cols) == 4:
            phos_df = dfph[cols].copy()

    return prot_df, rna_df, phos_df


def _worker_simulation(task_args):
    """Run one Morris perturbation simulation."""
    (
        sample_id,
        param_vector,
        original_shapes,
        sys_obj,
        idx_obj,
        times_p,
        times_r,
        times_ph,
        metric,
        keep_trajectory,
    ) = task_args

    p_new = _reconstruct_params(param_vector, original_shapes)

    # The System object is copied/pickled into the process. Updating it is local
    # to the worker process.
    sys_obj.update(**p_new)

    dfp, dfr, dfph = simulate_and_measure(
        sys_obj,
        idx_obj,
        times_p,
        times_r,
        times_ph,
    )

    y_val = _compute_scalar_metric(dfp, dfr, dfph, metric)

    if not np.isfinite(y_val):
        raise FloatingPointError(f"Non-finite sensitivity metric for sample {sample_id}: {y_val}")

    if keep_trajectory:
        prot_df, rna_df, phos_df = _subset_prediction_frames(dfp, dfr, dfph)
    else:
        prot_df, rna_df, phos_df = None, None, None

    return sample_id, float(y_val), prot_df, rna_df, phos_df


def _select_top_trajectories(trajectory_storage, max_items):
    """Keep the strongest response trajectories by scalar output value."""
    if not trajectory_storage:
        return []

    return sorted(
        trajectory_storage,
        key=lambda x: float(x["y_val"]),
        reverse=True,
    )[: int(max_items)]


def _make_task(
    sample_id,
    param_vector,
    original_shapes,
    sys_obj,
    idx_obj,
    metric,
    keep_trajectory=True,
):
    return (
        int(sample_id),
        np.asarray(param_vector, dtype=np.float64),
        original_shapes,
        sys_obj,
        idx_obj,
        TIME_POINTS_PROTEIN,
        TIME_POINTS_RNA,
        TIME_POINTS_PHOSPHO,
        metric,
        bool(keep_trajectory),
    )


def _get_process_context():
    """Use spawn on Linux/HPC when available; fall back to default elsewhere."""
    try:
        return mp.get_context("spawn")
    except ValueError:
        return mp.get_context()


def run_sensitivity_analysis(sys, idx, fitted_params, output_dir, metric="l2_norm"):
    """Run Morris perturbation sensitivity analysis.

    Returns:
        DataFrame with Morris sensitivity indices sorted by mu_star.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "[Sensitivity] Starting Morris analysis "
        "(N=%s, levels=%s, perturbation=%s).",
        SENSITIVITY_TRAJECTORIES,
        SENSITIVITY_LEVELS,
        SENSITIVITY_PERTURBATION,
    )
    logger.info("[Sensitivity] Metric: %s", metric)

    original_shapes = _infer_original_shapes(fitted_params)
    problem = compute_bounds(fitted_params, perturbation=SENSITIVITY_PERTURBATION)

    bounds_df = pd.DataFrame(
        {
            "Parameter": problem["names"],
            "lower": [b[0] for b in problem["bounds"]],
            "upper": [b[1] for b in problem["bounds"]],
            "nonnegative_constrained": [
                _is_nonnegative_parameter(name) for name in problem["names"]
            ],
        }
    )
    bounds_path = os.path.join(output_dir, "sensitivity_parameter_bounds.csv")
    bounds_df.to_csv(bounds_path, index=False)
    logger.info("[Sensitivity] Parameter bounds saved to %s", bounds_path)

    param_values = morris.sample(
        problem,
        N=int(SENSITIVITY_TRAJECTORIES),
        num_levels=int(SENSITIVITY_LEVELS),
        local_optimization=True,
        seed=int(SEED),
    )

    logger.info(
        "[Sensitivity] Generated %d Morris model evaluations for %d parameters.",
        len(param_values),
        problem["num_vars"],
    )

    results_y = np.full(len(param_values), np.nan, dtype=np.float64)
    trajectory_storage = []
    failed_rows = []

    cpu_count = os.cpu_count() or 1
    n_workers = max(1, min(8, int(cpu_count)))

    logger.info(
        "[Sensitivity] Simulating with %d worker processes "
        "(CPU count detected: %d).",
        n_workers,
        cpu_count,
    )

    tasks = [
        _make_task(
            sample_id=i,
            param_vector=param_values[i],
            original_shapes=original_shapes,
            sys_obj=sys,
            idx_obj=idx,
            metric=metric,
            keep_trajectory=True,
        )
        for i in range(len(param_values))
    ]

    mp_ctx = _get_process_context()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
        futures = {
            executor.submit(_worker_simulation, task): task[0]
            for task in tasks
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sensitivity simulations"):
            sample_id = futures[fut]

            try:
                i, y_val, dfp, dfr, dfph = fut.result()
                results_y[i] = float(y_val)

                if dfp is not None or dfr is not None or dfph is not None:
                    trajectory_storage.append(
                        {
                            "id": int(i),
                            "y_val": float(y_val),
                            "prot_df": dfp,
                            "rna_df": dfr,
                            "phos_df": dfph,
                        }
                    )

            except Exception as exc:
                failed_rows.append(
                    {
                        "sample_id": int(sample_id),
                        "failure_reason": str(exc),
                    }
                )
                logger.warning(
                    "[Sensitivity] Simulation %d failed: %s",
                    sample_id,
                    exc,
                )

    response_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(results_y), dtype=int),
            "response": results_y,
        }
    )
    response_path = os.path.join(output_dir, "sensitivity_model_responses.csv")
    response_df.to_csv(response_path, index=False)

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_path = os.path.join(output_dir, "sensitivity_failed_simulations.csv")
        failed_df.to_csv(failed_path, index=False)

        raise RuntimeError(
            f"Sensitivity analysis aborted because {len(failed_rows)} simulations failed. "
            f"See {failed_path}. Morris analysis requires the full design; "
            "dropping failed rows would invalidate the indices."
        )

    if not np.all(np.isfinite(results_y)):
        bad = int((~np.isfinite(results_y)).sum())
        raise RuntimeError(
            f"Sensitivity analysis aborted because {bad} simulations produced non-finite responses. "
            f"See {response_path}."
        )

    logger.info("[Sensitivity] Computing Morris indices.")

    si = analyze(
        problem,
        param_values,
        results_y,
        conf_level=0.95,
        print_to_console=False,
    )

    df_sens = pd.DataFrame(
        {
            "Parameter": problem["names"],
            "mu": si.get("mu", np.full(problem["num_vars"], np.nan)),
            "mu_star": si["mu_star"],
            "sigma": si["sigma"],
            "mu_star_conf": si["mu_star_conf"],
        }
    )

    df_sens = df_sens.sort_values("mu_star", ascending=False)

    out_csv = os.path.join(output_dir, "sensitivity_indices.csv")
    df_sens.to_csv(out_csv, index=False)
    logger.info("[Sensitivity] Indices saved to %s", out_csv)

    trajectory_storage = _select_top_trajectories(
        trajectory_storage,
        max_items=SENSITIVITY_TOP_CURVES,
    )

    traj_summary = pd.DataFrame(
        [
            {
                "id": tr["id"],
                "y_val": tr["y_val"],
            }
            for tr in trajectory_storage
        ]
    )
    traj_summary_path = os.path.join(output_dir, "sensitivity_trajectories.csv")
    traj_summary.to_csv(traj_summary_path, index=False)

    _plot_sensitivity_indices(df_sens.head(30), output_dir)
    _plot_perturbation_cloud(trajectory_storage, output_dir, idx)

    logger.info("[Sensitivity] Analysis complete.")

    return df_sens


def _plot_sensitivity_indices(df, out_dir):
    """Plot top Morris mu_star values."""
    if df is None or len(df) == 0:
        logger.info("[Sensitivity] No sensitivity indices available for plotting.")
        return

    df = df.sort_values("mu_star", ascending=True)

    fig_height = max(6, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(df))
    ax.barh(y, df["mu_star"].to_numpy(dtype=float))
    ax.set_yticks(y)
    ax.set_yticklabels(df["Parameter"].astype(str))
    ax.set_xlabel("mu_star (mean absolute elementary effect)")
    ax.set_title("Morris sensitivity analysis: top parameters")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_mu_star.png"), dpi=300)
    plt.close(fig)


def _plot_perturbation_cloud(
    trajectories,
    out_dir,
    idx,
    top_n_proteins=40,
    top_k_sites=6,
    draw_spaghetti=True,
    spaghetti_alpha=0.03,
):
    """Plot perturbation trajectory clouds for selected proteins."""
    sim_dir = os.path.join(out_dir, "sensitivity_perturbations")
    os.makedirs(sim_dir, exist_ok=True)

    prot_all = []
    rna_all = []
    phos_all = []

    for tr in trajectories:
        sim_id = tr["id"]

        d = tr.get("prot_df", None)
        if d is not None and len(d) > 0:
            dd = d.copy()
            dd["sim_id"] = sim_id
            prot_all.append(dd)

        d = tr.get("rna_df", None)
        if d is not None and len(d) > 0:
            dd = d.copy()
            dd["sim_id"] = sim_id
            rna_all.append(dd)

        d = tr.get("phos_df", None)
        if d is not None and len(d) > 0:
            dd = d.copy()
            dd["sim_id"] = sim_id
            phos_all.append(dd)

    if not prot_all and not rna_all and not phos_all:
        logger.info("[Sensitivity] No trajectories available for perturbation cloud plotting.")
        return

    prot_df = (
        pd.concat(prot_all, ignore_index=True)
        if prot_all
        else pd.DataFrame(columns=["protein", "time", "pred_fc", "sim_id"])
    )
    rna_df = (
        pd.concat(rna_all, ignore_index=True)
        if rna_all
        else pd.DataFrame(columns=["protein", "time", "pred_fc", "sim_id"])
    )
    phos_df = (
        pd.concat(phos_all, ignore_index=True)
        if phos_all
        else pd.DataFrame(columns=["protein", "psite", "time", "pred_fc", "sim_id"])
    )

    for d in (prot_df, rna_df):
        if len(d) > 0:
            d["protein"] = d["protein"].astype(str)
            d["time"] = pd.to_numeric(d["time"], errors="coerce")
            d["pred_fc"] = pd.to_numeric(d["pred_fc"], errors="coerce")
            d.dropna(subset=["time", "pred_fc"], inplace=True)

    if len(phos_df) > 0:
        phos_df["protein"] = phos_df["protein"].astype(str)
        phos_df["psite"] = phos_df["psite"].astype(str)
        phos_df["time"] = pd.to_numeric(phos_df["time"], errors="coerce")
        phos_df["pred_fc"] = pd.to_numeric(phos_df["pred_fc"], errors="coerce")
        phos_df.dropna(subset=["time", "pred_fc"], inplace=True)

    def _protein_score(d):
        if len(d) == 0:
            return pd.DataFrame(columns=["protein", "score"])

        g = (
            d.groupby(["protein", "time"])["pred_fc"]
            .std()
            .reset_index(name="std")
        )
        s = (
            g.groupby("protein")["std"]
            .max()
            .reset_index(name="score")
        )
        return s

    s_prot = _protein_score(prot_df)
    s_rna = _protein_score(rna_df)

    scores = pd.concat(
        [
            s_prot.assign(src="protein"),
            s_rna.assign(src="rna"),
        ],
        ignore_index=True,
    )

    if len(scores) == 0:
        proteins_to_plot = list(idx.proteins[:top_n_proteins])
    else:
        scores = (
            scores.sort_values("score", ascending=False)
            .drop_duplicates("protein")
        )
        proteins_to_plot = scores["protein"].head(top_n_proteins).tolist()

    def _summarize_band(d, group_cols):
        if d is None or len(d) == 0:
            return None

        q = (
            d.groupby(group_cols)["pred_fc"]
            .quantile([0.01, 0.05, 0.5, 0.95, 0.99])
            .unstack()
        )

        q = q.rename(
            columns={
                0.01: "q01",
                0.05: "q05",
                0.5: "med",
                0.95: "q95",
                0.99: "q99",
            }
        ).reset_index()

        return q

    def _plot_band(ax, qdf, xcol="time", label=None):
        if qdf is None or len(qdf) == 0:
            return

        qdf = qdf.sort_values(xcol)

        x = qdf[xcol].to_numpy(dtype=float)
        q01 = qdf["q01"].to_numpy(dtype=float)
        q05 = qdf["q05"].to_numpy(dtype=float)
        med = qdf["med"].to_numpy(dtype=float)
        q95 = qdf["q95"].to_numpy(dtype=float)
        q99 = qdf["q99"].to_numpy(dtype=float)

        ax.fill_between(x, q01, q99, alpha=0.12, linewidth=0)
        ax.fill_between(x, q05, q95, alpha=0.18, linewidth=0)
        ax.plot(x, med, linewidth=2, label=label)

    def _plot_spaghetti(ax, d):
        if not draw_spaghetti or d is None or len(d) == 0:
            return

        for _, sub in d.groupby("sim_id"):
            sub = sub.sort_values("time")
            ax.plot(
                sub["time"].to_numpy(dtype=float),
                sub["pred_fc"].to_numpy(dtype=float),
                alpha=spaghetti_alpha,
                linewidth=1,
            )

    for protein in proteins_to_plot:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        ax_p, ax_r, ax_ph = axes

        d_p = prot_df[prot_df["protein"] == protein].copy()
        if len(d_p) > 0:
            _plot_spaghetti(ax_p, d_p)
            q_p = _summarize_band(d_p, ["time"])
            _plot_band(ax_p, q_p, label="Median protein")

        ax_p.set_title(f"{protein} — protein cloud")
        ax_p.set_xlabel("Time")
        ax_p.set_ylabel("Predicted FC")
        ax_p.grid(True, alpha=0.25)

        d_r = rna_df[rna_df["protein"] == protein].copy()
        if len(d_r) > 0:
            _plot_spaghetti(ax_r, d_r)
            q_r = _summarize_band(d_r, ["time"])
            _plot_band(ax_r, q_r, label="Median RNA")

        ax_r.set_title(f"{protein} — RNA cloud")
        ax_r.set_xlabel("Time")
        ax_r.set_ylabel("Predicted FC")
        ax_r.grid(True, alpha=0.25)

        d_ph = phos_df[phos_df["protein"] == protein].copy()
        if len(d_ph) > 0:
            total = (
                d_ph.groupby(["sim_id", "time"], as_index=False)["pred_fc"]
                .sum()
            )

            if draw_spaghetti:
                for _, sub in total.groupby("sim_id"):
                    sub = sub.sort_values("time")
                    ax_ph.plot(
                        sub["time"].to_numpy(dtype=float),
                        sub["pred_fc"].to_numpy(dtype=float),
                        alpha=spaghetti_alpha,
                        linewidth=1,
                    )

            q_total = _summarize_band(total, ["time"])
            _plot_band(ax_ph, q_total, label="Median total phospho")

            site_var = (
                d_ph.groupby(["psite", "time"])["pred_fc"]
                .std()
                .reset_index(name="std")
                .groupby("psite")["std"]
                .max()
                .sort_values(ascending=False)
            )

            top_sites = site_var.head(top_k_sites).index.tolist()

            for site in top_sites:
                ds = d_ph[d_ph["psite"] == site]
                qs = _summarize_band(ds, ["time"])
                if qs is None or len(qs) == 0:
                    continue

                qs = qs.sort_values("time")
                ax_ph.plot(
                    qs["time"].to_numpy(dtype=float),
                    qs["med"].to_numpy(dtype=float),
                    linewidth=1.5,
                    alpha=0.9,
                    label=str(site),
                )

        ax_ph.set_title(f"{protein} — phospho cloud")
        ax_ph.set_xlabel("Time")
        ax_ph.set_ylabel("Predicted FC")
        ax_ph.grid(True, alpha=0.25)

        for ax in (ax_p, ax_r, ax_ph):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=8, frameon=True)

        fig.suptitle(f"Perturbation cloud: {protein}", fontsize=14)
        fig.tight_layout()

        safe_name = (
            str(protein)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

        out_path = os.path.join(sim_dir, f"cloud_{safe_name}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    logger.info(
        "[Sensitivity] Perturbation cloud plots saved to %s (n=%d).",
        sim_dir,
        len(proteins_to_plot),
    )