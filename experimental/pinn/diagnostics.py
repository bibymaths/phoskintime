# networkmodel/pinn/diagnostics.py
"""Diagnostics for PINN / NeuralODE neural parameters.

These reports describe neural RHS weights. They are not biological,
mechanistic, kinase, phosphosite, or transcription parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_default(x: Any):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _safe_array(x, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D-compatible array.")
    return arr


def _write_hist(values: np.ndarray, path: Path, *, title: str, xlabel: str) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite, bins=min(60, max(10, int(np.sqrt(finite.size)))))
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_barh_top(
    table: pd.DataFrame,
    path: Path,
    *,
    value_col: str,
    label_col: str,
    title: str,
    top_n: int = 40,
) -> None:
    if table.empty or value_col not in table.columns or label_col not in table.columns:
        return

    top = table.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top))))
    ax.barh(top[label_col].astype(str), top[value_col].astype(float))
    ax.set(title=title, xlabel=value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_pinn_parameter_reports(
    *,
    theta,
    lower,
    upper,
    spec,
    output_dir: str | Path,
    theta0=None,
    logger=None,
    report_subdir: str = "insights/pinn",
    bound_relative_tol: float = 0.02,
    top_n: int = 100,
) -> dict[str, str]:
    """Write PINN / NeuralODE neural-parameter diagnostics.

    Parameters
    ----------
    theta:
        Final optimized full theta vector. Must include the PINN neural slice.
    lower, upper:
        Full lower/upper optimizer bounds matching theta.
    spec:
        PinnParameterSpec containing nn_slice, n_neural_params, and parameter_names.
    output_dir:
        Run output directory.
    theta0:
        Optional initial full theta vector. If supplied, movement columns are added.
    logger:
        Optional logger.
    report_subdir:
        Subdirectory under output_dir.
    bound_relative_tol:
        Relative distance threshold for near-bound flags.
    top_n:
        Number of largest absolute neural parameters to write separately.

    Returns
    -------
    dict[str, str]
        Paths of generated report files.
    """
    if spec is None or getattr(spec, "n_neural_params", 0) <= 0:
        if logger is not None:
            logger.info("[PINN] No neural parameters detected; skipping PINN diagnostics.")
        return {}

    sl = spec.nn_slice
    if sl.start is None or sl.stop is None:
        raise ValueError("spec.nn_slice must have explicit start and stop.")

    theta = _safe_array(theta, name="theta")
    lower = _safe_array(lower, name="lower")
    upper = _safe_array(upper, name="upper")

    if theta.shape != lower.shape or theta.shape != upper.shape:
        raise ValueError(
            f"theta/lower/upper shape mismatch: {theta.shape}, {lower.shape}, {upper.shape}"
        )

    if sl.stop > theta.size:
        raise ValueError(
            f"PINN slice [{sl.start}, {sl.stop}) exceeds theta length {theta.size}."
        )

    values = theta[sl]
    lo = lower[sl]
    hi = upper[sl]

    theta0_values = None
    if theta0 is not None:
        theta0_arr = _safe_array(theta0, name="theta0")
        if theta0_arr.shape != theta.shape:
            raise ValueError(f"theta0 shape {theta0_arr.shape} does not match theta shape {theta.shape}.")
        theta0_values = theta0_arr[sl]

    names = list(getattr(spec, "parameter_names", ()) or ())
    if len(names) != values.size:
        names = [f"pinn_nn[{i}]" for i in range(values.size)]

    out_dir = Path(output_dir) / report_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    width = hi - lo
    denom = np.maximum(np.abs(width), 1e-12)

    distance_to_lower = values - lo
    distance_to_upper = hi - values
    relative_distance_to_lower = distance_to_lower / denom
    relative_distance_to_upper = distance_to_upper / denom

    table = pd.DataFrame(
        {
            "theta_index": np.arange(sl.start, sl.stop, dtype=int),
            "local_index": np.arange(values.size, dtype=int),
            "parameter": names,
            "parameter_group": "PINN_NN",
            "parameter_type": "neural_rhs_weight",
            "value": values,
            "abs_value": np.abs(values),
            "lower": lo,
            "upper": hi,
            "width": width,
            "distance_to_lower": distance_to_lower,
            "distance_to_upper": distance_to_upper,
            "relative_distance_to_lower": relative_distance_to_lower,
            "relative_distance_to_upper": relative_distance_to_upper,
            "near_lower_bound": relative_distance_to_lower <= bound_relative_tol,
            "near_upper_bound": relative_distance_to_upper <= bound_relative_tol,
            "outside_bounds": (values < lo) | (values > hi),
        }
    )

    table["near_any_bound"] = table["near_lower_bound"] | table["near_upper_bound"]

    if theta0_values is not None:
        table["initial_value"] = theta0_values
        table["delta"] = values - theta0_values
        table["abs_delta"] = np.abs(table["delta"])
    else:
        table["initial_value"] = np.nan
        table["delta"] = np.nan
        table["abs_delta"] = np.nan

    full_path = out_dir / "pinn_neural_parameters.csv"
    table.to_csv(full_path, index=False)

    top_abs = table.sort_values("abs_value", ascending=False).head(int(top_n)).copy()
    top_abs_path = out_dir / "pinn_top_abs_parameters.csv"
    top_abs.to_csv(top_abs_path, index=False)

    near_bounds = table.loc[table["near_any_bound"] | table["outside_bounds"]].copy()
    near_bounds_path = out_dir / "pinn_parameters_near_bounds.csv"
    near_bounds.to_csv(near_bounds_path, index=False)

    summary = {
        "mode": "pinn_neural_parameters",
        "description": (
            "PINN_NN parameters are neural RHS weights. They are not directly "
            "interpretable as biological rate constants."
        ),
        "n_parameters": int(values.size),
        "theta_slice": [int(sl.start), int(sl.stop)],
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
        "value_min": float(np.min(values)),
        "value_median": float(np.median(values)),
        "value_max": float(np.max(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "abs_median": float(np.median(np.abs(values))),
        "abs_max": float(np.max(np.abs(values))),
        "l1_norm": float(np.sum(np.abs(values))),
        "l2_norm": float(np.sqrt(np.sum(values * values))),
        "linf_norm": float(np.max(np.abs(values))),
        "lower_min": float(np.min(lo)),
        "upper_max": float(np.max(hi)),
        "n_near_lower_bound": int(table["near_lower_bound"].sum()),
        "n_near_upper_bound": int(table["near_upper_bound"].sum()),
        "n_near_any_bound": int(table["near_any_bound"].sum()),
        "fraction_near_any_bound": float(table["near_any_bound"].mean()),
        "n_outside_bounds": int(table["outside_bounds"].sum()),
    }

    if theta0_values is not None:
        delta = values - theta0_values
        summary.update(
            {
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta)),
                "delta_abs_mean": float(np.mean(np.abs(delta))),
                "delta_abs_max": float(np.max(np.abs(delta))),
                "delta_l2_norm": float(np.sqrt(np.sum(delta * delta))),
            }
        )

    summary_json_path = out_dir / "pinn_parameter_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, default=_json_default))

    summary_csv_path = out_dir / "pinn_parameter_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

    hist_path = out_dir / "pinn_parameter_hist.png"
    abs_hist_path = out_dir / "pinn_abs_parameter_hist.png"
    delta_hist_path = out_dir / "pinn_parameter_delta_hist.png"
    top_abs_plot_path = out_dir / "pinn_top_abs_parameters.png"

    _write_hist(
        values,
        hist_path,
        title="PINN neural parameter distribution",
        xlabel="neural parameter value",
    )
    _write_hist(
        np.abs(values),
        abs_hist_path,
        title="PINN neural parameter absolute values",
        xlabel="absolute neural parameter value",
    )
    if theta0_values is not None:
        _write_hist(
            values - theta0_values,
            delta_hist_path,
            title="PINN neural parameter movement",
            xlabel="final - initial",
        )

    _write_barh_top(
        top_abs,
        top_abs_plot_path,
        value_col="abs_value",
        label_col="parameter",
        title=f"Top {min(int(top_n), len(top_abs))} PINN neural parameters by absolute value",
        top_n=min(40, int(top_n)),
    )

    paths = {
        "full_table": str(full_path),
        "top_abs_table": str(top_abs_path),
        "near_bounds_table": str(near_bounds_path),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "hist": str(hist_path),
        "abs_hist": str(abs_hist_path),
        "top_abs_plot": str(top_abs_plot_path),
    }

    if theta0_values is not None:
        paths["delta_hist"] = str(delta_hist_path)

    if logger is not None:
        logger.info("[PINN] Neural parameter diagnostics written to %s", out_dir)
        logger.info(
            "[PINN] Neural parameters: n=%d l2=%.6g max_abs=%.6g near_bounds=%d outside_bounds=%d",
            summary["n_parameters"],
            summary["l2_norm"],
            summary["abs_max"],
            summary["n_near_any_bound"],
            summary["n_outside_bounds"],
        )

    return paths

def _state_metadata(idx, state_dim: int, model_code: int | None = None) -> pd.DataFrame:
    """Return state index/name/type metadata for networkmodel trajectories."""
    state_dim = int(state_dim)

    rows = [
        {
            "state_index": i,
            "state_name": f"state[{i}]",
            "state_type": "unknown",
            "protein": "",
            "psite": "",
        }
        for i in range(state_dim)
    ]

    proteins = list(getattr(idx, "proteins", []))
    offsets = np.asarray(getattr(idx, "offset_y", []), dtype=int)

    if len(proteins) == 0 or offsets.size == 0:
        return pd.DataFrame(rows)

    use_combinatorial = int(model_code) == 2 if model_code is not None else hasattr(idx, "n_states")

    for i, protein in enumerate(proteins):
        if i >= offsets.size:
            continue

        st = int(offsets[i])
        if 0 <= st < state_dim:
            rows[st] = {
                "state_index": st,
                "state_name": f"{protein}:RNA",
                "state_type": "rna",
                "protein": str(protein),
                "psite": "",
            }

        if use_combinatorial and hasattr(idx, "n_states"):
            ns = int(np.asarray(getattr(idx, "n_states"))[i])
            p0 = st + 1

            for q in range(ns):
                j = p0 + q
                if 0 <= j < state_dim:
                    rows[j] = {
                        "state_index": j,
                        "state_name": f"{protein}:phospho_state_{q}",
                        "state_type": "phospho_combinatorial_state",
                        "protein": str(protein),
                        "psite": "",
                    }

        else:
            n_sites = int(np.asarray(getattr(idx, "n_sites", []))[i]) if hasattr(idx, "n_sites") else 0

            p0 = st + 1
            if 0 <= p0 < state_dim:
                rows[p0] = {
                    "state_index": p0,
                    "state_name": f"{protein}:protein_unphosphorylated",
                    "state_type": "protein_unphosphorylated",
                    "protein": str(protein),
                    "psite": "",
                }

            sites = list(getattr(idx, "sites", [[]])[i]) if hasattr(idx, "sites") else []
            for s_idx in range(n_sites):
                j = st + 2 + s_idx
                psite = str(sites[s_idx]) if s_idx < len(sites) else f"site_{s_idx}"
                if 0 <= j < state_dim:
                    rows[j] = {
                        "state_index": j,
                        "state_name": f"{protein}:{psite}",
                        "state_type": "phosphosite",
                        "protein": str(protein),
                        "psite": psite,
                    }

    return pd.DataFrame(rows)


def _as_trajectory_array(trajectory, *, name: str = "trajectory") -> np.ndarray:
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D time-by-state array, got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}.")
    return arr


def write_pinn_rhs_reports(
    *,
    theta,
    spec,
    pinn_config,
    idx,
    time_grid,
    trajectory,
    output_dir: str | Path,
    logger=None,
    report_subdir: str = "insights/pinn",
    model_code: int | None = None,
    jacobian_stride: int = 1,
    max_jacobian_timepoints: int = 25,
) -> dict[str, str]:
    """Write PINN RHS and learned-coupling diagnostics.

    Files written
    -------------
    pinn_rhs_state_contributions.csv
        Long table of neural RHS values along the fitted trajectory.

    pinn_rhs_summary_by_state.csv
        Per-state summary of neural RHS magnitude.

    pinn_jacobian_summary.csv
        Average learned local coupling strength along the fitted trajectory:
        d f_target / d y_source.

    Interpretation
    --------------
    These are learned neural vector-field diagnostics. They are not kinase
    effects, TF effects, biochemical rates, or causal network edges.
    """
    if spec is None or getattr(spec, "n_neural_params", 0) <= 0:
        if logger is not None:
            logger.info("[PINN] No neural parameters detected; skipping PINN RHS diagnostics.")
        return {}

    if pinn_config is None or not getattr(pinn_config, "enabled", False):
        if logger is not None:
            logger.info("[PINN] PINN disabled; skipping PINN RHS diagnostics.")
        return {}

    mode = str(getattr(pinn_config, "mode", "off")).lower()
    if mode not in {"hybrid", "neuralode"}:
        if logger is not None:
            logger.info("[PINN] PINN mode %r has no neural RHS diagnostics.", mode)
        return {}

    import jax
    import jax.numpy as jnp

    from networkmodel.pinn.objective import _rebuild_model

    theta = _safe_array(theta, name="theta")
    Y = _as_trajectory_array(trajectory)
    times = np.asarray(time_grid, dtype=np.float64).reshape(-1)

    if times.size != Y.shape[0]:
        raise ValueError(
            f"time_grid length {times.size} does not match trajectory rows {Y.shape[0]}."
        )

    sl = spec.nn_slice
    if sl.start is None or sl.stop is None:
        raise ValueError("spec.nn_slice must have explicit start and stop.")
    if sl.stop > theta.size:
        raise ValueError(f"PINN slice [{sl.start}, {sl.stop}) exceeds theta length {theta.size}.")

    nn_flat = jnp.asarray(theta[sl], dtype=jnp.float64)
    nn_model = _rebuild_model(spec, nn_flat)

    t_scale = float(getattr(pinn_config, "t_scale", 1.0))
    y_scale = float(getattr(pinn_config, "y_scale", 1.0))
    if t_scale <= 0 or y_scale <= 0:
        raise ValueError("pinn_config.t_scale and pinn_config.y_scale must be positive.")

    state_meta = _state_metadata(idx, state_dim=Y.shape[1], model_code=model_code)

    def neural_rhs_one(ti, yi):
        ti = jnp.asarray(ti, dtype=jnp.float64)
        yi = jnp.asarray(yi, dtype=jnp.float64)
        return nn_model(ti / t_scale, yi / y_scale)

    rhs_values = np.asarray(
        jax.vmap(neural_rhs_one)(
            jnp.asarray(times, dtype=jnp.float64),
            jnp.asarray(Y, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )

    out_dir = Path(output_dir) / report_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Long RHS contribution table
    # ------------------------------------------------------------------
    T, S = rhs_values.shape

    state_index = np.tile(np.arange(S, dtype=int), T)
    time_long = np.repeat(times, S)
    rhs_long = rhs_values.reshape(-1)

    meta_long = state_meta.iloc[state_index].reset_index(drop=True)

    contrib = pd.DataFrame(
        {
            "time": time_long,
            "state_index": state_index,
            "state_name": meta_long["state_name"].to_numpy(),
            "state_type": meta_long["state_type"].to_numpy(),
            "protein": meta_long["protein"].to_numpy(),
            "psite": meta_long["psite"].to_numpy(),
            "rhs_value": rhs_long,
            "abs_rhs_value": np.abs(rhs_long),
            "pinn_mode": mode,
            "rhs_component": "neural_rhs",
        }
    )

    contrib_path = out_dir / "pinn_rhs_state_contributions.csv"
    contrib.to_csv(contrib_path, index=False)

    # ------------------------------------------------------------------
    # 2) Per-state summary
    # ------------------------------------------------------------------
    abs_rhs = np.abs(rhs_values)

    auc_abs = np.asarray(
        [
            float(np.trapezoid(abs_rhs[:, j], times)) if T > 1 else 0.0
            for j in range(S)
        ],
        dtype=np.float64,
    )

    summary = state_meta.copy()
    summary["mean_rhs"] = np.mean(rhs_values, axis=0)
    summary["mean_abs_rhs"] = np.mean(abs_rhs, axis=0)
    summary["max_abs_rhs"] = np.max(abs_rhs, axis=0)
    summary["auc_abs_rhs"] = auc_abs
    summary["signed_auc_rhs"] = np.asarray(
        [
            float(np.trapezoid(rhs_values[:, j], times)) if T > 1 else 0.0
            for j in range(S)
        ],
        dtype=np.float64,
    )
    summary["pinn_mode"] = mode

    summary = summary.sort_values(
        ["mean_abs_rhs", "max_abs_rhs", "auc_abs_rhs"],
        ascending=False,
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)

    summary = summary[
        [
            "state_name",
            "state_type",
            "state_index",
            "protein",
            "psite",
            "mean_rhs",
            "mean_abs_rhs",
            "max_abs_rhs",
            "auc_abs_rhs",
            "signed_auc_rhs",
            "rank",
            "pinn_mode",
        ]
    ]

    summary_path = out_dir / "pinn_rhs_summary_by_state.csv"
    summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------------
    # 3) Jacobian summary: d f_target / d y_source
    # ------------------------------------------------------------------
    jacobian_stride = max(1, int(jacobian_stride))
    candidate_idx = np.arange(0, T, jacobian_stride, dtype=int)

    if candidate_idx.size > int(max_jacobian_timepoints):
        lin = np.linspace(0, candidate_idx.size - 1, int(max_jacobian_timepoints))
        candidate_idx = candidate_idx[np.unique(np.round(lin).astype(int))]

    def jac_one(ti, yi):
        def f_y(y_inner):
            return neural_rhs_one(ti, y_inner)

        return jax.jacfwd(f_y)(yi)

    jac = np.asarray(
        jax.vmap(jac_one)(
            jnp.asarray(times[candidate_idx], dtype=jnp.float64),
            jnp.asarray(Y[candidate_idx], dtype=jnp.float64),
        ),
        dtype=np.float64,
    )

    # Shape: sampled_time x target_state x source_state
    mean_abs = np.mean(np.abs(jac), axis=0)
    max_abs = np.max(np.abs(jac), axis=0)
    mean_signed = np.mean(jac, axis=0)

    source_meta = state_meta.rename(
        columns={
            "state_index": "source_state_index",
            "state_name": "source_state",
            "state_type": "source_state_type",
            "protein": "source_protein",
            "psite": "source_psite",
        }
    )

    target_meta = state_meta.rename(
        columns={
            "state_index": "target_state_index",
            "state_name": "target_state",
            "state_type": "target_state_type",
            "protein": "target_protein",
            "psite": "target_psite",
        }
    )

    jac_rows = []
    for target in range(S):
        for source in range(S):
            jac_rows.append(
                {
                    "source_state_index": source,
                    "target_state_index": target,
                    "mean_abs_dtarget_dsource": float(mean_abs[target, source]),
                    "max_abs_dtarget_dsource": float(max_abs[target, source]),
                    "mean_signed_dtarget_dsource": float(mean_signed[target, source]),
                    "n_timepoints_used": int(candidate_idx.size),
                    "pinn_mode": mode,
                }
            )

    jac_summary = pd.DataFrame(jac_rows)
    jac_summary = jac_summary.merge(source_meta, on="source_state_index", how="left")
    jac_summary = jac_summary.merge(target_meta, on="target_state_index", how="left")

    jac_summary = jac_summary[
        [
            "source_state",
            "target_state",
            "source_state_index",
            "target_state_index",
            "source_state_type",
            "target_state_type",
            "source_protein",
            "target_protein",
            "source_psite",
            "target_psite",
            "mean_abs_dtarget_dsource",
            "max_abs_dtarget_dsource",
            "mean_signed_dtarget_dsource",
            "n_timepoints_used",
            "pinn_mode",
        ]
    ].sort_values(
        ["mean_abs_dtarget_dsource", "max_abs_dtarget_dsource"],
        ascending=False,
    )

    jac_path = out_dir / "pinn_jacobian_summary.csv"
    jac_summary.to_csv(jac_path, index=False)

    # ------------------------------------------------------------------
    # JSON manifest
    # ------------------------------------------------------------------
    manifest = {
        "description": (
            "PINN RHS diagnostics describe the learned neural vector field. "
            "They are not biological rate constants or causal network edges."
        ),
        "pinn_mode": mode,
        "n_timepoints": int(T),
        "n_states": int(S),
        "n_jacobian_timepoints": int(candidate_idx.size),
        "jacobian_time_indices": candidate_idx.tolist(),
        "paths": {
            "rhs_state_contributions": str(contrib_path),
            "rhs_summary_by_state": str(summary_path),
            "jacobian_summary": str(jac_path),
        },
    }

    manifest_path = out_dir / "pinn_rhs_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default))

    if logger is not None:
        logger.info("[PINN] RHS state contributions written to %s", contrib_path)
        logger.info("[PINN] RHS summary by state written to %s", summary_path)
        logger.info("[PINN] Jacobian summary written to %s", jac_path)
        logger.info(
            "[PINN] Top neural RHS state: %s | mean_abs_rhs=%.6g",
            str(summary.iloc[0]["state_name"]) if not summary.empty else "none",
            float(summary.iloc[0]["mean_abs_rhs"]) if not summary.empty else 0.0,
        )

    return {
        "rhs_state_contributions": str(contrib_path),
        "rhs_summary_by_state": str(summary_path),
        "jacobian_summary": str(jac_path),
        "manifest": str(manifest_path),
    }