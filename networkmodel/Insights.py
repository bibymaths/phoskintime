"""Combined diagnostics and trajectory-feature utilities for networkmodel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAYER_KEYS = {
    "protein": ["protein"],
    "rna": ["protein"],
    "phospho": ["protein", "psite"],
}


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def _log(logger, level: str, msg: str, *args) -> None:
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(msg, *args)


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _report_dir(output_dir: str | Path, report_subdir: str, name: str) -> Path:
    out = Path(output_dir) / report_subdir / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df.copy()


def _observed_set(df: pd.DataFrame, column: str) -> set[str]:
    if column not in df.columns:
        return set()
    return set(df[column].dropna().astype(str))


def _nnz(mat, axis: int) -> np.ndarray:
    """Return sparse non-zero counts by row (axis=1) or column (axis=0)."""
    if mat is None:
        return np.asarray([], dtype=int)
    compressed = mat.tocsr() if axis == 1 and hasattr(mat, "tocsr") else mat.tocsc() if hasattr(mat, "tocsc") else mat
    return np.diff(compressed.indptr).astype(int)


def _as_float_array(x) -> np.ndarray:
    return np.asarray([] if x is None else x, dtype=float)


def _hist(values, title: str, xlabel: str, path: Path) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(30, max(5, int(np.sqrt(values.size)))))
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _barh_top(df: pd.DataFrame, value_col: str, label_col: str, title: str, path: Path, top_n: int = 30) -> None:
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return
    top = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.24 * len(top))))
    ax.barh(top[label_col].astype(str), top[value_col].astype(float))
    ax.set(title=title, xlabel=value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _empty_trajectory_row(layer: str, source: str, keys: list[str], values: tuple) -> dict:
    row = {"layer": layer, "source": source, "entity": ":".join(map(str, values))}
    row.update(dict(zip(keys, values)))
    row.update(
        n_points=0,
        baseline=np.nan,
        final=np.nan,
        min=np.nan,
        max=np.nan,
        amplitude=np.nan,
        auc=np.nan,
        auc_centered=np.nan,
        time_to_peak=np.nan,
        time_to_min=np.nan,
        max_abs_slope=np.nan,
        net_change=np.nan,
        final_over_baseline=np.nan,
        recovery_fraction=np.nan,
        monotonicity=np.nan,
        n_direction_changes=0,
        response_class="empty",
    )
    return row


# -----------------------------------------------------------------------------
# Network diagnostics
# -----------------------------------------------------------------------------


def run_network_diagnostics(
        *,
        idx,
        W_global,
        tf_mat=None,
        kin_input=None,
        df_kin: pd.DataFrame | None = None,
        df_tf: pd.DataFrame | None = None,
        df_prot: pd.DataFrame | None = None,
        df_pho: pd.DataFrame | None = None,
        df_rna: pd.DataFrame | None = None,
        output_dir: str | Path,
        logger=None,
        report_subdir: str = "insights",
) -> dict:
    """Write topology, observability, kinase-support, and TF-support reports."""
    out_dir = _report_dir(output_dir, report_subdir, "network")
    df_kin, df_tf, df_prot, df_pho, df_rna = map(_frame, (df_kin, df_tf, df_prot, df_pho, df_rna))

    site_in_degree = _nnz(W_global, axis=1)
    kinase_out_degree = _nnz(W_global, axis=0)
    tf_target_degree = _nnz(tf_mat, axis=1) if tf_mat is not None else np.zeros(getattr(idx, "N", 0), dtype=int)
    tf_source_degree = _nnz(tf_mat, axis=0) if tf_mat is not None else np.zeros(getattr(idx, "N", 0), dtype=int)

    observed_sites = set()
    if {"protein", "psite"}.issubset(df_pho.columns):
        observed_sites = set(map(tuple, df_pho[["protein", "psite"]].dropna().astype(str).to_numpy()))

    site_rows, flat = [], 0
    for i, protein in enumerate(idx.proteins):
        for local_site, site in enumerate(idx.sites[i]):
            indeg = int(site_in_degree[flat]) if flat < len(site_in_degree) else 0
            site_rows.append(
                {
                    "protein": protein,
                    "psite": site,
                    "site_index": int(flat),
                    "local_site_index": int(local_site),
                    "n_upstream_kinases": indeg,
                    "observed_phosphosite": (str(protein), str(site)) in observed_sites,
                    "weak_flag": indeg == 0,
                }
            )
            flat += 1
    site_table = pd.DataFrame(site_rows)
    site_path = out_dir / "site_topology.csv"
    site_table.to_csv(site_path, index=False)

    obs_prot = _observed_set(df_prot, "protein")
    obs_rna = _observed_set(df_rna, "protein")
    obs_pho_prot = _observed_set(df_pho, "protein")

    protein_rows = []
    for i, protein in enumerate(idx.proteins):
        sub = site_table.loc[site_table["protein"] == protein] if not site_table.empty else pd.DataFrame()
        protein_rows.append(
            {
                "protein": protein,
                "protein_index": int(i),
                "n_sites": int(idx.n_sites[i]),
                "state_offset": int(idx.offset_y[i]),
                "site_offset": int(idx.offset_s[i]),
                "state_block_size": int(idx.block(i).stop - idx.block(i).start),
                "n_site_kinase_edges": int(sub["n_upstream_kinases"].sum()) if not sub.empty else 0,
                "n_observed_sites": int(sub["observed_phosphosite"].sum()) if not sub.empty else 0,
                "tf_in_degree": int(tf_target_degree[i]) if i < len(tf_target_degree) else 0,
                "tf_out_degree": int(tf_source_degree[i]) if i < len(tf_source_degree) else 0,
                "observed_protein": protein in obs_prot,
                "observed_rna": protein in obs_rna,
                "observed_phospho": protein in obs_pho_prot,
                "proxy_target": getattr(idx, "proxy_map", {}).get(protein, ""),
            }
        )
    protein_table = pd.DataFrame(protein_rows)
    if not protein_table.empty:
        obs_cols = ["observed_protein", "observed_rna", "observed_phospho"]
        protein_table["observed_any"] = protein_table[obs_cols].any(axis=1)
        protein_table["weak_flag"] = (~protein_table["observed_any"]) | (
                (protein_table["n_sites"] > 0) & (protein_table["n_site_kinase_edges"] == 0)
        )
    protein_path = out_dir / "protein_observability.csv"
    protein_table.to_csv(protein_path, index=False)

    k_times = _as_float_array(getattr(kin_input, "grid", []))
    k_mat = _as_float_array(getattr(kin_input, "Kmat", []))
    kinase_rows = []
    for j, kinase in enumerate(idx.kinases):
        values = k_mat[j, :] if k_mat.ndim == 2 and j < k_mat.shape[0] else np.asarray([], dtype=float)
        finite = values[np.isfinite(values)]
        kinase_rows.append(
            {
                "kinase": kinase,
                "kinase_index": int(j),
                "n_downstream_sites": int(kinase_out_degree[j]) if j < len(kinase_out_degree) else 0,
                "has_input_trajectory": finite.size > 0,
                "input_min": float(np.min(finite)) if finite.size else np.nan,
                "input_median": float(np.median(finite)) if finite.size else np.nan,
                "input_max": float(np.max(finite)) if finite.size else np.nan,
                "input_dynamic_range": float(np.max(finite) - np.min(finite)) if finite.size else np.nan,
            }
        )
    kinase_table = pd.DataFrame(kinase_rows)
    if not kinase_table.empty:
        kinase_table["weak_flag"] = kinase_table["n_downstream_sites"].eq(0)
    kinase_path = out_dir / "kinase_support.csv"
    kinase_table.to_csv(kinase_path, index=False)

    tf_rows = []
    if not df_tf.empty and {"tf", "target"}.issubset(df_tf.columns):
        tf_edges = df_tf[["tf", "target"]].dropna().astype(str)
        src_counts = tf_edges.groupby("tf").size()
        tgt_counts = tf_edges.groupby("target").size()
        for name in sorted(set(src_counts.index) | set(tgt_counts.index)):
            tf_rows.append(
                {
                    "node": name,
                    "n_targets": int(src_counts.get(name, 0)),
                    "n_regulators": int(tgt_counts.get(name, 0)),
                    "in_idx": name in getattr(idx, "p2i", {}),
                }
            )
    tf_table = pd.DataFrame(tf_rows)
    tf_path = out_dir / "tf_support.csv"
    tf_table.to_csv(tf_path, index=False)

    _hist(site_table.get("n_upstream_kinases", pd.Series(dtype=float)), "Upstream kinases per phosphosite",
          "n upstream kinases", out_dir / "site_indegree_hist.png")
    _hist(kinase_table.get("n_downstream_sites", pd.Series(dtype=float)), "Downstream phosphosites per kinase",
          "n downstream sites", out_dir / "kinase_outdegree_hist.png")

    summary = {
        "n_proteins": int(getattr(idx, "N", len(getattr(idx, "proteins", [])))),
        "n_kinases": int(len(getattr(idx, "kinases", []))),
        "n_sites": int(getattr(idx, "total_sites", len(site_table))),
        "state_dim": int(getattr(idx, "state_dim", 0)),
        "w_global_shape": list(getattr(W_global, "shape", (0, 0))),
        "w_global_nnz": int(getattr(W_global, "nnz", 0)),
        "tf_shape": list(getattr(tf_mat, "shape", (0, 0))) if tf_mat is not None else [0, 0],
        "tf_nnz": int(getattr(tf_mat, "nnz", 0)) if tf_mat is not None else 0,
        "n_sites_without_upstream_kinase": int(site_table.get("n_upstream_kinases", pd.Series(dtype=int)).eq(0).sum()),
        "n_kinases_without_downstream_site": int(
            kinase_table.get("n_downstream_sites", pd.Series(dtype=int)).eq(0).sum()),
        "n_unobserved_model_proteins": int(
            (~protein_table.get("observed_any", pd.Series(dtype=bool))).sum()) if not protein_table.empty else 0,
        "n_proxy_tfs": int(len(getattr(idx, "proxy_map", {}))),
        "time_grid_kinase": k_times.tolist(),
        "paths": {
            "site_topology": str(site_path),
            "protein_observability": str(protein_path),
            "kinase_support": str(kinase_path),
            "tf_support": str(tf_path),
        },
    }
    summary_path = out_dir / "network_diagnostics_summary.json"
    _write_json(summary_path, summary)
    summary["paths"]["summary"] = str(summary_path)

    _log(logger, "info", "[Insights] Network diagnostics written to %s", out_dir)
    if summary["n_sites_without_upstream_kinase"]:
        _log(logger, "warning", "[Insights][Network] %d modeled sites have zero upstream kinases.",
             summary["n_sites_without_upstream_kinase"])
    if summary["n_unobserved_model_proteins"]:
        _log(logger, "warning", "[Insights][Network] %d model proteins are not directly observed.",
             summary["n_unobserved_model_proteins"])
    return summary


# -----------------------------------------------------------------------------
# Parameter diagnostics
# -----------------------------------------------------------------------------


def build_parameter_names(idx, slices: Mapping[str, slice]) -> list[str]:
    """Return raw-theta parameter names matching the existing slice layout."""
    total = max((int(sl.stop) for sl in slices.values()), default=0)
    names = np.empty(total, dtype=object)
    for group, sl in slices.items():
        size = int(sl.stop - sl.start)
        if group == "c_k":
            labels = [f"c_k[{k}]" for k in idx.kinases]
        elif group == "Dp_i":
            labels = [f"Dp_i[{p}_{s}]" for p, sites in zip(idx.proteins, idx.sites) for s in sites]
        elif group == "tf_scale":
            labels = ["tf_scale"]
        else:
            labels = [f"{group}[{p}]" for p in idx.proteins]
        names[sl] = labels if len(labels) == size else [f"{group}[{i}]" for i in range(size)]
    return [str(x) for x in names]


def _parameter_labels(group: str, values, idx) -> tuple[np.ndarray, list[str], str]:
    arr = np.ravel(np.asarray(values, dtype=float))
    if group == "c_k":
        labels, entity_type = list(getattr(idx, "kinases", [])), "kinase"
    elif group == "Dp_i":
        labels, entity_type = [f"{p}:{s}" for p, sites in zip(idx.proteins, idx.sites) for s in sites], "phosphosite"
    elif group == "tf_scale":
        labels, entity_type = ["global"], "global"
    else:
        labels, entity_type = list(getattr(idx, "proteins", [])), "protein"
    if len(labels) != arr.size:
        labels, entity_type = [str(i) for i in range(arr.size)], "index"
    return arr, labels, entity_type


def _physical_parameter_table(params: Mapping[str, object], idx) -> pd.DataFrame:
    rows = []
    for group, values in params.items():
        arr, labels, entity_type = _parameter_labels(group, values, idx)
        rows.extend(
            {
                "parameter_group": group,
                "entity_type": entity_type,
                "entity": labels[i],
                "local_index": int(i),
                "value": float(value),
            }
            for i, value in enumerate(arr)
        )
    return pd.DataFrame(rows)


def _raw_bound_table(theta, lower, upper, names, slices, tol: float) -> pd.DataFrame:
    if theta is None or lower is None or upper is None:
        return pd.DataFrame()
    theta = np.ravel(np.asarray(theta, dtype=float))
    lower = np.ravel(np.asarray(lower, dtype=float))
    upper = np.ravel(np.asarray(upper, dtype=float))
    n = min(theta.size, lower.size, upper.size, len(names))
    if n == 0:
        return pd.DataFrame()

    groups = np.asarray([""] * n, dtype=object)
    for group, sl in slices.items():
        groups[max(0, sl.start): min(n, sl.stop)] = group

    width = upper[:n] - lower[:n]
    dist_lower = theta[:n] - lower[:n]
    dist_upper = upper[:n] - theta[:n]
    denom = np.maximum(np.abs(width), 1e-12)
    table = pd.DataFrame(
        {
            "theta_index": np.arange(n, dtype=int),
            "parameter": names[:n],
            "parameter_group": groups,
            "theta": theta[:n],
            "lower": lower[:n],
            "upper": upper[:n],
            "width": width,
            "distance_to_lower": dist_lower,
            "distance_to_upper": dist_upper,
            "relative_distance_to_lower": dist_lower / denom,
            "relative_distance_to_upper": dist_upper / denom,
        }
    )
    table["near_lower_bound"] = table["relative_distance_to_lower"].le(tol)
    table["near_upper_bound"] = table["relative_distance_to_upper"].le(tol)
    table["near_any_bound"] = table["near_lower_bound"] | table["near_upper_bound"]
    table["fixed_or_degenerate_bound"] = table["width"].abs().le(1e-12)
    return table


def run_parameter_diagnostics(
        *,
        params: Mapping[str, object],
        idx,
        slices: Mapping[str, slice],
        output_dir: str | Path,
        theta=None,
        lower=None,
        upper=None,
        bound_relative_tol: float = 0.02,
        logger=None,
        report_subdir: str = "insights",
) -> dict:
    """Write tidy parameter tables, group summaries, and optimizer-bound flags."""
    out_dir = _report_dir(output_dir, report_subdir, "parameters")

    physical = _physical_parameter_table(params, idx)
    physical_path = out_dir / "physical_parameters_tidy.csv"
    physical.to_csv(physical_path, index=False)

    group_summary = (
        physical.groupby("parameter_group")["value"].agg(n="count", mean="mean", std="std", min="min", median="median",
                                                         max="max").reset_index()
        if not physical.empty
        else pd.DataFrame()
    )
    group_summary_path = out_dir / "physical_parameter_group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    raw_bounds = _raw_bound_table(theta, lower, upper, build_parameter_names(idx, slices), slices, bound_relative_tol)
    raw_bound_path = out_dir / "raw_theta_bound_flags.csv"
    raw_bounds.to_csv(raw_bound_path, index=False)

    near_bounds = raw_bounds.loc[raw_bounds["near_any_bound"] | raw_bounds[
        "fixed_or_degenerate_bound"]].copy() if not raw_bounds.empty else pd.DataFrame()
    near_bound_path = out_dir / "raw_theta_parameters_near_bounds.csv"
    near_bounds.to_csv(near_bound_path, index=False)

    if not physical.empty:
        for group, sub in physical.groupby("parameter_group"):
            safe = str(group).replace("/", "_").replace(" ", "_")
            _hist(sub["value"], f"Physical parameter distribution: {group}", "value",
                  out_dir / f"parameter_distribution_{safe}.png")

    summary = {
        "n_physical_parameters": int(len(physical)),
        "n_raw_theta_parameters": int(len(raw_bounds)),
        "n_raw_parameters_near_bounds": int(raw_bounds["near_any_bound"].sum()) if not raw_bounds.empty else 0,
        "n_raw_fixed_or_degenerate_bounds": int(
            raw_bounds["fixed_or_degenerate_bound"].sum()) if not raw_bounds.empty else 0,
        "paths": {
            "physical_parameters": str(physical_path),
            "physical_group_summary": str(group_summary_path),
            "raw_theta_bound_flags": str(raw_bound_path),
            "raw_theta_parameters_near_bounds": str(near_bound_path),
        },
    }
    summary_path = out_dir / "parameter_diagnostics_summary.json"
    _write_json(summary_path, summary)
    summary["paths"]["summary"] = str(summary_path)

    _log(logger, "info", "[Insights] Parameter diagnostics written to %s", out_dir)
    if summary["n_raw_parameters_near_bounds"]:
        _log(logger, "warning", "[Insights][Parameters] %d raw parameters are close to optimizer bounds.",
             summary["n_raw_parameters_near_bounds"])
    return summary


# -----------------------------------------------------------------------------
# Trajectory features
# -----------------------------------------------------------------------------


def _trajectory_feature_row(layer: str, keys: list[str], key_values, sub: pd.DataFrame, value_col: str,
                            source: str) -> dict:
    key_values = key_values if isinstance(key_values, tuple) else (key_values,)
    sub = sub.sort_values("time")
    t = pd.to_numeric(sub["time"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]

    if y.size == 0:
        return _empty_trajectory_row(layer, source, keys, key_values)

    baseline, final = float(y[0]), float(y[-1])
    y_min, y_max = float(np.min(y)), float(np.max(y))
    i_peak, i_min = int(np.argmax(y)), int(np.argmin(y))
    dy, dt = np.diff(y), np.diff(t)
    valid_dt = np.abs(dt) > 0
    slopes = dy[valid_dt] / dt[valid_dt] if dy.size and np.any(valid_dt) else np.asarray([], dtype=float)
    signs = np.sign(dy[np.abs(dy) > 1e-12])
    n_changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    monotonicity = float(abs(np.sum(dy)) / (np.sum(np.abs(dy)) + 1e-12)) if dy.size else 1.0

    up, down = y_max - baseline, y_min - baseline
    if up >= abs(down):
        dominant = up
        response_class = "induced" if dominant > 0 else "flat"
        recovery_fraction = (y_max - final) / (abs(up) + 1e-12)
    else:
        dominant = down
        response_class = "repressed" if dominant < 0 else "flat"
        recovery_fraction = (final - y_min) / (abs(down) + 1e-12)
    if abs(dominant) < 1e-8:
        response_class = "flat"

    row = {"layer": layer, "source": source, "entity": ":".join(map(str, key_values))}
    row.update(dict(zip(keys, key_values)))
    row.update(
        n_points=int(y.size),
        baseline=baseline,
        final=final,
        min=y_min,
        max=y_max,
        amplitude=float(y_max - y_min),
        auc=float(np.trapezoid(y, t)) if y.size > 1 else 0.0,
        auc_centered=float(np.trapezoid(y - baseline, t)) if y.size > 1 else 0.0,
        time_to_peak=float(t[i_peak]),
        time_to_min=float(t[i_min]),
        max_abs_slope=float(np.max(np.abs(slopes))) if slopes.size else 0.0,
        net_change=float(final - baseline),
        final_over_baseline=float(final / max(abs(baseline), 1e-12)),
        recovery_fraction=float(recovery_fraction),
        monotonicity=monotonicity,
        n_direction_changes=n_changes,
        response_class=response_class,
    )
    return row


def compute_trajectory_features(df: pd.DataFrame | None, *, layer: str, value_col: str, source: str) -> pd.DataFrame:
    """Return one feature row per trajectory entity."""
    if df is None or df.empty:
        return pd.DataFrame()
    if layer not in LAYER_KEYS:
        raise ValueError(f"Unsupported layer: {layer!r}")

    keys = LAYER_KEYS[layer]
    missing = set(keys + ["time", value_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {layer} trajectory features: {sorted(missing)}")

    group_key = keys[0] if len(keys) == 1 else keys
    rows = [_trajectory_feature_row(layer, keys, key_values, sub, value_col, source) for key_values, sub in
            df.groupby(group_key, dropna=False)]
    return pd.DataFrame(rows)


def run_trajectory_feature_export(
        *,
        df_prot_pred: pd.DataFrame | None,
        df_rna_pred: pd.DataFrame | None,
        df_pho_pred: pd.DataFrame | None,
        output_dir: str | Path,
        df_prot_obs: pd.DataFrame | None = None,
        df_rna_obs: pd.DataFrame | None = None,
        df_pho_obs: pd.DataFrame | None = None,
        logger=None,
        report_subdir: str = "insights",
) -> dict:
    """Write trajectory-feature tables for predicted and observed series."""
    out_dir = _report_dir(output_dir, report_subdir, "trajectory_features")
    specs = (
        ("protein", df_prot_pred, "pred_fc", "predicted"),
        ("rna", df_rna_pred, "pred_fc", "predicted"),
        ("phospho", df_pho_pred, "pred_fc", "predicted"),
        ("protein", df_prot_obs, "fc", "observed"),
        ("rna", df_rna_obs, "fc", "observed"),
        ("phospho", df_pho_obs, "fc", "observed"),
    )
    frames = [compute_trajectory_features(df, layer=layer, value_col=value_col, source=source) for
              layer, df, value_col, source in specs if df is not None and not df.empty]
    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    feature_path = out_dir / "trajectory_features.csv"
    features.to_csv(feature_path, index=False)

    predicted = features.loc[features["source"].eq("predicted")].copy() if not features.empty else pd.DataFrame()
    ranked_path = out_dir / "top_dynamic_entities.csv"
    if predicted.empty:
        pd.DataFrame().to_csv(ranked_path, index=False)
    else:
        predicted.sort_values(["amplitude", "max_abs_slope"], ascending=False).to_csv(ranked_path, index=False)
        for layer, sub in predicted.groupby("layer"):
            _barh_top(sub, "amplitude", "entity", f"Top dynamic {layer} trajectories",
                      out_dir / f"top_{layer}_amplitudes.png")

    summary = {
        "n_feature_rows": int(len(features)),
        "n_predicted_rows": int(len(predicted)),
        "paths": {"features": str(feature_path), "top_dynamic_entities": str(ranked_path)},
    }
    if not predicted.empty:
        summary["top_predicted_entities"] = predicted.sort_values("amplitude", ascending=False).head(20)[
            ["layer", "entity", "amplitude", "time_to_peak", "response_class"]
        ].to_dict(orient="records")

    summary_path = out_dir / "trajectory_feature_summary.json"
    _write_json(summary_path, summary)
    summary["paths"]["summary"] = str(summary_path)

    _log(logger, "info", "[Insights] Trajectory features written to %s", out_dir)
    return summary


# -----------------------------------------------------------------------------
# Optional one-call wrapper
# -----------------------------------------------------------------------------


def run_model_insights(
        *,
        idx,
        W_global,
        output_dir: str | Path,
        params: Mapping[str, object] | None = None,
        slices: Mapping[str, slice] | None = None,
        theta=None,
        lower=None,
        upper=None,
        tf_mat=None,
        kin_input=None,
        df_kin: pd.DataFrame | None = None,
        df_tf: pd.DataFrame | None = None,
        df_prot: pd.DataFrame | None = None,
        df_pho: pd.DataFrame | None = None,
        df_rna: pd.DataFrame | None = None,
        df_prot_pred: pd.DataFrame | None = None,
        df_rna_pred: pd.DataFrame | None = None,
        df_pho_pred: pd.DataFrame | None = None,
        logger=None,
        report_subdir: str = "insights",
) -> dict:
    """Run available network, parameter, and trajectory reports from one call."""
    reports = {
        "network": run_network_diagnostics(
            idx=idx,
            W_global=W_global,
            tf_mat=tf_mat,
            kin_input=kin_input,
            df_kin=df_kin,
            df_tf=df_tf,
            df_prot=df_prot,
            df_pho=df_pho,
            df_rna=df_rna,
            output_dir=output_dir,
            logger=logger,
            report_subdir=report_subdir,
        )
    }
    if params is not None and slices is not None:
        reports["parameters"] = run_parameter_diagnostics(
            params=params,
            idx=idx,
            slices=slices,
            theta=theta,
            lower=lower,
            upper=upper,
            output_dir=output_dir,
            logger=logger,
            report_subdir=report_subdir,
        )
    if any(df is not None and not df.empty for df in (df_prot_pred, df_rna_pred, df_pho_pred, df_prot, df_rna, df_pho)):
        reports["trajectory_features"] = run_trajectory_feature_export(
            df_prot_pred=df_prot_pred,
            df_rna_pred=df_rna_pred,
            df_pho_pred=df_pho_pred,
            df_prot_obs=df_prot,
            df_rna_obs=df_rna,
            df_pho_obs=df_pho,
            output_dir=output_dir,
            logger=logger,
            report_subdir=report_subdir,
        )
    return reports
