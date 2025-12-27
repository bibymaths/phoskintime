#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global ODE Dual-Fit (Pymoo / UNSGA3) - Elementwise Multi-Objective Optimization
"""

import argparse
import json
import os
import re
import multiprocessing as mp
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import warnings
from scipy.integrate import odeint, ODEintWarning

warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", message="Excess work done on this call")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------
# Numba
# ------------------------------

from numba import njit

# ------------------------------
# Pymoo
# ------------------------------
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.core.problem import StarmapParallelization
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
# ------------------------------
# Global Constants
# ------------------------------
TIME_POINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    dtype=float
)
TIME_POINTS_RNA = np.array(
    [4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    dtype=float
)

# ------------------------------
# Biology bounds (physical) then mapped to raw via inv_softplus
# ------------------------------
BOUNDS_CONFIG = {
    "c_k":      (1e-6, 5.0),
    "A_i":      (1e-6, 2.0),
    "B_i":      (1e-6, 2.0),
    "C_i":      (1e-6, 2.0),
    "D_i":      (1e-6, 2.0),
    "E_i":      (1e-6, 2.0),
    "tf_scale": (1e-6, 0.5),
}

def softplus(x):
    # stable softplus
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def inv_softplus(y):
    y = np.maximum(y, 1e-12)
    return np.log(np.expm1(y))

def normalize_fc_to_t0(df):
    df = df.copy()
    # assumes each protein has a row at time==0
    t0 = df[df["time"] == 0.0].set_index("protein")["fc"]
    df["fc"] = df.apply(lambda r: r["fc"] / t0.get(r["protein"], np.nan), axis=1)
    return df.dropna(subset=["fc"])

def save_pareto_3d(res, selected_solution=None, output_dir="out_moo"):
    """
    Saves a high-quality 3D Scatter plot of the Pareto Front.
    Highlights the 'selected' balanced solution if provided.
    """
    print("[Output] Generating 3D Pareto Plot...")

    # 1. Setup Plot
    # Angle (elevation, azimuth) for best viewing
    plot = Scatter(
        plot_3d=True,
        angle=(45, 45),
        labels=["Prot MSE", "RNA MSE", "Reg Loss"],
        figsize=(10, 8),
        title=("Pareto Front", {'pad': 20})
    )

    # 2. Add Data
    # All solutions in the front
    plot.add(res.F, color="grey", alpha=0.6, s=30, label="Pareto Solutions")

    # Highlight the picked solution (Red Star)
    if selected_solution is not None:
        plot.add(selected_solution, color="red", s=150, marker="*", label="Selected")

    # 3. Save
    # Pymoo plots wrap Matplotlib, so we can save easily
    save_path = os.path.join(output_dir, "pareto_front_3d.png")
    plot.save(save_path)
    print(f"[Output] Saved: {save_path}")


def save_parallel_coordinates(res, selected_solution=None, output_dir="out_moo"):
    """
    Saves a Parallel Coordinate Plot (PCP).
    Great for visualizing trade-offs across normalized axes.
    """
    print("[Output] Generating Parallel Coordinate Plot...")

    # 1. Setup Plot
    # normalize_each_axis=True is CRITICAL because MSE errors and Reg loss
    # might have vastly different magnitudes (e.g. 1000 vs 0.1).
    plot = PCP(
        title=("Objective Trade-offs", {'pad': 20}),
        labels=["Prot MSE", "RNA MSE", "Reg Loss"],
        normalize_each_axis=True,
        figsize=(12, 6),
        legend=(True, {'loc': "upper left"})
    )

    # 2. Styling
    plot.set_axis_style(color="grey", alpha=0.5)

    # 3. Add Data
    # Background solutions (faint)
    plot.add(res.F, color="grey", alpha=0.2, linewidth=1)

    # Highlight Selected (Bold Blue)
    if selected_solution is not None:
        plot.add(selected_solution, linewidth=4, color="blue", label="Selected")

    # 4. Save
    save_path = os.path.join(output_dir, "pareto_pcp.png")
    plot.save(save_path)
    print(f"[Output] Saved: {save_path}")


def create_convergence_video(res, output_dir="out_moo", filename="optimization_history.mp4"):
    """
    Creates an animation of the Pareto Front evolution using standard Matplotlib.
    Saves as .gif (universal) or .mp4 (if ffmpeg is installed).
    """
    print("[Output] Rendering Optimization Video...")

    if not res.history:
        print("[Warning] No history found. Cannot create video.")
        return

    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Pre-determine bounds so the camera doesn't jump around
    all_F = np.vstack([e.pop.get("F") for e in res.history])
    min_f = all_F.min(axis=0)
    max_f = all_F.max(axis=0)

    def update(frame_idx):
        ax.clear()

        # Get data for this generation
        entry = res.history[frame_idx]
        gen = entry.n_gen
        F = entry.pop.get("F")

        # Plot
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c="blue", s=10, alpha=0.6, label="Population")

        # Reference (Final Result) - Ghosted
        if res.F is not None:
            ax.scatter(res.F[:, 0], res.F[:, 1], res.F[:, 2], c="red", s=5, alpha=0.1)

        # Styling
        ax.set_title(f"Optimization History - Gen {gen}")
        ax.set_xlabel("Prot MSE")
        ax.set_ylabel("RNA MSE")
        ax.set_zlabel("Reg Loss")
        ax.set_xlim(min_f[0], max_f[0])
        ax.set_ylim(min_f[1], max_f[1])
        ax.set_zlim(min_f[2], max_f[2])
        ax.view_init(elev=45, azim=45)

    # Create Animation
    # We skip frames to make it render faster (every 5th gen)
    frames = list(range(0, len(res.history), 5))
    if len(res.history) - 1 not in frames:
        frames.append(len(res.history) - 1)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)

    # Save (Try MP4 via ffmpeg, fallback to GIF via Pillow)
    save_path = os.path.join(output_dir, filename)

    try:
        # Try saving highly compressed MP4
        ani.save(save_path, writer='ffmpeg', fps=5, dpi=150)
        print(f"[Output] Video saved: {save_path}")
    except Exception:
        # Fallback to GIF (universally supported, no ffmpeg needed)
        gif_path = save_path.replace(".mp4", ".gif")
        print("[System] FFMPEG not found. Falling back to GIF...")
        ani.save(gif_path, writer='pillow', fps=5, dpi=100)
        print(f"[Output] Video saved: {gif_path}")

    plt.close()

def export_pareto_front_to_excel(
    res,
    sys,
    idx,
    slices,
    output_path,
    weights=(1.0, 1.0, 1.0),   # (w_prot, w_rna, w_reg) used for scalar score + ranking
    top_k_trajectories=None,   # None = export trajectories for all solutions; else only top K by scalar score
    t_points_p=None,
    t_points_r=None,
    rtol=1e-5,
    atol=1e-7,
    mxstep=5000,
):
    """
    Export all Pareto solutions into one Excel workbook.

    Writes sheets:
      - summary: objectives + scalar score + rank + weights
      - params_genes: per-protein parameters for each sol_id (long format)
      - params_kinases: per-kinase parameters for each sol_id (long format)
      - traj_protein: trajectories per sol_id (long format)
      - traj_rna: trajectories per sol_id (long format)

    Notes:
      - This can get HUGE if you have many Pareto points. Use top_k_trajectories.
      - The function assumes res.X and res.F exist.
    """
    X = np.asarray(res.X)
    F = np.asarray(res.F)

    if t_points_p is None:
        t_points_p = TIME_POINTS
    if t_points_r is None:
        t_points_r = TIME_POINTS_RNA

    w_prot, w_rna, w_reg = map(float, weights)

    # ---- Summary + ranking ----
    df_summary = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "reg_loss"])
    df_summary.insert(0, "sol_id", np.arange(len(df_summary), dtype=int))
    df_summary["w_prot"] = w_prot
    df_summary["w_rna"] = w_rna
    df_summary["w_reg"] = w_reg

    # scalar score for ranking / convenience
    df_summary["scalar_score"] = (
        w_prot * df_summary["prot_mse"]
        + w_rna * df_summary["rna_mse"]
        + w_reg * df_summary["reg_loss"]
    )

    # rank: 1 = best
    df_summary["rank"] = df_summary["scalar_score"].rank(method="dense").astype(int)
    df_summary.sort_values(["scalar_score", "sol_id"], inplace=True, ignore_index=True)

    # which solutions get trajectories exported
    if top_k_trajectories is None:
        sol_ids_for_traj = df_summary["sol_id"].tolist()
    else:
        sol_ids_for_traj = df_summary.nsmallest(int(top_k_trajectories), "scalar_score")["sol_id"].tolist()

    # ---- Collect params + trajectories ----
    params_genes_rows = []
    params_kin_rows = []
    traj_p_list = []
    traj_r_list = []

    # Cache these for speed
    proteins = idx.proteins
    kinases = idx.kinases

    for sol_id in range(X.shape[0]):
        theta = X[sol_id].astype(float)
        p = unpack_params(theta, slices)

        # update system parameters
        sys.update(**p)

        # ----- parameters: genes (A,B,C,D,E + tf_scale) -----
        # long-form rows: sol_id, protein, param, value
        # (You can switch to wide format later in Excel easily)
        tf_scale_val = float(p["tf_scale"])
        for i, g in enumerate(proteins):
            params_genes_rows.append((sol_id, g, "A_i", float(p["A_i"][i])))
            params_genes_rows.append((sol_id, g, "B_i", float(p["B_i"][i])))
            params_genes_rows.append((sol_id, g, "C_i", float(p["C_i"][i])))
            params_genes_rows.append((sol_id, g, "D_i", float(p["D_i"][i])))
            params_genes_rows.append((sol_id, g, "E_i", float(p["E_i"][i])))
            params_genes_rows.append((sol_id, g, "tf_scale", tf_scale_val))

        # ----- parameters: kinases (c_k) -----
        for j, k in enumerate(kinases):
            params_kin_rows.append((sol_id, k, float(p["c_k"][j])))

        # ----- trajectories (optional / top-K) -----
        if sol_id in sol_ids_for_traj:
            # use your existing measurement function (calls simulate_odeint internally)
            dfp, dfr = simulate_and_measure(sys, idx, t_points_p, t_points_r)

            if dfp is not None and not dfp.empty:
                dfp = dfp.copy()
                dfp.insert(0, "sol_id", sol_id)
                traj_p_list.append(dfp)

            if dfr is not None and not dfr.empty:
                dfr = dfr.copy()
                dfr.insert(0, "sol_id", sol_id)
                traj_r_list.append(dfr)

    df_params_genes = pd.DataFrame(params_genes_rows, columns=["sol_id", "protein", "param", "value"])
    df_params_kin = pd.DataFrame(params_kin_rows, columns=["sol_id", "kinase", "c_k"])

    df_traj_p = pd.concat(traj_p_list, ignore_index=True) if traj_p_list else pd.DataFrame(
        columns=["sol_id", "protein", "time", "pred_fc"]
    )
    df_traj_r = pd.concat(traj_r_list, ignore_index=True) if traj_r_list else pd.DataFrame(
        columns=["sol_id", "protein", "time", "pred_fc"]
    )

    # ---- Write Excel ----
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_params_genes.to_excel(writer, sheet_name="params_genes", index=False)
        df_params_kin.to_excel(writer, sheet_name="params_kinases", index=False)
        df_traj_p.to_excel(writer, sheet_name="traj_protein", index=False)
        df_traj_r.to_excel(writer, sheet_name="traj_rna", index=False)

    print(f"[Output] Pareto export saved: {output_path}")
    print(f"[Output] Solutions: {len(df_summary)} | Traj exported for: {len(sol_ids_for_traj)}")

def save_gene_timeseries_plots(
    gene: str,
    df_prot_obs: pd.DataFrame,
    df_prot_pred: pd.DataFrame,
    df_rna_obs: pd.DataFrame,
    df_rna_pred: pd.DataFrame,
    output_dir: str,
    prot_times: np.ndarray = None,
    rna_times: np.ndarray = None,
    filename_prefix: str = "ts",
    dpi: int = 300,
):
    """
    Save a 2-panel time-series plot for ONE gene symbol:
      - Top: Protein observed vs predicted FC across TIME_POINTS
      - Bottom: mRNA observed vs predicted FC across TIME_POINTS_RNA

    Expected inputs:
      df_*_obs  columns:  protein, time, fc
      df_*_pred columns: protein, time, pred_fc   (or fc_pred; handled)

    Output:
      {output_dir}/{filename_prefix}_{gene}.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- normalize column names for predicted dfs ----
    def _norm_pred(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "pred_fc" in df.columns and "fc_pred" not in df.columns:
            df.rename(columns={"pred_fc": "fc_pred"}, inplace=True)
        if "fc_pred" not in df.columns:
            raise ValueError("Pred df must have 'pred_fc' or 'fc_pred' column.")
        return df

    prot_pred = _norm_pred(df_prot_pred)
    rna_pred  = _norm_pred(df_rna_pred)

    # ---- subset one gene ----
    p_obs = df_prot_obs[df_prot_obs["protein"] == gene].copy()
    p_pre = prot_pred[prot_pred["protein"] == gene].copy()
    r_obs = df_rna_obs[df_rna_obs["protein"] == gene].copy()
    r_pre = rna_pred[rna_pred["protein"] == gene].copy()

    if p_obs.empty and p_pre.empty and r_obs.empty and r_pre.empty:
        # nothing to plot
        return None

    # optional: restrict to known grids
    if prot_times is not None:
        p_obs = p_obs[p_obs["time"].isin(prot_times)]
        p_pre = p_pre[p_pre["time"].isin(prot_times)]
    if rna_times is not None:
        r_obs = r_obs[r_obs["time"].isin(rna_times)]
        r_pre = r_pre[r_pre["time"].isin(rna_times)]

    # ensure numeric + sorted
    for df, col in [(p_obs, "fc"), (r_obs, "fc"), (p_pre, "fc_pred"), (r_pre, "fc_pred")]:
        if not df.empty:
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["time", col], inplace=True)
            df.sort_values("time", inplace=True)

    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    ax_p, ax_r = axes

    # Protein panel
    if not p_obs.empty:
        ax_p.plot(p_obs["time"].to_numpy(), p_obs["fc"].to_numpy(), marker="o", linewidth=2, label="Protein obs")
    if not p_pre.empty:
        ax_p.plot(p_pre["time"].to_numpy(), p_pre["fc_pred"].to_numpy(), marker="o", linewidth=2, label="Protein pred")
    ax_p.set_title(f"{gene} — Protein")
    ax_p.set_xlabel("Time")
    ax_p.set_ylabel("FC")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend()

    # RNA panel
    if not r_obs.empty:
        ax_r.plot(r_obs["time"].to_numpy(), r_obs["fc"].to_numpy(), marker="o", linewidth=2, label="mRNA obs")
    if not r_pre.empty:
        ax_r.plot(r_pre["time"].to_numpy(), r_pre["fc_pred"].to_numpy(), marker="o", linewidth=2, label="mRNA pred")
    ax_r.set_title(f"{gene} — mRNA")
    ax_r.set_xlabel("Time")
    ax_r.set_ylabel("FC")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend()

    fig.suptitle(f"Observed vs Predicted Time Series — {gene}", y=0.98)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{filename_prefix}_{gene}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_goodness_of_fit(df_prot_obs, df_prot_pred, df_rna_obs, df_rna_pred, output_dir):
    """
    Generates a Goodness of Fit (Parity Plot) for Protein and RNA data.
    """

    # 1. Prepare Data
    # Merge on protein + time to align points
    # Note the suffixes: _obs and _pred
    mp = df_prot_obs.merge(df_prot_pred, on=["protein", "time"], suffixes=('_obs', '_pred'))
    mr = df_rna_obs.merge(df_rna_pred, on=["protein", "time"], suffixes=('_obs', '_pred'))

    # Combine into one plotting structure
    mp["Type"] = "Protein"
    mr["Type"] = "RNA"
    combined = pd.concat([mp, mr], ignore_index=True)

    # 2. Critical Renaming Step
    # The merge created 'pred_fc' (from df_prot_pred) or 'fc_pred' depending on input names.
    # Let's standardize everything to 'fc_obs' and 'fc_pred' to be safe.

    # Standardize Observation Column
    if "fc" in combined.columns:
        combined.rename(columns={"fc": "fc_obs"}, inplace=True)

    # Standardize Prediction Column (The cause of your error)
    if "pred_fc" in combined.columns:
        combined.rename(columns={"pred_fc": "fc_pred"}, inplace=True)

    # Drop NaNs just in case
    combined = combined.dropna(subset=["fc_obs", "fc_pred"])

    # 3. Setup Plot
    sns.set_style("whitegrid")
    g = sns.FacetGrid(combined, col="Type", height=6, sharex=False, sharey=False)

    def scatter_with_metrics(x, y, **kwargs):
        # Scatter points
        plt.scatter(x, y, alpha=0.5, edgecolor='w', s=30, **kwargs)

        # Identity line (y=x)
        lims = [min(min(x), min(y)), max(max(x), max(y))]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label="Perfect Fit")

        # Regression with 95% CI
        sns.regplot(x=x, y=y, scatter=False, ci=95,
                    line_kws={"color": "red", "alpha": 0.5, "lw": 2}, label="95% CI Fit")

        # Calculate Metrics
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r2 = r_value ** 2
        rmse = np.sqrt(np.mean((y - x) ** 2))

        # Annotate
        ax = plt.gca()
        stats = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- FIX IS HERE: Use "fc_pred" instead of "pred_fc" ---
    g.map(scatter_with_metrics, "fc_obs", "fc_pred")

    g.set_axis_labels("Observed FC", "Predicted FC")
    g.fig.suptitle("Goodness of Fit: Global ODE Model", y=1.05)

    # 4. Save
    out_path = os.path.join(output_dir, "goodness_of_fit.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved Goodness of Fit plot to: {out_path}")


def plot_gof_from_pareto_excel(
        excel_path: str,
        output_dir: str,
        plot_goodness_of_fit_func,
        df_prot_obs_all: pd.DataFrame,  # your df_prot (columns: protein,time,fc)
        df_rna_obs_all: pd.DataFrame,  # your df_rna  (columns: protein,time,fc)
        traj_protein_sheet: str = "traj_protein",
        traj_rna_sheet: str = "traj_rna",
        summary_sheet: str = "summary",
        top_k: int = None,
        only_solutions=None,  # iterable of sol_id
        score_col: str = "scalar_score",
        make_subdirs: bool = True,
):
    """
    Uses the Excel produced by export_pareto_front_to_excel (your current version):
      - summary
      - traj_protein (sol_id, protein, time, pred_fc)
      - traj_rna     (sol_id, protein, time, pred_fc)

    Observations are NOT in Excel, so we take them from df_prot_obs_all / df_rna_obs_all.
    """
    os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    for s in [traj_protein_sheet, traj_rna_sheet, summary_sheet]:
        if s not in xls.sheet_names:
            raise ValueError(f"Missing sheet '{s}' in {excel_path}. Found: {xls.sheet_names}")

    df_sum = pd.read_excel(xls, sheet_name=summary_sheet)
    if "sol_id" not in df_sum.columns:
        raise ValueError(f"'{summary_sheet}' must contain 'sol_id' column.")
    if score_col not in df_sum.columns:
        # still ok, but then top_k by score_col can't work
        if top_k is not None:
            raise ValueError(f"top_k requested but '{summary_sheet}' has no '{score_col}' column.")

    # choose sol_ids
    sol_ids = df_sum["sol_id"].astype(int).tolist()

    if only_solutions is not None:
        only = set(int(x) for x in only_solutions)
        sol_ids = [sid for sid in sol_ids if sid in only]

    if top_k is not None:
        df_rank = df_sum.sort_values(score_col, ascending=True)
        sol_ids = df_rank["sol_id"].astype(int).head(int(top_k)).tolist()

    if not sol_ids:
        raise ValueError("No solutions selected to plot.")

    # load trajectories once
    traj_p = pd.read_excel(xls, sheet_name=traj_protein_sheet)
    traj_r = pd.read_excel(xls, sheet_name=traj_rna_sheet)

    # sanity required columns
    for name, df in [("traj_protein", traj_p), ("traj_rna", traj_r)]:
        need = {"sol_id", "protein", "time", "pred_fc"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{name}' missing columns: {sorted(missing)}")

    # ensure types
    traj_p["sol_id"] = pd.to_numeric(traj_p["sol_id"], errors="coerce").astype("Int64")
    traj_r["sol_id"] = pd.to_numeric(traj_r["sol_id"], errors="coerce").astype("Int64")
    traj_p["time"] = pd.to_numeric(traj_p["time"], errors="coerce")
    traj_r["time"] = pd.to_numeric(traj_r["time"], errors="coerce")

    # observed must have protein,time,fc
    for df_name, df in [("df_prot_obs_all", df_prot_obs_all), ("df_rna_obs_all", df_rna_obs_all)]:
        for col in ["protein", "time", "fc"]:
            if col not in df.columns:
                raise ValueError(f"{df_name} must have columns: protein,time,fc. Missing: {col}")

    # loop solutions
    for sid in sol_ids:
        sid = int(sid)

        sub_p = traj_p[traj_p["sol_id"] == sid].copy()
        sub_r = traj_r[traj_r["sol_id"] == sid].copy()

        if sub_p.empty and sub_r.empty:
            continue

        # build pred dfs in the format your plot_goodness_of_fit expects:
        # df_*_pred must have columns: protein,time,pred_fc
        df_prot_pred = sub_p[["protein", "time", "pred_fc"]].copy()
        df_rna_pred = sub_r[["protein", "time", "pred_fc"]].copy()

        # observed: subset to the times/proteins present in preds (keeps merge tight)
        if not df_prot_pred.empty:
            keys_p = df_prot_pred[["protein", "time"]].drop_duplicates()
            df_prot_obs = df_prot_obs_all.merge(keys_p, on=["protein", "time"], how="inner")
        else:
            df_prot_obs = df_prot_obs_all.iloc[0:0].copy()

        if not df_rna_pred.empty:
            keys_r = df_rna_pred[["protein", "time"]].drop_duplicates()
            df_rna_obs = df_rna_obs_all.merge(keys_r, on=["protein", "time"], how="inner")
        else:
            df_rna_obs = df_rna_obs_all.iloc[0:0].copy()

        sol_dir = output_dir
        if make_subdirs:
            sol_dir = os.path.join(output_dir, f"sol_{sid:05d}")
            os.makedirs(sol_dir, exist_ok=True)

        plot_goodness_of_fit_func(
            df_prot_obs=df_prot_obs,
            df_prot_pred=df_prot_pred,
            df_rna_obs=df_rna_obs,
            df_rna_pred=df_rna_pred,
            output_dir=sol_dir
        )

    print(f"[Output] GoF plots generated for {len(sol_ids)} solutions into: {output_dir}")

def export_results(sys, idx, df_prot_obs, df_rna_obs, df_pred_p, df_pred_r, output_dir):
    """
    Exports results using PRE-CALCULATED prediction dataframes.
    Avoids re-running the simulation.
    """
    print("[Output] Exporting Trajectories...")

    # 1. Merge Protein
    # Rename cols to ensure clean merge
    obs_p = df_prot_obs.rename(columns={"fc": "fc_obs"})
    pred_p = df_pred_p.rename(columns={"pred_fc": "fc_pred"})
    merged_p = obs_p.merge(pred_p, on=["protein", "time"], how="outer")
    merged_p["type"] = "Protein"

    # 2. Merge RNA
    obs_r = df_rna_obs.rename(columns={"fc": "fc_obs"})
    pred_r = df_pred_r.rename(columns={"pred_fc": "fc_pred"})
    merged_r = obs_r.merge(pred_r, on=["protein", "time"], how="outer")
    merged_r["type"] = "RNA"

    # 3. Combine & Save
    full_traj = pd.concat([merged_p, merged_r], ignore_index=True)
    full_traj = full_traj[["type", "protein", "time", "fc_obs", "fc_pred"]]
    full_traj.sort_values(["type", "protein", "time"], inplace=True)

    full_traj.to_csv(os.path.join(output_dir, "model_trajectories.csv"), index=False)

    # --- Export Parameters ---
    print("[Output] Exporting Parameters...")

    df_params = pd.DataFrame({
        "Protein_Gene": idx.proteins,
        "Synthesis_A": sys.A_i,
        "mRNA_Degradation_B": sys.B_i,
        "Translation_C": sys.C_i,
        "Protein_Degradation_D": sys.D_i
    })
    df_params["Global_TF_Scale"] = sys.tf_scale
    df_params.to_csv(os.path.join(output_dir, "model_parameters_genes.csv"), index=False)

    df_kin_params = pd.DataFrame({
        "Kinase": idx.kinases,
        "Activity_Scale_ck": sys.c_k
    })
    df_kin_params.to_csv(os.path.join(output_dir, "model_parameters_kinases.csv"), index=False)

    print(f"[Output] Exports saved to {output_dir}")

def build_random_transitions(idx):
    """
    Precompute random phosphorylation transitions for all proteins.

    Returns arrays for Numba:
      trans_from, trans_to, trans_site  (flattened)
      trans_off[i], trans_n[i] per protein i

    Interpretation:
      for protein i:
        transitions are in slice [trans_off[i] : trans_off[i]+trans_n[i]]
        each transition uses site index 'j' (0..ns-1) to pick rate S_all[s_start + j]
    """
    trans_from = []
    trans_to = []
    trans_site = []
    trans_off = np.zeros(idx.N, dtype=np.int32)
    trans_n = np.zeros(idx.N, dtype=np.int32)

    cur = 0
    for i in range(idx.N):
        ns = int(idx.n_sites[i])
        trans_off[i] = cur

        if ns == 0:
            trans_n[i] = 0
            continue

        nstates = 1 << ns
        for m in range(nstates):
            for j in range(ns):
                if (m & (1 << j)) == 0:
                    mp = m | (1 << j)
                    trans_from.append(m)
                    trans_to.append(mp)
                    trans_site.append(j)

        n_i = len(trans_from) - cur
        trans_n[i] = n_i
        cur += n_i

    return (
        np.asarray(trans_from, dtype=np.int32),
        np.asarray(trans_to, dtype=np.int32),
        np.asarray(trans_site, dtype=np.int32),
        trans_off,
        trans_n,
    )


@njit(cache=True)
def time_bucket(t, grid):
    # stepwise hold bucket index j
    if t <= grid[0]:
        return 0
    if t >= grid[-1]:
        return grid.size - 1
    j = np.searchsorted(grid, t, side="right") - 1
    if j < 0:
        j = 0
    if j >= grid.size:
        j = grid.size - 1
    return j

# ------------------------------
# 1. Numba Optimized Kernel (RHS hot loop)
# ------------------------------
@njit(fastmath=True, cache=True)
def fast_rhs_loop(
    y, dy,
    A_i, B_i, C_i, D_i, E_i, tf_scale,
    TF_inputs, S_cache, jb,
    offset_y, offset_s,
    n_sites, n_states,
    trans_from, trans_to, trans_site, trans_off, trans_n
):
    N = len(A_i)

    for i in range(N):
        y_start = offset_y[i]
        s_start = offset_s[i]
        ns = n_sites[i]

        # indices
        idx_R = y_start
        idx_P0 = y_start + 1  # mask=0 starts here

        R = y[idx_R]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]
        Ei = E_i[i]

        u = TF_inputs[i]
        if u < -1.0: u = -1.0
        if u > 1.0:  u = 1.0
        synth = Ai * (1.0 + tf_scale * u)

        # RNA
        dy[idx_R] = synth - Bi * R

        # No sites: reduces to simple protein production/decay
        if ns == 0:
            P0 = y[idx_P0]
            dy[idx_P0] = Ci * R - Di * P0
            continue

        nstates = n_states[i]

        # Translation goes into unphosphorylated state (mask 0)
        # decay applies to ALL protein states
        dy[idx_P0] += Ci * R

        for m in range(nstates):
            Pm = y[idx_P0 + m]
            dy[idx_P0 + m] += -Di * Pm

        # PHOSPHORYLATION forward transitions: m -> m|bit(j)
        off = trans_off[i]
        ntr = trans_n[i]
        for k in range(ntr):
            frm = trans_from[off + k]
            to  = trans_to[off + k]
            j   = trans_site[off + k]

            rate = S_cache[s_start + j, jb]
            flux = rate * y[idx_P0 + frm]

            dy[idx_P0 + frm] -= flux
            dy[idx_P0 + to]  += flux

        # DEPHOSPHORYLATION (minimal model):
        # remove one phosphate at a time with rate Ei per phosphorylated site present.
        # This preserves “Ei” meaning without inventing per-site Ei.
        for m in range(1, nstates):
            Pm = y[idx_P0 + m]
            if Pm <= 0.0:
                continue
            for j in range(ns):
                if (m & (1 << j)) != 0:
                    to = m & ~(1 << j)
                    flux = Ei * Pm
                    dy[idx_P0 + m]  -= flux
                    dy[idx_P0 + to] += flux

# ------------------------------
# ODEINT + NJIT RHS + NJIT FD Jacobian (drop-in)
# ------------------------------
@njit(cache=True, fastmath=True)
def csr_matvec(indptr, indices, data, x, n_rows):
    out = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        s = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            s += data[p] * x[indices[p]]
        out[i] = s
    return out


@njit(cache=True, fastmath=True)
def kin_eval_step(t, grid, Kmat):
    """
    Stepwise hold (no interpolation), matching your KinaseInput.eval behavior.
    Returns vector K(t) of size n_kinases.
    """
    if t <= grid[0]:
        return Kmat[:, 0].copy()
    if t >= grid[-1]:
        return Kmat[:, -1].copy()

    j = np.searchsorted(grid, t, side="right") - 1
    if j < 0:
        j = 0
    if j >= grid.size:
        j = grid.size - 1
    return Kmat[:, j].copy()


@njit(cache=True, fastmath=True)
def rhs_nb(
    y,
    t,
    # params
    c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
    # kinase grid for bucketing
    kin_grid,
    # PRECOMPUTED phosphorylation rates: shape (total_sites, len(kin_grid))
    S_cache,
    # TF CSR (rows = n_proteins, cols = n_proteins)
    TF_indptr, TF_indices, TF_data, n_TF_rows,
    offset_y, offset_s, n_sites, n_states,
    trans_from, trans_to, trans_site, trans_off, trans_n,
    tf_deg,
):
    dy = np.zeros_like(y)

    # pick bucket (stepwise hold)
    jb = time_bucket(t, kin_grid)

    # build protein totals (P_vec)
    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        y_start = offset_y[i]
        nst = n_states[i]
        p0 = y_start + 1
        tot = 0.0
        for m in range(nst):
            tot += y[p0 + m]
        P_vec[i] = tot

    # TF inputs
    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)
    for i in range(n_TF_rows):
        TF_inputs[i] /= tf_deg[i]

    # core dynamics using cached phosphorylation rates
    fast_rhs_loop(
        y, dy,
        A_i, B_i, C_i, D_i, E_i, tf_scale,
        TF_inputs,
        S_cache, jb,
        offset_y, offset_s,
        n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n
    )
    return dy


def rhs_odeint(y, t, *args):
    # odeint expects f(y, t, ...)
    return rhs_nb(y, t, *args)


def fd_jacobian_odeint(y, t, *args):
    # odeint expects J(y, t, ...) with J[i, j] = df_i/dy_j
    y_arr = np.asarray(y, dtype=np.float64)
    return fd_jacobian_nb_core(y_arr, t, *args)


@njit(cache=True, fastmath=True)
def fd_jacobian_nb_core(
        y,
        t,
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg,
        eps=1e-8
):
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb(
        y, t,
        c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
        kin_grid, S_cache,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        offset_y, offset_s, n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n,
        tf_deg
    )

    for j in range(n):
        y_pert = y.copy()
        h = eps * (1.0 if abs(y[j]) < 1.0 else abs(y[j]))
        y_pert[j] += h

        fj = rhs_nb(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, E_i, tf_scale,
            kin_grid, S_cache,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            offset_y, offset_s, n_sites, n_states,
            trans_from, trans_to, trans_site, trans_off, trans_n,
            tf_deg
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J

# ------------------------------
# 2. Parallel W Builder
# ------------------------------
def _build_single_W(args):
    """
    Build W block for one protein for SEQUENTIAL phosphorylation.

    rows = stages j = 0..ns-1 (transition P_j -> P_{j+1})
    stage j corresponds to the psite sites_i[j]
    cols = kinases
    data = 1.0 (unweighted edges), like your current builder
    """
    p, interactions, sites_i, k2i, n_kinases = args

    # IMPORTANT: sequential needs a deterministic site order
    # If your idx.sites[i] is not stable, enforce sorting here.
    # If you already enforce ordering upstream, you can remove this.
    sites_i = list(sites_i)
    # sites_i.sort()  # optional: uncomment if you want alphabetical order

    site_to_stage = {s: j for j, s in enumerate(sites_i)}

    sub = interactions[interactions["protein"] == p]

    rows, cols = [], []
    for _, r in sub.iterrows():
        s = r["psite"]
        k = r["kinase"]
        if s in site_to_stage and k in k2i:
            rows.append(site_to_stage[s])      # stage index
            cols.append(k2i[k])                # kinase index

    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sites_i), n_kinases))

def build_W_parallel(interactions: pd.DataFrame, idx, n_cores=4) -> sparse.csr_matrix:
    """
    Build global W for sequential phosphorylation.

    Global W shape:
      (sum_i n_sites[i]) x (n_kinases)

    For each protein i, its block has n_sites[i] rows:
      row j -> rate k_j for transition P_j -> P_{j+1}
    """
    print(f"[Model] Building SEQUENTIAL W matrices in parallel using {n_cores} cores...")

    tasks = [
        (p, interactions, idx.sites[i], idx.k2i, len(idx.kinases))
        for i, p in enumerate(idx.proteins)
    ]

    if n_cores <= 1:
        W_list = list(map(_build_single_W, tasks))
    else:
        with mp.Pool(n_cores) as pool:
            W_list = pool.map(_build_single_W, tasks)

    print("[Model] Stacking Global SEQUENTIAL W matrix...")
    return sparse.vstack(W_list).tocsr()


# ------------------------------
# 3. Data Loading (Robust)
# ------------------------------
def _normcols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def load_data(args):
    # 1) Kinase net
    print(f"[Data] Loading Kinase Net: {args.kinase_net}")
    df_kin = pd.read_csv(args.kinase_net)
    df_kin = _normcols(df_kin)

    rows = []
    pcol = _find_col(df_kin, ["geneid", "protein"])
    scol = _find_col(df_kin, ["psite", "site"])
    kcol = _find_col(df_kin, ["kinase", "k"])

    for _, r in df_kin.iterrows():
        ks = str(r[kcol]).strip('{}').split(',')
        for k in ks:
            k = k.strip()
            if k:
                rows.append((str(r[pcol]).strip(), str(r[scol]).strip(), k))
    df_kin_clean = pd.DataFrame(rows, columns=["protein", "psite", "kinase"]).drop_duplicates()

    # 2) TF net
    print(f"[Data] Loading TF Net: {args.tf_net}")
    df_tf = pd.read_csv(args.tf_net)
    df_tf = _normcols(df_tf)

    scol = _find_col(df_tf, ["source", "tf"])
    tcol = _find_col(df_tf, ["target", "gene"])
    df_tf_clean = pd.DataFrame({
        "tf": df_tf[scol].astype(str).str.strip(),
        "target": df_tf[tcol].astype(str).str.strip()
    }).drop_duplicates()

    # 3) MS
    print(f"[Data] Loading MS: {args.ms}")
    df_ms = pd.read_csv(args.ms)
    df_ms = _normcols(df_ms)

    gcol = _find_col(df_ms, ["geneid", "protein"])
    scol = _find_col(df_ms, ["psite", "site"])

    if scol and scol in df_ms.columns:
        df_ms[scol] = df_ms[scol].fillna("").astype(str).str.strip().replace({"nan": "", "NaN": ""})
    else:
        scol = "psite"
        df_ms[scol] = ""

    xcols = sorted([c for c in df_ms.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))
    if len(xcols) >= 2:
        tidy = df_ms[[gcol, scol] + xcols].melt(id_vars=[gcol, scol], var_name="xcol", value_name="fc")
        tidy["protein"] = tidy[gcol].astype(str).str.strip()
        tidy["psite"] = tidy[scol]
        x_idx = tidy["xcol"].str.replace("x", "", regex=False).astype(int) - 1
        tidy["time"] = TIME_POINTS[x_idx.to_numpy()]
    else:
        tcol = _find_col(df_ms, ["time", "t"])
        mcol = _find_col(df_ms, ["mean", "fc"])
        tidy = df_ms[[gcol, scol, tcol, mcol]].copy()
        tidy.columns = ["protein", "psite", "time", "fc"]

    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    tidy = tidy.dropna(subset=["fc", "time"])
    is_prot = tidy["psite"].str.len().eq(0)
    df_prot = tidy.loc[is_prot, ["protein", "time", "fc"]].reset_index(drop=True)

    # 4) RNA
    print(f"[Data] Loading RNA: {args.rna}")
    df_rna_raw = pd.read_csv(args.rna)
    df_rna_raw = _normcols(df_rna_raw)

    gcol = _find_col(df_rna_raw, ["geneid", "mrna"])
    xcols = sorted([c for c in df_rna_raw.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))
    if not xcols:
        xcols = sorted([c for c in df_rna_raw.columns if str(c).startswith("x")],
                       key=lambda x: int(re.findall(r"\d+", x)[0]))

    tidy_r = df_rna_raw[[gcol] + xcols].melt(id_vars=[gcol], var_name="xcol", value_name="fc")
    t_map = {c: TIME_POINTS_RNA[i] for i, c in enumerate(xcols) if i < len(TIME_POINTS_RNA)}
    tidy_r["time"] = tidy_r["xcol"].map(t_map)
    tidy_r["protein"] = tidy_r[gcol].astype(str).str.strip()
    tidy_r["fc"] = pd.to_numeric(tidy_r["fc"], errors="coerce")
    df_rna = tidy_r.dropna(subset=["fc", "time"])[["protein", "time", "fc"]].reset_index(drop=True)

    return df_kin_clean, df_tf_clean, df_prot, df_rna


# ------------------------------
# 4. Model Structure
# ------------------------------
class Index:
    def __init__(self, interactions: pd.DataFrame):
        self.proteins = sorted(interactions["protein"].unique().tolist())
        self.p2i = {p: i for i, p in enumerate(self.proteins)}

        self.sites = [
            interactions.loc[interactions["protein"] == p, "psite"]
            .dropna().astype(str).unique().tolist()
            for p in self.proteins
        ]

        self.kinases = sorted(interactions["kinase"].unique().tolist())
        self.k2i = {k: i for i, k in enumerate(self.kinases)}
        self.N = len(self.proteins)

        self.n_sites  = np.array([len(s) for s in self.sites], dtype=np.int32)
        self.n_states = np.array([1 << int(ns) for ns in self.n_sites], dtype=np.int32)

        self.offset_y = np.zeros(self.N, dtype=np.int32)
        self.offset_s = np.zeros(self.N, dtype=np.int32)

        cy = 0
        cs = 0
        for i in range(self.N):
            self.offset_y[i] = cy
            self.offset_s[i] = cs
            cy += 1 + self.n_states[i]   # R + 2^ns
            cs += self.n_sites[i]        # site rates

        self.state_dim = cy
        print(f"[Model] RANDOM PHOS: {self.N} proteins, state_dim={self.state_dim}")

    def block(self, i: int) -> slice:
        start = self.offset_y[i]
        end = start + 1 + self.n_states[i]
        return slice(start, end)

def build_tf_matrix(tf_net, idx):
    rows, cols = [], []
    for _, r in tf_net.iterrows():
        if r["tf"] in idx.p2i and r["target"] in idx.p2i:
            rows.append(idx.p2i[r["target"]])
            cols.append(idx.p2i[r["tf"]])
    data = np.ones(len(rows), float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(idx.N, idx.N))

class KinaseInput:
    def __init__(self, kinases, df_fc):
        self.grid = TIME_POINTS
        self.Kmat = np.ones((len(kinases), len(self.grid)), float)
        if not df_fc.empty:
            for i, k in enumerate(kinases):
                sub = df_fc[df_fc["protein"] == k]
                if not sub.empty:
                    mp_fc = dict(zip(sub["time"], sub["fc"]))
                    for j, t in enumerate(self.grid):
                        if t in mp_fc:
                            self.Kmat[i, j] = max(mp_fc[t], 1e-6)

    def eval(self, t):
        if t <= self.grid[0]:
            return self.Kmat[:, 0]
        if t >= self.grid[-1]:
            return self.Kmat[:, -1]
        j = int(np.searchsorted(self.grid, t, side="right") - 1)
        return self.Kmat[:, j]

class System:
    def __init__(self, idx, W_global, tf_mat, kin_input, defaults, tf_deg):
        self.idx = idx
        self.W_global = W_global
        self.tf_mat = tf_mat
        self.kin = kin_input

        self.c_k = defaults["c_k"]
        self.A_i = defaults["A_i"]
        self.B_i = defaults["B_i"]
        self.C_i = defaults["C_i"]
        self.D_i = defaults["D_i"]
        self.E_i = defaults["E_i"]
        self.tf_scale = defaults["tf_scale"]
        # --- CSR buffers for njit RHS ---
        W = self.W_global.tocsr()
        self.W_indptr = W.indptr.astype(np.int32)
        self.W_indices = W.indices.astype(np.int32)
        self.W_data = W.data.astype(np.float64)
        self.n_W_rows = W.shape[0]

        TF = self.tf_mat.tocsr()
        self.TF_indptr = TF.indptr.astype(np.int32)
        self.TF_indices = TF.indices.astype(np.int32)
        self.TF_data = TF.data.astype(np.float64)
        self.n_TF_rows = TF.shape[0]   # == idx.N

        # Kinase input arrays
        self.kin_grid = np.asarray(self.kin.grid, dtype=np.float64)
        self.kin_Kmat = np.asarray(self.kin.Kmat, dtype=np.float64)

        # Degree of target TFs
        self.tf_deg = tf_deg

        # Random transitions
        (self.trans_from,
         self.trans_to,
         self.trans_site,
         self.trans_off,
         self.trans_n) = build_random_transitions(idx)

    def update(self, c_k, A_i, B_i, C_i, D_i, E_i, tf_scale):
        self.c_k = c_k
        self.A_i = A_i
        self.B_i = B_i
        self.C_i = C_i
        self.D_i = D_i
        self.E_i = E_i
        self.tf_scale = tf_scale

    def rhs(self, t, y):
        dy = np.zeros_like(y)

        Kt = self.kin.eval(t) * self.c_k
        S_all = self.W_global.dot(Kt)

        P_vec = np.zeros(self.idx.N, dtype=np.float64)
        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            nstates = int(self.idx.n_states[i])
            tot = 0.0
            for m in range(nstates):
                tot += y[st + 1 + m]
            P_vec[i] = tot

        TF_inputs = self.tf_mat.dot(P_vec)
        TF_inputs = TF_inputs / self.tf_deg

        fast_rhs_loop(
            y, dy,
            self.A_i, self.B_i, self.C_i, self.D_i, self.E_i, self.tf_scale,
            TF_inputs, S_all,
            self.idx.offset_y, self.idx.offset_s, self.idx.n_sites,
            self.trans_from, self.trans_to, self.trans_site, self.trans_off, self.trans_n
        )
        return dy

    def y0(self):
        y = np.zeros(self.idx.state_dim, float)
        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            y[st] = 1.0  # R
            y[st + 1 + 0] = 1.0  # Pmask0 (unphosphorylated)
            nstates = int(self.idx.n_states[i])
            if nstates > 1:
                y[st + 1 + 1: st + 1 + nstates] = 0.01
        return y

    def odeint_args(self, S_cache):
        """
        Returns args tuple matching rhs_nb / fd_jacobian_nb_core signature.
        """
        return (
            self.c_k.astype(np.float64),
            self.A_i.astype(np.float64),
            self.B_i.astype(np.float64),
            self.C_i.astype(np.float64),
            self.D_i.astype(np.float64),
            self.E_i.astype(np.float64),
            float(self.tf_scale),

            self.kin_grid,
            np.asarray(S_cache, dtype=np.float64),

            self.TF_indptr, self.TF_indices, self.TF_data, int(self.n_TF_rows),
            self.idx.offset_y.astype(np.int32),
            self.idx.offset_s.astype(np.int32),
            self.idx.n_sites.astype(np.int32),
            self.idx.n_states.astype(np.int32),

            self.trans_from,
            self.trans_to,
            self.trans_site,
            self.trans_off,
            self.trans_n,
            self.tf_deg,
        )

# ------------------------------
# 5. Param packing/unpacking (raw -> physical via softplus)
# ------------------------------
def init_raw_params(defaults):
    vecs = []
    slices = {}
    bounds = []
    curr = 0

    for k in ["c_k", "A_i", "B_i", "C_i", "D_i", "E_i"]:
        raw = inv_softplus(defaults[k])
        vecs.append(raw)
        length = len(raw)
        slices[k] = slice(curr, curr + length)
        curr += length

        phys_min, phys_max = BOUNDS_CONFIG[k]
        raw_min = inv_softplus(np.array([phys_min]))[0]
        raw_max = inv_softplus(np.array([phys_max]))[0]
        bounds.extend([(raw_min, raw_max)] * length)

    raw_tf = inv_softplus(np.array([defaults["tf_scale"]]))
    vecs.append(raw_tf)
    slices["tf_scale"] = slice(curr, curr + 1)

    phys_min, phys_max = BOUNDS_CONFIG["tf_scale"]
    raw_min = inv_softplus(np.array([phys_min]))[0]
    raw_max = inv_softplus(np.array([phys_max]))[0]
    bounds.append((raw_min, raw_max))

    theta0 = np.concatenate(vecs)
    xl = np.array([b[0] for b in bounds], dtype=float)
    xu = np.array([b[1] for b in bounds], dtype=float)
    return theta0, slices, xl, xu

def unpack_params(theta, slices):
    return {
        "c_k": softplus(theta[slices["c_k"]]),
        "A_i": softplus(theta[slices["A_i"]]),
        "B_i": softplus(theta[slices["B_i"]]),
        "C_i": softplus(theta[slices["C_i"]]),
        "D_i": softplus(theta[slices["D_i"]]),
        "E_i": softplus(theta[slices["E_i"]]),
        "tf_scale": softplus(theta[slices["tf_scale"]])[0]
    }


# ------------------------------
# 6. Fast loss data (Numba)
# ------------------------------
def prepare_fast_loss_data(idx, df_prot, df_rna, time_grid):
    t_map = {t: i for i, t in enumerate(time_grid)}

    def get_indices(df, p2i_map):
        p_idxs = np.array([p2i_map[p] for p in df["protein"].values], dtype=np.int32)
        t_idxs = np.array([t_map[t] for t in df["time"].values], dtype=np.int32)
        obs = df["fc"].values.astype(np.float64)
        ws = df["w"].values.astype(np.float64)
        return p_idxs, t_idxs, obs, ws

    p_prot, t_prot, obs_prot, w_prot = get_indices(df_prot, idx.p2i)
    p_rna, t_rna, obs_rna, w_rna = get_indices(df_rna, idx.p2i)

    prot_map = np.zeros((idx.N, 2), dtype=np.int32)
    for i in range(idx.N):
        sl = idx.block(i)
        prot_map[i, 0] = sl.start
        prot_map[i, 1] = idx.n_states[i]

    return {
        "p_prot": p_prot, "t_prot": t_prot, "obs_prot": obs_prot, "w_prot": w_prot,
        "p_rna": p_rna, "t_rna": t_rna, "obs_rna": obs_rna, "w_rna": w_rna,
        "prot_map": prot_map,
        "n_p": max(1, len(obs_prot)),
        "n_r": max(1, len(obs_rna))
    }

@njit(fastmath=True, cache=True, nogil=True)
def jit_loss_core(Y, p_prot, t_prot, obs_prot, w_prot,
                  p_rna, t_rna, obs_rna, w_rna,
                  prot_map, rna_base_idx):
    loss_p = 0.0
    for k in range(len(p_prot)):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1

        tot_t = 0.0
        tot_0 = 0.0
        for m in range(nstates):
            tot_t += Y[t_idx, p0 + m]
            tot_0 += Y[0, p0 + m]
        pred_fc = max(tot_t, 1e-9) / max(tot_0, 1e-9)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * (diff * diff)

    loss_r = 0.0
    for k in range(len(p_rna)):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]
        pred_fc = max(R_t, 1e-9) / max(R_b, 1e-9)

        diff = obs_rna[k] - pred_fc
        loss_r += w_rna[k] * (diff * diff)

    return loss_p, loss_r


# ------------------------------
# 7. Pymoo Elementwise Problem
# ------------------------------
class GlobalODE_MOO(ElementwiseProblem):
    """
    Elementwise multiobjective:
      F = [prot_mse, rna_mse, reg_loss]
    """
    def __init__(self, sys, slices, loss_data, defaults, lambdas, time_grid,
                 xl, xu, fail_value=1e12, elementwise_runner=None):
        super().__init__(
            n_var=len(xl),
            n_obj=3,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            elementwise_runner=elementwise_runner
        )
        self.sys = sys
        self.slices = slices
        self.loss_data = loss_data
        self.defaults = defaults
        self.lambdas = lambdas
        self.time_grid = time_grid
        self.fail_value = float(fail_value)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1) unpack + update
        p = unpack_params(x, self.slices)
        self.sys.update(**p)

        # 2) reg term (same as before, but keep as separate objective)
        reg = 0.0
        count = 0
        for k in ["A_i", "B_i", "C_i", "D_i", "E_i"]:
            diff = (p[k] - self.defaults[k]) / (self.defaults[k] + 1e-6)
            reg += float(np.sum(diff * diff))
            count += diff.size
        reg_loss = self.lambdas["prior"] * (reg / max(1, count))

        # 3) simulate (odeint + njit RHS + njit Jacobian)
        try:
            Y = simulate_odeint(
                self.sys,
                self.time_grid,
                rtol=1e-5,
                atol=1e-6,
                mxstep=5000
            )
        except Exception:
            out["F"] = np.array([self.fail_value, self.fail_value, self.fail_value], dtype=float)
            return

        if Y is None or Y.size == 0 or not np.all(np.isfinite(Y)):
            out["F"] = np.array([self.fail_value, self.fail_value, self.fail_value], dtype=float)
            return

        # Ensure (T, state_dim) contiguous
        Y = np.ascontiguousarray(Y)
        loss_p_sum, loss_r_sum = jit_loss_core(
            Y,
            self.loss_data["p_prot"], self.loss_data["t_prot"], self.loss_data["obs_prot"], self.loss_data["w_prot"],
            self.loss_data["p_rna"], self.loss_data["t_rna"], self.loss_data["obs_rna"], self.loss_data["w_rna"],
            self.loss_data["prot_map"], self.loss_data["rna_base_idx"]
        )

        prot_mse = loss_p_sum / self.loss_data["n_p"]
        rna_mse = loss_r_sum / self.loss_data["n_r"]

        out["F"] = np.array([prot_mse, rna_mse, reg_loss], dtype=float)


# ------------------------------
# 8. Post-processing: pick a single solution and export
# ------------------------------
def simulate_odeint(sys, t_eval, rtol=1e-5, atol=1e-6, mxstep=5000):
    """
    Returns Y with shape (T, state_dim), matching your usage (like sol.y.T).
    """
    y0 = sys.y0().astype(np.float64)
    # Kt_mat shape: (n_kinases, n_grid)
    Kt_mat = sys.kin_Kmat * sys.c_k[:, None]
    # S_cache shape: (total_sites, n_grid)
    S_cache = sys.W_global.dot(Kt_mat)
    S_cache = np.asarray(S_cache, dtype=np.float64)  # force ndarray
    S_cache = np.ascontiguousarray(S_cache)
    args = sys.odeint_args(S_cache)
    xs = odeint(
        rhs_odeint,
        y0,
        t_eval.astype(np.float64),
        args=args,
        Dfun=fd_jacobian_odeint,
        col_deriv=False,
        # rtol=rtol,
        # atol=atol,
        # mxstep=mxstep,
    )
    return np.ascontiguousarray(xs, dtype=np.float64)

def simulate_and_measure(sys, idx, t_points_p, t_points_r):
    times = np.unique(np.concatenate([t_points_p, t_points_r]))
    Y = simulate_odeint(sys, times, rtol=1e-5, atol=1e-7, mxstep=5000)

    # baseline indices
    prot_b = int(np.where(times == 0.0)[0][0])
    rna_b  = int(np.where(times == 4.0)[0][0])   # IMPORTANT

    rows_p, rows_r = [], []
    for i, p in enumerate(idx.proteins):
        st = idx.offset_y[i]
        nstates = int(idx.n_states[i])
        p0 = st + 1
        tot = 0.0
        for m in range(nstates):
            tot += Y[:, p0 + m]
        fc_p = np.maximum(tot, 1e-9) / np.maximum(tot[prot_b], 1e-9)
        rows_p.append(pd.DataFrame({"protein": p, "time": times, "pred_fc": fc_p}))

        # RNA baseline at t=4
        R = Y[:, st]
        fc_r = np.maximum(R, 1e-9) / np.maximum(R[rna_b], 1e-9)
        rows_r.append(pd.DataFrame({"protein": p, "time": times, "pred_fc": fc_r}))

    df_p = pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame()
    df_r = pd.concat(rows_r, ignore_index=True) if rows_r else pd.DataFrame()

    df_p = df_p[df_p["time"].isin(t_points_p)]
    df_r = df_r[df_r["time"].isin(t_points_r)]
    return df_p, df_r

# ------------------------------
# 9. Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinase-net", required=True)
    parser.add_argument("--tf-net", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--output-dir", default="out_unsga3")
    parser.add_argument("--cores", type=int, default=os.cpu_count())

    # Pymoo
    parser.add_argument("--n-gen", type=int, default=1000)
    parser.add_argument("--pop", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)

    # Loss weights
    parser.add_argument("--lambda-prior", type=float, default=1e-3)
    parser.add_argument("--lambda-rna", type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load
    df_kin, df_tf, df_prot, df_rna = load_data(args)
    df_prot = normalize_fc_to_t0(df_prot)
    base = df_rna[df_rna["time"] == 4.0].set_index("protein")["fc"]
    df_rna["fc"] = df_rna.apply(lambda r: r["fc"] / base.get(r["protein"], np.nan), axis=1)
    df_rna = df_rna.dropna(subset=["fc"])

    # 2) Model index
    idx = Index(df_kin)

    # Restrict obs to model proteins
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()

    # Weights (early emphasis)
    all_times = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    wmap = {t: 1.0 + (all_times.max() - t) / all_times.max() for t in all_times}
    df_prot["w"] = df_prot["time"].map(wmap).fillna(1.0)
    df_rna["w"] = df_rna["time"].map(wmap).fillna(1.0)

    # 3) Build W + TF
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf, idx)
    kin_in = KinaseInput(idx.kinases, df_prot)

    tf_deg = np.asarray(tf_mat.sum(axis=1)).ravel().astype(np.float64)  # row sums = in-degree per target
    tf_deg[tf_deg == 0.0] = 1.0

    # 4) Defaults/system
    defaults = {
        "c_k": np.ones(len(idx.kinases)),
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1
    }
    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    # 5) Precompute loss data on solver time grid
    solver_times = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    rna_base_time = 4.0
    rna_base_idx = int(np.where(solver_times == rna_base_time)[0][0])

    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, solver_times)
    loss_data["rna_base_idx"] = np.int32(rna_base_idx)

    # 6) Decision vector bounds (raw space)
    theta0, slices, xl, xu = init_raw_params(defaults)
    lambdas = {"rna": args.lambda_rna, "prior": args.lambda_prior}

    # 7) Pymoo parallel runner (optional)
    runner = None
    pool = None
    if args.cores > 1:
        pool = mp.Pool(args.cores)
        runner = StarmapParallelization(pool.starmap)
        print(f"[Pymoo] Parallel evaluation enabled with {args.cores} workers.")
    else:
        print("[Pymoo] Parallel evaluation disabled (or unavailable).")

    # 8) Problem
    problem = GlobalODE_MOO(
        sys=sys,
        slices=slices,
        loss_data=loss_data,
        defaults=defaults,
        lambdas=lambdas,
        time_grid=solver_times,
        xl=xl,
        xu=xu,
        elementwise_runner=runner
    )

    # 9) UNSGA3 needs reference directions for n_obj=3
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    algorithm = UNSGA3(pop_size=args.pop, ref_dirs=ref_dirs)

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=args.n_gen,
        n_max_evals=100000
    )

    print(f"[Fit] UNSGA3: pop={args.pop}, n_gen={args.n_gen}, n_var={problem.n_var}, n_obj={problem.n_obj}")
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=args.seed,
        save_history=True,
        verbose=True
    )

    if pool is not None:
        pool.close()
        pool.join()


    # 10) Save Pareto set
    X = res.X
    F = res.F
    np.save(os.path.join(args.output_dir, "pareto_X.npy"), X)
    np.save(os.path.join(args.output_dir, "pareto_F.npy"), F)

    # Also write a CSV summary
    df_pareto = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "reg_loss"])
    df_pareto.to_csv(os.path.join(args.output_dir, "pareto_F.csv"), index=False)
    print(f"[Output] Saved Pareto front: {len(df_pareto)} solutions")

    excel_path = os.path.join(args.output_dir, "pareto_front.xlsx")

    export_pareto_front_to_excel(
        res=res,
        sys=sys,
        idx=idx,
        slices=slices,
        output_path=excel_path,
        weights=(1.0, args.lambda_rna, args.lambda_prior),
        top_k_trajectories=None,
    )

    print(f"[Output] Saved Pareto front Excel: {excel_path}")

    plot_gof_from_pareto_excel(
        excel_path=excel_path,
        output_dir=os.path.join(args.output_dir, "gof_all"),
        plot_goodness_of_fit_func=plot_goodness_of_fit,
        df_prot_obs_all=df_prot,
        df_rna_obs_all=df_rna,
        top_k=None,
        score_col="scalar_score",
    )

    print(f"[Output] Saved Goodness of Fit plots for all Pareto solutions.")

    # 11) Pick one solution
    F = res.F
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    w = np.array([1.0, args.lambda_rna, args.lambda_prior])
    I = np.argmin((Fn * w).sum(axis=1))
    theta_best = X[I].astype(float)
    F_best = F[I]
    params = unpack_params(theta_best, slices)
    sys.update(**params)

    # 12) Export picked solution
    dfp, dfr = simulate_and_measure(sys, idx, TIME_POINTS, TIME_POINTS_RNA)
    # Save raw preds
    if dfp is not None: dfp.to_csv(os.path.join(args.output_dir, "pred_prot_picked.csv"), index=False)
    if dfr is not None: dfr.to_csv(os.path.join(args.output_dir, "pred_rna_picked.csv"), index=False)

    p_out = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in params.items()}
    with open(os.path.join(args.output_dir, "fitted_params_picked.json"), "w") as f:
        json.dump(p_out, f, indent=2)

    # Write picked objective values
    picked = {"prot_mse": float(F[I, 0]), "rna_mse": float(F[I, 1]), "reg_loss": float(F[I, 2]),
              "scalar_score": float(F[I, 0] + args.lambda_rna * F[I, 1] + F[I, 2])}
    with open(os.path.join(args.output_dir, "picked_objectives.json"), "w") as f:
        json.dump(picked, f, indent=2)

    print("[Done] Picked solution:")
    print(json.dumps(picked, indent=2))

    plot_goodness_of_fit(df_prot, dfp, df_rna, dfr, output_dir=args.output_dir)
    print("[Done] Goodness of Fit plot saved.")

    ts_dir = os.path.join(args.output_dir, "timeseries_plots")
    for g in idx.proteins:
        save_gene_timeseries_plots(
            gene=g,
            df_prot_obs=df_prot,
            df_prot_pred=dfp,
            df_rna_obs=df_rna,
            df_rna_pred=dfr,
            output_dir=ts_dir,
            prot_times=TIME_POINTS,
            rna_times=TIME_POINTS_RNA,
            filename_prefix="fit"
        )

    print("[Done] Time series plots saved.")

    # Use the NEW export function (pass dfp, dfr directly)
    if dfp is not None and dfr is not None:
        export_results(sys, idx, df_prot, df_rna, dfp, dfr, args.output_dir)

    print("[Done] Exported results saved.")

    # --- VISUALIZATION BLOCK ---

    # 1. 3D Pareto Front
    #
    save_pareto_3d(res, selected_solution=F_best, output_dir=args.output_dir)
    print("[Done] 3D Pareto plot saved.")

    # 2. Parallel Coordinate Plot
    #
    save_parallel_coordinates(res, selected_solution=F_best, output_dir=args.output_dir)
    print("[Done] Parallel Coordinate plot saved.")

    # 3. Convergence Video
    create_convergence_video(res, output_dir=args.output_dir)
    print("[Done] Convergence video saved.")

if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    main()
