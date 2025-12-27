#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global ODE Dual-Fit (Pymoo / UNSGA3) - Elementwise Multi-Objective Optimization
------------------------------------------------------------------------------

Replaces scipy.optimize.minimize(SLSQP) with pymoo multi-objective optimization:
  Objectives (minimize):
    f1 = Protein MSE
    f2 = RNA MSE
    f3 = Prior regularization loss

Key properties:
- ElementwiseProblem (network problem evaluated per candidate)
- UNSGA3 (a.k.a. "UNSA3" in your message) with 3 objectives
- Numba JIT for RHS hot loop + loss core
- Works with your existing model structure + bounds via raw (inv_softplus) parametrization

Run:
  python global_optim_pymoo_unsga3.py \
    --kinase-net input2.csv \
    --tf-net CollecTRI.csv \
    --ms MS_Gaussian_updated_09032023.csv \
    --rna Rout_LimmaTable.csv \
    --output-dir out_unsga3 \
    --cores 4 \
    --n-gen 200 \
    --pop 120 \
    --seed 1 \
    --lambda-prior 0.001 \
    --lambda-rna 1.0

Notes:
- Multiobjective returns a Pareto front. This script also selects a "picked" solution
  by minimizing: f1 + lambda_rna*f2 + f3 (same scalarization idea as before) and exports it.
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
    "c_k":      (0.0, 1.0),
    "A_i":      (1e-4, 2.0),
    "B_i":      (0.001, 2.0),
    "C_i":      (0.01, 2.0),
    "D_i":      (1e-4, 2.0),
    "tf_scale": (0.0, 1.0)
}

def softplus(x):
    # stable softplus
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def inv_softplus(y):
    y = np.maximum(y, 1e-12)
    return np.log(np.expm1(y))


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
# ------------------------------
# 1. Numba Optimized Kernel (RHS hot loop)
# ------------------------------
@njit(fastmath=True, cache=True, nogil=True)
def fast_rhs_loop(y, dy, A_i, B_i, C_i, D_i, tf_scale, TF_inputs, S_all,
                  offset_y, offset_s, n_sites):
    N = len(A_i)

    for i in range(N):
        y_start = offset_y[i]
        idx_R = y_start
        idx_P = y_start + 1

        s_start = offset_s[i]
        ns = n_sites[i]

        R = y[idx_R]
        P = y[idx_P]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]

        synth = Ai * (1.0 + tf_scale * TF_inputs[i])
        dy[idx_R] = synth - Bi * R

        if ns == 0:
            dy[idx_P] = Ci * R - Di * P
        else:
            sum_S = 0.0
            sum_Ps = 0.0
            for j in range(ns):
                s_rate = S_all[s_start + j]
                ps_val = y[y_start + 2 + j]

                sum_S += s_rate
                sum_Ps += ps_val
                dy[y_start + 2 + j] = s_rate * P - (1.0 + Di) * ps_val

            dy[idx_P] = Ci * R - (Di + sum_S) * P + sum_Ps

# ------------------------------
# ODEINT + NJIT RHS + NJIT FD Jacobian (drop-in)
# ------------------------------
from numba import njit

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
    c_k, A_i, B_i, C_i, D_i, tf_scale,
    # kinase input (stepwise)
    kin_grid, kin_Kmat,
    # W_global CSR (rows = total_sites, cols = n_kinases)
    W_indptr, W_indices, W_data, n_W_rows,
    # TF CSR (rows = n_proteins, cols = n_proteins)
    TF_indptr, TF_indices, TF_data, n_TF_rows,
    # maps
    p_indices,
    offset_y, offset_s, n_sites
):
    dy = np.zeros_like(y)

    # Kinase vector at time t, scaled by c_k
    Kt = kin_eval_step(t, kin_grid, kin_Kmat)
    for i in range(Kt.size):
        Kt[i] *= c_k[i]

    # Site phosphorylation rates S_all = W_global * Kt   (len = total_sites)
    S_all = csr_matvec(W_indptr, W_indices, W_data, Kt, n_W_rows)

    # TF inputs per protein from protein abundance P_vec
    # P_vec is y at the per-protein protein index positions (length = n_proteins)
    P_vec = np.zeros(n_TF_rows, dtype=np.float64)
    for i in range(n_TF_rows):
        P_vec[i] = y[p_indices[i]]

    TF_inputs = csr_matvec(TF_indptr, TF_indices, TF_data, P_vec, n_TF_rows)

    # Core dynamics (your hot loop)
    fast_rhs_loop(
        y, dy,
        A_i, B_i, C_i, D_i, tf_scale,
        TF_inputs, S_all,
        offset_y, offset_s, n_sites
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
    c_k, A_i, B_i, C_i, D_i, tf_scale,
    kin_grid, kin_Kmat,
    W_indptr, W_indices, W_data, n_W_rows,
    TF_indptr, TF_indices, TF_data, n_TF_rows,
    p_indices,
    offset_y, offset_s, n_sites,
    eps=1e-8
):
    """
    Forward-diff Jacobian of rhs_nb wrt y: J[i, j] = d f_i / d y_j
    """
    n = y.size
    J = np.empty((n, n), dtype=np.float64)

    f0 = rhs_nb(
        y, t,
        c_k, A_i, B_i, C_i, D_i, tf_scale,
        kin_grid, kin_Kmat,
        W_indptr, W_indices, W_data, n_W_rows,
        TF_indptr, TF_indices, TF_data, n_TF_rows,
        p_indices,
        offset_y, offset_s, n_sites
    )

    for j in range(n):
        y_pert = y.copy()
        h = eps * (1.0 if abs(y[j]) < 1.0 else abs(y[j]))
        y_pert[j] += h

        fj = rhs_nb(
            y_pert, t,
            c_k, A_i, B_i, C_i, D_i, tf_scale,
            kin_grid, kin_Kmat,
            W_indptr, W_indices, W_data, n_W_rows,
            TF_indptr, TF_indices, TF_data, n_TF_rows,
            p_indices,
            offset_y, offset_s, n_sites
        )

        invh = 1.0 / h
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) * invh

    return J

# ------------------------------
# 2. Parallel W Builder
# ------------------------------
def _build_single_W(args):
    p, interactions, sites_i, k2i, n_kinases = args

    sub = interactions[interactions["protein"] == p]
    site_map = {s: r for r, s in enumerate(sites_i)}
    rows, cols = [], []

    for _, r in sub.iterrows():
        if r["psite"] in site_map and r["kinase"] in k2i:
            rows.append(site_map[r["psite"]])
            cols.append(k2i[r["kinase"]])

    data = np.ones(len(rows), float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sites_i), n_kinases))

def build_W_parallel(interactions: pd.DataFrame, idx, n_cores=4) -> sparse.csr_matrix:
    print(f"[Model] Building W matrices in parallel using {n_cores} cores...")

    tasks = [
        (p, interactions, idx.sites[i], idx.k2i, len(idx.kinases))
        for i, p in enumerate(idx.proteins)
    ]

    if n_cores <= 1:
        W_list = list(map(_build_single_W, tasks))
    else:
        with mp.Pool(n_cores) as pool:
            W_list = pool.map(_build_single_W, tasks)

    print("[Model] Stacking Global W matrix...")
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
        self.sites = [interactions.loc[interactions["protein"] == p, "psite"].unique().tolist() for p in self.proteins]
        self.kinases = sorted(interactions["kinase"].unique().tolist())
        self.k2i = {k: i for i, k in enumerate(self.kinases)}
        self.N = len(self.proteins)

        self.n_sites = np.array([len(s) for s in self.sites], dtype=np.int32)
        self.offset_y = np.zeros(self.N, dtype=np.int32)
        self.offset_s = np.zeros(self.N, dtype=np.int32)

        curr_y = 0
        curr_s = 0
        for i in range(self.N):
            self.offset_y[i] = curr_y
            self.offset_s[i] = curr_s
            curr_y += 2 + self.n_sites[i]
            curr_s += self.n_sites[i]

        self.state_dim = curr_y
        print(f"[Model] {self.N} proteins, {len(self.kinases)} kinases, {self.state_dim} state variables.")

    def block(self, i: int) -> slice:
        start = self.offset_y[i]
        end = start + 2 + self.n_sites[i]
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
    def __init__(self, idx, W_global, tf_mat, kin_input, defaults):
        self.idx = idx
        self.W_global = W_global
        self.tf_mat = tf_mat
        self.kin = kin_input
        self.p_indices = self.idx.offset_y + 1

        self.c_k = defaults["c_k"]
        self.A_i = defaults["A_i"]
        self.B_i = defaults["B_i"]
        self.C_i = defaults["C_i"]
        self.D_i = defaults["D_i"]
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

        # p_indices must be int32 for njit indexing
        self.p_indices = self.p_indices.astype(np.int32)


    def update(self, c_k, A_i, B_i, C_i, D_i, tf_scale):
        self.c_k = c_k
        self.A_i = A_i
        self.B_i = B_i
        self.C_i = C_i
        self.D_i = D_i
        self.tf_scale = tf_scale

    def rhs(self, t, y):
        dy = np.zeros_like(y)

        Kt = self.kin.eval(t) * self.c_k
        S_all = self.W_global.dot(Kt)

        P_vec = y[self.p_indices]
        TF_inputs = self.tf_mat.dot(P_vec)

        fast_rhs_loop(
            y, dy,
            self.A_i, self.B_i, self.C_i, self.D_i, self.tf_scale,
            TF_inputs, S_all,
            self.idx.offset_y, self.idx.offset_s, self.idx.n_sites
        )
        return dy

    def y0(self):
        y = np.zeros(self.idx.state_dim, float)
        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            y[st] = 1.0
            y[st + 1] = 1.0
            ns = self.idx.n_sites[i]
            if ns > 0:
                y[st + 2: st + 2 + ns] = 0.01
        return y

    def odeint_args(self):
        """
        Returns args tuple matching rhs_nb / fd_jacobian_nb_core signature.
        IMPORTANT: order must match rhs_nb(...).
        """
        return (
            self.c_k.astype(np.float64),
            self.A_i.astype(np.float64),
            self.B_i.astype(np.float64),
            self.C_i.astype(np.float64),
            self.D_i.astype(np.float64),
            float(self.tf_scale),

            self.kin_grid,
            self.kin_Kmat,

            self.W_indptr, self.W_indices, self.W_data, int(self.n_W_rows),

            self.TF_indptr, self.TF_indices, self.TF_data, int(self.n_TF_rows),

            self.p_indices,

            self.idx.offset_y.astype(np.int32),
            self.idx.offset_s.astype(np.int32),
            self.idx.n_sites.astype(np.int32),
        )


# ------------------------------
# 5. Param packing/unpacking (raw -> physical via softplus)
# ------------------------------
def init_raw_params(defaults):
    vecs = []
    slices = {}
    bounds = []
    curr = 0

    for k in ["c_k", "A_i", "B_i", "C_i", "D_i"]:
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
        prot_map[i, 1] = idx.n_sites[i]

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
                  prot_map):
    loss_p = 0.0
    for k in range(len(p_prot)):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        P_t = Y[t_idx, start + 1]
        Ps_t = 0.0
        for s in range(n_sites):
            Ps_t += Y[t_idx, start + 2 + s]
        tot_t = P_t + Ps_t

        P_0 = Y[0, start + 1]
        Ps_0 = 0.0
        for s in range(n_sites):
            Ps_0 += Y[0, start + 2 + s]
        tot_0 = P_0 + Ps_0

        pred_fc = max(tot_t, 1e-9) / max(tot_0, 1e-9)
        diff = pred_fc - obs_prot[k]
        loss_p += w_prot[k] * (diff * diff)

    loss_r = 0.0
    for k in range(len(p_rna)):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_0 = Y[0, start]
        pred_fc = max(R_t, 1e-9) / max(R_0, 1e-9)

        diff = pred_fc - obs_rna[k]
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
        for k in ["A_i", "B_i", "C_i", "D_i"]:
            diff = p[k] - self.defaults[k]
            reg += float(np.sum(diff * diff))
        reg_loss = (reg * self.lambdas["prior"]) / max(1, len(x))

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
            self.loss_data["prot_map"]
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
    args = sys.odeint_args()

    xs = odeint(
        rhs_odeint,
        y0,
        t_eval.astype(np.float64),
        args=args,
        Dfun=fd_jacobian_odeint,
        col_deriv=False,
        rtol=rtol,
        atol=atol,
        mxstep=mxstep,
    )
    return np.ascontiguousarray(xs, dtype=np.float64)

def simulate_and_measure(sys, idx, t_points_p, t_points_r):
    times = np.unique(np.concatenate([t_points_p, t_points_r]))
    try:
        Y = simulate_odeint(sys, times, rtol=1e-5, atol=1e-7, mxstep=5000)
    except Exception:
        return None, None

    if Y is None or Y.size == 0 or not np.all(np.isfinite(Y)):
        return None, None

    rows_p = []
    for i, p in enumerate(idx.proteins):
        st = idx.offset_y[i]
        ns = idx.n_sites[i]
        P = Y[:, st + 1]
        Ps = Y[:, st + 2: st + 2 + ns].sum(axis=1) if ns > 0 else 0.0
        tot = P + Ps
        fc = np.maximum(tot, 1e-9) / np.maximum(tot[0], 1e-9)
        rows_p.append(pd.DataFrame({"protein": p, "time": times, "pred_fc": fc}))

    rows_r = []
    for i, p in enumerate(idx.proteins):
        st = idx.offset_y[i]
        R = Y[:, st]
        fc = np.maximum(R, 1e-9) / np.maximum(R[0], 1e-9)
        rows_r.append(pd.DataFrame({"protein": p, "time": times, "pred_fc": fc}))

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
    parser.add_argument("--n-gen", type=int, default=100)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)

    # Loss weights
    parser.add_argument("--lambda-prior", type=float, default=0.001)
    parser.add_argument("--lambda-rna", type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load
    df_kin, df_tf, df_prot, df_rna = load_data(args)

    # 2) Model index
    idx = Index(df_kin)

    # Restrict obs to model proteins
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()

    # Weights (early emphasis)
    wmap = {t: 1.0 + (max(TIME_POINTS) - t) / max(TIME_POINTS) for t in TIME_POINTS}
    df_prot["w"] = df_prot["time"].map(wmap).fillna(1.0)
    df_rna["w"] = 1.0

    # 3) Build W + TF
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf, idx)
    kin_in = KinaseInput(idx.kinases, df_prot)

    # 4) Defaults/system
    defaults = {
        "c_k": np.ones(len(idx.kinases)),
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "tf_scale": 0.1
    }
    sys = System(idx, W_global, tf_mat, kin_in, defaults)

    # 5) Precompute loss data on solver time grid
    solver_times = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, solver_times)

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

    # 11) Pick one solution
    F = res.F
    weights = np.array([1.0, 1.0, 0.2])  # Weights for Prot, RNA, Reg
    # Normalize objectives to 0-1 range so weights are fair
    norm_matrix = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-9)
    # Calculate the single scalar score for every solution
    weighted_scores = np.sum(norm_matrix * weights, axis=1)
    # Find the index of the best score
    best_i = np.argmin(weighted_scores)
    theta_best = X[best_i].astype(float)
    F_best = F[best_i]
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
    picked = {"prot_mse": float(F[best_i, 0]), "rna_mse": float(F[best_i, 1]), "reg_loss": float(F[best_i, 2]),
              "scalar_score": float(F[best_i, 0] + args.lambda_rna * F[best_i, 1] + F[best_i, 2])}
    with open(os.path.join(args.output_dir, "picked_objectives.json"), "w") as f:
        json.dump(picked, f, indent=2)

    print("[Done] Picked solution:")
    print(json.dumps(picked, indent=2))

    plot_goodness_of_fit(df_prot, dfp, df_rna, dfr, output_dir=args.output_dir)
    print("[Done] Goodness of Fit plot saved.")


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
