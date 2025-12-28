import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from scipy.stats import linregress

from phoskintime_global.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO, MODEL
from phoskintime_global.params import unpack_params
from phoskintime_global.simulate import simulate_and_measure
from phoskintime_global.jacspeedup import build_S_cache_into

def build_site_meta(idx):
    """
    Returns parallel arrays of length idx.total_sites:
      site_protein[s] = protein name for global site s
      site_psite[s]   = psite label for global site s
      site_local[s]   = local site index within that protein
    """
    total = int(idx.total_sites)
    site_protein = np.empty(total, dtype=object)
    site_psite   = np.empty(total, dtype=object)
    site_local   = np.empty(total, dtype=np.int32)

    for i, prot in enumerate(idx.proteins):
        off = int(idx.offset_s[i])
        for j, psite in enumerate(idx.sites[i]):
            s = off + j
            site_protein[s] = prot
            site_psite[s]   = psite
            site_local[s]   = j

    return site_protein, site_psite, site_local

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
        labels=["Prot MSE", "RNA MSE", "Phospho MSE"],
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
        labels=["Prot MSE", "RNA MSE", "Phospho MSE"],
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
        ax.set_zlabel("Phospho MSE")
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
        ani.save(save_path, writer='ffmpeg', fps=5, dpi=300)
        print(f"[Output] Video saved: {save_path}")
    except Exception:
        # Fallback to GIF (universally supported, no ffmpeg needed)
        gif_path = save_path.replace(".mp4", ".gif")
        print("[System] FFMPEG not found. Falling back to GIF...")
        ani.save(gif_path, writer='pillow', fps=5, dpi=300)
        print(f"[Output] Video saved: {gif_path}")

    plt.close()


def export_pareto_front_to_excel(
        res,
        sys,
        idx,
        slices,
        output_path,
        weights=(1.0, 1.0, 1.0),  # (w_prot, w_rna, w_phos) used for scalar score + ranking
        top_k_trajectories=None,  # None = export trajectories for all solutions; else only top K by scalar score
        t_points_p=None,
        t_points_r=None,
        t_points_ph=None,
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
      - traj_phospho: trajectories per sol_id (long format)

    Notes:
      - This can get HUGE if you have many Pareto points. Use top_k_trajectories.
      - The function assumes res.X and res.F exist.
    """
    X = np.asarray(res.X)
    F = np.asarray(res.F)

    if t_points_p is None:
        t_points_p = TIME_POINTS_PROTEIN
    if t_points_r is None:
        t_points_r = TIME_POINTS_RNA
    if t_points_ph is None:
        t_points_ph = TIME_POINTS_PHOSPHO

    w_prot, w_rna, w_phos = map(float, weights)

    # ---- Summary + ranking ----
    df_summary = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "phospho_mse"])
    df_summary.insert(0, "sol_id", np.arange(len(df_summary), dtype=int))
    df_summary["w_prot"] = w_prot
    df_summary["w_rna"] = w_rna
    df_summary["w_phos"] = w_phos

    # scalar score for ranking / convenience
    df_summary["scalar_score"] = (
            w_prot * df_summary["prot_mse"]
            + w_rna * df_summary["rna_mse"]
            + w_phos * df_summary["phospho_mse"]
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
    deg_site_rows = []
    traj_p_list = []
    traj_r_list = []
    traj_ph_list = []

    # Cache these for speed
    proteins = idx.proteins
    kinases = idx.kinases

    site_protein, site_psite, site_local = build_site_meta(idx)

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

        # ---- per-site degradation rates ----
        # Prefer params dict if present, else sys attribute set by update().
        deg_vec = (
                p.get("deg_site", None)
                or p.get("kdeg_site", None)
                or getattr(sys, "deg_site", None)
                or getattr(sys, "kdeg_site", None)
        )

        if deg_vec is not None:
            deg_vec = np.asarray(deg_vec, dtype=float).reshape(-1)
            if deg_vec.size != idx.total_sites:
                raise ValueError(f"deg_vec has size {deg_vec.size}, expected idx.total_sites={idx.total_sites}")

            # store long format: one row per site per solution
            for s in range(idx.total_sites):
                deg_site_rows.append((
                    sol_id,
                    int(s),  # global site id
                    site_protein[s],
                    site_psite[s],
                    int(site_local[s]),
                    float(deg_vec[s]),
                ))

        # ----- parameters: kinases (c_k) -----
        for j, k in enumerate(kinases):
            params_kin_rows.append((sol_id, k, float(p["c_k"][j])))

        # ----- trajectories (optional / top-K) -----
        if sol_id in sol_ids_for_traj:
            # use your existing measurement function (calls simulate_odeint internally)
            dfp, dfr, dfph = simulate_and_measure(sys, idx, t_points_p, t_points_r, t_points_ph)

            if dfp is not None and not dfp.empty:
                dfp = dfp.copy()
                dfp.insert(0, "sol_id", sol_id)
                traj_p_list.append(dfp)

            if dfr is not None and not dfr.empty:
                dfr = dfr.copy()
                dfr.insert(0, "sol_id", sol_id)
                traj_r_list.append(dfr)

            if dfph is not None and not dfph.empty:
                dfph = dfph.copy()
                dfph.insert(0, "sol_id", sol_id)
                traj_ph_list.append(dfph)

    df_params_genes = pd.DataFrame(params_genes_rows, columns=["sol_id", "protein", "param", "value"])
    df_params_kin = pd.DataFrame(params_kin_rows, columns=["sol_id", "kinase", "c_k"])

    df_traj_p = pd.concat(traj_p_list, ignore_index=True) if traj_p_list else pd.DataFrame(
        columns=["sol_id", "protein", "time", "pred_fc"]
    )
    df_traj_r = pd.concat(traj_r_list, ignore_index=True) if traj_r_list else pd.DataFrame(
        columns=["sol_id", "protein", "time", "pred_fc"]
    )
    df_traj_ph = pd.concat(traj_ph_list, ignore_index=True) if traj_ph_list else pd.DataFrame(
        columns=["sol_id", "protein", "psite", "time", "pred_fc"]
    )
    df_deg_sites = pd.DataFrame(
        deg_site_rows,
        columns=["sol_id", "site_id", "protein", "psite", "local_site", "k_deg"]
    )

    # ---- Write Excel ----
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_params_genes.to_excel(writer, sheet_name="params_genes", index=False)
        df_deg_sites.to_excel(writer, sheet_name="deg_sites", index=False)
        df_params_kin.to_excel(writer, sheet_name="params_kinases", index=False)
        df_traj_p.to_excel(writer, sheet_name="traj_protein", index=False)
        df_traj_r.to_excel(writer, sheet_name="traj_rna", index=False)
        df_traj_ph.to_excel(writer, sheet_name="traj_phospho", index=False)

    print(f"[Output] Pareto export saved: {output_path}")
    print(f"[Output] Solutions: {len(df_summary)} | Traj exported for: {len(sol_ids_for_traj)}")


def _standardize_merged_fc(df, obs_suffix="_obs", pred_suffix="_pred"):
    """
    Given a merged df with suffixes, produce columns: fc_obs, fc_pred.
    Supports common input names: fc, pred_fc, fc_obs/fc_pred already present.
    """
    if {"fc_obs", "fc_pred"}.issubset(df.columns):
        return df

    candidates = ["fc", "pred_fc", "obs_fc", "meas_fc"]

    obs_col = None
    pred_col = None
    for base in candidates:
        c_obs = f"{base}{obs_suffix}"
        c_pred = f"{base}{pred_suffix}"
        if c_obs in df.columns and c_pred in df.columns:
            obs_col, pred_col = c_obs, c_pred
            break

    if obs_col is None and pred_col is None:
        if "fc" in df.columns and "pred_fc" in df.columns:
            obs_col, pred_col = "fc", "pred_fc"
        elif "fc_obs" in df.columns and "pred_fc" in df.columns:
            obs_col, pred_col = "fc_obs", "pred_fc"
        elif "fc" in df.columns and "fc_pred" in df.columns:
            obs_col, pred_col = "fc", "fc_pred"

    if obs_col is None or pred_col is None:
        raise KeyError(
            f"Could not infer observed/predicted FC columns. "
            f"Columns seen: {list(df.columns)}"
        )

    out = df.copy()
    out.rename(columns={obs_col: "fc_obs", pred_col: "fc_pred"}, inplace=True)
    return out

def plot_goodness_of_fit(df_prot_obs, df_prot_pred,
                         df_rna_obs, df_rna_pred,
                         df_phos_obs, df_phos_pred,
                         output_dir, file_prefix=""):

    # Merge and standardize
    mp = df_prot_obs.merge(df_prot_pred, on=["protein", "time"], suffixes=("_obs", "_pred"))
    mr = df_rna_obs.merge(df_rna_pred, on=["protein", "time"], suffixes=("_obs", "_pred"))
    mph = df_phos_obs.merge(df_phos_pred, on=["protein", "psite", "time"], suffixes=("_obs", "_pred"))

    mp = _standardize_merged_fc(mp)
    mr = _standardize_merged_fc(mr)
    mph = _standardize_merged_fc(mph)

    mp["Type"] = "Protein"
    mr["Type"] = "RNA"
    mph["Type"] = "Phosphorylation"

    combined = pd.concat([mp, mr, mph], ignore_index=True)
    combined = combined.dropna(subset=["fc_obs", "fc_pred"])

    sns.set_style("whitegrid")
    g = sns.FacetGrid(combined, col="Type", height=6, sharex=False, sharey=False)

    def scatter_with_metrics(x, y, **kwargs):
        ax = plt.gca()

        ax.scatter(x, y, alpha=0.5, s=30, edgecolors="w", linewidths=0.3)

        # identity line
        xmin = min(np.min(x), np.min(y))
        xmax = max(np.max(x), np.max(y))
        ax.plot([xmin, xmax], [xmin, xmax], "k--", alpha=0.75, zorder=0)

        # regression line
        sns.regplot(x=x, y=y, scatter=False, ci=95,
                    line_kws={"alpha": 0.6, "lw": 2},
                    ax=ax)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r2 = r_value ** 2
        rmse = float(np.sqrt(np.mean((np.asarray(y) - np.asarray(x)) ** 2)))

        ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    g.map(scatter_with_metrics, "fc_obs", "fc_pred")
    g.set_axis_labels("Observed FC", "Predicted FC")
    g.fig.suptitle("Goodness of Fit: Global ODE Model", y=1.05)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_prefix}goodness_of_fit.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved Goodness of Fit plot to: {out_path}")


def plot_gof_from_pareto_excel(
        excel_path: str,
        output_dir: str,
        plot_goodness_of_fit_func,
        df_prot_obs_all: pd.DataFrame,
        df_rna_obs_all: pd.DataFrame,
        df_phos_obs_all: pd.DataFrame,
        traj_protein_sheet: str = "traj_protein",
        traj_rna_sheet: str = "traj_rna",
        traj_phospho_sheet: str = "traj_phospho",
        summary_sheet: str = "summary",
        top_k: int = None,
        only_solutions=None,
        score_col: str = "scalar_score",
):
    """
    Uses the Excel produced by export_pareto_front_to_excel (your current version):
      - summary
      - traj_protein (sol_id, protein, time, pred_fc)
      - traj_rna     (sol_id, protein, time, pred_fc)
      - traj_phospho (sol_id, protein, psite, time, pred_fc)

    Observations are NOT in Excel, so we take them from df_prot_obs_all / df_rna_obs_all.
    """
    os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    for s in [traj_protein_sheet, traj_rna_sheet, traj_phospho_sheet, summary_sheet]:
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
    traj_ph = pd.read_excel(xls, sheet_name=traj_phospho_sheet)

    # sanity required columns
    for name, df in [("traj_protein", traj_p), ("traj_rna", traj_r)]:
        need = {"sol_id", "protein", "time", "pred_fc"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{name}' missing columns: {sorted(missing)}")

    for name, df in [("traj_phospho", traj_ph)]:
        need = {"sol_id", "protein", "psite", "time", "pred_fc"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{name}' missing columns: {sorted(missing)}")

    # ensure types
    traj_p["sol_id"] = pd.to_numeric(traj_p["sol_id"], errors="coerce").astype("Int64")
    traj_r["sol_id"] = pd.to_numeric(traj_r["sol_id"], errors="coerce").astype("Int64")
    traj_ph["sol_id"] = pd.to_numeric(traj_ph["sol_id"], errors="coerce").astype("Int64")

    traj_p["time"] = pd.to_numeric(traj_p["time"], errors="coerce")
    traj_r["time"] = pd.to_numeric(traj_r["time"], errors="coerce")
    traj_ph["time"] = pd.to_numeric(traj_ph["time"], errors="coerce")

    # observed must have protein,time,fc
    for df_name, df in [("df_prot_obs_all", df_prot_obs_all), ("df_rna_obs_all", df_rna_obs_all)]:
        for col in ["protein", "time", "fc"]:
            if col not in df.columns and not df.empty:
                raise ValueError(f"{df_name} must have columns: protein,time,fc. Missing: {col}")

    for df_name, df in [("df_phos_obs_all", df_phos_obs_all)]:
        for col in ["protein", "psite", "time", "fc"]:
            if col not in df.columns and not df.empty:
                raise ValueError(f"{df_name} must have columns: protein,psite,time,fc. Missing: {col}")

    # loop solutions
    for sid in sol_ids:
        sid = int(sid)

        sub_p = traj_p[traj_p["sol_id"] == sid].copy()
        sub_r = traj_r[traj_r["sol_id"] == sid].copy()
        sub_ph = traj_ph[traj_ph["sol_id"] == sid].copy()

        if sub_p.empty and sub_r.empty:
            continue

        # build pred dfs in the format your plot_goodness_of_fit expects:
        # df_*_pred must have columns: protein,time,pred_fc
        df_prot_pred = sub_p[["protein", "time", "pred_fc"]].copy()
        df_rna_pred = sub_r[["protein", "time", "pred_fc"]].copy()
        df_phos_pred = sub_ph[["protein", "psite", "time", "pred_fc"]].copy()

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

        if not df_phos_pred.empty:
            keys_ph = df_phos_pred[["protein", "psite", "time"]].drop_duplicates()
            df_phos_obs = df_phos_obs_all.merge(keys_ph, on=["protein", "psite", "time"], how="inner")
        else:
            df_phos_obs = df_phos_obs_all.iloc[0:0].copy()

        sol_dir = output_dir

        plot_goodness_of_fit_func(
            df_prot_obs=df_prot_obs,
            df_prot_pred=df_prot_pred,
            df_rna_obs=df_rna_obs,
            df_rna_pred=df_rna_pred,
            df_phos_obs=df_phos_obs,
            df_phos_pred=df_phos_pred,
            output_dir=sol_dir,
            file_prefix=f"sol_{sid}_"
        )

    print(f"[Output] GoF plots generated for {len(sol_ids)} solutions into: {output_dir}")


def export_results(
    sys,
    idx,
    df_prot_obs,
    df_rna_obs,
    df_phos_obs,
    df_pred_p,
    df_pred_r,
    df_pred_ph,
    output_dir,
):
    """
    Export pre-computed observed + predicted trajectories and model parameters.

    Writes:
      - model_trajectories.csv
      - model_parameters_genes.csv
      - model_parameters_genes_psites.csv   (long format: protein x psite)
      - model_parameters_kinases.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    def _merge_traj(obs_df, pred_df, on_cols, traj_type):
        obs = obs_df.rename(columns={"fc": "fc_obs"}).copy()
        pred = pred_df.rename(columns={"pred_fc": "fc_pred"}).copy()
        merged = obs.merge(pred, on=on_cols, how="outer")
        merged["type"] = traj_type
        if "psite" not in merged.columns:
            merged["psite"] = np.nan
        return merged

    print("[Output] Exporting Trajectories...")

    merged_p = _merge_traj(df_prot_obs, df_pred_p, on_cols=["protein", "time"], traj_type="Protein")
    merged_r = _merge_traj(df_rna_obs,  df_pred_r, on_cols=["protein", "time"], traj_type="RNA")
    merged_ph = _merge_traj(df_phos_obs, df_pred_ph, on_cols=["protein", "psite", "time"], traj_type="Phosphorylation")

    full_traj = pd.concat([merged_p, merged_r, merged_ph], ignore_index=True)
    full_traj = full_traj[["type", "protein", "psite", "time", "fc_obs", "fc_pred"]]
    full_traj.sort_values(["type", "protein", "psite", "time"], inplace=True)

    full_traj.to_csv(os.path.join(output_dir, "model_trajectories.csv"), index=False)

    print("[Output] Exporting Parameters...")

    # --- Gene/protein-level parameters (one row per protein) ---
    N = len(idx.proteins)

    dp_mean = np.full(N, np.nan, dtype=float)
    dp_min = np.full(N, np.nan, dtype=float)
    dp_max = np.full(N, np.nan, dtype=float)

    for i in range(N):
        ns = int(idx.n_sites[i])
        if ns > 0:
            s0 = int(idx.offset_s[i])
            dps = np.asarray(sys.Dp_i[s0:s0 + ns], dtype=float)  # per-site Dp for this protein
            dp_mean[i] = float(np.mean(dps))
            dp_min[i] = float(np.min(dps))
            dp_max[i] = float(np.max(dps))

    df_params_genes = pd.DataFrame(
        {
            "Protein_Gene": idx.proteins,
            "Synthesis_A": sys.A_i,
            "mRNA_Degradation_B": sys.B_i,
            "Translation_C": sys.C_i,
            "Protein_Degradation_D": sys.D_i,
            "Phospho_Degradation_Dp_mean": dp_mean,
            "Phospho_Degradation_Dp_min": dp_min,
            "Phospho_Degradation_Dp_max": dp_max,
            "De-Phosphorylation_E": sys.E_i,
            "Global_TF_Scale": np.full(N, sys.tf_scale, dtype=float),
        }
    )

    df_params_genes.to_csv(os.path.join(output_dir, "model_parameters_genes.csv"), index=False)

    # --- Gene/protein x psite parameters (long format) ---
    # psites come from observed phos dataframe (can switch to df_pred_ph if preferred)
    psites_by_protein = (
        df_phos_obs[["protein", "psite"]]
        .dropna()
        .drop_duplicates()
        .groupby("protein")["psite"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    rows = []
    for i, prot in enumerate(idx.proteins):
        ns = int(idx.n_sites[i])
        s0 = int(idx.offset_s[i])

        if ns == 0:
            # no sites -> one row with NaNs for site-specific values
            rows.append(
                {
                    "Protein_Gene": prot,
                    "psite": np.nan,
                    "Synthesis_A": sys.A_i[i],
                    "mRNA_Degradation_B": sys.B_i[i],
                    "Translation_C": sys.C_i[i],
                    "Protein_Degradation_D": sys.D_i[i],
                    "Phospho_Degradation_Dp": np.nan,
                    "De-Phosphorylation_E": sys.E_i[i],
                    "Global_TF_Scale": sys.tf_scale,
                }
            )
            continue

        # map psite -> local index j
        site_to_j = {ps: j for j, ps in enumerate(idx.sites[i])}

        psite_list = psites_by_protein.get(prot, [])
        for psite in psite_list:
            j = site_to_j.get(psite, None)
            dp_val = float(sys.Dp_i[s0 + j]) if j is not None else np.nan

            rows.append(
                {
                    "Protein_Gene": prot,
                    "psite": psite,
                    "Synthesis_A": sys.A_i[i],
                    "mRNA_Degradation_B": sys.B_i[i],
                    "Translation_C": sys.C_i[i],
                    "Protein_Degradation_D": sys.D_i[i],
                    "Phospho_Degradation_Dp": dp_val,
                    "De-Phosphorylation_E": sys.E_i[i],
                    "Global_TF_Scale": sys.tf_scale,
                }
            )

    df_params_genes_psites = pd.DataFrame(rows)
    df_params_genes_psites.to_csv(
        os.path.join(output_dir, "model_parameters_genes_psites.csv"),
        index=False,
    )

    # --- Kinase parameters ---
    df_kin_params = pd.DataFrame(
        {
            "Kinase": idx.kinases,
            "Activity_Scale_ck": sys.c_k,
        }
    )
    df_kin_params.to_csv(os.path.join(output_dir, "model_parameters_kinases.csv"), index=False)

    print(f"[Output] Exports saved to {output_dir}")


def save_gene_timeseries_plots(
    gene: str,
    df_prot_obs: pd.DataFrame,
    df_prot_pred: pd.DataFrame,
    df_rna_obs: pd.DataFrame,
    df_rna_pred: pd.DataFrame,
    df_phos_obs: pd.DataFrame,
    df_phos_pred: pd.DataFrame,
    output_dir: str,
    prot_times: np.ndarray = None,
    rna_times: np.ndarray = None,
    phos_times: np.ndarray = None,
    filename_prefix: str = "ts",
    dpi: int = 300,
    phos_mode: str = "per_psite",   # "mean" or "per_psite"
    max_psites: int = None,      # only used for per_psite
):
    """
    Save a 3-panel time-series plot for ONE gene symbol:
      - Protein observed vs predicted (fc vs fc_pred)
      - RNA observed vs predicted
      - Phosphorylation observed vs predicted (either mean across psites or per-psite lines)

    Expected inputs:
      Protein/RNA obs columns:  protein, time, fc
      Protein/RNA pred columns: protein, time, pred_fc OR fc_pred

      Phospho obs columns:  protein, psite, time, fc
      Phospho pred columns: protein, psite, time, pred_fc OR fc_pred

    Output:
      {output_dir}/{filename_prefix}_{gene}.png
    """
    os.makedirs(output_dir, exist_ok=True)

    def _norm_pred(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "fc_pred" not in df.columns and "pred_fc" in df.columns:
            df.rename(columns={"pred_fc": "fc_pred"}, inplace=True)
        if "fc_pred" not in df.columns:
            raise ValueError(f"Pred df must have 'pred_fc' or 'fc_pred'. Got: {list(df.columns)}")
        return df

    def _clean_ts(df: pd.DataFrame, value_col: str, time_col: str = "time") -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        out.dropna(subset=[time_col, value_col], inplace=True)
        out.sort_values(time_col, inplace=True)
        return out

    def _lighten(color, amount=0.65):
        r, g, b = mcolors.to_rgb(color)
        return (1 - amount) * r + amount, (1 - amount) * g + amount, (1 - amount) * b + amount

    # Normalize preds
    prot_pred = _norm_pred(df_prot_pred) if df_prot_pred is not None else pd.DataFrame()
    rna_pred  = _norm_pred(df_rna_pred)  if df_rna_pred  is not None else pd.DataFrame()
    phos_pred = _norm_pred(df_phos_pred) if df_phos_pred is not None else pd.DataFrame()

    # Subset gene
    p_obs = df_prot_obs[df_prot_obs["protein"] == gene].copy() if df_prot_obs is not None else pd.DataFrame()
    p_pre = prot_pred[prot_pred["protein"] == gene].copy() if not prot_pred.empty else pd.DataFrame()

    r_obs = df_rna_obs[df_rna_obs["protein"] == gene].copy() if df_rna_obs is not None else pd.DataFrame()
    r_pre = rna_pred[rna_pred["protein"] == gene].copy() if not rna_pred.empty else pd.DataFrame()

    ph_obs = df_phos_obs[df_phos_obs["protein"] == gene].copy() if df_phos_obs is not None else pd.DataFrame()
    ph_pre = phos_pred[phos_pred["protein"] == gene].copy() if not phos_pred.empty else pd.DataFrame()

    # If truly nothing exists, bail
    if p_obs.empty and p_pre.empty and r_obs.empty and r_pre.empty and ph_obs.empty and ph_pre.empty:
        return None

    # Optional time filtering
    if prot_times is not None:
        p_obs = p_obs[p_obs["time"].isin(prot_times)]
        p_pre = p_pre[p_pre["time"].isin(prot_times)]
    if rna_times is not None:
        r_obs = r_obs[r_obs["time"].isin(rna_times)]
        r_pre = r_pre[r_pre["time"].isin(rna_times)]
    if phos_times is not None:
        ph_obs = ph_obs[ph_obs["time"].isin(phos_times)]
        ph_pre = ph_pre[ph_pre["time"].isin(phos_times)]

    # Clean numeric + sort
    p_obs = _clean_ts(p_obs, "fc")
    p_pre = _clean_ts(p_pre, "fc_pred")
    r_obs = _clean_ts(r_obs, "fc")
    r_pre = _clean_ts(r_pre, "fc_pred")

    # phospho cleaning (needs psite too, but same numeric cleaning)
    if not ph_obs.empty:
        ph_obs = _clean_ts(ph_obs, "fc")
    if not ph_pre.empty:
        ph_pre = _clean_ts(ph_pre, "fc_pred")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=False)
    ax_p, ax_r, ax_ph = axes
    prot_c = "C0"
    rna_c = "C1"
    phos_c = "C2"

    obs_alpha = 0.35
    pred_alpha = 1.0

    prot_obs_c = _lighten(prot_c, 0.65)
    rna_obs_c = _lighten(rna_c, 0.65)
    phos_obs_c = _lighten(phos_c, 0.65)

    # Protein
    if not p_obs.empty:
        ax_p.plot(p_obs["time"].to_numpy(), p_obs["fc"].to_numpy(),
                  marker="o", linewidth=2, label="obs",
                  color=prot_obs_c, alpha=obs_alpha)
    if not p_pre.empty:
        ax_p.plot(p_pre["time"].to_numpy(), p_pre["fc_pred"].to_numpy(),
                  marker="o", linewidth=2, label="pred",
                  color=prot_c, alpha=pred_alpha)
    ax_p.set_title(f"{gene} — Protein")
    ax_p.set_xlabel("Time")
    ax_p.set_ylabel("FC")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend()

    # RNA
    if not r_obs.empty:
        ax_r.plot(r_obs["time"].to_numpy(), r_obs["fc"].to_numpy(),
                  marker="o", linewidth=2, label="obs",
                  color=rna_obs_c, alpha=obs_alpha)
    if not r_pre.empty:
        ax_r.plot(r_pre["time"].to_numpy(), r_pre["fc_pred"].to_numpy(),
                  marker="o", linewidth=2, label="pred",
                  color=rna_c, alpha=pred_alpha)

    ax_r.set_title(f"{gene} — mRNA")
    ax_r.set_xlabel("Time")
    ax_r.set_ylabel("FC")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend()

    # Phospho
    ax_ph.set_title(f"{gene} — Phosphorylation")
    ax_ph.set_xlabel("Time")
    ax_ph.set_ylabel("FC")
    ax_ph.grid(True, alpha=0.3)

    if phos_mode not in {"mean", "per_psite"}:
        raise ValueError("phos_mode must be 'mean' or 'per_psite'")

    if not ph_obs.empty or not ph_pre.empty:
        if phos_mode == "mean":
            # mean across psites at each time
            if not ph_obs.empty:
                obs_mean = ph_obs.groupby("time", as_index=False)["fc"].mean()
                ax_ph.plot(obs_mean["time"].to_numpy(), obs_mean["fc"].to_numpy(),
                           marker="o", linewidth=2, label="obs (mean)",
                           color=phos_obs_c, alpha=obs_alpha)

            if not ph_pre.empty:
                pre_mean = ph_pre.groupby("time", as_index=False)["fc_pred"].mean()
                ax_ph.plot(pre_mean["time"].to_numpy(), pre_mean["fc_pred"].to_numpy(),
                           marker="o", linewidth=2, label="pred (mean)",
                           color=phos_c, alpha=pred_alpha)
        else:
            # per-psite lines (capped)
            psites = sorted(set(ph_obs["psite"].unique()).union(set(ph_pre["psite"].unique()))) if ("psite" in ph_obs.columns or "psite" in ph_pre.columns) else []

            # if len(psites) > max_psites:
            #     psites = psites[:max_psites]

            for ps in psites:
                ps_color = f"C{hash(ps) % 10}"
                ps_obs_color = _lighten(ps_color, 0.65)

                if not ph_obs.empty:
                    subo = ph_obs[ph_obs["psite"] == ps]
                    if not subo.empty:
                        ax_ph.plot(subo["time"].to_numpy(), subo["fc"].to_numpy(),
                                   marker="o", linewidth=1, label=f"obs {ps}",
                                   color=ps_obs_color, alpha=obs_alpha)

                if not ph_pre.empty:
                    subp = ph_pre[ph_pre["psite"] == ps]
                    if not subp.empty:
                        ax_ph.plot(subp["time"].to_numpy(), subp["fc_pred"].to_numpy(),
                                   marker="o", linewidth=1, label=f"pred {ps}",
                                   color=ps_color, alpha=pred_alpha)

        ax_ph.legend(ncol=2, fontsize=8)

    fig.suptitle(f"Observed vs Predicted Time Series — {gene}", y=0.995)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{filename_prefix}_{gene}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def scan_prior_reg(out_dir):
    F = np.load(os.path.join(out_dir, "pareto_F.npy"))
    X = np.load(os.path.join(out_dir, "pareto_X.npy"))  # not used here, but kept for consistency

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"Expected F shape (n, 3) = [prot_mse, rna_mse, phospho_mse]. Got {F.shape}")

    lambda_prot_grid  = np.logspace(-2, 2, 9)   # 0.01 .. 100
    lambda_rna_grid   = np.logspace(-2, 2, 9)   # 0.01 .. 100
    lambda_phos_grid  = np.logspace(-2, 2, 9)   # 0.01 .. 100
    lambda_prior_grid = np.logspace(-4, 0, 9)   # 1e-4 .. 1

    prot = F[:, 0].astype(float)
    rna  = F[:, 1].astype(float)
    phos = F[:, 2].astype(float)

    rows = []
    for lprot in lambda_prot_grid:
        for lrna in lambda_rna_grid:
            for լph in lambda_phos_grid:
                base = (float(lprot) * prot) + (float(lrna) * rna) + (float(լph) * phos)

                for lprior in lambda_prior_grid:
                    if float(lprior) <= 0:
                        raise ValueError("lambda_prior must be > 0 to preserve ordering / meaning.")

                    score = float(lprior) * base
                    best_i = int(np.argmin(score))
                    best_score = float(score[best_i])

                    rows.append({
                        "lambda_prot": float(lprot),
                        "lambda_rna": float(lrna),
                        "lambda_phospho": float(լph),
                        "lambda_prior": float(lprior),
                        "best_i": best_i,
                        "best_score": best_score,
                        "prot_mse": float(prot[best_i]),
                        "rna_mse": float(rna[best_i]),
                        "phospho_mse": float(phos[best_i]),
                    })

    df = pd.DataFrame(rows).sort_values(
        ["lambda_prot", "lambda_rna", "lambda_phospho", "lambda_prior"],
        ignore_index=True
    )
    df.to_csv(os.path.join(out_dir, "lambda_scan.csv"), index=False)

    # also save the unique picked solutions (often repeats)
    uniq = df.drop_duplicates("best_i").copy()
    uniq.to_csv(os.path.join(out_dir, "lambda_scan_unique_picks.csv"), index=False)

    # “recommended” choice: pick the best (prot, then rna, then phos) among unique picks
    cand = uniq.sort_values(["prot_mse", "rna_mse", "phospho_mse"], ignore_index=True).iloc[0]
    rec = {
        "lambda_prot": float(cand["lambda_prot"]),
        "lambda_rna": float(cand["lambda_rna"]),
        "lambda_phospho": float(cand["lambda_phospho"]),
        "lambda_prior": float(cand["lambda_prior"]),
        "best_i": int(cand["best_i"]),
        "objectives": {
            "prot_mse": float(cand["prot_mse"]),
            "rna_mse": float(cand["rna_mse"]),
            "phospho_mse": float(cand["phospho_mse"]),
        },
        "note": "lambda_prior is a global multiplier; it does not change best_i for fixed F (only rescales best_score).",
    }
    with open(os.path.join(out_dir, "lambda_scan_recommended.json"), "w") as f:
        json.dump(rec, f, indent=2)

    print("Wrote:")
    print(" - lambda_scan.csv")
    print(" - lambda_scan_unique_picks.csv")
    print(" - lambda_scan_recommended.json")

    return df, uniq, rec

def export_S_rates(sys, idx, output_dir, filename="S_rates_picked.csv", long=True):
    """
    Export phosphorylation drive S for optimized parameters.
    S is per-site and per time-bin (TIME_POINTS_PROTEIN / sys.kin_grid).

    long=True  -> columns: protein, psite, time, S
    long=False -> wide: protein, psite, S_t0, S_t1, ...
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- compute S matrix: shape (total_sites, n_bins) ----
    if MODEL == 2:
        # Ensure cache matches current optimized c_k
        build_S_cache_into(sys.S_cache, sys.W_indptr, sys.W_indices, sys.W_data, sys.kin_Kmat, sys.c_k)
        S_mat = np.asarray(sys.S_cache, dtype=np.float64)
        times = np.asarray(sys.kin_grid, dtype=float)
    else:
        # Dense kinase signal scaled by c_k
        K_scaled = (np.asarray(sys.kin_Kmat, dtype=np.float64) *
                    np.asarray(sys.c_k, dtype=np.float64)[:, None])      # (n_kin, n_bins)
        # Sparse (total_sites x n_kin) dot dense (n_kin x n_bins) -> (total_sites x n_bins)
        S_mat = sys.W_global.dot(K_scaled)
        S_mat = np.asarray(S_mat, dtype=np.float64)
        times = np.asarray(sys.kin_grid, dtype=float)

    # ---- build (protein, psite) mapping in the exact row order of W / S_mat ----
    proteins = []
    psites = []
    for i, p in enumerate(idx.proteins):
        for s in idx.sites[i]:
            proteins.append(p)
            psites.append(s)

    if len(proteins) != S_mat.shape[0]:
        raise RuntimeError(
            f"Row mapping mismatch: mapped {len(proteins)} sites but S_mat has {S_mat.shape[0]} rows. "
            f"Check idx.sites ordering vs W construction."
        )

    if long:
        # long format: repeat each site across all time bins
        n_sites, n_bins = S_mat.shape
        df = pd.DataFrame({
            "protein": np.repeat(np.array(proteins, dtype=object), n_bins),
            "psite":   np.repeat(np.array(psites, dtype=object),   n_bins),
            "time":    np.tile(times, n_sites),
            "S":       S_mat.reshape(-1),
        })
    else:
        # wide format: one row per site, one column per time
        cols = [f"S_t{t:g}" for t in times]
        df = pd.DataFrame(S_mat, columns=cols)
        df.insert(0, "psite", psites)
        df.insert(0, "protein", proteins)

    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"[Output] Saved S rates to: {out_path}")
    return df

def plot_s_rates_report(
    csv_path: str | Path,
    out_pdf: str | Path = "S_rates_report.pdf",
    *,
    time_col: str = "time",
    value_col: str = "S",
    protein_col: str = "protein",
    psite_col: str = "psite",
    log_x: bool = True,
    # keep plots readable for many sites
    top_k_sites_per_protein: int | None = 24,   # rank by AUC and keep only top K per protein in "small multiples"
    max_sites_per_page: int = 12,               # small-multiples pagination
    ncols: int = 3,                              # small-multiples grid columns
    normalize_per_site: bool = False,            # if True: plot S/Smax to compare kinetics
    heatmap_per_protein: bool = True,
    heatmap_cap_sites: int = 80,                 # cap number of rows in a heatmap (rank by AUC)
    agg_duplicates: str = "mean",                # if repeated (protein,psite,time)
    dpi: int = 150,
) -> Path:
    """
    Robust plotting for S_rates_picked.csv (protein, psite, time, S):
      - Global summary pages (AUC top sites, early-vs-late scatter)
      - For each protein:
          * optional heatmap (sites x time) capped
          * small-multiples time series pages (paginated), ranked by AUC
    Outputs a single multi-page PDF (no clumped mega-figure).

    Returns
    -------
    Path to saved PDF.
    """
    csv_path = Path(csv_path)
    out_pdf = Path(out_pdf)

    df = pd.read_csv(csv_path)

    need = {protein_col, psite_col, time_col, value_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Got: {list(df.columns)}")

    # coerce + clean
    df = df[[protein_col, psite_col, time_col, value_col]].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[protein_col, psite_col, time_col, value_col])

    # aggregate duplicates (same protein, site, time)
    if agg_duplicates:
        df = (
            df.groupby([protein_col, psite_col, time_col], as_index=False)[value_col]
              .agg(agg_duplicates)
        )

    # sort
    df.sort_values([protein_col, psite_col, time_col], inplace=True)

    # compute AUC per (protein, psite) to rank sites
    def _auc(g: pd.DataFrame) -> float:
        t = g[time_col].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)
        if t.size < 2:
            return float(y[0]) if y.size else 0.0
        # trapz assumes t sorted
        return float(np.trapz(y, t))

    auc_df = (
        df.groupby([protein_col, psite_col], as_index=False)
          .apply(_auc)
          .reset_index()
    )
    # pandas groupby.apply output can be awkward depending on version
    if "level_0" in auc_df.columns and 0 in auc_df.columns:
        auc_df = auc_df.rename(columns={"level_0": protein_col, "level_1": psite_col, 0: "AUC"})
    elif 0 in auc_df.columns:
        auc_df = auc_df.rename(columns={0: "AUC"})
    elif "AUC" not in auc_df.columns:
        # fallback: rebuild robustly
        rows = []
        for (p, s), g in df.groupby([protein_col, psite_col]):
            rows.append((p, s, _auc(g)))
        auc_df = pd.DataFrame(rows, columns=[protein_col, psite_col, "AUC"])

    auc_df["AUC"] = pd.to_numeric(auc_df["AUC"], errors="coerce").fillna(0.0)
    auc_df.sort_values(["AUC"], ascending=False, inplace=True)

    # early vs late means for global scatter
    tmin = float(df[time_col].min())
    tmax = float(df[time_col].max())
    # choose early/late cutoffs that match your grids
    early_cut = 2.0
    late_cut = 120.0

    early = (
        df[df[time_col] <= early_cut]
        .groupby([protein_col, psite_col])[value_col]
        .mean()
        .rename("early_S")
    )
    late = (
        df[df[time_col] >= late_cut]
        .groupby([protein_col, psite_col])[value_col]
        .mean()
        .rename("late_S")
    )
    el = pd.concat([early, late], axis=1).dropna().reset_index()

    # ---------- plotting helpers ----------
    def _apply_xscale(ax):
        if log_x:
            # avoid log(0): shift zeros to smallest positive tick if needed
            ax.set_xscale("symlog" if (df[time_col] == 0).any() else "log")

    def _small_multiples_for_protein(pdf: PdfPages, prot: str, sub: pd.DataFrame):
        # rank this protein's sites by AUC
        sub_auc = auc_df[auc_df[protein_col] == prot].sort_values("AUC", ascending=False)
        sites_ranked = sub_auc[psite_col].tolist()
        if top_k_sites_per_protein is not None:
            sites_ranked = sites_ranked[: int(top_k_sites_per_protein)]

        # paginate
        pages = max(1, math.ceil(len(sites_ranked) / max_sites_per_page))
        for page in range(pages):
            chunk = sites_ranked[page * max_sites_per_page : (page + 1) * max_sites_per_page]
            if not chunk:
                continue

            n = len(chunk)
            nrows = math.ceil(n / ncols)

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(3.8 * ncols, 2.7 * nrows),
                squeeze=False
            )
            axes = axes.ravel()

            for ax_i, site in enumerate(chunk):
                ax = axes[ax_i]
                g = sub[sub[psite_col] == site]
                t = g[time_col].to_numpy(dtype=float)
                y = g[value_col].to_numpy(dtype=float)

                if normalize_per_site:
                    mx = float(np.max(y)) if y.size else 1.0
                    y = y / (mx if mx > 0 else 1.0)

                ax.plot(t, y, marker="o", linewidth=1.5, markersize=3)
                _apply_xscale(ax)
                ax.set_title(f"{prot}  {site}", fontsize=9)
                ax.grid(True, alpha=0.25)

                if ax_i % ncols == 0:
                    ax.set_ylabel("S" if not normalize_per_site else "S / max(S)")
                ax.set_xlabel("time")

            # turn off unused axes
            for j in range(n, len(axes)):
                axes[j].axis("off")

            fig.suptitle(
                f"{prot} — site time series"
                + (" (normalized)" if normalize_per_site else "")
                + (f" — page {page+1}/{pages}" if pages > 1 else ""),
                fontsize=12
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    def _heatmap_for_protein(pdf: PdfPages, prot: str, sub: pd.DataFrame):
        # pivot sites x times
        sub_auc = auc_df[auc_df[protein_col] == prot].sort_values("AUC", ascending=False)
        sites_ranked = sub_auc[psite_col].tolist()
        if heatmap_cap_sites is not None:
            sites_ranked = sites_ranked[: int(heatmap_cap_sites)]

        sub2 = sub[sub[psite_col].isin(sites_ranked)]
        piv = sub2.pivot(index=psite_col, columns=time_col, values=value_col)

        # reorder columns by time
        piv = piv.reindex(sorted(piv.columns), axis=1)

        # optionally normalize rows
        mat = piv.to_numpy(dtype=float)
        if normalize_per_site:
            row_max = np.nanmax(mat, axis=1)
            row_max[row_max <= 0] = 1.0
            mat = mat / row_max[:, None]

        fig, ax = plt.subplots(figsize=(10.5, max(3.5, 0.18 * mat.shape[0])))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_title(
            f"{prot} — heatmap (top {len(sites_ranked)} sites by AUC)"
            + (" — normalized" if normalize_per_site else ""),
            fontsize=12
        )
        ax.set_ylabel("psite")
        ax.set_xlabel("time")

        # ticks
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index.tolist(), fontsize=7)

        xt = piv.columns.to_list()
        ax.set_xticks(np.arange(len(xt)))
        ax.set_xticklabels([str(x) for x in xt], rotation=45, ha="right", fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("S" if not normalize_per_site else "S / max(S)")

        fig.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

    # ---------- build PDF report ----------
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Page 1: AUC top sites (global)
        top_n = min(30, len(auc_df))
        top = auc_df.head(top_n).copy()
        labels = (top[protein_col].astype(str) + " " + top[psite_col].astype(str)).tolist()

        fig, ax = plt.subplots(figsize=(11, 0.35 * top_n + 2.5))
        ax.barh(range(top_n)[::-1], top["AUC"].to_numpy()[::-1])
        ax.set_yticks(range(top_n)[::-1])
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("AUC of S over time")
        ax.set_title(f"Top {top_n} sites by total signaling (AUC)")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

        # Page 2: Early vs Late scatter (global)
        if not el.empty:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            x = el["early_S"].to_numpy(dtype=float)
            y = el["late_S"].to_numpy(dtype=float)
            ax.scatter(x, y, s=20, alpha=0.7)
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            ax.set_xlabel(f"Early mean S (t ≤ {early_cut})")
            ax.set_ylabel(f"Late mean S (t ≥ {late_cut})")
            ax.set_title("Early vs Late signaling per site")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

        # Per-protein pages
        for prot, sub in df.groupby(protein_col, sort=True):
            if heatmap_per_protein:
                _heatmap_for_protein(pdf, prot, sub)
            _small_multiples_for_protein(pdf, prot, sub)

    return out_pdf
