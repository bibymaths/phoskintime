import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
import seaborn as sns
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from scipy.stats import linregress

from phoskintime_global.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA
from phoskintime_global.params import unpack_params
from phoskintime_global.simulate import simulate_and_measure
from phoskintime_global.utils import pick_best_lamdas


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
        weights=(1.0, 1.0, 1.0),  # (w_prot, w_rna, w_reg) used for scalar score + ranking
        top_k_trajectories=None,  # None = export trajectories for all solutions; else only top K by scalar score
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
        t_points_p = TIME_POINTS_PROTEIN
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


def plot_goodness_of_fit(df_prot_obs, df_prot_pred, df_rna_obs, df_rna_pred, output_dir, file_prefix=""):
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
        traj_protein_sheet: str = "traj_protein",
        traj_rna_sheet: str = "traj_rna",
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

        plot_goodness_of_fit_func(
            df_prot_obs=df_prot_obs,
            df_prot_pred=df_prot_pred,
            df_rna_obs=df_rna_obs,
            df_rna_pred=df_rna_pred,
            output_dir=sol_dir,
            file_prefix=f"sol_{sid}_"
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
      - Top: Protein observed vs predicted FC across TIME_POINTS_PROTEIN
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
    rna_pred = _norm_pred(df_rna_pred)

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

def scan_prior_reg(out_dir):
    F = np.load(os.path.join(out_dir, "pareto_F.npy"))
    X = np.load(os.path.join(out_dir, "pareto_X.npy"))

    # ---- define scan grid (simple log grids) ----
    lambda_rna_grid   = np.logspace(-2, 2, 9)     # 0.01 .. 100
    lambda_prior_grid = np.logspace(-4, 0, 9)     # 1e-4 .. 1

    rows = []
    for lr in lambda_rna_grid:
        for lp in lambda_prior_grid:
            weights = np.array([1.0, lr, lp], dtype=float)
            best_i, best_score = pick_best_lamdas(F, weights)

            rows.append({
                "lambda_rna": lr,
                "lambda_prior": lp,
                "best_i": best_i,
                "best_score": best_score,
                "prot_mse": float(F[best_i, 0]),
                "rna_mse": float(F[best_i, 1]),
                "reg_loss": float(F[best_i, 2]),
            })

    df = pd.DataFrame(rows).sort_values(["lambda_rna", "lambda_prior"])
    df.to_csv(os.path.join(out_dir, "lambda_scan.csv"), index=False)

    # also save the unique picked solutions (often repeats)
    uniq = df.drop_duplicates("best_i").copy()
    uniq.to_csv(os.path.join(out_dir, "lambda_scan_unique_picks.csv"), index=False)

    # write one “recommended” choice: lowest prot_mse among solutions with rna_mse not crazy
    # (adjust this rule to taste)
    cand = uniq.sort_values(["prot_mse", "rna_mse", "reg_loss"]).head(1).iloc[0]
    rec = {
        "lambda_rna": float(cand["lambda_rna"]),
        "lambda_prior": float(cand["lambda_prior"]),
        "best_i": int(cand["best_i"]),
        "objectives": {
            "prot_mse": float(cand["prot_mse"]),
            "rna_mse": float(cand["rna_mse"]),
            "reg_loss": float(cand["reg_loss"]),
        }
    }
    with open(os.path.join(out_dir, "lambda_scan_recommended.json"), "w") as f:
        json.dump(rec, f, indent=2)

    print("Wrote:")
    print(" - lambda_scan.csv")
    print(" - lambda_scan_unique_picks.csv")
    print(" - lambda_scan_recommended.json")

def plot_hypervolume(out_path, gen_history, hv_history):
    plt.figure(figsize=(6, 4))
    plt.plot(gen_history, hv_history, lw=2)
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()