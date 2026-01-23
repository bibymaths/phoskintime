import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from SALib.sample import morris
from SALib.analyze.morris import analyze
from tqdm import tqdm

from global_model.config import SENSITIVITY_TRAJECTORIES, SENSITIVITY_LEVELS, SENSITIVITY_PERTURBATION, \
    SENSITIVITY_TOP_CURVES, RESULTS_DIR, SEED
from global_model.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO
from global_model.simulate import simulate_and_measure

from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


def compute_bounds(params_dict, perturbation=SENSITIVITY_PERTURBATION):
    """
    Generates [lower, upper] bounds for each parameter in the dictionary.
    """
    bounds = []
    names = []

    # Iterate in specific order to match decision vector reconstruction later if needed
    # But here we work with the dictionary keys for SALib
    for key, value in params_dict.items():
        if isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                lb = v * (1 - perturbation)
                ub = v * (1 + perturbation)
                if abs(v) < 1e-6:  # Handle zero or near-zero
                    lb, ub = 0.0, 0.01
                bounds.append([max(0.0, lb), ub])
                names.append(f"{key}_{i}")
        else:
            v = float(value)
            lb = v * (1 - perturbation)
            ub = v * (1 + perturbation)
            if abs(v) < 1e-6:
                lb, ub = 0.0, 0.01
            bounds.append([max(0.0, lb), ub])
            names.append(key)

    return {"num_vars": len(names), "names": names, "bounds": bounds}


def _reconstruct_params(param_vector, names_map, original_shapes):
    """
    Reconstructs the parameter dictionary from the flat Morris vector.
    """
    p_out = {}
    curr = 0

    for key, shape in original_shapes.items():
        if shape == ():  # Scalar
            p_out[key] = param_vector[curr]
            curr += 1
        else:
            # Array
            size = np.prod(shape)
            arr = np.array(param_vector[curr: curr + size])
            p_out[key] = arr
            curr += size

    return p_out


def _compute_scalar_metric(df_prot, df_rna, df_phos, metric="total_signal"):
    """
    Compresses the complex time-series output into a single scalar Y for SALib.
    """
    # 1. Concatenate all signal columns
    # We prioritize Protein > Phospho > RNA for signal magnitude usually
    v_p = df_prot["pred_fc"].values if df_prot is not None else np.array([])
    v_r = df_rna["pred_fc"].values if df_rna is not None else np.array([])
    v_ph = df_phos["pred_fc"].values if df_phos is not None else np.array([])

    combined = np.concatenate([v_p, v_r, v_ph])

    if len(combined) == 0:
        return 0.0

    if metric == "total_signal":
        return np.sum(combined)
    elif metric == "mean":
        return np.mean(combined)
    elif metric == "variance":
        return np.var(combined)
    elif metric == "l2_norm":
        return np.linalg.norm(combined)
    else:
        return np.sum(combined)


def _worker_simulation(task_args):
    """
    Worker function for parallel execution.
    """
    (idx, param_vector, names_map, original_shapes, sys, idx_sys, times_p, times_r, times_ph, metric) = task_args

    # 1. Rebuild params
    p_new = _reconstruct_params(param_vector, names_map, original_shapes)

    # 2. Update System
    # Note: System object is pickled. This is heavy but necessary for multiprocessing.
    sys.update(**p_new)

    # 3. Simulate
    dfp, dfr, dfph = simulate_and_measure(sys, idx_sys, times_p, times_r, times_ph)

    # 4. Compute Scalar Y (Sensitivity Target)
    y_val = _compute_scalar_metric(dfp, dfr, dfph, metric)

    # 5. Compute Goodness of Fit (RMSE) against DATA (if data was passed, assuming it's in sys)
    # We assume sys has _ic_data attached or we pass it. For brevity, we return raw preds.

    return idx, y_val, dfp, dfr, dfph


def run_sensitivity_analysis(sys, idx, fitted_params, output_dir, metric="total_signal"):
    """
    Main driver for Morris Sensitivity Analysis.
    """
    logger.info(f"[Sensitivity] Starting Morris Analysis (N={SENSITIVITY_TRAJECTORIES}, p={SENSITIVITY_LEVELS})...")
    logger.info(f"[Sensitivity] Metric: {metric}")

    # 1. Define Problem
    # We need to flatten the fitted_params dictionary into a list for SALib
    # And keep track of shapes to reconstruct it inside the worker
    original_shapes = {}
    for k, v in fitted_params.items():
        if isinstance(v, np.ndarray) or isinstance(v, list):
            original_shapes[k] = np.shape(v)
        else:
            original_shapes[k] = ()

    problem = compute_bounds(fitted_params)

    # 2. Sample Parameter Space (Morris)
    param_values = morris.sample(problem, N=SENSITIVITY_TRAJECTORIES, num_levels=SENSITIVITY_LEVELS,
                                 local_optimization=True, seed=SEED)
    logger.info(f"[Sensitivity] Generated {len(param_values)} trajectories.")

    # 3. Parallel Execution
    tasks = []
    # We need to pass the system object.
    # Warning: 'sys' might be large. If pickle fails, we need a lighter pickling strategy.
    for i in range(len(param_values)):
        tasks.append((
            i,
            param_values[i],
            problem['names'],
            original_shapes,
            sys,
            idx,
            TIME_POINTS_PROTEIN,
            TIME_POINTS_RNA,
            TIME_POINTS_PHOSPHO,
            metric
        ))

    results_y = np.zeros(len(param_values))
    trajectory_storage = []  # To store top K plots

    # Use fewer workers than cores to prevent memory overflow with large System objects
    n_workers = max(1, int(os.cpu_count() * 0.75))

    logger.info(f"[Sensitivity] simulating with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_simulation, t): t[0] for t in tasks}

        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Simulating"):
            i, y_val, dfp, dfr, dfph = fut.result()
            results_y[i] = y_val

            # Store lightweight summary for plotting later
            # (We don't want to keep ALL dataframes in memory for 1000s of runs)
            # Just keep the "Protein" curves for the top K check
            if dfp is not None:
                trajectory_storage.append({
                    "id": i,
                    "params": param_values[i],
                    "y_val": y_val,
                    "prot_df": dfp[["protein", "time", "pred_fc"]].copy() if dfp is not None else None,
                    "rna_df": dfr[["protein", "time", "pred_fc"]].copy() if dfr is not None else None,
                    "phos_df": dfph[["protein", "psite", "time", "pred_fc"]].copy() if dfph is not None else None
                })

    # 4. Analyze Indices (Morris)
    logger.info("[Sensitivity] Computing Morris Indices...")
    Si = analyze(problem, param_values, results_y, conf_level=0.95, print_to_console=False)

    # Convert to DataFrame
    df_sens = pd.DataFrame({
        "Parameter": problem['names'],
        "mu_star": Si['mu_star'],
        "sigma": Si['sigma'],
        "mu_star_conf": Si['mu_star_conf']
    })

    # Sort by influence
    df_sens = df_sens.sort_values("mu_star", ascending=False)

    # Save CSV
    out_csv = os.path.join(output_dir, "sensitivity_indices.csv")
    df_sens.to_csv(out_csv, index=False)
    logger.info(f"[Sensitivity] Indices saved to {out_csv}")

    trajectory_storage = sorted(trajectory_storage, key=lambda x: x["y_val"], reverse=True)
    trajectory_storage = trajectory_storage[:SENSITIVITY_TOP_CURVES]
    traj_df = pd.DataFrame(trajectory_storage)
    traj_df.to_csv(os.path.join(output_dir, "sensitivity_trajectories.csv"), index=False)

    # 5. Plotting Sensitivity (Top 30 Parameters)
    _plot_sensitivity_indices(df_sens.head(30), output_dir)

    # 6. Plotting Trajectory Cloud (Perturbation Analysis)
    # We plot the spread of the model predictions around the mean
    _plot_perturbation_cloud(trajectory_storage, output_dir, idx)

    return df_sens


def _plot_sensitivity_indices(df, out_dir):
    """
    Bar chart of Mu_Star (Influence).
    """
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x="mu_star", y="Parameter", palette="viridis", legend=False, hue="Parameter")
    plt.title("Morris Sensitivity Analysis (Top Parameters)")
    plt.xlabel("mu_star (Mean Absolute Influence)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_mu_star.png"), dpi=300)
    plt.close()


def _plot_perturbation_cloud(
    trajectories,
    out_dir,
    idx,
    top_n_proteins=40,
    top_k_sites=6,
    draw_spaghetti=True,
    spaghetti_alpha=0.03,
):
    """
    For each selected protein, plot a 3-panel perturbation cloud:
      (1) Protein pred_fc
      (2) RNA pred_fc
      (3) Phospho: total phospho + top-k variable psites

    Outputs:
      out_dir/sensitivity_perturbations/cloud_<PROTEIN>.png
    """
    sim_dir = os.path.join(out_dir, "sensitivity_perturbations")
    os.makedirs(sim_dir, exist_ok=True)

    # -------------------------
    # 1) Concatenate all modalities
    # -------------------------
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

    prot_df = pd.concat(prot_all, ignore_index=True) if prot_all else pd.DataFrame(columns=["protein","time","pred_fc","sim_id"])
    rna_df  = pd.concat(rna_all,  ignore_index=True) if rna_all  else pd.DataFrame(columns=["protein","time","pred_fc","sim_id"])
    phos_df = pd.concat(phos_all, ignore_index=True) if phos_all else pd.DataFrame(columns=["protein","psite","time","pred_fc","sim_id"])

    # Ensure numeric
    for d in (prot_df, rna_df):
        if len(d) > 0:
            d["time"] = d["time"].astype(float)
            d["pred_fc"] = d["pred_fc"].astype(float)

    if len(phos_df) > 0:
        phos_df["time"] = phos_df["time"].astype(float)
        phos_df["pred_fc"] = phos_df["pred_fc"].astype(float)
        phos_df["psite"] = phos_df["psite"].astype(str)

    # -------------------------
    # 2) Choose proteins to plot (systematic, not arbitrary)
    #    Score = max std across time of median trajectory (protein modality preferred)
    # -------------------------
    def _protein_score(d, col="pred_fc"):
        if len(d) == 0:
            return pd.DataFrame(columns=["protein", "score"])
        # variability across sims, then aggregate across time
        g = d.groupby(["protein", "time"])[col].std().reset_index(name="std")
        s = g.groupby("protein")["std"].max().reset_index(name="score")
        return s

    s_prot = _protein_score(prot_df)
    s_rna  = _protein_score(rna_df)
    # prefer protein modality; fallback to RNA if protein missing
    scores = pd.concat([s_prot.assign(src="prot"), s_rna.assign(src="rna")], ignore_index=True)

    if len(scores) == 0:
        # last resort: use idx order
        proteins_to_plot = idx.proteins[:top_n_proteins]
    else:
        # For each protein keep best score across modalities
        scores = scores.sort_values("score", ascending=False).drop_duplicates("protein")
        proteins_to_plot = scores["protein"].head(top_n_proteins).tolist()

    # -------------------------
    # 3) Helpers: quantile bands + optional spaghetti
    # -------------------------
    def _summarize_band(d, group_cols):
        """
        Returns per-time summary: median, q05, q95, q01, q99.
        """
        if len(d) == 0:
            return None
        q = d.groupby(group_cols)["pred_fc"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).unstack()
        q = q.rename(columns={0.01:"q01", 0.05:"q05", 0.5:"med", 0.95:"q95", 0.99:"q99"}).reset_index()
        return q

    def _plot_band(ax, qdf, xcol="time", label=None):
        ax.fill_between(qdf[xcol], qdf["q99"], qdf["q01"], alpha=0.12, linewidth=0, label=None)
        ax.fill_between(qdf[xcol], qdf["q95"], qdf["q05"], alpha=0.18, linewidth=0, label=None)
        ax.plot(qdf[xcol], qdf["med"], linewidth=2, label=label)

    def _plot_spaghetti(ax, d, entity_cols):
        # Light lines per sim
        if not draw_spaghetti or len(d) == 0:
            return
        # no seaborn required; keep it fast and deterministic
        for sim_id, sub in d.groupby("sim_id"):
            sub = sub.sort_values("time")
            ax.plot(sub["time"].values, sub["pred_fc"].values, alpha=spaghetti_alpha, linewidth=1)

    # -------------------------
    # 4) Plot per protein: (RNA, Protein, Phospho total + top sites)
    # -------------------------
    for p in proteins_to_plot:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        axP, axR, axPH = axes  # Protein, RNA, Phospho

        # ---- Protein modality
        dP = prot_df[prot_df["protein"] == p].copy()
        if len(dP) > 0:
            _plot_spaghetti(axP, dP, ["protein"])
            qP = _summarize_band(dP, ["time"])
            _plot_band(axP, qP, label="Median (Protein)")
        axP.set_title(f"{p} — Protein cloud")
        axP.set_xlabel("Time (min)")
        axP.set_ylabel("Pred FC")
        axP.grid(True, alpha=0.25)

        # ---- RNA modality
        dR = rna_df[rna_df["protein"] == p].copy()
        if len(dR) > 0:
            _plot_spaghetti(axR, dR, ["protein"])
            qR = _summarize_band(dR, ["time"])
            _plot_band(axR, qR, label="Median (RNA)")
        axR.set_title(f"{p} — RNA cloud")
        axR.set_xlabel("Time (min)")
        axR.set_ylabel("Pred FC")
        axR.grid(True, alpha=0.25)

        # ---- Phospho modality: total phospho + top-k sites by variance
        dPH = phos_df[phos_df["protein"] == p].copy()
        if len(dPH) > 0:
            # Total phospho per sim/time: sum over psites
            tot = dPH.groupby(["sim_id", "time"], as_index=False)["pred_fc"].sum()
            if draw_spaghetti:
                for sim_id, sub in tot.groupby("sim_id"):
                    sub = sub.sort_values("time")
                    axPH.plot(sub["time"].values, sub["pred_fc"].values, alpha=spaghetti_alpha, linewidth=1)

            qT = _summarize_band(tot, ["time"])
            _plot_band(axPH, qT, label="Median (Total phospho)")

            # Pick top-k variable sites
            # Compute variability across sims per site (max std across time)
            site_var = (
                dPH.groupby(["psite", "time"])["pred_fc"].std()
                   .reset_index(name="std")
                   .groupby("psite")["std"].max()
                   .sort_values(ascending=False)
            )
            top_sites = site_var.head(top_k_sites).index.tolist()

            for s in top_sites:
                ds = dPH[dPH["psite"] == s]
                qs = _summarize_band(ds, ["time"])
                # line only (no extra fill, to avoid clutter)
                axPH.plot(qs["time"].values, qs["med"].values, linewidth=1.5, alpha=0.9, label=str(s))

        axPH.set_title(f"{p} — Phospho cloud")
        axPH.set_xlabel("Time (min)")
        axPH.set_ylabel("Pred FC (sum or site)")
        axPH.grid(True, alpha=0.25)

        # ---- Legend: avoid duplicates and keep it small
        # Only one legend per axis; Phospho legend can get big
        axP.legend(loc="best", fontsize=8, frameon=True, ncol=1)
        axR.legend(loc="best", fontsize=8, frameon=True, ncol=1)
        axPH.legend(loc="best", fontsize=7, frameon=True, ncol=1)

        plt.suptitle(f"Perturbation cloud: {p}", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(sim_dir, f"cloud_{p}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

    logger.info(f"[Sensitivity] Perturbation cloud plots saved to: {sim_dir} (n={len(proteins_to_plot)})")
