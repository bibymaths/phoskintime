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

# Import Global Model components
from global_model.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO
from global_model.simulate import simulate_and_measure
from global_model.params import unpack_params
from config.config import setup_logger

logger = setup_logger()

# --- Configuration Constants ---
NUM_TRAJECTORIES = 120  # Total Morris trajectories (N)
NUM_LEVELS = 4  # Number of grid levels for Morris (p)
PERTURBATION = 0.2  # +/- 20% bounds around optimal
TOP_K_PLOTS = 50  # Number of best trajectories to plot
METRIC_TYPES = ["total_signal", "mean", "variance", "l2_norm"]


def compute_bounds(params_dict, perturbation=PERTURBATION):
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
    logger.info(f"[Sensitivity] Starting Morris Analysis (N={NUM_TRAJECTORIES}, p={NUM_LEVELS})...")
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
    param_values = morris.sample(problem, N=NUM_TRAJECTORIES, num_levels=NUM_LEVELS)
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
                    "rna_df": dfr[["gene", "time", "pred_fc"]].copy() if dfr is not None else None,
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


def _plot_perturbation_cloud(trajectories, out_dir, idx):
    """
    Plots the 'cloud' of protein trajectories generated by the sensitivity scan.
    This shows the robustness of the model solution.
    """
    # Combine all stored protein frames
    all_dfs = []
    for t in trajectories:
        d = t["prot_df"]
        d["sim_id"] = t["id"]
        all_dfs.append(d)

    if not all_dfs:
        return

    big_df = pd.concat(all_dfs)

    # Pick a few key proteins to plot (e.g. Kinases or Hubs)
    # Plotting all 200 is messy. Let's pick the top 6 by variance in the cloud.

    # Calculate variance per protein across simulations
    stats = big_df.groupby("protein")["pred_fc"].std().reset_index().sort_values("pred_fc", ascending=False)
    top_prots = stats.head(6)["protein"].tolist()

    sim_dir = os.path.join(out_dir, "sensitivity_perturbations")
    os.makedirs(sim_dir, exist_ok=True)

    for p in top_prots:
        sub = big_df[big_df["protein"] == p]

        plt.figure(figsize=(8, 6))
        # Plot individual lines lightly
        sns.lineplot(
            data=sub, x="time", y="pred_fc", units="sim_id", estimator=None,
            color="gray", alpha=0.1, linewidth=1
        )
        # Plot mean trend
        sns.lineplot(
            data=sub, x="time", y="pred_fc", color="red", linewidth=2, label="Mean Response"
        )

        plt.title(f"Parameter Sensitivity Cloud: {p}")
        plt.xlabel("Time (min)")
        plt.ylabel("Fold Change")
        plt.legend()
        plt.savefig(os.path.join(sim_dir, f"cloud_{p}.png"), dpi=200)
        plt.close()