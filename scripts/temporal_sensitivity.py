#!/usr/bin/env python

"""
Temporal Sensitivity Analysis Script

This script performs global sensitivity analysis (GSA) on a mechanistic model to identify the most influential parameters.
It uses the SALib library for Sobol' indices and Saltelli sampling to quantify parameter importance.

Usage: python temporal_sensitivity.py --results-dir results --samples 128

License: BSD-3-Clause
Author: Abhinav Mishra
"""

import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path

from SALib.analyze import sobol
from SALib.sample import saltelli
import plotly.express as px

from global_model.config import RNA_DATA_FILE, KINASE_NET_FILE, TF_NET_FILE, MS_DATA_FILE, TIME_POINTS_PROTEIN
from global_model.network import Index, KinaseInput, System
from global_model.buildmat import build_W_parallel, build_tf_matrix
from global_model.simulate import simulate_and_measure
from global_model.utils import normcols, find_col

from config.config import setup_logger

logger = setup_logger()


# --- 1. DATA LOADING HELPER ---
def setup_gsa_system(results_dir, rna_file, kin_net, tf_net, ms_data):
    """
    Reconstructs the System object using the saved parameters and original input files.
    """
    params_path = Path(results_dir) / "fitted_params_picked.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find fitted parameters at: {params_path}")

    # 1. Load optimized parameters
    with open(params_path, "r") as f:
        best_params_dict = json.load(f)

    # 2. Replicate Kinase Net Loading (Expansion + Normalization)
    df_k_raw = normcols(pd.read_csv(kin_net))
    pcol = find_col(df_k_raw, ["geneid", "protein", "gene"])
    scol = find_col(df_k_raw, ["psite", "site"])
    kcol = find_col(df_k_raw, ["kinase", "k"])

    rows = []
    for _, r in df_k_raw.iterrows():
        ks = str(r[kcol]).strip('{}').split(',')
        for k in ks:
            k = k.strip()
            if k:
                rows.append((str(r[pcol]).strip().upper(),
                             str(r[scol]).strip(),
                             k.strip().upper()))
    df_kin_clean = pd.DataFrame(rows, columns=["protein", "psite", "kinase"]).drop_duplicates()

    # 3. Replicate TF Net Loading (Normalization)
    df_t_raw = normcols(pd.read_csv(tf_net))
    tf_scol = find_col(df_t_raw, ["source", "tf"])
    tf_tcol = find_col(df_t_raw, ["target", "gene"])
    df_tf_clean = pd.DataFrame({
        "tf": df_t_raw[tf_scol].astype(str).str.strip().str.upper(),
        "target": df_t_raw[tf_tcol].astype(str).str.strip().str.upper()
    }).drop_duplicates()

    # 4. Replicate MS Data Loading (Normalization for KinaseInput)
    df_ms_raw = normcols(pd.read_csv(ms_data))
    gcol = find_col(df_ms_raw, ["geneid", "protein"])
    scol_ms = find_col(df_ms_raw, ["psite", "site"])

    # Identify time columns x1, x2...
    xcols = sorted([c for c in df_ms_raw.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))
    tidy = df_ms_raw[[gcol, scol_ms] + xcols].melt(id_vars=[gcol, scol_ms], var_name="xcol", value_name="fc")
    tidy["protein"] = tidy[gcol].astype(str).str.strip().str.upper()
    x_idx = tidy["xcol"].str.replace("x", "", regex=False).astype(int) - 1
    tidy["time"] = np.array(TIME_POINTS_PROTEIN)[x_idx.to_numpy()]

    # KinaseInput needs the protein-level tidy data
    df_ms_clean = tidy[tidy[scol_ms].isna() | (tidy[scol_ms] == "")].copy()

    # 5. Initialize Model Objects
    idx = Index(df_kin_clean, tf_interactions=df_tf_clean)
    W_global = build_W_parallel(df_kin_clean, idx, n_cores=-1)
    tf_mat = build_tf_matrix(df_tf_clean, idx)
    kin_in = KinaseInput(idx.kinases, df_ms_clean)

    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    sys = System(idx, W_global, tf_mat, kin_in, best_params_dict, tf_deg)

    return sys, idx, df_tf_clean


# --- 2. WORKER & ANALYSIS ---
def temporal_gsa_worker(args):
    """
    Runs a single simulation for a specific parameter set (Saltelli sample).
    """
    sys, idx, x_values, param_names, target_protein, time_grid = args

    # Copy current parameters to modify them
    new_ck = np.array(sys.c_k)
    new_Ei = np.array(sys.E_i)

    # Map the sampled values (x_values) back to the system parameters
    for val, name in zip(x_values, param_names):
        if name.startswith("ck_"):
            new_ck[idx.k2i[name.replace("ck_", "")]] = val
        elif name.startswith("Ei_"):
            new_Ei[idx.p2i[name.replace("Ei_", "")]] = val

    # Update system with new perturbed parameters
    sys.update(c_k=new_ck, E_i=new_Ei, A_i=sys.A_i, B_i=sys.B_i,
               C_i=sys.C_i, D_i=sys.D_i, Dp_i=sys.Dp_i, tf_scale=sys.tf_scale)

    # Run simulation
    dfp, _, _ = simulate_and_measure(sys, idx, time_grid, [], [])

    # Extract trajectory for the target protein
    if dfp is not None and not dfp.empty:
        # Ensure we get the values sorted by time
        res = dfp[dfp['protein'] == target_protein].sort_values('time')['pred_fc'].values
        if len(res) == len(time_grid):
            return res

    # Fallback if simulation fails or protein missing
    return np.zeros(len(time_grid))


def run_full_gsa(sys, idx, df_tf, target_protein, n_samples=128):
    """
    Performs Sobol sensitivity analysis for a single target protein.
    """
    time_grid = [0.0, 5.0, 15.0, 30.0, 60.0, 90.0, 120.0]
    param_names = []
    bounds = []

    # 1. Define Problem (Parameters to vary)
    # Vary Kinase Activities (c_k)
    for k in idx.kinases:
        val = sys.c_k[idx.k2i[k]]
        param_names.append(f"ck_{k}")
        bounds.append([val * 0.5, val * 1.5])  # +/- 50%

    # Vary TF Efficacies (E_i) - Only for proteins that act as TFs
    for p in idx.proteins:
        if p in df_tf['tf'].values:
            val = sys.E_i[idx.p2i[p]]
            param_names.append(f"Ei_{p}")
            bounds.append([val * 0.5, val * 1.5])  # +/- 50%

    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}

    # 2. Generate Samples
    # Note: calc_second_order=False saves computation time (N * (D + 2))
    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

    # 3. Run Simulations in Parallel
    tasks = [(sys, idx, x, param_names, target_protein, time_grid) for x in param_values]

    # Determine safe core count
    n_cores = max(1, os.cpu_count() - 1)

    with mp.Pool(processes=n_cores) as pool:
        Y_trajectories = np.array(pool.map(temporal_gsa_worker, tasks))

    # 4. Analyze Sensitivity per Time Point
    temporal_results = []
    for t_idx, t in enumerate(time_grid):
        # Y_trajectories shape: (n_samples, n_time_points)
        # We analyze the column corresponding to the current time point
        Si = sobol.analyze(problem, Y_trajectories[:, t_idx], calc_second_order=False)
        temporal_results.append({'time': t, 'ST': Si['ST'], 'names': param_names})

    return temporal_results, problem


# --- 3. PLOTTING ---
def plot_temporal_heatmap(temporal_results, target_protein, output_dir):
    """
    Generates and saves a heatmap of Total Sensitivity Indices (ST).
    """
    data = []
    for entry in temporal_results:
        t = entry['time']
        for name, st_val in zip(entry['names'], entry['ST']):
            data.append({'Time (min)': t, 'Parameter': name, 'ST': max(0, st_val)})

    df = pd.DataFrame(data)

    # Filter for top 20 most influential parameters to keep plot readable
    top_params = df.groupby('Parameter')['ST'].max().nlargest(20).index
    df_pivot = df[df['Parameter'].isin(top_params)].pivot(index='Parameter', columns='Time (min)', values='ST')

    fig = px.imshow(
        df_pivot,
        color_continuous_scale='Viridis',
        aspect="auto",
        title=f"Temporal Sensitivity (ST) Drivers for {target_protein}",
        labels=dict(x="Time (min)", y="Parameter", color="Total Sensitivity")
    )

    filename = output_dir / f"gsa_{target_protein}.html"
    fig.write_html(str(filename))
    return filename


def main():
    parser = argparse.ArgumentParser(description="Run Global Sensitivity Analysis (GSA) for all proteins.")
    parser.add_argument("--results-dir", required=True, help="Directory containing fitted model parameters.")
    parser.add_argument("--samples", type=int, default=128,
                        help="Number of samples for Saltelli sequence (default: 128).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return

    # 1. Setup Output Directory
    gsa_dir = results_dir / "global_sensitivity_analysis"
    gsa_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created at: {gsa_dir}")

    # 2. Reconstruct System
    logger.info("Reconstructing system from parameters...")
    try:
        sys_obj, idx, df_tf_clean = setup_gsa_system(
            results_dir=results_dir,
            rna_file=RNA_DATA_FILE,
            kin_net=KINASE_NET_FILE,
            tf_net=TF_NET_FILE,
            ms_data=MS_DATA_FILE
        )
    except Exception as e:
        logger.error(f"Failed to setup system: {e}")
        return

    proteins = idx.proteins
    total_proteins = len(proteins)
    logger.info(f"Starting GSA for {total_proteins} proteins. Samples per protein: {args.samples}")

    # 3. Run GSA Sequentially for all Proteins
    for i, protein in enumerate(proteins):
        logger.info(f"[{i + 1}/{total_proteins}] Running GSA for {protein}...")

        try:
            results, _ = run_full_gsa(
                sys=sys_obj,
                idx=idx,
                df_tf=df_tf_clean,
                target_protein=protein,
                n_samples=args.samples
            )

            saved_path = plot_temporal_heatmap(results, protein, gsa_dir)
            logger.info(f"    Saved plot to {saved_path.name}")

        except Exception as e:
            logger.error(f"    Failed GSA for {protein}: {e}")

    logger.info("✅ Global Sensitivity Analysis complete for all proteins.")


if __name__ == "__main__":
    # Ensure safe multiprocessing on different OSs
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    main()
