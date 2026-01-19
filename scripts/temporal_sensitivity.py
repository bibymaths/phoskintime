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

# Core Model Imports
from global_model.config import RNA_DATA_FILE, KINASE_NET_FILE, TF_NET_FILE, MS_DATA_FILE, TIME_POINTS_PROTEIN
from global_model.network import Index, KinaseInput, System
from global_model.buildmat import build_W_parallel, build_tf_matrix
from global_model.simulate import simulate_and_measure
from global_model.utils import _normcols, _find_col

from config.config import setup_logger

logger = setup_logger()


# --- 1. DATA LOADING HELPER (REPLICATING YOUR MODULE LOGIC) ---
def setup_gsa_system(results_dir, rna_file, kin_net, tf_net, ms_data):
    # 1. Load optimized parameters
    with open(Path(results_dir) / "fitted_params_picked.json", "r") as f:
        best_params_dict = json.load(f)

    # 2. Replicate Kinase Net Loading (Expansion + Normalization)
    df_k_raw = _normcols(pd.read_csv(kin_net))
    pcol = _find_col(df_k_raw, ["geneid", "protein", "gene"])
    scol = _find_col(df_k_raw, ["psite", "site"])
    kcol = _find_col(df_k_raw, ["kinase", "k"])

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
    df_t_raw = _normcols(pd.read_csv(tf_net))
    tf_scol = _find_col(df_t_raw, ["source", "tf"])
    tf_tcol = _find_col(df_t_raw, ["target", "gene"])
    df_tf_clean = pd.DataFrame({
        "tf": df_t_raw[tf_scol].astype(str).str.strip().str.upper(),
        "target": df_t_raw[tf_tcol].astype(str).str.strip().str.upper()
    }).drop_duplicates()

    # 4. Replicate MS Data Loading (Normalization for KinaseInput)
    df_ms_raw = _normcols(pd.read_csv(ms_data))
    gcol = _find_col(df_ms_raw, ["geneid", "protein"])
    scol_ms = _find_col(df_ms_raw, ["psite", "site"])

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
    sys, idx, x_values, param_names, target_protein, time_grid = args
    new_ck = np.array(sys.c_k)
    new_Ei = np.array(sys.E_i)

    for val, name in zip(x_values, param_names):
        if name.startswith("ck_"):
            new_ck[idx.k2i[name.replace("ck_", "")]] = val
        elif name.startswith("Ei_"):
            new_Ei[idx.p2i[name.replace("Ei_", "")]] = val

    sys.update(c_k=new_ck, E_i=new_Ei, A_i=sys.A_i, B_i=sys.B_i,
               C_i=sys.C_i, D_i=sys.D_i, Dp_i=sys.Dp_i, tf_scale=sys.tf_scale)

    dfp, _, _ = simulate_and_measure(sys, idx, time_grid, [], [])
    if dfp is not None and not dfp.empty:
        res = dfp[dfp['protein'] == target_protein].sort_values('time')['pred_fc'].values
        if len(res) == len(time_grid):
            return res
    return np.zeros(len(time_grid))


def run_full_gsa(sys, idx, df_tf, target_protein, n_samples=128):
    time_grid = [0.0, 5.0, 15.0, 30.0, 60.0, 90.0, 120.0]
    param_names = []
    bounds = []

    for k in idx.kinases:
        val = sys.c_k[idx.k2i[k]]
        param_names.append(f"ck_{k}")
        bounds.append([val * 0.5, val * 1.5])

    # Note: df_tf here is the cleaned version with 'tf' and 'target' columns
    for p in idx.proteins:
        if p in df_tf['tf'].values:
            val = sys.E_i[idx.p2i[p]]
            param_names.append(f"Ei_{p}")
            bounds.append([val * 0.5, val * 1.5])

    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    param_values = saltelli.sample(problem, n_samples)

    tasks = [(sys, idx, x, param_names, target_protein, time_grid) for x in param_values]
    with mp.Pool(processes=os.cpu_count()) as pool:
        Y_trajectories = np.array(pool.map(temporal_gsa_worker, tasks))

    temporal_results = []
    for t_idx, t in enumerate(time_grid):
        Si = sobol.analyze(problem, Y_trajectories[:, t_idx])
        temporal_results.append({'time': t, 'ST': Si['ST'], 'names': param_names})
    return temporal_results, problem


# --- 3. PLOTTING ---
def plot_temporal_heatmap(temporal_results, target_protein):
    data = []
    for entry in temporal_results:
        t = entry['time']
        for name, st_val in zip(entry['names'], entry['ST']):
            data.append({'Time (min)': t, 'Parameter': name, 'ST': max(0, st_val)})

    df = pd.DataFrame(data)
    top_params = df.groupby('Parameter')['ST'].max().nlargest(20).index
    df_pivot = df[df['Parameter'].isin(top_params)].pivot(index='Parameter', columns='Time (min)', values='ST')

    fig = px.imshow(df_pivot, color_continuous_scale='Viridis', aspect="auto",
                    title=f"Temporal Sensitivity (ST) Drivers for {target_protein}")
    fig.write_html(f"gsa_{target_protein}.html")
    return fig


if __name__ == "__main__":
    BASE_DIR = "./"
    RES_DIR = Path(BASE_DIR) / "results_global_distributive"

    # Reconstruct system with identical module logic
    sys, idx, df_tf_clean = setup_gsa_system(
        results_dir=RES_DIR,
        rna_file=RNA_DATA_FILE,
        kin_net=KINASE_NET_FILE,
        tf_net=TF_NET_FILE,
        ms_data=MS_DATA_FILE
    )

    TARGET_PROTEIN = "EGFR"
    results, problem = run_full_gsa(sys, idx, df_tf_clean, target_protein=TARGET_PROTEIN)
    fig = plot_temporal_heatmap(results, TARGET_PROTEIN)
    fig.show()