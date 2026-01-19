import os
import json
import re

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import correlate

from global_model.config import TIME_POINTS_PROTEIN
# Import core model modules
from global_model.network import Index, KinaseInput, System
from global_model.buildmat import build_W_parallel, build_tf_matrix
from global_model.simulate import simulate_and_measure
from global_model.utils import _normcols, _find_col
from config.config import setup_logger
logger = setup_logger()

# --- 1. THE DISCOVERY ENGINE ---

def run_mechanistic_discovery(sys, idx, df_tf, results_dir):
    """
    Analyzes the optimized system to extract 4 biological insights.
    """
    logger.info("🎬 Starting Mechanistic Discovery...")

    # A. PRE-SIMULATION: Required for Kinetic Lag and Saturation
    # We use a high-resolution time grid for lag detection
    time_grid = np.linspace(0, 120, 61)
    dfp, dfr, _ = simulate_and_measure(sys, idx, time_grid, time_grid, [])

    # ---------------------------------------------------------
    # INSIGHT 1: Signaling Refractory Period (Flash vs Stable)
    # ---------------------------------------------------------
    refractory_data = []
    for i, prot in enumerate(idx.proteins):
        site_start = idx.offset_s[i]
        n_sites = idx.n_sites[i]

        # S-Flux Reset (Dp_i) + State Reset (D_i)
        avg_dephos = np.mean(sys.Dp_i[site_start: site_start + n_sites]) if n_sites > 0 else 0
        deactivation = sys.D_i[i]
        # Protein Stability
        degradation = sys.B_i[i]

        flash_index = (avg_dephos + deactivation) / (degradation + 1e-9)
        refractory_data.append({
            'Protein': prot,
            'Signal_Reset': avg_dephos + deactivation,
            'Protein_Stability': degradation,
            'Flash_Index': flash_index
        })
    df_refractory = pd.DataFrame(refractory_data).sort_values('Flash_Index', ascending=False)

    # ---------------------------------------------------------
    # INSIGHT 2: Kinetic Lag (Translational Bottleneck)
    # ---------------------------------------------------------
    lag_data = []
    for p in idx.proteins:
        sub_r = dfr[dfr['protein'] == p].sort_values('time')
        sub_p = dfp[dfp['protein'] == p].sort_values('time')

        if sub_r['pred_fc'].max() > 1.1:  # Only analyze responders
            rna_sig = sub_r['pred_fc'].values - 1
            prot_sig = sub_p['pred_fc'].values - 1

            # Cross-correlation to find lag
            correlation = correlate(prot_sig, rna_sig)
            lags = np.arange(-len(time_grid) + 1, len(time_grid))
            lag_idx = np.argmax(correlation)
            lag_min = lags[lag_idx] * (time_grid[1] - time_grid[0])

            lag_data.append({
                'Protein': p,
                'Lag_Minutes': max(0, lag_min),
                'RNA_Peak': sub_r['pred_fc'].max(),
                'Prot_Peak': sub_p['pred_fc'].max()
            })
    df_lag = pd.DataFrame(lag_data).sort_values('Lag_Minutes', ascending=False)

    # ---------------------------------------------------------
    # INSIGHT 3: Transcriptional Saturation (Digital Switches)
    # ---------------------------------------------------------
    sat_data = []
    for i, prot in enumerate(idx.proteins):
        alpha = sys.E_i[i]  # Transcriptional Efficacy
        max_rna = dfr[dfr['protein'] == prot]['pred_fc'].max()

        # High Alpha + Low Output = Saturation
        sat_index = alpha / (max_rna + 1e-9)
        sat_data.append({
            'Protein': prot,
            'TF_Efficacy_Alpha': alpha,
            'Max_mRNA_FC': max_rna,
            'Saturation_Index': sat_index
        })
    df_saturation = pd.DataFrame(sat_data).sort_values('Saturation_Index', ascending=False)

    # ---------------------------------------------------------
    # INSIGHT 4: Feedback Gain (The Revolving Door)
    # ---------------------------------------------------------
    loop_data = []
    W_csc = sys.W_global.tocsc()
    for _, row in df_tf.iterrows():
        tf, target = row['tf'], row['target']
        # Check if TF targets a Kinase
        if target in idx.kinases:
            # TF -> Kinase Efficacy (Alpha)
            alpha = sys.E_i[idx.p2i[target]]

            # Check if that Kinase phosphorylates the TF back (Signaling Beta)
            k_idx = idx.k2i[target]
            tf_idx = idx.p2i[tf]
            s_start = idx.offset_s[tf_idx]
            s_end = s_start + idx.n_sites[tf_idx]

            # Sum of weights from this Kinase to the TF's sites
            beta = np.sum(W_csc[s_start:s_end, k_idx].toarray())

            if beta > 0:
                loop_data.append({
                    'TF': tf,
                    'Kinase': target,
                    'Loop_Gain': alpha * beta,
                    'Efficacy_Alpha': alpha,
                    'Signaling_Beta': beta
                })

    # Fix: Handle empty loop_data
    if loop_data:
        df_loops = pd.DataFrame(loop_data).sort_values('Loop_Gain', ascending=False)
    else:
        # Create an empty dataframe with the expected columns so the Excel writer doesn't fail
        df_loops = pd.DataFrame(columns=['TF', 'Kinase', 'Loop_Gain', 'Efficacy_Alpha', 'Signaling_Beta'])
        print("ℹ️ No feedback loops (TF -> Kinase -> TF) detected in the current network.")

    # --- SAVE REPORT ---
    with pd.ExcelWriter(Path(results_dir) / "Mechanistic_Discovery_Report.xlsx") as writer:
        df_refractory.to_excel(writer, sheet_name="1_Refractory_Period")
        df_lag.to_excel(writer, sheet_name="2_Kinetic_Lag")
        df_saturation.to_excel(writer, sheet_name="3_Saturation")
        df_loops.to_excel(writer, sheet_name="4_Feedback_Gain")

    logger.info(f"✅ Report saved to {results_dir}/Mechanistic_Discovery_Report.xlsx")


# --- 2. THE MAIN LOADER ---

if __name__ == "__main__":
    RES_DIR = "./results_global_combinatorial"

    # Replicate your specific loading logic
    with open(Path(RES_DIR) / "fitted_params_picked.json", "r") as f:
        best_params = json.load(f)

        # =========================================================================
        # 1. Load and Clean Kinase Network
        # =========================================================================
        logger.info(f"[Data] Loading Kinase Net: input2.csv")
        df_kin_raw = _normcols(pd.read_csv("./data/input2.csv"))

        # Map topology columns
        pcol = _find_col(df_kin_raw, ["geneid", "protein", "gene"])
        scol = _find_col(df_kin_raw, ["psite", "site"])
        kcol = _find_col(df_kin_raw, ["kinase", "k"])

        # Expand kinase sets {K1, K2} -> rows and Normalize
        rows = []
        for _, r in df_kin_raw.iterrows():
            ks = str(r[kcol]).strip('{}').split(',')
            for k in ks:
                k = k.strip()
                if k:
                    prot = str(r[pcol]).strip().upper()
                    site = str(r[scol]).strip()
                    kin = k.strip().upper()
                    rows.append((prot, site, kin))

        df_kin_clean = pd.DataFrame(rows, columns=["protein", "psite", "kinase"]).drop_duplicates()

        # =========================================================================
        # 2. Load and Clean TF Network
        # =========================================================================
        logger.info(f"[Data] Loading TF Net: input4.csv")
        df_tf_raw = _normcols(pd.read_csv("./data/input4.csv"))

        tf_scol = _find_col(df_tf_raw, ["source", "tf"])
        tf_tcol = _find_col(df_tf_raw, ["target", "gene"])

        # Normalize strings (upper case + strip)
        df_tf_clean = pd.DataFrame({
            "tf": df_tf_raw[tf_scol].astype(str).str.strip().str.upper(),
            "target": df_tf_raw[tf_tcol].astype(str).str.strip().str.upper()
        }).drop_duplicates()

        # =========================================================================
        # 3. Load MS Data for Kinase Activity Input
        # =========================================================================
        # 1. Load and normalize columns
        df_ms_raw = pd.read_csv("./data/input1.csv")
        df_ms_raw = _normcols(df_ms_raw)  # Standardize to lowercase/clean names

        # 2. Identify the core columns
        gcol = _find_col(df_ms_raw, ["geneid", "protein"])
        scol = _find_col(df_ms_raw, ["psite", "site"])

        # 3. Identify time columns (x1, x2, ...)
        xcols = sorted([c for c in df_ms_raw.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))

        # 4. Melt from wide to tidy format
        df_tidy = df_ms_raw[[gcol, scol] + xcols].melt(
            id_vars=[gcol, scol],
            var_name="xcol",
            value_name="fc"
        )

        # 5. Map 'x' indices to actual TIME_POINTS_PROTEIN values
        x_idx = df_tidy["xcol"].str.replace("x", "", regex=False).astype(int) - 1
        df_tidy["time"] = np.array(TIME_POINTS_PROTEIN)[x_idx.to_numpy()]
        df_tidy["protein"] = df_tidy[gcol].astype(str).str.strip().str.upper()

        # 6. Filter for protein-level data only (Kinase activity is based on total protein)
        # We look for empty or NaN psites
        df_ms_clean = df_tidy[df_tidy[scol].isna() | (df_tidy[scol].astype(str).str.strip() == "")].copy()

        # Ensure fc is numeric
        df_ms_clean["fc"] = pd.to_numeric(df_ms_clean["fc"], errors="coerce")
        # =========================================================================
        # 4. Reconstruct System Objects
        # =========================================================================
        # 1. Initialize Model Index
        idx = Index(df_kin_clean, tf_interactions=df_tf_clean)

        # 2. Build Matrices (W for Signaling, TF for Transcription)
        W_global = build_W_parallel(df_kin_clean, idx, n_cores=-1)
        tf_mat = build_tf_matrix(df_tf_clean, idx)

        # 3. Setup Kinase Input (Driving force of the ODE)
        kin_in = KinaseInput(idx.kinases, df_ms_clean)

        # 4. Calculate TF degree normalization (Absolute sum of TF impacts per gene)
        tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
        tf_deg[tf_deg < 1e-12] = 1.0

        # 5. Build the Final System
        # Passing the loaded 'best_params' (from JSON) as the system defaults
        sys = System(idx, W_global, tf_mat, kin_in, best_params, tf_deg)

        # =========================================================================
        # 5. RUN DISCOVERY
        # =========================================================================
        logger.info("🚀 System Reconstructed. Running Mechanistic Discovery Analysis...")
        run_mechanistic_discovery(sys, idx, df_tf_clean, RES_DIR)