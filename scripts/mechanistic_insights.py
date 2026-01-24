#!/usr/bin/env python
"""
Mechanistic Discovery Script

This script analyzes the optimized model to extract 4 biological insights:
1. Refractory Period (Flash vs Stable signaling).
2. Kinetic Lag (Time delay between protein signal and RNA response).
3. Transcriptional Saturation (Digital switching behavior).
4. Feedback Gain (Revolving door loops).

Usage: python mechanistic_insights.py --kinase-net input2.csv --tf-net input4.csv --ms input1.csv --rna input3.csv --phospho input5.csv --kinopt fitted_params_picked.json --tfopt fitted_params_picked.json --output-dir output

License: BSD-3-Clause
Author: Abhinav Mishra
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate

from global_model.config import (
    TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO,
    RESULTS_DIR, KINASE_NET_FILE, TF_NET_FILE, MS_DATA_FILE,
    RNA_DATA_FILE, PHOSPHO_DATA_FILE, KINOPT_RESULTS_FILE, TFOPT_RESULTS_FILE, CORES
)
from global_model.network import Index, KinaseInput, System
from global_model.buildmat import build_W_parallel, build_tf_matrix
from global_model.simulate import simulate_and_measure
from global_model.io import load_data
from global_model.params import unpack_params
from config.config import setup_logger

logger = setup_logger()


# --- 1. THE DISCOVERY ENGINE ---

def run_mechanistic_discovery(ode_sys: System, idx: Index, df_tf: pd.DataFrame, results_dir: str):
    """
    Analyzes the optimized system to extract 4 biological insights.

    1. Refractory Period (Flash vs Stable signaling).
    2. Kinetic Lag (Time delay between protein signal and RNA response).
    3. Transcriptional Saturation (Digital switching behavior).
    4. Feedback Gain (Revolving door loops).

    Args:
        ode_sys (System): The fully constructed and parameterized ODE system.
        idx (Index): The network topology index.
        df_tf (pd.DataFrame): The cleaned TF network dataframe.
        results_dir (str): Directory to save the output Excel report.
    """
    logger.info("🎬 Starting Mechanistic Discovery...")

    # A. PRE-SIMULATION: Required for Kinetic Lag and Saturation
    # We use a high-resolution time grid for lag detection (0 to 120 min)
    time_grid = np.linspace(0, 120, 61)
    # Simulate the system using the high-res grid
    dfp, dfr, _ = simulate_and_measure(ode_sys, idx, time_grid, time_grid, [])

    # ---------------------------------------------------------
    # INSIGHT 1: Signaling Refractory Period (Flash vs Stable)
    # ---------------------------------------------------------
    # Identifies proteins that signal briefly ("flash") vs those that sustain signal.
    refractory_data = []
    for i, prot in enumerate(idx.proteins):
        site_start = idx.offset_s[i]
        n_sites = idx.n_sites[i]

        # Calculate average dephosphorylation rate for the protein's sites
        avg_dephos = np.mean(ode_sys.Dp_i[site_start: site_start + n_sites]) if n_sites > 0 else 0

        # Deactivation (State D_i) and Degradation (B_i/Stability)
        deactivation = ode_sys.D_i[i]
        degradation = ode_sys.B_i[
            i]  # Note: In some model versions B_i is RNA deg, check model definition. Assuming Protein degradation context here.

        # Flash Index: Ratio of "Turn Off" speed to "Disappear" speed
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
    # Calculates the time delay between peak protein activation and peak RNA response.
    lag_data = []
    for p in idx.proteins:
        sub_r = dfr[dfr['protein'] == p].sort_values('time')
        sub_p = dfp[dfp['protein'] == p].sort_values('time')

        # Only analyze responders (max fold change > 1.1) to avoid noise
        if sub_r['pred_fc'].max() > 1.1:
            rna_sig = sub_r['pred_fc'].values - 1
            prot_sig = sub_p['pred_fc'].values - 1

            # Cross-correlation to find the time shift (lag)
            correlation = correlate(prot_sig, rna_sig)
            lags = np.arange(-len(time_grid) + 1, len(time_grid))
            lag_idx = np.argmax(correlation)
            lag_min = lags[lag_idx] * (time_grid[1] - time_grid[0])

            lag_data.append({
                'Protein': p,
                'Lag_Minutes': max(0, lag_min),  # Lag cannot be negative in this causal model
                'RNA_Peak': sub_r['pred_fc'].max(),
                'Prot_Peak': sub_p['pred_fc'].max()
            })
    df_lag = pd.DataFrame(lag_data).sort_values('Lag_Minutes', ascending=False)

    # ---------------------------------------------------------
    # INSIGHT 3: Transcriptional Saturation (Digital Switches)
    # ---------------------------------------------------------
    # Identifies genes where Transcription Factor efficacy (Alpha) is high,
    # but output is capped, indicating a switch-like or saturated response.
    sat_data = []
    for i, prot in enumerate(idx.proteins):
        alpha = ode_sys.E_i[i]  # Transcriptional Efficacy / Initial Condition Multiplier
        max_rna = dfr[dfr['protein'] == prot]['pred_fc'].max()

        # Saturation Index
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
    # Identifies loops where a TF regulates a Kinase, which in turn phosphorylates that TF.
    loop_data = []
    W_csc = ode_sys.W_global.tocsc()

    # Iterate through TF network interactions
    for _, row in df_tf.iterrows():
        tf, target = row['tf'], row['target']

        # Check if the Target Gene is actually a Kinase in our model
        if target in idx.kinases:
            # Alpha: Strength of TF -> Kinase transcription
            # Note: Need target index in protein list
            if target in idx.p2i:
                alpha = ode_sys.E_i[idx.p2i[target]]
            else:
                continue

            # Beta: Strength of Kinase -> TF phosphorylation (Signaling feedback)
            # Check if Kinase exists in kinase index and TF exists in protein index
            if target in idx.k2i and tf in idx.p2i:
                k_idx = idx.k2i[target]
                tf_idx = idx.p2i[tf]
                s_start = idx.offset_s[tf_idx]
                s_end = s_start + idx.n_sites[tf_idx]

                # Sum of weights from this Kinase to all sites on the TF
                # W_global is (Sites x Kinases)
                beta = np.sum(W_csc[s_start:s_end, k_idx].toarray())

                if beta > 0:
                    loop_data.append({
                        'TF': tf,
                        'Kinase': target,
                        'Loop_Gain': alpha * beta,
                        'Efficacy_Alpha': alpha,
                        'Signaling_Beta': beta
                    })

    if loop_data:
        df_loops = pd.DataFrame(loop_data).sort_values('Loop_Gain', ascending=False)
    else:
        df_loops = pd.DataFrame(columns=['TF', 'Kinase', 'Loop_Gain', 'Efficacy_Alpha', 'Signaling_Beta'])
        logger.info("ℹ️ No feedback loops (TF -> Kinase -> TF) detected in the current network.")

    # --- SAVE REPORT ---
    out_path = Path(results_dir) / "Mechanistic_Discovery_Report.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        df_refractory.to_excel(writer, sheet_name="1_Refractory_Period")
        df_lag.to_excel(writer, sheet_name="2_Kinetic_Lag")
        df_saturation.to_excel(writer, sheet_name="3_Saturation")
        df_loops.to_excel(writer, sheet_name="4_Feedback_Gain")

    logger.info(f"✅ Report saved to {out_path}")


# --- 2. THE MAIN LOADER (Logic from runner.py) ---

def main():
    parser = argparse.ArgumentParser(description="Run Mechanistic Discovery on a fitted model.")

    # Input File Arguments (Defaulting to config constants)
    parser.add_argument("--kinase-net", default=KINASE_NET_FILE, help="Path to input2.csv")
    parser.add_argument("--tf-net", default=TF_NET_FILE, help="Path to input4.csv")
    parser.add_argument("--ms", default=MS_DATA_FILE, help="Path to input1.csv (MS data)")
    parser.add_argument("--rna", default=RNA_DATA_FILE, help="Path to input3.csv (RNA data)")
    parser.add_argument("--phospho", default=PHOSPHO_DATA_FILE, help="Path to input5.csv (Phospho data)")
    parser.add_argument("--kinopt", default=KINOPT_RESULTS_FILE)
    parser.add_argument("--tfopt", default=TFOPT_RESULTS_FILE)

    # Optimization Result Arguments
    parser.add_argument("--output-dir", required=True, help="Directory containing fitted_params_picked.json")
    parser.add_argument("--cores", type=int, default=CORES)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        logger.error(f"Output directory not found: {out_dir}")
        sys.exit(1)

    params_file = out_dir / "fitted_params_picked.json"
    if not params_file.exists():
        logger.error(f"Fitted parameters not found at: {params_file}")
        sys.exit(1)

    logger.info("============================================================")
    logger.info("Mechanistic Discovery Analysis")
    logger.info("============================================================")

    # -------------------------------------------------------------------------
    # 1. Load Data (Standardized Logic)
    # -------------------------------------------------------------------------
    df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(args)

    # -------------------------------------------------------------------------
    # 2. Mechanistic Phospho Filter (Logic from runner.py)
    # -------------------------------------------------------------------------
    # Ensure strict matching between phospho-data sites and kinase-network sites
    for _df in (df_kin, df_pho):
        _df["protein"] = _df["protein"].astype(str).str.strip()
    df_kin["psite"] = df_kin["psite"].astype(str).str.strip()
    df_pho["psite"] = df_pho["psite"].astype(str).str.strip()

    kin_site_pairs = set(zip(df_kin["protein"].values, df_kin["psite"].values))

    n_before = len(df_pho)
    pairs = list(zip(df_pho["protein"].values, df_pho["psite"].values))
    keep = np.fromiter(((p, s) in kin_site_pairs for (p, s) in pairs), dtype=bool, count=len(pairs))
    df_pho = df_pho.loc[keep].copy()

    logger.info(f"[Phospho] Mechanistic site filter: {n_before} → {len(df_pho)} (dropped {n_before - len(df_pho)})")

    # -------------------------------------------------------------------------
    # 3. Sophisticated TF Handling & Proxying (Logic from runner.py)
    # -------------------------------------------------------------------------
    if df_tf is None or df_tf.empty:
        df_tf_model = df_tf
        idx = Index(df_kin, tf_interactions=df_tf_model, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)
        logger.info("[TF] TF net empty; building Index without TF edges.")
    else:
        proteins_with_sites = set(df_kin["protein"].unique())  # signaling layer “state-capable” proteins
        kinase_set = set(df_kin["kinase"].unique())

        # Target universe: anything the model scores or represents
        target_universe = (
                set(df_kin["protein"].unique())
                | set(df_kin["kinase"].unique())
                | set(df_prot["protein"].unique())
                | set(df_rna["protein"].unique())
                | set(df_pho["protein"].unique())
        )

        # Filter TF edges
        df_tf_model = df_tf[df_tf["target"].isin(target_universe)].copy()
        orphan_tfs = sorted(set(df_tf_model["tf"].unique()) - proteins_with_sites)

        # Build proxy map
        TF_PROXY_MAP = {}

        def _proxy_score(orphan: str, candidate: str) -> float:
            score = 0.0
            if tf_beta_map and orphan in tf_beta_map:
                score += float(tf_beta_map[orphan])
            if kin_beta_map and candidate in kin_beta_map:
                score += float(kin_beta_map[candidate])
            return score

        for orphan in orphan_tfs:
            targets = df_tf_model.loc[df_tf_model["tf"] == orphan, "target"].astype(str)
            cand1 = [t for t in targets if t in kinase_set]
            cand2 = [t for t in targets if t in proteins_with_sites]
            candidates = cand1 if cand1 else cand2
            if candidates:
                best = sorted(candidates, key=lambda c: (-_proxy_score(orphan, c), c))[0]
                TF_PROXY_MAP[orphan] = best

        # Rewrite TF names
        if TF_PROXY_MAP:
            df_tf_model["tf"] = df_tf_model["tf"].replace(TF_PROXY_MAP)

        # Enforce representability
        df_tf_model = df_tf_model[df_tf_model["tf"].isin(proteins_with_sites)].copy()

        # Build Index
        idx = Index(df_kin, tf_interactions=df_tf_model, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)

        logger.info(f"[TF] Final TF edges for model: {len(df_tf_model)}")

    # -------------------------------------------------------------------------
    # 4. Restrict Observations (Logic from runner.py)
    # -------------------------------------------------------------------------
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    # -------------------------------------------------------------------------
    # 5. Build Matrices & System (Logic from runner.py)
    # -------------------------------------------------------------------------
    # Build Kinase Input
    df_prot_kin = df_prot[df_prot["protein"].isin(idx.kinases)].copy()
    kin_in = KinaseInput(idx.kinases, df_prot_kin)

    # Build Sparse Matrices
    logger.info("[Matrix] Building W_global and TF matrix...")
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf_model, idx, tf_beta_map=tf_beta_map, kin_beta_map=kin_beta_map)

    # Calculate TF degree normalization
    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    # Initialize Defaults
    # Note: We create defaults just to initialize the object structure.
    # The actual values will be overwritten by the JSON load immediately after.
    c_k_init = np.array([max(0.01, float(kin_beta_map.get(k, 1.0))) for k in idx.kinases])
    defaults = {
        "c_k": c_k_init,
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1
    }

    # Construct System
    ode_sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    # -------------------------------------------------------------------------
    # 6. Load Fitted Parameters
    # -------------------------------------------------------------------------
    logger.info(f"Loading fitted parameters from: {params_file}")
    with open(params_file, "r") as f:
        fitted_params = json.load(f)

    # Update system with the optimized parameters
    # The unpack logic from runner.py is implicit here as sys.update accepts dict
    ode_sys.update(**fitted_params)
    logger.info("✅ System successfully reconstructed and parameterized.")

    # -------------------------------------------------------------------------
    # 7. Run Discovery
    # -------------------------------------------------------------------------
    run_mechanistic_discovery(ode_sys, idx, df_tf_model, str(out_dir))


if __name__ == "__main__":
    main()
