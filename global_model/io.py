"""
Data Loading and Pre-processing Module.

This script is responsible for ingesting raw CSV/Excel files containing biological
network definitions (Kinases, TFs) and experimental data (MS Proteomics, RNA-seq).
It harmonizes identifiers, merges prior knowledge (Alpha/Beta weights), and reshapes
time-series data into a standardized 'tidy' format for the modeling pipeline.


"""

import os
import pandas as pd
import re
from global_model.utils import _normcols, _find_col, process_and_scale_raw_data
from global_model.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, RESULTS_DIR, SCALING_METHOD
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


def load_data(args):
    """
    Loads, cleans, and merges all required data files for the model.

    Process:
    1.  **Networks:** Loads Kinase-Substrate and TF-Gene interactions.
        -   Expands set-based notations (e.g., "{K1, K2}").
        -   Merges 'Alpha' (edge strength) and 'Beta' (node activity prior) values from optimization results.
    2.  **Experimental Data:** Loads MS and RNA-seq data.
        -   Scales data (e.g., log-transformation, normalization).
        -   Reshapes from "wide" (time points as columns) to "long" (time as a variable).

    Args:
        args (Namespace): Configuration namespace containing file paths (kinase_net, tf_net, ms, rna, kinopt, tfopt).

    Returns:
        tuple: A tuple containing:
            - df_kin_clean (pd.DataFrame): Kinase network with columns [protein, psite, kinase, alpha].
            - df_tf_clean (pd.DataFrame): TF network with columns [tf, target, alpha].
            - df_prot (pd.DataFrame): Protein abundance data [protein, time, fc].
            - df_pho (pd.DataFrame): Phosphorylation data [protein, psite, time, fc].
            - df_rna (pd.DataFrame): RNA abundance data [protein, time, fc].
            - kin_beta_map (dict): Prior activity assumptions for kinases {kinase: beta}.
            - tf_beta_map (dict): Prior activity assumptions for TFs {tf: beta}.
    """
    # =========================================================================
    # 1. Load Kinase Network & Merge Alphas
    # =========================================================================
    logger.info(f"[Data] Loading Kinase Net: {args.kinase_net}")
    df_kin = pd.read_csv(args.kinase_net)
    df_kin = _normcols(df_kin)

    # Map topology columns dynamically
    pcol = _find_col(df_kin, ["geneid", "protein", "gene"])
    scol = _find_col(df_kin, ["psite", "site"])
    kcol = _find_col(df_kin, ["kinase", "k"])

    # Expand kinase sets {K1, K2} -> rows
    # Some databases list multiple kinases for one site in a single cell (e.g., "{Akt1, Akt2}").
    # We explode these into distinct rows.
    rows = []
    for _, r in df_kin.iterrows():
        ks = str(r[kcol]).strip('{}').split(',')
        for k in ks:
            k = k.strip()
            if k:
                # Normalize strings here (upper case + strip) to ensure consistent matching
                prot = str(r[pcol]).strip().upper()
                site = str(r[scol]).strip()  # Keep site case (e.g. S_123) or normalize if needed
                kin = k.strip().upper()
                rows.append((prot, site, kin))

    df_kin_clean = pd.DataFrame(rows, columns=["protein", "psite", "kinase"]).drop_duplicates()

    # --- Merge Kinase Alpha Values ---
    # Try to load 'Alpha' (Interaction Strength) from previous optimization steps if available.
    alpha_path = args.kinopt if args.kinopt and os.path.exists(args.kinopt) else None

    if alpha_path:
        logger.info(f"[Data] Loading Kinase Alphas from: {alpha_path}")
        try:
            df_ka = pd.read_excel(alpha_path, sheet_name="Alpha Values")
            # Map specific columns: Gene, Psite, Kinase, Alpha
            df_ka = df_ka.rename(columns={"Gene": "protein", "Psite": "psite", "Kinase": "kinase", "Alpha": "alpha"})

            # Normalize strings for merge
            for c in ["protein", "kinase"]:
                if c in df_ka.columns:
                    df_ka[c] = df_ka[c].astype(str).str.strip().str.upper()
            if "psite" in df_ka.columns:
                df_ka["psite"] = df_ka["psite"].astype(str).str.strip()

            # Merge existing network with loaded alphas
            df_kin_clean = df_kin_clean.merge(df_ka[["protein", "psite", "kinase", "alpha"]],
                                              on=["protein", "psite", "kinase"],
                                              how="left")

            # Fill missing alphas with 1.0 (default assumption of equal strength)
            df_kin_clean["alpha"] = df_kin_clean["alpha"].fillna(1.0)

        except Exception as e:
            logger.info(f"[Data] Warning: Could not load Kinase Alphas: {e}")
            df_kin_clean["alpha"] = 1.0
    else:
        df_kin_clean["alpha"] = 1.0

    # --- Load Kinase Beta Values (Priors) ---
    # Betas represent inherent kinase activity multipliers (e.g., from expression levels or mutations).
    kin_beta_map = {}
    if alpha_path:
        try:
            df_kb = pd.read_excel(alpha_path, sheet_name="Beta Values")
            # Map: Kinase, Beta (Global has Psite=NaN)
            df_kb = df_kb.rename(columns={"Kinase": "kinase", "Beta": "beta", "Psite": "psite"})

            # Normalize kinase names
            df_kb["kinase"] = df_kb["kinase"].astype(str).str.strip().str.upper()

            # Filter for global betas (rows where psite is empty/NaN)
            mask_global = df_kb["psite"].isna() | (df_kb["psite"].astype(str).str.strip() == "")
            df_kb_global = df_kb[mask_global]

            kin_beta_map = dict(zip(df_kb_global["kinase"], df_kb_global["beta"]))
            logger.info(f"       Found {len(kin_beta_map)} kinase priors.")
        except Exception as e:
            logger.info(f"[Data] Warning: Could not load Kinase Betas: {e}")

    # =========================================================================
    # 2. Load TF Network & Merge Alphas
    # =========================================================================
    logger.info(f"[Data] Loading TF Net: {args.tf_net}")
    df_tf = pd.read_csv(args.tf_net)
    df_tf = _normcols(df_tf)

    scol = _find_col(df_tf, ["source", "tf"])
    tcol = _find_col(df_tf, ["target", "gene"])

    # Normalize strings (upper case + strip)
    df_tf_clean = pd.DataFrame({
        "tf": df_tf[scol].astype(str).str.strip().str.upper(),
        "target": df_tf[tcol].astype(str).str.strip().str.upper()
    }).drop_duplicates()

    # --- Merge TF Alpha Values ---
    tf_alpha_path = args.tfopt if args.tfopt and os.path.exists(args.tfopt) else None

    if tf_alpha_path:
        logger.info(f"[Data] Loading TF Alphas from: {tf_alpha_path}")
        try:
            df_ta = pd.read_excel(tf_alpha_path, sheet_name="Alpha Values")
            # Map: mRNA -> target, TF -> tf, Value -> alpha
            df_ta = df_ta.rename(columns={"mRNA": "target", "TF": "tf", "Value": "alpha"})

            # Normalize strings for merge
            for c in ["target", "tf"]:
                if c in df_ta.columns:
                    df_ta[c] = df_ta[c].astype(str).str.strip().str.upper()

            # Merge
            df_tf_clean = df_tf_clean.merge(df_ta[["target", "tf", "alpha"]],
                                            on=["target", "tf"],
                                            how="left")

            # Fill missing with 1.0
            df_tf_clean["alpha"] = df_tf_clean["alpha"].fillna(1.0)

        except Exception as e:
            logger.info(f"[Data] Warning: Could not load TF Alphas: {e}")
            df_tf_clean["alpha"] = 1.0
    else:
        df_tf_clean["alpha"] = 1.0

    # --- Load TF Beta Values ---
    tf_beta_map = {}
    if tf_alpha_path:
        try:
            df_tb = pd.read_excel(tf_alpha_path, sheet_name="Beta Values")
            # Map: TF, Value -> Beta
            df_tb = df_tb.rename(columns={"TF": "tf", "Value": "beta", "PSite": "psite"})

            # Normalize names
            df_tb["tf"] = df_tb["tf"].astype(str).str.strip().str.upper()

            # Filter global
            mask_global = df_tb["psite"].isna() | (df_tb["psite"].astype(str).str.strip() == "")
            df_tb_global = df_tb[mask_global]

            tf_beta_map = dict(zip(df_tb_global["tf"], df_tb_global["beta"]))
            logger.info(f"       Found {len(tf_beta_map)} TF priors.")
        except Exception as e:
            logger.info(f"[Data] Warning: Could not load TF Betas: {e}")

    # =========================================================================
    # 3. Load MS Data (Proteins + Phospho)
    # =========================================================================
    # 
    logger.info(f"[Data] Loading MS: {args.ms}")
    df_ms_raw = pd.read_csv(args.ms)

    # Process raw data: normalize, log-transform if needed, and handle replicates
    df_ms = process_and_scale_raw_data(
        df_ms_raw,
        time_points=TIME_POINTS_PROTEIN,
        id_cols=["GeneID", "Psite"],
        scale_method=SCALING_METHOD
    )
    df_ms = _normcols(df_ms)

    gcol = _find_col(df_ms, ["geneid", "protein"])
    scol = _find_col(df_ms, ["psite", "site"])

    # Ensure psite column exists for uniform processing
    if scol and scol in df_ms.columns:
        df_ms[scol] = df_ms[scol].fillna("").astype(str).str.strip().replace({"nan": "", "NaN": ""})
    else:
        df_ms["psite"] = ""
        scol = "psite"

    # Identify time columns (e.g., x1, x2, x3...)
    xcols = sorted([c for c in df_ms.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))

    # Reshape logic: Wide (cols x1,x2..) -> Long (rows with 'time' column)
    if len(xcols) >= 2:
        tidy = df_ms[[gcol, scol] + xcols].melt(id_vars=[gcol, scol], var_name="xcol", value_name="fc")
        # Normalize Protein Names to Upper Case
        tidy["protein"] = tidy[gcol].astype(str).str.strip().str.upper()
        tidy["psite"] = tidy[scol]
        # Map 'x1' -> Index 0 -> TimePoint[0]
        x_idx = tidy["xcol"].str.replace("x", "", regex=False).astype(int) - 1
        tidy["time"] = TIME_POINTS_PROTEIN[x_idx.to_numpy()]
    else:
        # Handling pre-melted data
        tcol = _find_col(df_ms, ["time", "t"])
        mcol = _find_col(df_ms, ["mean", "fc"])
        tidy = df_ms[[gcol, scol, tcol, mcol]].copy()
        tidy.columns = ["protein", "psite", "time", "fc"]
        # Normalize Protein Names to Upper Case
        tidy["protein"] = tidy["protein"].astype(str).str.strip().str.upper()
        tidy["psite"] = tidy["psite"]

    # Final cleanup of numeric types
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    tidy["time"] = pd.to_numeric(tidy["time"], errors="coerce")
    tidy = tidy.dropna(subset=["fc", "time"])

    # Split into Protein data (no psite) and Phospho data (has psite)
    is_prot = tidy["psite"].str.len().eq(0)
    df_prot = tidy.loc[is_prot, ["protein", "time", "fc"]].reset_index(drop=True)
    df_pho = tidy.loc[~is_prot, ["protein", "psite", "time", "fc"]].reset_index(drop=True)

    # =========================================================================
    # 4. Load & Process RNA Data
    # =========================================================================
    logger.info(f"[Data] Loading RNA: {args.rna}")

    # 1. Read and Clean Headers
    df_rna_raw = pd.read_csv(args.rna)
    df_rna_raw = _normcols(df_rna_raw)

    # 2. Identify the Gene/Protein ID column
    gcol = _find_col(df_rna_raw, ["geneid", "mrna", "gene"])

    # 3. Rename it to 'protein' so the rest of the pipeline understands it
    df_rna_raw = df_rna_raw.rename(columns={gcol: "protein"})

    # 4. Process (Scale -> Map Time -> Melt)
    # The function returns the final tidy dataframe directly.
    df_rna = process_and_scale_raw_data(
        df_rna_raw,
        time_points=TIME_POINTS_RNA,
        id_cols=["protein"],
        scale_method=SCALING_METHOD
    )

    # df_rna is now ready. It has columns: ["protein", "time", "fc"]
    logger.info(f"[Data] Loaded {len(df_rna)} RNA points.")

    return df_kin_clean, df_tf_clean, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map