import os
import pandas as pd
import re
from phoskintime_global.utils import _normcols, _find_col
from phoskintime_global.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA


def load_data(args):
    # =========================================================================
    # 1. Load Kinase Network & Merge Alphas
    # =========================================================================
    print(f"[Data] Loading Kinase Net: {args.kinase_net}")
    df_kin = pd.read_csv(args.kinase_net)
    df_kin = _normcols(df_kin)

    # Map topology columns
    pcol = _find_col(df_kin, ["geneid", "protein", "gene"])
    scol = _find_col(df_kin, ["psite", "site"])
    kcol = _find_col(df_kin, ["kinase", "k"])

    # Expand kinase sets {K1, K2} -> rows
    rows = []
    for _, r in df_kin.iterrows():
        ks = str(r[kcol]).strip('{}').split(',')
        for k in ks:
            k = k.strip()
            if k:
                # Normalize strings here (upper case + strip)
                prot = str(r[pcol]).strip().upper()
                site = str(r[scol]).strip()  # Keep site case (e.g. S_123) or normalize if needed
                kin = k.strip().upper()
                rows.append((prot, site, kin))

    df_kin_clean = pd.DataFrame(rows, columns=["protein", "psite", "kinase"]).drop_duplicates()

    # --- Merge Kinase Alpha Values ---
    # Try to load from args or infer
    alpha_path = args.kinopt if args.kinopt and os.path.exists(args.kinopt) else None

    if alpha_path:
        print(f"[Data] Loading Kinase Alphas from: {alpha_path}")
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

            # Merge
            df_kin_clean = df_kin_clean.merge(df_ka[["protein", "psite", "kinase", "alpha"]],
                                              on=["protein", "psite", "kinase"],
                                              how="left")

            # Fill missing with 1.0
            df_kin_clean["alpha"] = df_kin_clean["alpha"].fillna(1.0)

        except Exception as e:
            print(f"[Data] Warning: Could not load Kinase Alphas: {e}")
            df_kin_clean["alpha"] = 1.0
    else:
        df_kin_clean["alpha"] = 1.0

    # --- Load Kinase Beta Values (Priors) ---
    kin_beta_map = {}
    if alpha_path:
        try:
            df_kb = pd.read_excel(alpha_path, sheet_name="Beta Values")
            # Map: Kinase, Beta (Global has Psite=NaN)
            df_kb = df_kb.rename(columns={"Kinase": "kinase", "Beta": "beta", "Psite": "psite"})

            # Normalize kinase names
            df_kb["kinase"] = df_kb["kinase"].astype(str).str.strip().str.upper()

            # Filter for global betas (empty psite)
            mask_global = df_kb["psite"].isna() | (df_kb["psite"].astype(str).str.strip() == "")
            df_kb_global = df_kb[mask_global]

            kin_beta_map = dict(zip(df_kb_global["kinase"], df_kb_global["beta"]))
            print(f"       Found {len(kin_beta_map)} kinase priors.")
        except Exception as e:
            print(f"[Data] Warning: Could not load Kinase Betas: {e}")

    # =========================================================================
    # 2. Load TF Network & Merge Alphas
    # =========================================================================
    print(f"[Data] Loading TF Net: {args.tf_net}")
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
        print(f"[Data] Loading TF Alphas from: {tf_alpha_path}")
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
            print(f"[Data] Warning: Could not load TF Alphas: {e}")
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
            print(f"       Found {len(tf_beta_map)} TF priors.")
        except Exception as e:
            print(f"[Data] Warning: Could not load TF Betas: {e}")

    # =========================================================================
    # 3. Load MS Data (Proteins + Phospho)
    # =========================================================================
    print(f"[Data] Loading MS: {args.ms}")
    df_ms = pd.read_csv(args.ms)
    df_ms = _normcols(df_ms)

    gcol = _find_col(df_ms, ["geneid", "protein"])
    scol = _find_col(df_ms, ["psite", "site"])

    if scol and scol in df_ms.columns:
        df_ms[scol] = df_ms[scol].fillna("").astype(str).str.strip().replace({"nan": "", "NaN": ""})
    else:
        df_ms["psite"] = ""
        scol = "psite"

    # Identify time columns
    xcols = sorted([c for c in df_ms.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))

    if len(xcols) >= 2:
        tidy = df_ms[[gcol, scol] + xcols].melt(id_vars=[gcol, scol], var_name="xcol", value_name="fc")
        # Normalize Protein Names to Upper Case
        tidy["protein"] = tidy[gcol].astype(str).str.strip().str.upper()
        tidy["psite"] = tidy[scol]
        x_idx = tidy["xcol"].str.replace("x", "", regex=False).astype(int) - 1
        tidy["time"] = TIME_POINTS_PROTEIN[x_idx.to_numpy()]
    else:
        tcol = _find_col(df_ms, ["time", "t"])
        mcol = _find_col(df_ms, ["mean", "fc"])
        tidy = df_ms[[gcol, scol, tcol, mcol]].copy()
        tidy.columns = ["protein", "psite", "time", "fc"]
        # Normalize Protein Names to Upper Case
        tidy["protein"] = tidy["protein"].astype(str).str.strip().str.upper()
        tidy["psite"] = tidy["psite"]

    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    tidy["time"] = pd.to_numeric(tidy["time"], errors="coerce")
    tidy = tidy.dropna(subset=["fc", "time"])

    is_prot = tidy["psite"].str.len().eq(0)
    df_prot = tidy.loc[is_prot, ["protein", "time", "fc"]].reset_index(drop=True)
    df_pho = tidy.loc[~is_prot, ["protein", "psite", "time", "fc"]].reset_index(drop=True)

    # =========================================================================
    # 4. Load RNA Data
    # =========================================================================
    print(f"[Data] Loading RNA: {args.rna}")
    df_rna_raw = pd.read_csv(args.rna)
    df_rna_raw = _normcols(df_rna_raw)

    gcol = _find_col(df_rna_raw, ["geneid", "mrna", "gene"])
    xcols = sorted([c for c in df_rna_raw.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))

    tidy_r = df_rna_raw[[gcol] + xcols].melt(id_vars=[gcol], var_name="xcol", value_name="fc")
    t_map = {c: TIME_POINTS_RNA[i] for i, c in enumerate(xcols) if i < len(TIME_POINTS_RNA)}
    tidy_r["time"] = tidy_r["xcol"].map(t_map)
    # Normalize Protein Names to Upper Case
    tidy_r["protein"] = tidy_r[gcol].astype(str).str.strip().str.upper()
    tidy_r["fc"] = pd.to_numeric(tidy_r["fc"], errors="coerce")

    df_rna = tidy_r.dropna(subset=["fc", "time"])[["protein", "time", "fc"]].reset_index(drop=True)

    # DEBUG: Check overlap
    # model_proteins = set(df_prot["protein"].unique())
    # tf_tfs = set(df_tf_clean["tf"].unique())
    # tf_targets = set(df_tf_clean["target"].unique())
    #
    # print(f"\n[DEBUG] Model has {len(model_proteins)} proteins from MS.")
    # print(f"[DEBUG] TF Net has {len(tf_tfs)} TFs and {len(tf_targets)} targets.")
    # print(f"[DEBUG] Intersection TFs: {len(model_proteins.intersection(tf_tfs))}")
    # print(f"[DEBUG] Intersection Targets: {len(model_proteins.intersection(tf_targets))}")

    return df_kin_clean, df_tf_clean, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map