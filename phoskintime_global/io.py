import pandas as pd

from phoskintime_global.utils import _normcols, _find_col


def load_data(args):
    # 1) Kinase net
    print(f"[Data] Loading Kinase Net: {args.kinase_net}")
    df_kin = pd.read_csv(args.kinase_net)
    df_kin = _normcols(df_kin)

    rows = []
    pcol = _find_col(df_kin, ["geneid", "protein"])
    scol = _find_col(df_kin, ["psite", "site"])
    kcol = _find_col(df_kin, ["kinase", "k"])

    for _, r in df_kin.iterrows():
        ks = str(r[kcol]).strip('{}').split(',')
        for k in ks:
            k = k.strip()
            if k:
                rows.append((str(r[pcol]).strip(), str(r[scol]).strip(), k))
    df_kin_clean = pd.DataFrame(rows, columns=["protein", "psite", "kinase"]).drop_duplicates()

    # 2) TF net
    print(f"[Data] Loading TF Net: {args.tf_net}")
    df_tf = pd.read_csv(args.tf_net)
    df_tf = _normcols(df_tf)

    scol = _find_col(df_tf, ["source", "tf"])
    tcol = _find_col(df_tf, ["target", "gene"])
    df_tf_clean = pd.DataFrame({
        "tf": df_tf[scol].astype(str).str.strip(),
        "target": df_tf[tcol].astype(str).str.strip()
    }).drop_duplicates()

    # 3) MS
    print(f"[Data] Loading MS: {args.ms}")
    df_ms = pd.read_csv(args.ms)
    df_ms = _normcols(df_ms)

    gcol = _find_col(df_ms, ["geneid", "protein"])
    scol = _find_col(df_ms, ["psite", "site"])

    if scol and scol in df_ms.columns:
        df_ms[scol] = df_ms[scol].fillna("").astype(str).str.strip().replace({"nan": "", "NaN": ""})
    else:
        scol = "psite"
        df_ms[scol] = ""

    xcols = sorted([c for c in df_ms.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))
    if len(xcols) >= 2:
        tidy = df_ms[[gcol, scol] + xcols].melt(id_vars=[gcol, scol], var_name="xcol", value_name="fc")
        tidy["protein"] = tidy[gcol].astype(str).str.strip()
        tidy["psite"] = tidy[scol]
        x_idx = tidy["xcol"].str.replace("x", "", regex=False).astype(int) - 1
        tidy["time"] = TIME_POINTS[x_idx.to_numpy()]
    else:
        tcol = _find_col(df_ms, ["time", "t"])
        mcol = _find_col(df_ms, ["mean", "fc"])
        tidy = df_ms[[gcol, scol, tcol, mcol]].copy()
        tidy.columns = ["protein", "psite", "time", "fc"]

    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")
    tidy = tidy.dropna(subset=["fc", "time"])
    is_prot = tidy["psite"].str.len().eq(0)
    df_prot = tidy.loc[is_prot, ["protein", "time", "fc"]].reset_index(drop=True)

    # 4) RNA
    print(f"[Data] Loading RNA: {args.rna}")
    df_rna_raw = pd.read_csv(args.rna)
    df_rna_raw = _normcols(df_rna_raw)

    gcol = _find_col(df_rna_raw, ["geneid", "mrna"])
    xcols = sorted([c for c in df_rna_raw.columns if re.fullmatch(r"x\d+", str(c))], key=lambda x: int(x[1:]))
    if not xcols:
        xcols = sorted([c for c in df_rna_raw.columns if str(c).startswith("x")],
                       key=lambda x: int(re.findall(r"\d+", x)[0]))

    tidy_r = df_rna_raw[[gcol] + xcols].melt(id_vars=[gcol], var_name="xcol", value_name="fc")
    t_map = {c: TIME_POINTS_RNA[i] for i, c in enumerate(xcols) if i < len(TIME_POINTS_RNA)}
    tidy_r["time"] = tidy_r["xcol"].map(t_map)
    tidy_r["protein"] = tidy_r[gcol].astype(str).str.strip()
    tidy_r["fc"] = pd.to_numeric(tidy_r["fc"], errors="coerce")
    df_rna = tidy_r.dropna(subset=["fc", "time"])[["protein", "time", "fc"]].reset_index(drop=True)

    return df_kin_clean, df_tf_clean, df_prot, df_rna
