#!/usr/bin/env python3
"""
tfopt_network_readout.py  (corrected for your data layout)

Inputs:
  - data/tfopt_results.xlsx
      * Observed sheet: columns = mRNA, x1..x9
      * Estimated sheet: columns = mRNA, x1..x9
      * Alpha Values sheet: TF -> mRNA weights (already loads correctly)
      * Beta Values sheet: TF beta0 (PSite empty) + TF psite betas (already loads correctly)
  - data/input1.csv
      * TF / phospho time series: columns = TF (or Gene/GeneID), PSite, x1..x9
        - PSite empty/NaN => TF protein series
        - PSite present   => TF phosphosite series
  - data/input4.csv
      * Network edges: columns = Source, Target
        - Source = TF
        - Target = mRNA

Outputs (CSV) in ./results_scripts/figures_tfopt:
  - tfopt_tf_load.csv
  - tfopt_target_dominant_tfs.csv
  - tfopt_knockout_effects.csv
  - tfopt_tf_psite_stats.csv

Notes:
  - Uses scipy.integrate.trapezoid (no np.trapz).
  - Ignores input2.csv/input3.csv as requested.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

# -----------------------
# CONFIG
# -----------------------
XLSX_PATH = Path("data/tfopt_results.xlsx")
TF_SERIES_CSV = Path("data/input1.csv")
NETWORK_EDGES_CSV = Path("data/input4.csv")

OUT_DIR = Path("./results_scripts/figures_tfopt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# mRNA grid (tfopt)
MRNA_TIME_COLS = [f"x{i}" for i in range(1, 10)]
MRNA_TIME_POINTS = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960], dtype=float)

# TF grid (input1.csv)
TF_TIME_COLS = [f"x{i}" for i in range(1, 15)]
TF_TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
                           30.0, 60.0, 120.0, 240.0, 480.0, 960.0], dtype=float)


# -----------------------
# HELPERS
# -----------------------
def _require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Have: {list(df.columns)}")


def _standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_first_sheet(xl: pd.ExcelFile, candidates: list[str]) -> str:
    sheets = set(xl.sheet_names)
    for c in candidates:
        if c in sheets:
            return c
    raise ValueError(f"Could not find any sheet in {candidates}. Available: {xl.sheet_names}")


def auc_abs(y: np.ndarray, t: np.ndarray) -> float:
    return float(trapezoid(np.abs(y), t))


def peak_abs(y: np.ndarray) -> float:
    return float(np.max(np.abs(y)))


def compute_time_windows(t: np.ndarray):
    q1 = np.quantile(t, 1 / 3)
    q2 = np.quantile(t, 2 / 3)
    early = np.where(t <= q1)[0]
    mid = np.where((t > q1) & (t <= q2))[0]
    late = np.where(t > q2)[0]
    return early, mid, late


# -----------------------
# LOADERS
# -----------------------
def load_observed_estimated(xl: pd.ExcelFile) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_name = _find_first_sheet(xl, ["Observed", "observed", "OBSERVED"])
    est_name = _find_first_sheet(xl, ["Estimated", "estimated", "ESTIMATED"])

    obs = _standardize_colnames(pd.read_excel(xl, obs_name))
    est = _standardize_colnames(pd.read_excel(xl, est_name))

    # Must be: mRNA, x1..x9
    _require_cols(obs, ["mRNA"] + MRNA_TIME_COLS, obs_name)
    _require_cols(est, ["mRNA"] + MRNA_TIME_COLS, est_name)

    obs = obs.set_index("mRNA")[MRNA_TIME_COLS].astype(float)
    est = est.set_index("mRNA")[MRNA_TIME_COLS].astype(float)
    return obs, est


def load_alpha(xl: pd.ExcelFile) -> pd.DataFrame:
    alpha_name = _find_first_sheet(xl, ["Alpha Values", "Alpha", "alpha", "ALPHA"])
    a = _standardize_colnames(pd.read_excel(xl, alpha_name))

    # Be permissive: map to (mRNA, TF, Value)
    colmap = {}
    for c in a.columns:
        lc = str(c).strip().lower()
        if lc in {"mrna", "target", "gene", "geneid"}:
            colmap[c] = "mRNA"
        elif lc in {"tf", "source", "regulator"}:
            colmap[c] = "TF"
        elif lc in {"value", "alpha", "a"}:
            colmap[c] = "Value"
    a = a.rename(columns=colmap)

    _require_cols(a, ["mRNA", "TF", "Value"], alpha_name)
    a["mRNA"] = a["mRNA"].astype(str)
    a["TF"] = a["TF"].astype(str)
    a["Value"] = a["Value"].astype(float)
    return a[["mRNA", "TF", "Value"]]


def load_beta(xl: pd.ExcelFile) -> pd.DataFrame:
    beta_name = _find_first_sheet(xl, ["Beta Values", "Beta", "beta", "BETA"])
    b = _standardize_colnames(pd.read_excel(xl, beta_name))

    colmap = {}
    for c in b.columns:
        lc = str(c).strip().lower()
        if lc in {"tf", "gene", "geneid", "source"}:
            colmap[c] = "TF"
        elif lc in {"psite", "psites", "site"}:
            colmap[c] = "PSite"
        elif lc in {"value", "beta", "b"}:
            colmap[c] = "Value"
    b = b.rename(columns=colmap)

    _require_cols(b, ["TF", "PSite", "Value"], beta_name)
    b["TF"] = b["TF"].astype(str)
    b["PSite"] = b["PSite"].astype(object)
    b["Value"] = b["Value"].astype(float)
    return b[["TF", "PSite", "Value"]]


def load_tf_series_input1(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    input1.csv provides TF protein and TF phosphosite time series.

    Expected columns:
      - TF identifier column: TF or Gene or GeneID (auto-detected)
      - PSite column: PSite (auto-detected)
      - x1..x9

    Returns:
      tf_prot: index TF, columns x1..x9
      tf_ps: columns TF, PSite, x1..x9
    """
    df = _standardize_colnames(pd.read_csv(path))
    # detect TF id column
    tf_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in {"tf", "gene", "geneid", "source"}:
            tf_col = c
            break
    if tf_col is None:
        raise ValueError(f"{path} needs a TF/Gene column. Have: {list(df.columns)}")

    # detect PSite column
    psite_col = None
    for c in df.columns:
        if c.lower() == "psite":
            psite_col = c
            break
    if psite_col is None:
        raise ValueError(f"{path} needs PSite column. Have: {list(df.columns)}")

    _require_cols(df, [tf_col, psite_col] + TF_TIME_COLS, str(path))

    df = df.rename(columns={tf_col: "TF", psite_col: "PSite"})
    df["TF"] = df["TF"].astype(str)
    df["PSite"] = df["PSite"].astype(object)

    psite_str = df["PSite"].astype(str).str.strip()
    is_prot = df["PSite"].isna() | (psite_str == "") | (psite_str.str.lower().isin({"nan", "none"}))

    def _interp_row(y14: np.ndarray) -> np.ndarray:
        """
        Cubic interpolation from TF time grid (14 points, 0–960 min)
        onto mRNA grid (9 points, 4–960 min).
        """
        f = interp1d(
            TF_TIME_POINTS,
            y14,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return f(MRNA_TIME_POINTS)

    # TF protein
    prot_df = df[is_prot].drop_duplicates(subset=["TF"], keep="last").copy()
    prot_y14 = prot_df[TF_TIME_COLS].to_numpy(dtype=float)
    prot_y9 = np.vstack([_interp_row(row) for row in prot_y14])
    tf_prot = pd.DataFrame(prot_y9, index=prot_df["TF"].astype(str).to_list(), columns=MRNA_TIME_COLS)

    # TF psites
    ps_df = df[~is_prot].copy()
    ps_y14 = ps_df[TF_TIME_COLS].to_numpy(dtype=float)
    ps_y9 = np.vstack([_interp_row(row) for row in ps_y14])
    tf_ps = ps_df[["TF", "PSite"]].copy()
    tf_ps["TF"] = tf_ps["TF"].astype(str)
    tf_ps["PSite"] = tf_ps["PSite"].astype(str)
    for j, c in enumerate(MRNA_TIME_COLS):
        tf_ps[c] = ps_y9[:, j].astype(float)

    return tf_prot, tf_ps


def restrict_alpha_to_network(alpha: pd.DataFrame, edges_csv: Path) -> pd.DataFrame:
    edges = _standardize_colnames(pd.read_csv(edges_csv))
    _require_cols(edges, ["Source", "Target"], str(edges_csv))
    edges["Source"] = edges["Source"].astype(str)
    edges["Target"] = edges["Target"].astype(str)

    allowed = set(zip(edges["Target"], edges["Source"]))  # (mRNA, TF)
    a = alpha[alpha.apply(lambda r: (r["mRNA"], r["TF"]) in allowed, axis=1)].copy()
    return a


# -----------------------
# CORE
# -----------------------
def build_tf_latent_activity(tf_prot: pd.DataFrame, tf_ps: pd.DataFrame, beta: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    A_j(t) = beta0_j * TF_protein_j(t) + sum_k beta_{k,j} * PSite_{k,j}(t)

    beta rows:
      - PSite empty/NaN => beta0 (protein component)
      - PSite present   => psite beta
    """
    # fast lookup for psite series
    ps_map = {(r.TF, r.PSite): np.array([getattr(r, c) for c in MRNA_TIME_COLS], dtype=float)
              for r in tf_ps.itertuples(index=False)}

    latent: dict[str, np.ndarray] = {}
    for tf, btf in beta.groupby("TF", sort=False):
        # beta0
        ps_str = btf["PSite"].astype(str).str.strip()
        is_b0 = btf["PSite"].isna() | (ps_str == "") | (ps_str.str.lower().isin({"nan", "none"}))
        beta0 = float(btf.loc[is_b0, "Value"].iloc[0]) if is_b0.any() else 0.0

        prot = tf_prot.loc[tf].to_numpy(dtype=float) if tf in tf_prot.index else np.zeros(len(MRNA_TIME_COLS), float)
        y = beta0 * prot

        # psite terms
        for r in btf.loc[~is_b0].itertuples(index=False):
            key = (tf, str(r.PSite))
            if key in ps_map:
                y = y + float(r.Value) * ps_map[key]
            # else: missing psite series -> contributes 0

        latent[tf] = y

    return latent


def compute_predictions(alpha: pd.DataFrame, latent: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Rhat_i(t) = sum_j alpha_{i,j} * A_j(t)
    alpha: columns (mRNA, TF, Value)
    """
    pred: dict[str, np.ndarray] = {}
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        y = np.zeros(len(MRNA_TIME_COLS), float)
        for r in grp.itertuples(index=False):
            tf = r.TF
            if tf not in latent:
                continue
            y += float(r.Value) * latent[tf]
        pred[mrna] = y
    return pred


def main():
    xl = pd.ExcelFile(XLSX_PATH)

    obs, est = load_observed_estimated(xl)
    alpha = load_alpha(xl)
    beta = load_beta(xl)

    tf_prot, tf_ps = load_tf_series_input1(TF_SERIES_CSV)

    # Enforce the TF->mRNA network as ground truth
    alpha = restrict_alpha_to_network(alpha, NETWORK_EDGES_CSV)

    # Latent TF activity + reconstructed predictions
    latent = build_tf_latent_activity(tf_prot, tf_ps, beta)
    pred = compute_predictions(alpha, latent)

    # -----------------------
    # 1) TF psite stats
    # -----------------------
    b2 = beta.copy()
    ps_str = b2["PSite"].astype(str).str.strip()
    has_ps = ~(b2["PSite"].isna() | (ps_str == "") | (ps_str.str.lower().isin({"nan", "none"})))
    psite_stats = (
        b2.assign(has_psite=has_ps)
        .groupby("TF", as_index=False)
        .agg(
            n_beta=("Value", "size"),
            n_psites=("has_psite", "sum"),
            has_any_psite=("has_psite", "max"),
        )
    )
    psite_stats.to_csv(OUT_DIR / "tfopt_tf_psite_stats.csv", index=False)

    # -----------------------
    # 2) TF load (kinopt-like)
    # -----------------------
    # total_load_auc_abs(TF) = sum_{targets i} |alpha_{i,TF}| * AUC(|A_TF|)
    latent_auc = {tf: auc_abs(y, MRNA_TIME_POINTS) for tf, y in latent.items()}
    latent_peak = {tf: peak_abs(y) for tf, y in latent.items()}

    # bound hits (tfopt bounds are -4..4 per your description)
    BOUND = 4.0
    btf = beta.copy()
    btf["at_bound"] = np.isclose(np.abs(btf["Value"].to_numpy()), BOUND, atol=1e-6)

    targets_per_tf = alpha.groupby("TF")["mRNA"].nunique().to_dict()

    tf_rows = []
    for tf in sorted(set(alpha["TF"].astype(str))):
        a_tf = alpha[alpha["TF"] == tf]
        total_load = float(np.sum(np.abs(a_tf["Value"].to_numpy(dtype=float))) * latent_auc.get(tf, 0.0))

        b_sub = btf[btf["TF"] == tf]
        frac_bound = float(b_sub["at_bound"].mean()) if len(b_sub) else 0.0
        n_bound = int(b_sub["at_bound"].sum()) if len(b_sub) else 0

        tf_rows.append({
            "TF": tf,
            "n_targets": int(targets_per_tf.get(tf, 0)),
            "total_load_auc_abs": total_load,
            "frac_beta_at_bound": frac_bound,
            "n_beta_at_bound": n_bound,
            "latent_auc_abs": float(latent_auc.get(tf, 0.0)),
            "latent_peak_abs": float(latent_peak.get(tf, 0.0)),
        })

    tf_load = pd.DataFrame(tf_rows).sort_values("total_load_auc_abs", ascending=False)
    tf_load.to_csv(OUT_DIR / "tfopt_tf_load.csv", index=False)

    # -----------------------
    # 3) Target dominance (overall + early/mid/late)
    # -----------------------
    early_idx, mid_idx, late_idx = compute_time_windows(MRNA_TIME_POINTS)

    dom_rows = []
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        contrib = []
        for r in grp.itertuples(index=False):
            tf = r.TF
            if tf not in latent:
                continue
            contrib.append((tf, float(r.Value) * latent[tf]))

        if not contrib:
            continue

        aucs = [(tf, auc_abs(s, MRNA_TIME_POINTS)) for tf, s in contrib]
        aucs.sort(key=lambda x: x[1], reverse=True)
        dom_tf, dom_val = aucs[0]
        total = sum(v for _, v in aucs) + 1e-12
        dom_share = dom_val / total

        def win_dom(idxs):
            t = MRNA_TIME_POINTS[idxs]
            w = [(tf, float(trapezoid(np.abs(s[idxs]), t))) for tf, s in contrib]
            w.sort(key=lambda x: x[1], reverse=True)
            tf0, v0 = w[0]
            tot = sum(v for _, v in w) + 1e-12
            return tf0, v0 / tot

        dE, sE = win_dom(early_idx)
        dM, sM = win_dom(mid_idx)
        dL, sL = win_dom(late_idx)

        # Observed/Estimated series are from xlsx; reconstructed is `pred`
        obs_y = obs.loc[mrna].to_numpy(dtype=float) if mrna in obs.index else np.zeros(len(MRNA_TIME_COLS))
        est_y = est.loc[mrna].to_numpy(dtype=float) if mrna in est.index else np.zeros(len(MRNA_TIME_COLS))
        pred_y = pred.get(mrna, np.zeros(len(MRNA_TIME_COLS)))

        dom_rows.append({
            "mRNA": mrna,
            "n_TFs": int(len(grp)),
            "dominant_overall": dom_tf,
            "dominant_overall_share": float(dom_share),
            "dominant_early": dE,
            "dominant_early_share": float(sE),
            "dominant_mid": dM,
            "dominant_mid_share": float(sM),
            "dominant_late": dL,
            "dominant_late_share": float(sL),
            "recon_auc_abs": auc_abs(pred_y, MRNA_TIME_POINTS),
            "recon_peak_abs": peak_abs(pred_y),
            "est_auc_abs": auc_abs(est_y, MRNA_TIME_POINTS),
            "est_peak_abs": peak_abs(est_y),
            "obs_auc_abs": auc_abs(obs_y, MRNA_TIME_POINTS),
            "obs_peak_abs": peak_abs(obs_y),
        })

    dom_df = pd.DataFrame(dom_rows).sort_values("dominant_overall_share", ascending=False)
    dom_df.to_csv(OUT_DIR / "tfopt_target_dominant_tfs.csv", index=False)

    # -----------------------
    # 4) KO effects per target (remove one TF at a time from reconstructed prediction)
    # -----------------------
    ko_rows = []
    for mrna, grp in alpha.groupby("mRNA", sort=False):
        base = pred.get(mrna, np.zeros(len(MRNA_TIME_COLS)))
        base_auc = auc_abs(base, MRNA_TIME_POINTS)
        base_peak = peak_abs(base)

        for r in grp.itertuples(index=False):
            tf = r.TF
            if tf not in latent:
                continue
            ko_series = base - float(r.Value) * latent[tf]
            ko_auc = auc_abs(ko_series, MRNA_TIME_POINTS)
            ko_peak = peak_abs(ko_series)

            ko_rows.append({
                "mRNA": mrna,
                "KnockedTF": tf,
                "alpha": float(r.Value),
                "baseline_auc_abs": float(base_auc),
                "baseline_peak_abs": float(base_peak),
                "ko_auc_abs": float(ko_auc),
                "ko_peak_abs": float(ko_peak),
                "delta_auc_abs": float(base_auc - ko_auc),
                "delta_peak_abs": float(base_peak - ko_peak),
            })

    ko_df = pd.DataFrame(ko_rows)
    ko_df["abs_delta_auc"] = ko_df["delta_auc_abs"].abs()
    ko_df["ko_rank_target"] = ko_df.groupby("mRNA")["abs_delta_auc"].rank(ascending=False, method="dense")
    ko_df = ko_df.drop(columns=["abs_delta_auc"]).sort_values(["mRNA", "ko_rank_target"])
    ko_df.to_csv(OUT_DIR / "tfopt_knockout_effects.csv", index=False)

    print("Wrote:")
    print(" -", OUT_DIR / "tfopt_tf_psite_stats.csv")
    print(" -", OUT_DIR / "tfopt_tf_load.csv")
    print(" -", OUT_DIR / "tfopt_target_dominant_tfs.csv")
    print(" -", OUT_DIR / "tfopt_knockout_effects.csv")


if __name__ == "__main__":
    main()
