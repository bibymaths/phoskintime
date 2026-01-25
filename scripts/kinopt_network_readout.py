#!/usr/bin/env python3
"""
Kinopt network readout using the *actual* model equation:

  P_i(t) = sum_j Q_{i,j} * alpha_{i,j} * ( sum_k beta_{k,j} * P^k_j(t) )

In the Excel, we assume:
- Alpha Values sheet encodes the active edges (Q_{i,j}=1): one row per (Gene, Psite, Kinase) with Alpha.
  => Target index i is (Gene, Psite) unless your sheet uses Gene only; script supports both.
- Beta Values sheet encodes beta_{k,j}: one row per (Kinase, Psite) with Beta.
  => Here "Psite" refers to the *kinase's own site* k on kinase j.
- Observed / Estimated sheets contain time series for (Gene, Psite) rows.

What the script outputs:
1) Reconstructed latent kinase activities A_j(t) = sum_k beta_{k,j} * P^k_j(t)
   - and which kinase sites dominate A_j(t).
2) Target decompositions C_{i<-j}(t) = alpha_{i,j} * A_j(t)
   - dominant kinases early/mid/late per target.
3) In-silico knockout sensitivity:
   - Remove kinase j for a target i by setting alpha_{i,j}=0
   - Optionally renormalize remaining alphas to sum to 1 (default: True)
   - Effect sizes: ΔAUC, Δpeak of predicted target curve.

IMPORTANT INTERPRETATION NOTES:
- This is not "Frechet analysis". This is a mechanistic, model-consistent signal-flow audit.
- High dominance_overall_share => target is effectively controlled by one kinase (fragile / single-driver).
- High kinase total_load_auc_abs => the network routes much of its fitted signal through that kinase (hub load).
- Beta at bounds (±4) means the kinase's internal site-mixture is extreme; interpret as within-kinase tension/selection.

Run:
  python kinopt_network_readout.py

Outputs in ./out:
  - kinopt_latent_kinase_activity_summary.csv
  - kinopt_target_dominant_kinases.csv
  - kinopt_kinase_load.csv
  - kinopt_knockout_effects.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# ------------------------
# Config
# ------------------------
XLSX = "data/kinopt_results.xlsx"
OUTDIR = Path("./results_scripts")
OUTDIR.mkdir(parents=True, exist_ok=True)

BETA_BOUND = 4.0  # confirmed by you
BOUND_TOL = 1e-9
EPS = 1e-12

# Use observed kinase sites P^k_j(t) by default (as in your equation).
# If you want, you can switch to "Estimated" for kinase-site signals.
KINASE_SITE_SOURCE = "Observed"  # "Observed" or "Estimated"

# Knockout mode:
# - If True: renormalize remaining alphas for that target so sum_j alpha_{i,j}=1 continues to hold.
# - If False: do not renormalize (interpretable as removing that kinase without redistributing weight).
KNOCKOUT_RENORMALIZE_ALPHA = True


# ------------------------
# Helpers
# ------------------------
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"Could not find any of {candidates} in columns={list(df.columns)}")


def _detect_time_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """
    Prefer columns whose names parse as float (timepoint headers).
    Fallback: numeric dtype columns not in exclude.
    """
    time_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = str(c).strip()
        try:
            float(s)
            time_cols.append(c)
        except Exception:
            pass
    if not time_cols:
        # fallback to numeric columns
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
        time_cols = num_cols
    if not time_cols:
        raise ValueError("No timepoint columns detected.")
    # sort by numeric value
    time_cols = sorted(time_cols, key=lambda x: float(str(x).strip()))
    return time_cols


def _time_array(time_cols: list[str]) -> np.ndarray:
    t = np.array([float(str(c).strip()) for c in time_cols], dtype=np.float64)
    return t


def _auc(t: np.ndarray, y: np.ndarray) -> float:
    return float(trapezoid(y, t))


def _windows(nT: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Early/mid/late windows by indices. Works for nT >= 4.
    For Kinopt (14 timepoints), early captures 0–1min region reasonably.
    """
    early = np.arange(0, min(4, nT))
    late = np.arange(max(0, nT - 4), nT)
    mid = np.setdiff1d(np.arange(nT), np.union1d(early, late))
    if len(mid) == 0:
        mid = np.arange(nT)
    return early, mid, late


def _as_key(g: str, p: str) -> tuple[str, str]:
    return (str(g), str(p))


# ------------------------
# Core computation
# ------------------------
def main():
    # Load
    obs = pd.read_excel(XLSX, sheet_name="Observed")
    est = pd.read_excel(XLSX, sheet_name="Estimated")
    alpha = pd.read_excel(XLSX, sheet_name="Alpha Values")
    beta = pd.read_excel(XLSX, sheet_name="Beta Values")

    # Identify identifier columns in time series sheets
    gene_col_obs = _find_col(obs, ["Gene", "GeneID", "Protein", "mRNA"])
    psite_col_obs = _find_col(obs, ["Psite", "PSite", "Position", "Site"])
    gene_col_est = _find_col(est, ["Gene", "GeneID", "Protein", "mRNA"])
    psite_col_est = _find_col(est, ["Psite", "PSite", "Position", "Site"])

    # Time columns
    TIME_COLS = [f"x{i}" for i in range(1, 15)]
    TIME_POINTS_MIN = np.array(
        [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
        dtype=np.float64
    )

    time_cols = TIME_COLS

    # safety checks
    missing = [c for c in time_cols if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing time columns in Observed sheet: {missing}")
    missing = [c for c in time_cols if c not in est.columns]
    if missing:
        raise ValueError(f"Missing time columns in Estimated sheet: {missing}")

    if len(time_cols) != len(TIME_POINTS_MIN):
        raise ValueError(f"TIME_COLS ({len(time_cols)}) != TIME_POINTS_MIN ({len(TIME_POINTS_MIN)})")

    t_min = TIME_POINTS_MIN
    t = t_min / (t_min.max() + EPS)  # normalized time for AUC/load calculations
    nT = len(time_cols)

    early_idx, mid_idx, late_idx = _windows(nT)

    # Build row-indexable maps for observed/estimated trajectories keyed by (Gene,Psite)
    def build_ts_map(df: pd.DataFrame, gene_col: str, psite_col: str) -> dict[tuple[str, str], np.ndarray]:
        m = {}
        for _, r in df[[gene_col, psite_col] + time_cols].iterrows():
            key = _as_key(r[gene_col], r[psite_col])
            y = r[time_cols].to_numpy(dtype=np.float64)
            m[key] = y
        return m

    obs_map = build_ts_map(obs, gene_col_obs, psite_col_obs)
    est_map = build_ts_map(est, gene_col_est, psite_col_est)

    kinase_site_map = obs_map if KINASE_SITE_SOURCE.lower() == "observed" else est_map

    # Alpha edges
    gene_col_a = _find_col(alpha, ["Gene", "GeneID", "Protein"])
    psite_col_a = _find_col(alpha, ["Psite", "PSite", "Position", "Site"])
    kinase_col_a = _find_col(alpha, ["Kinase"])
    alpha_col_a = _find_col(alpha, ["Alpha", "alpha"])

    edges = alpha[[gene_col_a, psite_col_a, kinase_col_a, alpha_col_a]].copy()
    edges.rename(columns={gene_col_a: "Gene", psite_col_a: "Psite", kinase_col_a: "Kinase", alpha_col_a: "Alpha"},
                 inplace=True)
    edges["Gene"] = edges["Gene"].astype(str)
    edges["Psite"] = edges["Psite"].astype(str)
    edges["Kinase"] = edges["Kinase"].astype(str)
    edges["Alpha"] = pd.to_numeric(edges["Alpha"], errors="coerce").fillna(0.0).astype(np.float64)

    # Beta (kinase internal site-mixture)
    kinase_col_b = _find_col(beta, ["Kinase"])
    psite_col_b = _find_col(beta, ["Psite", "PSite", "Position", "Site"])
    beta_col_b = _find_col(beta, ["Beta", "beta"])

    b = beta[[kinase_col_b, psite_col_b, beta_col_b]].copy()
    b.rename(columns={kinase_col_b: "Kinase", psite_col_b: "KinaseSite", beta_col_b: "Beta"}, inplace=True)
    b["Kinase"] = b["Kinase"].astype(str)
    b["KinaseSite"] = b["KinaseSite"].astype(str)
    b["Beta"] = pd.to_numeric(b["Beta"], errors="coerce").fillna(0.0).astype(np.float64)
    b["beta_at_bound"] = (b["Beta"].abs() >= (BETA_BOUND - BOUND_TOL))

    # ------------------------
    # 1) Reconstruct latent kinase activity A_j(t)
    # ------------------------
    # A_j(t) = sum_k beta_{k,j} * P^k_j(t)
    latent = {}  # kinase -> A_j(t)
    latent_meta_rows = []

    for kin, grp in b.groupby("Kinase", sort=False):
        A = np.zeros((nT,), dtype=np.float64)
        # Track which kinase sites exist and how much they contribute (by |beta| * AUC(|site|))
        site_rows = []
        for _, rr in grp.iterrows():
            ksite = rr["KinaseSite"]
            beta_kj = float(rr["Beta"])
            y_site = kinase_site_map.get(_as_key(kin, ksite))

            if y_site is None:
                # Missing kinase site trajectory; skip contribution (but log it)
                site_rows.append(
                    {"Kinase": kin, "KinaseSite": ksite, "Beta": beta_kj, "beta_at_bound": bool(rr["beta_at_bound"]),
                     "has_timeseries": False, "site_auc_abs": np.nan}
                )
                continue

            A += beta_kj * y_site
            site_rows.append(
                {"Kinase": kin, "KinaseSite": ksite, "Beta": beta_kj, "beta_at_bound": bool(rr["beta_at_bound"]),
                 "has_timeseries": True, "site_auc_abs": _auc(t, np.abs(y_site))}
            )

        latent[kin] = A

        # Summarize dominance of kinase sites in forming A_j(t) using a proxy importance score:
        # importance ~ |beta| * AUC(|site|)
        df_site = pd.DataFrame(site_rows)
        if df_site["has_timeseries"].any():
            df_site["importance"] = df_site["Beta"].abs() * df_site["site_auc_abs"].fillna(0.0)
            df_site_ok = df_site[df_site["has_timeseries"]].sort_values("importance", ascending=False)
            top_site = df_site_ok.iloc[0]["KinaseSite"]
            top_import = float(df_site_ok.iloc[0]["importance"])
            tot_import = float(df_site_ok["importance"].sum() + EPS)
            top_share = float(top_import / tot_import)
        else:
            top_site, top_share = None, np.nan

        latent_meta_rows.append(
            {
                "Kinase": kin,
                "n_beta_sites": int(len(grp)),
                "n_beta_at_bound": int(grp["beta_at_bound"].sum()),
                "frac_beta_at_bound": float(grp["beta_at_bound"].mean()),
                "top_kinase_site": top_site,
                "top_kinase_site_share": top_share,
                "latent_auc_abs": _auc(t, np.abs(latent[kin])),
                "latent_peak_abs": float(np.max(np.abs(latent[kin]))),
            }
        )

    latent_summary = pd.DataFrame(latent_meta_rows).sort_values(
        ["latent_auc_abs", "frac_beta_at_bound"], ascending=[False, False]
    )
    latent_summary.to_csv(OUTDIR / "kinopt_latent_kinase_activity_summary.csv", index=False)

    # ------------------------
    # 2) Target decomposition: contributions C_{i<-j}(t) = alpha_{i,j} * A_j(t)
    # ------------------------
    target_rows = []
    contrib_rows = []  # (optional, can be huge; we keep summary only)

    for (gene, psite), grp in edges.groupby(["Gene", "Psite"], sort=False):
        # compute contributions for each kinase
        kin_list = grp["Kinase"].tolist()
        a = grp["Alpha"].to_numpy(dtype=np.float64)

        # Build contribution matrix [nKin, nT]
        C = np.zeros((len(kin_list), nT), dtype=np.float64)
        missing_latent = 0
        for i, kin in enumerate(kin_list):
            Aj = latent.get(kin)
            if Aj is None:
                # If no beta/latent for this kinase, we cannot represent it under the model equation.
                # Set to 0 and mark missing.
                Aj = np.zeros((nT,), dtype=np.float64)
                missing_latent += 1
            C[i, :] = a[i] * Aj

        P_hat = C.sum(axis=0)  # model-consistent predicted target curve

        # dominance by window using |C|
        absC = np.abs(C)
        abs_total = absC.sum(axis=0) + EPS
        frac = absC / abs_total

        def dominant(idx: np.ndarray) -> tuple[str, float]:
            w = frac[:, idx].sum(axis=1)
            j = int(np.argmax(w))
            return kin_list[j], float(w[j] / (w.sum() + EPS))

        dom_overall_idx = int(np.argmax(absC.sum(axis=1)))
        dom_overall = kin_list[dom_overall_idx]
        dom_overall_share = float(absC.sum(axis=1)[dom_overall_idx] / (absC.sum() + EPS))

        dom_early, dom_early_share = dominant(early_idx)
        dom_mid, dom_mid_share = dominant(mid_idx)
        dom_late, dom_late_share = dominant(late_idx)

        # compare to observed target if available (not for fit scoring; for context)
        obs_y = obs_map.get(_as_key(gene, psite))
        obs_auc_abs = _auc(t, np.abs(obs_y)) if obs_y is not None else np.nan
        obs_peak_abs = float(np.max(np.abs(obs_y))) if obs_y is not None else np.nan

        target_rows.append(
            {
                "Gene": gene,
                "Psite": psite,
                "n_kinases": int(len(kin_list)),
                "n_missing_kinase_latent": int(missing_latent),
                "dominant_overall": dom_overall,
                "dominant_overall_share": dom_overall_share,
                "dominant_early": dom_early,
                "dominant_early_share": dom_early_share,
                "dominant_mid": dom_mid,
                "dominant_mid_share": dom_mid_share,
                "dominant_late": dom_late,
                "dominant_late_share": dom_late_share,
                "pred_auc_abs": _auc(t, np.abs(P_hat)),
                "pred_peak_abs": float(np.max(np.abs(P_hat))),
                "obs_auc_abs": obs_auc_abs,
                "obs_peak_abs": obs_peak_abs,
            }
        )

    target_dom = pd.DataFrame(target_rows).sort_values(
        ["dominant_overall_share", "n_kinases"], ascending=[False, True]
    )
    target_dom.to_csv(OUTDIR / "kinopt_target_dominant_kinases.csv", index=False)

    # ------------------------
    # 3) Kinase load across network: sum over targets of AUC(|alpha_{i,j} * A_j(t)|)
    # ------------------------
    edge_load_rows = []
    for _, r in edges.iterrows():
        gene, psite, kin, aij = r["Gene"], r["Psite"], r["Kinase"], float(r["Alpha"])
        Aj = latent.get(kin)
        if Aj is None:
            continue
        contrib = aij * Aj
        edge_load_rows.append(
            {
                "Kinase": kin,
                "Gene": gene,
                "Psite": psite,
                "Alpha": aij,
                "edge_load_auc_abs": _auc(t, np.abs(contrib)),
            }
        )

    edge_load = pd.DataFrame(edge_load_rows)
    kinase_load = (
        edge_load.groupby("Kinase")
        .agg(
            n_targets=("edge_load_auc_abs", "size"),
            total_load_auc_abs=("edge_load_auc_abs", "sum"),
        )
        .reset_index()
        .merge(
            latent_summary[["Kinase", "frac_beta_at_bound", "n_beta_at_bound", "latent_auc_abs", "latent_peak_abs"]],
            on="Kinase",
            how="left",
        )
        .sort_values(["total_load_auc_abs", "n_targets"], ascending=[False, False])
    )
    kinase_load.to_csv(OUTDIR / "kinopt_kinase_load.csv", index=False)

    # ------------------------
    # 4) Knockout effects per target: remove kinase j by alpha_{i,j}=0
    # ------------------------
    ko_rows = []
    for (gene, psite), grp in edges.groupby(["Gene", "Psite"], sort=False):
        kin_list = grp["Kinase"].tolist()
        a = grp["Alpha"].to_numpy(dtype=np.float64)

        # baseline prediction
        C = np.zeros((len(kin_list), nT), dtype=np.float64)
        for i, kin in enumerate(kin_list):
            Aj = latent.get(kin)
            if Aj is None:
                Aj = np.zeros((nT,), dtype=np.float64)
            C[i, :] = a[i] * Aj
        baseline = C.sum(axis=0)
        base_auc = _auc(t, np.abs(baseline))
        base_peak_abs = float(np.max(np.abs(baseline)))

        for i, kin in enumerate(kin_list):
            a_ko = a.copy()
            a_ko[i] = 0.0
            if KNOCKOUT_RENORMALIZE_ALPHA:
                s = a_ko.sum()
                if s > EPS:
                    a_ko = a_ko / s  # maintain sum_j alpha_{i,j}=1
                # if s==0, all were zero; keep zeros.

            # recompute prediction
            y = np.zeros((nT,), dtype=np.float64)
            for ii, kin2 in enumerate(kin_list):
                Aj = latent.get(kin2)
                if Aj is None:
                    Aj = np.zeros((nT,), dtype=np.float64)
                y += a_ko[ii] * Aj

            ko_auc = _auc(t, np.abs(y))
            ko_peak_abs = float(np.max(np.abs(y)))

            ko_rows.append(
                {
                    "Gene": gene,
                    "Psite": psite,
                    "KnockedKinase": kin,
                    "renormalize_alpha": bool(KNOCKOUT_RENORMALIZE_ALPHA),
                    "delta_auc_abs": base_auc - ko_auc,
                    "delta_peak_abs": base_peak_abs - ko_peak_abs,
                    "baseline_auc_abs": base_auc,
                    "baseline_peak_abs": base_peak_abs,
                }
            )

    ko = pd.DataFrame(ko_rows)
    ko["ko_rank_site"] = ko.groupby(["Gene", "Psite"])["delta_auc_abs"].rank(ascending=False, method="min")
    ko = ko.sort_values(["Gene", "Psite", "ko_rank_site"])
    ko.to_csv(OUTDIR / "kinopt_knockout_effects.csv", index=False)

    # ------------------------
    # Console quick view
    # ------------------------
    print("Wrote:")
    print(" -", OUTDIR / "kinopt_latent_kinase_activity_summary.csv")
    print(" -", OUTDIR / "kinopt_target_dominant_kinases.csv")
    print(" -", OUTDIR / "kinopt_kinase_load.csv")
    print(" -", OUTDIR / "kinopt_knockout_effects.csv")
    print()
    print("Top kinases by total load:")
    print(kinase_load.head(10).to_string(index=False))
    print()
    print("Most fragile targets (dominant_overall_share near 1):")
    print(target_dom.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
