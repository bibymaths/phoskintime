"""Normalize input data, transform positive parameters, and compute optimization bounds; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from numba import njit
from config.config import setup_logger
from networkmodel.config import MODEL, RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def _normcols(df):
    """Handle internal normcols"""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _base_idx(times, t0):
    """Handle internal base idx"""
    return np.int32(int(np.argmin(np.abs(times - float(t0)))))


def _find_col(df, cands):
    """Handle internal find col"""
    for c in cands:
        if c in df.columns:
            return c
    return None


def normcols(df):
    """Normalize column labels
    
    Args:
        df: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    return _normcols(df)


def find_col(df, cands):
    """Find the first matching column name
    
    Args:
        df: Input value used by this routine.
        cands: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    return _find_col(df, cands)


def normalize_fc_to_t0(df):
    """Normalize fold-change values to the baseline time
    
    Args:
        df: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    df = df.copy()

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["fc"] = pd.to_numeric(df["fc"], errors="coerce")
    df = df.dropna(subset=["time", "fc"])

    has_psite = "psite" in df.columns
    if has_psite:
        df["psite"] = (
            df["psite"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": ""})
        )

    keys = ["protein"] + (["psite"] if has_psite else [])

    t0 = (
        df.loc[df["time"].eq(0.0), keys + ["fc"]]
        .drop_duplicates(subset=keys, keep="last")
        .set_index(keys)["fc"]
    )

    def _norm_row(r):
        k = (r["protein"], r["psite"]) if has_psite else r["protein"]
        base = t0.get(k, np.nan)
        if not np.isfinite(base) or base == 0.0:
            return np.nan
        return r["fc"] / base

    df["fc"] = df.apply(_norm_row, axis=1)
    return df.dropna(subset=["fc"])


def process_and_scale_raw_data(df, time_points, id_cols, scale_method='fc_start', epsilon=1e-3, report=None):
    """Convert wide raw data into scaled tidy time-series data
    
    Args:
        df: Input value used by this routine.
        time_points: Input value used by this routine.
        id_cols: Input value used by this routine.
        scale_method: Input value used by this routine.
        epsilon: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=id_cols + ['time', 'fc'])

    # --- A. Identify and Sort X-Columns ---
    x_cols = [c for c in df.columns if re.fullmatch(r"x\d+", str(c))]

    if not x_cols:
        logger.warning(f"No 'x' columns found. Returning empty.")
        return pd.DataFrame()

    x_cols.sort(key=lambda c: int(c[1:]))

    if len(x_cols) > len(time_points):
        x_cols = x_cols[:len(time_points)]

    t_map = {xc: tp for xc, tp in zip(x_cols, time_points)}

    # --- B. Scale Data (Non-Negative Matrix Operations) ---
    df_work = df.copy()

    # Ensure numeric
    for c in x_cols:
        df_work[c] = pd.to_numeric(df_work[c], errors='coerce')

    # 0. Raw / None (Just formatting)
    if scale_method in ['raw', 'none']:
        pass  # Do nothing to values

    # 1. Standard Fold-Change (x / x0): divide by the series' own value at the baseline time (the first time
    #    point of this grid; networkmodel.baselines.BASELINE_TIMES). Zero, near-zero or non-finite baselines are
    #    never divided by: those series are skipped and counted (N7).
    elif scale_method == 'fc_start':
        from networkmodel.baselines import DATA_BASELINE_NEAR_ZERO

        start_vals = df_work[x_cols[0]]
        bad = ~np.isfinite(start_vals.to_numpy(dtype=float)) | (np.abs(start_vals.to_numpy(dtype=float))
                                                                 < DATA_BASELINE_NEAR_ZERO)
        n_bad = int(bad.sum())
        if n_bad:
            logger.warning(f"[Scaled] fc_start: {n_bad} of {len(df_work)} series have a zero, near-zero or missing "
                           f"baseline at t={t_map[x_cols[0]]} and are skipped (N7).")
        df_work = df_work.loc[~bad].copy()
        start_vals = start_vals.loc[~bad]
        df_work[x_cols] = df_work[x_cols].div(start_vals, axis=0)
        if report is not None:
            report.update({"scale_method": "fc_start", "baseline_time": float(t_map[x_cols[0]]),
                           "series_total": int(len(bad)), "skipped_series": n_bad})

    # 2. Robust Fold-Change
    elif scale_method == 'robust_fc':
        baseline = df_work[x_cols[0]] + epsilon
        df_work[x_cols] = df_work[x_cols].div(baseline, axis=0)

    # 3. Max Scaling (Peak Normalization)
    elif scale_method == 'max_scale':
        peaks = df_work[x_cols].max(axis=1).replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(peaks, axis=0)

    # 4. Mean Scaling
    elif scale_method == 'mean_scale':
        means = df_work[x_cols].mean(axis=1).replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(means, axis=0)

    # 5. L2 Norm Scaling
    elif scale_method == 'l2_norm':
        l2 = np.sqrt(np.square(df_work[x_cols]).sum(axis=1)).replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(l2, axis=0)

    # --- C. Melt to Tidy Format ---
    valid_ids = [c for c in id_cols if c in df_work.columns]

    melted = df_work[valid_ids + x_cols].melt(
        id_vars=valid_ids,
        value_vars=x_cols,
        var_name="xcol",
        value_name="fc"
    )

    melted["time"] = melted["xcol"].map(t_map)
    melted = melted.dropna(subset=["fc", "time"])

    # Text normalization
    if "protein" in melted.columns:
        melted["protein"] = melted["protein"].astype(str).str.strip().str.upper()
    if "psite" in melted.columns:
        melted["psite"] = melted["psite"].fillna("").astype(str).str.strip()

    logger.info(f"[Scaled] Processed {len(melted)} rows with {len(valid_ids)} IDs and {len(x_cols)} time points.")
    logger.info(f"[Scaled] Scaling method: {scale_method}")

    return melted[valid_ids + ["time", "fc"]]


@njit(cache=True, fastmath=True, nogil=True)
def softplus(x):
    """Apply a numerically stable softplus transform
    
    Args:
        x: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    out = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if xi > 20.0:
            out[i] = xi
        else:
            out[i] = np.log1p(np.exp(xi))
    return out


@njit(cache=True, fastmath=True, nogil=True)
def inv_softplus(y):
    """Apply the inverse softplus transform
    
    Args:
        y: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    out = np.empty_like(y)
    for i in range(y.size):
        yi = y[i]
        if yi < 1e-12:
            yi = 1e-12
        out[i] = np.log(np.expm1(yi))
    return out


# N22 (P3 decision 1): the first-order rate constants (unit 1/min) and the observation grid that resolves each.
#   B_i  mRNA degradation                      -> RNA grid
#   D_i  protein degradation                   -> protein grid
#   E_i  dephosphorylation (site -> unphospho) -> phospho grid
#   Dp_i extra decay of a phosphorylated state -> phospho grid
#   c_k  phosphorylation rate per unit kinase drive (K(0) = 1 since N7) -> phospho grid (lower bound only; the
#        upper bound is the c_k rule below, decision 3)
# Not rates of this kind: A_i (zero-order mRNA synthesis, FC/min), C_i (translation, coupled to R; decision 2),
# tf_scale (dimensionless gain).
DESIGN_RATE_GRIDS = {"B_i": "rna", "D_i": "protein", "E_i": "phospho", "Dp_i": "phospho", "c_k": "phospho"}
# delta: the smallest change the data resolve, as a fraction. Owner rule: the median relative SD across replicates if
# replicate data exist; the networkmodel inputs carry no replicates, so delta = 0.1 is an ASSUMPTION.
DESIGN_DELTA_ASSUMED = 0.1


def design_rate_bounds(time_points=None, delta=None):
    """Sampling-design bounds for first-order rate constants (N22, P3 decision 1).

    A rate k is resolvable if its process completes at least delta of its course over the observed span T
    (k >= -ln(1 - delta) / T) and at most 1 - delta within the shortest sampling interval dt_min
    (k <= -ln(delta) / dt_min). T = max(t) - min(t) and dt_min = min(diff(t)) of the configured observation grid of
    the modality that sees the rate (DESIGN_RATE_GRIDS).

    Args:
        time_points: dict with "protein", "rna", "phospho" grids (minutes); default: networkmodel.config grids.
        delta: resolution fraction; default DESIGN_DELTA_ASSUMED (assumption: no replicate data).

    Returns:
        dict with delta, delta_source, per-grid T/dt_min/k_min/k_max and per-group bounds and half-lives (minutes).
    """
    if time_points is None:
        from networkmodel.config import TIME_POINTS_PHOSPHO, TIME_POINTS_PROTEIN, TIME_POINTS_RNA

        time_points = {"protein": TIME_POINTS_PROTEIN, "rna": TIME_POINTS_RNA, "phospho": TIME_POINTS_PHOSPHO}
    if delta is None:
        delta, source = DESIGN_DELTA_ASSUMED, "assumption: no replicate data in the networkmodel inputs"
    else:
        source = "given"
    delta = float(delta)
    if not 0.0 < delta < 0.5:
        raise ValueError(f"design delta must lie in (0, 0.5), got {delta}")
    grids = {}
    for name in ("protein", "rna", "phospho"):
        t = np.unique(np.asarray(time_points[name], dtype=float))
        if t.size < 2:
            raise ValueError(f"time grid {name!r} needs at least two distinct points for the design bounds")
        T, dt = float(t.max() - t.min()), float(np.min(np.diff(t)))
        grids[name] = {"T_min": T, "dt_min_min": dt, "k_min": float(-np.log1p(-delta) / T),
                       "k_max": float(-np.log(delta) / dt)}
    groups = {}
    for g, grid in DESIGN_RATE_GRIDS.items():
        k_lo, k_hi = grids[grid]["k_min"], grids[grid]["k_max"]
        groups[g] = {"grid": grid, "k_min": k_lo, "k_max": k_hi, "half_life_max_min": float(np.log(2) / k_lo),
                     "half_life_min_min": float(np.log(2) / k_hi)}
    return {"delta": delta, "delta_source": source, "grids": grids, "groups": groups,
            "formula": "k_min = -ln(1-delta)/T, k_max = -ln(delta)/dt_min"}


def calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in, *, time_points=None, delta=None):
    """Calculate the optimizer bounds (physical) of every parameter group.

    Rate constants (DESIGN_RATE_GRIDS) take the sampling-design bounds of design_rate_bounds (N22).

    Args:
        idx: network index (idx.N).
        df_prot: protein data (column fc) of the model proteins.
        df_rna: RNA data (column fc) of the model proteins.
        tf_mat: TF matrix (sparse).
        kin_in: kinase drive (Kmat).
        time_points: observation grids for the design bounds (default: networkmodel.config).
        delta: design resolution fraction (default: DESIGN_DELTA_ASSUMED).

    Returns:
        dict group -> (min, max), physical.
    """
    _design_info = design_rate_bounds(time_points, delta)
    design = _design_info["groups"]
    logger.info("=" * 60)
    logger.info("[Bounds] CALCULATION: Performing Topological & Kinetic Analysis")

    # 1. DATA RANGE (fold changes of the model proteins), widened by s = 1.5 on each side; always contains 1.
    # Empty modality: range [1/5, 5] (former fallback of 5.0).
    s_buf = 1.5

    def _fc_range(df):
        if df is None or df.empty:
            return 1.0 / 5.0, 5.0
        fc = np.asarray(df["fc"], dtype=float)
        fc = fc[np.isfinite(fc) & (fc > 0)]
        if fc.size == 0:
            return 1.0 / 5.0, 5.0
        return min(1.0, float(fc.min())) / s_buf, max(1.0, float(fc.max())) * s_buf

    r_lo, r_hi = _fc_range(df_rna)
    p_lo, p_hi = _fc_range(df_prot)

    # 2./3. Rates B_i, D_i: sampling design. A_i and C_i are set after the model adjustments (below), because their
    # coupling uses the final tf_scale and Dp_i bounds (pre-merge item 2).
    b_min, b_max = design["B_i"]["k_min"], design["B_i"]["k_max"]
    d_min, d_max = design["D_i"]["k_min"], design["D_i"]["k_max"]

    # 4. TOPOLOGICAL SENSITIVITY (tf_scale; E_i is a rate since N22)
    # For large networks (200+ nodes), additive inputs can be huge.
    n_edges = tf_mat.nnz
    avg_density = n_edges / max(1, idx.N)

    if avg_density < 2.0:
        # Sparse (Sub-network): Allow higher gain
        tf_scale_min, tf_scale_max = 0.5, 5.0
    else:
        # Dense (Global): Clamp gain hard to prevent feedback explosions
        tf_scale_min, tf_scale_max = 0.1, 2.5  # Reduced from 10.0

    # 5. SIGNALING VELOCITY (c_k, Dp_i)
    # Phospho is working well, keep these bounds wide to maintain signal fit.
    dp_min, dp_max = design["Dp_i"]["k_min"], design["Dp_i"]["k_max"]  # sampling design (N22)

    # c_k upper bound: continuous rule clip(0.1 / var(Kmat), 5, 15) (N25/N23, P3 decision 3). It replaces the step
    # 15 if var < 0.02 else 5, which jumped at the threshold (N7 moved the ABL2 data across it); the end points and
    # the threshold value (0.1 / 0.02 = 5) are kept. Flatter drives allow a larger gain.
    kin_variance = float(np.var(kin_in.Kmat))
    ck_max = 15.0 if kin_variance <= 0.0 else float(np.clip(0.1 / kin_variance, 5.0, 15.0))

    bounds_dict = {
        "c_k": (design["c_k"]["k_min"], ck_max),  # lower: sampling design (N22)
        "A_i": (np.nan, np.nan),  # set below (coupled)
        "B_i": (b_min, b_max),
        "C_i": (np.nan, np.nan),  # set below (coupled)
        "D_i": (d_min, d_max),
        "Dp_i": (dp_min, dp_max),
        "E_i": (design["E_i"]["k_min"], design["E_i"]["k_max"]),  # sampling design (N22)
        "tf_scale": (tf_scale_min, tf_scale_max)
    }

    # Model-specific adjustments
    # MODEL: 0 distributive, 1 sequential, 2 combinatorial, 4 saturating
    if MODEL == 1:
        # Sequential: tighter kinase gain, faster phospho turnover
        # Dp_i: sampling design for every model (N22); the former per-model re-set to (0.01, 5.0) is removed.
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, max(3.0, 0.75 * ck_hi))

    elif MODEL == 2:
        # Combinatorial: clamp hard (many transitions per state)
        # Dp_i: sampling design for every model (N22); the former per-model re-set to (0.01, 5.0) is removed.
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, min(2.5, ck_hi))
        # E_i: sampling design for every model (N22); the former clamp to 2.5 / 8.0 is removed.

    elif MODEL == 4:
        # Saturating: bounded forward flux allows more kinase gain and TF scale
        # Dp_i: sampling design for every model (N22); the former per-model re-set to (0.01, 5.0) is removed.
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, min(10.0, 1.5 * ck_hi))

        tf_lo, tf_hi = bounds_dict["tf_scale"]
        if avg_density >= 2.0:
            bounds_dict["tf_scale"] = (tf_lo, max(tf_hi, 6.0))
        else:
            bounds_dict["tf_scale"] = (tf_lo, max(tf_hi, 10.0))

    # A_i, C_i coupled to the rate bounds and the data range (N22 decision 2; pre-merge item 2).
    # mRNA: R_ss = A f / B, f the TF modulation of backend._synth: u = u_raw/(1+|u_raw|) in (-1, 1); activation
    # f = 1 + tf u/(1+u) in [1, 1 + tf/2), repression f = 1/(1 + tf |u|) in (1/(1 + tf), 1]. With tf_max the final
    # tf_scale upper bound and any TF edge: A_i = [B_min R_lo / (1 + tf_max/2), B_max R_hi (1 + tf_max)]; without TF
    # edges f = 1 and A_i = [B_min R_lo, B_max R_hi].
    tf_max = float(bounds_dict["tf_scale"][1])
    f_hi, f_lo = (1.0 + tf_max / 2.0, 1.0 / (1.0 + tf_max)) if tf_mat.nnz > 0 else (1.0, 1.0)
    bounds_dict["A_i"] = (b_min * r_lo / f_hi, b_max * r_hi / f_lo)
    # Protein: the observable is the total T over all states; dT/dt = trans - D T - sum_j Dp_j x_j. E_i moves a
    # site back to the unphosphorylated pool inside T, so it has no net term in T (it is not an outflux of the
    # observable). T_ss = trans / (D + Dp_eff), 0 <= Dp_eff <= kappa Dp_max: kappa = 1 for models 0, 1, 4 (one Dp per
    # state, the states sum to T), kappa = max sites per protein for model 2 (a state decays D + sum of its Dp).
    # C_i = [D_min P_lo / R_hi, (D_max + kappa Dp_max) P_hi / R_lo]; saturating model (translation C R/(1+R)): the
    # same times (1 + R_hi) and (1 + R_lo).
    dp_hi = float(bounds_dict["Dp_i"][1])
    kappa = max([int(x) for x in getattr(idx, "n_sites", [1])] + [1]) if MODEL == 2 else 1
    c_min, c_max = d_min * p_lo / r_hi, (d_max + kappa * dp_hi) * p_hi / r_lo
    if MODEL == 4:
        c_min, c_max = c_min * (1.0 + r_hi), c_max * (1.0 + r_lo)
    bounds_dict["C_i"] = (c_min, c_max)

    # Log the final bounds of all eight groups (after model-specific adjustments) (N2).
    logger.info("-" * 60)
    logger.info(f"{'Param':<10} | {'Min':<10} | {'Max':<10}")
    logger.info("-" * 60)
    for name in ("A_i", "B_i", "C_i", "D_i", "E_i", "c_k", "tf_scale", "Dp_i"):
        lo, hi = bounds_dict[name]
        logger.info(f"{name:<10} | {lo:<10.4g} | {hi:<10.4g}")
    logger.info("[Bounds] Rate constants (1/min) from the sampling design (N22): delta=%g (%s); %s",
                _design_info["delta"], _design_info["delta_source"], _design_info["formula"])
    for name, g in design.items():
        logger.info("[Bounds]   %-5s grid=%-7s k=[%.4g, %.4g] half-life=[%.4g, %.4g] min", name, g["grid"],
                    g["k_min"], g["k_max"], g["half_life_min_min"], g["half_life_max_min"])
    logger.info("=" * 60)

    return bounds_dict


# N24 (P3 decision 4): an edge weight counts as absent if |W| <= ZERO_INFLUENCE_THRESHOLD * max(1, max |W|).
ZERO_INFLUENCE_THRESHOLD = 1e-12
ZERO_INFLUENCE_LABEL = "not identifiable: no data influence"


def zero_influence_parameters(idx, W_global, tf_mat, threshold=ZERO_INFLUENCE_THRESHOLD):
    """Detect parameters without data influence at network construction (N24, P3 decision 4).

    c_k[k] enters the model only through S = W @ (c_k * K) and, when kinase k drives a model protein (driver map:
    the protein itself or an orphan proxied by k), through that protein's TF output. It has no data influence if every
    edge of k is absent (|W| <= threshold * max(1, max |W|)) and no protein it drives has a TF edge.
    A site whose edges are all absent has no kinase inflow (KinOpt alpha = 0); it is reported, not masked: its Dp_i
    still sets the decay of the site's initial value.

    Returns:
        dict: threshold, c_k (kinase names, in idx.kinases order), sites_without_kinase_inflow (protein_site names).
    """
    W = W_global.toarray() if hasattr(W_global, "toarray") else np.asarray(W_global, dtype=float)
    W = np.abs(np.asarray(W, dtype=float))
    tol = float(threshold) * max(1.0, float(W.max()) if W.size else 0.0)
    present = W > tol
    TF = tf_mat.toarray() if hasattr(tf_mat, "toarray") else np.asarray(tf_mat, dtype=float)
    tf_out = np.abs(np.asarray(TF, dtype=float)).sum(axis=0) > 0  # column i: protein i regulates someone
    drives_tf = np.zeros(len(idx.kinases), dtype=bool)
    drivers = {}
    for k_name in idx.kinases:
        if k_name in idx.p2i:
            drivers[idx.p2i[k_name]] = idx.k2i[k_name]
    for orphan, proxy in getattr(idx, "proxy_map", {}).items():
        if orphan in idx.p2i and proxy in idx.k2i:
            drivers[idx.p2i[orphan]] = idx.k2i[proxy]
    for p_i, k_i in drivers.items():
        if p_i < tf_out.size and tf_out[p_i]:
            drives_tf[k_i] = True
    has_edge = present.any(axis=0) if present.size else np.zeros(len(idx.kinases), dtype=bool)
    ck = [k for j, k in enumerate(idx.kinases) if not has_edge[j] and not drives_tf[j]]
    site_names = [f"{p}_{s}" for p, sites in zip(idx.proteins, idx.sites) for s in sites]
    inflow = present.any(axis=1) if present.size else np.zeros(len(site_names), dtype=bool)
    no_inflow = [n for n, ok in zip(site_names, inflow) if not ok]
    # E_i of a protein without phosphosites: E_i acts only on site states (every model), so its data gradient is 0.
    n_sites = getattr(idx, "n_sites", None)
    e_i = [] if n_sites is None else [p for p, ns in zip(idx.proteins, n_sites) if int(ns) == 0]
    return {"threshold": float(threshold), "c_k": ck, "E_i": e_i, "sites_without_kinase_inflow": no_inflow}


def apply_zero_influence_mask(zero_influence, idx, slices, theta0, xl, xu):
    """Fix the zero-influence c_k and E_i at their start values (lower = upper = theta0; N24, P3 decision 4).

    Returns:
        (xl, xu, mask, names): new bound arrays (inputs untouched), the boolean mask over theta and parameter names.
    """
    xl, xu = np.array(xl, dtype=float, copy=True), np.array(xu, dtype=float, copy=True)
    mask = np.zeros(np.asarray(theta0).shape[0], dtype=bool)
    names = []
    sl = slices["c_k"]
    for k_name in zero_influence["c_k"]:
        j = sl.start + idx.k2i[k_name]
        mask[j] = True
        names.append(f"c_k[{k_name}]")
    if "E_i" in slices:
        for p_name in zero_influence.get("E_i", []):
            j = slices["E_i"].start + list(idx.proteins).index(p_name)
            mask[j] = True
            names.append(f"E_i[{p_name}]")
    xl[mask] = np.asarray(theta0, dtype=float)[mask]
    xu[mask] = np.asarray(theta0, dtype=float)[mask]
    return xl, xu, mask, names


# N22 (P3 decision 5): a value within this relative distance (physical) of a bound is not a point estimate.
BOUND_PROXIMITY_REL = 0.01


def _fmt_half_life(minutes):
    """Half-life in readable units: seconds below 2 min, minutes below 2 h, hours otherwise."""
    if minutes < 2.0:
        return f"{minutes * 60.0:.1f} s"
    if minutes < 120.0:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.1f} h"


def parameter_status(parameter_names, theta_raw, xl, xu, slices, design_info, fixed_names=(),
                     rel=BOUND_PROXIMITY_REL):
    """Status of every fitted parameter (N22, P3 decision 5).

    status: "estimate"; "below_resolvable_rate" / "above_resolvable_rate" (within rel of a sampling-design bound:
    label "below resolvable rate (half-life > X)" / "above resolvable rate (half-life < X)"); "at_bound" (within rel
    of a bound that is not a design bound); "fixed_no_data_influence" (masked, N24). Only "estimate" is a point
    estimate. Physical values (softplus of raw).

    Returns:
        DataFrame: parameter, group, value, lower, upper, status, label, is_estimate.
    """
    th = np.asarray(theta_raw, dtype=float)
    lo = np.logaddexp(0.0, np.asarray(xl, dtype=float))  # softplus, as the model uses it
    hi = np.logaddexp(0.0, np.asarray(xu, dtype=float))
    val = np.logaddexp(0.0, th)
    group_of = {}
    for g, sl in slices.items():
        for j in range(sl.start, sl.stop):
            group_of[j] = g
    design = design_info["groups"] if design_info else {}
    fixed = set(fixed_names)
    rows = []
    for j, name in enumerate(parameter_names):
        g = group_of.get(j, "")
        v, a, b = float(val[j]), float(lo[j]), float(hi[j])
        status, label = "estimate", "estimate"
        if name in fixed:
            status, label = "fixed_no_data_influence", ZERO_INFLUENCE_LABEL
        else:
            near_lo = abs(v - a) <= rel * max(abs(a), 1e-300)
            near_hi = b > a and abs(b - v) <= rel * abs(b)
            dg = design.get(g)
            design_lo = dg is not None and np.isclose(a, dg["k_min"], rtol=1e-9)
            design_hi = dg is not None and g != "c_k" and np.isclose(b, dg["k_max"], rtol=1e-9)
            if near_lo and design_lo and g == "c_k":
                # c_k is a gain on a production rate (S = sum_k W c_k K), not a decay: state its time constant
                status = "below_resolvable_rate"
                label = (f"below resolvable rate (time constant 1/c_k > {_fmt_half_life(1.0 / dg['k_min'])} "
                         "at unit drive and weight)")
            elif near_lo and design_lo:
                status = "below_resolvable_rate"
                label = f"below resolvable rate (half-life > {_fmt_half_life(dg['half_life_max_min'])})"
            elif near_hi and design_hi:
                status = "above_resolvable_rate"
                label = f"above resolvable rate (half-life < {_fmt_half_life(dg['half_life_min_min'])})"
            elif near_lo or near_hi:
                status = "at_bound"
                label = f"at optimizer bound ({'lower' if near_lo else 'upper'}; not a design bound)"
        rows.append({"parameter": name, "group": g, "value": v, "lower": a, "upper": b, "status": status,
                     "label": label, "is_estimate": status == "estimate"})
    return pd.DataFrame(rows)


def _nondegenerate(mask_xl, mask_xu, eps=1e-14):
    """Handle internal nondegenerate"""
    return (mask_xu - mask_xl) > eps


def get_optimized_sets(idx, slices, xl, xu, eps=1e-14):
    """Report which parameter groups are optimized
    
    Args:
        idx: Input value used by this routine.
        slices: Input value used by this routine.
        xl: Input value used by this routine.
        xu: Input value used by this routine.
        eps: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
    """
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    N = idx.N

    # Protein-level params (length N each)
    protein_free = np.zeros(N, dtype=bool)
    for key in ("A_i", "B_i", "C_i", "D_i", "E_i"):
        if key not in slices:
            continue
        sl = slices[key]  # slice in theta
        free = _nondegenerate(xl[sl], xu[sl], eps=eps)
        if free.size != N:
            raise ValueError(f"{key} slice size {free.size} != idx.N {N}")
        protein_free |= free

    opt_proteins = set(np.array(idx.proteins)[protein_free].tolist())

    # Kinase activities c_k (length nK)
    opt_kinases = set()
    if "c_k" in slices:
        sl = slices["c_k"]
        free = _nondegenerate(xl[sl], xu[sl], eps=eps)
        opt_kinases = set(np.array(idx.kinases)[free].tolist())

    # Site-level params Dp_i (length total_sites)
    opt_sites = set()
    if "Dp_i" in slices:
        sl = slices["Dp_i"]
        free = _nondegenerate(xl[sl], xu[sl], eps=eps)

        # Build flattened site labels in the same order as Dp_i indexing
        flat_labels = []
        for i, p in enumerate(idx.proteins):
            for s in idx.sites[i]:
                flat_labels.append(f"{p}_{s}")
        flat_labels = np.array(flat_labels)

        if free.size != flat_labels.size:
            raise ValueError(f"Dp_i slice size {free.size} != total_sites {flat_labels.size}")

        opt_sites = set(flat_labels[free].tolist())

    return opt_proteins, opt_sites, opt_kinases
