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


def process_and_scale_raw_data(df, time_points, id_cols, scale_method='fc_start', epsilon=1e-3):
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

    # 1. Standard Fold-Change (x / x0)
    elif scale_method == 'fc_start':
        start_vals = df_work[x_cols[0]].replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(start_vals, axis=0)

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


def calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in):
    """Calculate biologically constrained optimizer bounds
    
    Args:
        idx: Input value used by this routine.
        df_prot: Input value used by this routine.
        df_rna: Input value used by this routine.
        tf_mat: Input value used by this routine.
        kin_in: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    logger.info("=" * 60)
    logger.info("[Bounds] CALCULATION: Performing Topological & Kinetic Analysis")

    # 1. ANALYZE DATA DYNAMIC RANGE
    # Cap the maximum Fold Change the model is ALLOWED to reach.
    max_prot_fc = df_prot['fc'].max() if (df_prot is not None and not df_prot.empty) else 5.0
    max_rna_fc = df_rna['fc'].max() if (df_rna is not None and not df_rna.empty) else 5.0

    # Safety buffer: Tighter than before (1.5x instead of 2.0x)
    safe_prot_max = max(2.0, max_prot_fc * 1.5)
    safe_rna_max = max(2.0, max_rna_fc * 1.5)

    # 2. mRNA KINETICS (A_i, B_i)
    # Bounds: 0.005 (~2.3h) to 0.1 (~7min)
    # We raise the floor to prevent 'stuck' mRNA
    b_min, b_max = 0.005, 0.15

    # A_i (Basal Synthesis)
    # Synthesis must balances Degradation at steady state (FC=1).
    # A_i approx B_i. We allow slight deviation for basal repression/activation.
    a_min = b_min * 0.1
    a_max = b_max * safe_rna_max

    # 3. PROTEIN KINETICS (C_i, D_i)
    # CRITICAL FIX: Raise the degradation floor.
    # If proteins degrade too slowly, errors accumulate over 960 minutes.
    # New Floor: 0.01 (approx 70 min half-life). Nothing lives forever.
    d_min, d_max = 0.01, 0.10

    # C_i (Translation)
    # Strict linkage: Synthesis cannot exceed Degradation * Max_Fold_Change
    # C_max = D_max * Max_FC.
    # If D=0.1 and MaxFC=5, C cannot exceed 0.5.
    c_min = d_min * 0.1
    c_max = d_max * safe_prot_max

    # 4. TOPOLOGICAL SENSITIVITY (E_i, tf_scale)
    # For large networks (200+ nodes), additive inputs can be huge.
    n_edges = tf_mat.nnz
    avg_density = n_edges / max(1, idx.N)

    if avg_density < 2.0:
        # Sparse (Sub-network): Allow higher gain
        e_max = 20.0
        tf_scale_min, tf_scale_max = 0.5, 5.0
    else:
        # Dense (Global): Clamp gain hard to prevent feedback explosions
        e_max = 5.0  # Reduced from 20.0
        tf_scale_min, tf_scale_max = 0.1, 2.5  # Reduced from 10.0

    # 5. SIGNALING VELOCITY (c_k, Dp_i)
    # Phospho is working well, keep these bounds wide to maintain signal fit.
    dp_min, dp_max = 0.01, 5.0

    # Check input variance. If inputs are flat, we need higher c_k.
    kin_variance = np.var(kin_in.Kmat)
    ck_max = 15.0 if kin_variance < 0.02 else 5.0

    bounds_dict = {
        "c_k": (0.01, ck_max),
        "A_i": (a_min, a_max),
        "B_i": (b_min, b_max),
        "C_i": (c_min, c_max),
        "D_i": (d_min, d_max),
        "Dp_i": (dp_min, dp_max),
        "E_i": (0.0, e_max),
        "tf_scale": (tf_scale_min, tf_scale_max)
    }

    # Model-specific adjustments
    # MODEL: 0 distributive, 1 sequential, 2 combinatorial, 4 saturating
    if MODEL == 1:
        # Sequential: tighter kinase gain, faster phospho turnover
        bounds_dict["Dp_i"] = (0.01, 5.0)
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, max(3.0, 0.75 * ck_hi))

    elif MODEL == 2:
        # Combinatorial: clamp hard (many transitions per state)
        bounds_dict["Dp_i"] = (0.01, 5.0)
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, min(2.5, ck_hi))

        # E_i is dephosph/backflow; keep smaller to avoid stiff cycling
        e_lo, e_hi = bounds_dict["E_i"]
        if avg_density >= 2.0:
            bounds_dict["E_i"] = (e_lo, min(e_hi, 2.5))
        else:
            bounds_dict["E_i"] = (e_lo, min(e_hi, 8.0))

    elif MODEL == 4:
        # Saturating: bounded forward flux allows more kinase gain and TF scale
        bounds_dict["Dp_i"] = (0.01, 5.0)
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, min(10.0, 1.5 * ck_hi))

        tf_lo, tf_hi = bounds_dict["tf_scale"]
        if avg_density >= 2.0:
            bounds_dict["tf_scale"] = (tf_lo, max(tf_hi, 6.0))
        else:
            bounds_dict["tf_scale"] = (tf_lo, max(tf_hi, 10.0))

    logger.info("-" * 60)
    logger.info(f"{'Param':<10} | {'Min':<10} | {'Max':<10} | {'Constraint Logic'}")
    logger.info("-" * 60)
    logger.info(f"{'C_i':<10} | {c_min:<10.4f} | {c_max:<10.4f} | {'Linked to D_max * MaxFC'}")
    logger.info(f"{'D_i':<10} | {d_min:<10.4f} | {d_max:<10.4f} | {'Floor raised to 0.01'}")
    logger.info(f"{'tf_scale':<10} | {tf_scale_min:<10.3f} | {tf_scale_max:<10.3f} | {'Clamped for stability'}")
    logger.info("=" * 60)

    return bounds_dict


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
