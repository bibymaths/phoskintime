#!/usr/bin/env python3
"""
Compute discrete Fréchet distance per row between "Observed" and "Estimated" curves
from the same two Excel files:

  - tfopt_results.xlsx
  - kinopt_results.xlsx

Assumptions (kept minimal, but explicit):
- Each workbook contains sheets named exactly: "Observed" and "Estimated".
- Both sheets have the same rows in the same order (or share a stable row identifier column).
- Curve values are stored across multiple numeric columns per row (wide format).
- If there are non-numeric identifier columns, we automatically exclude them.

Interpretation guidance (practical):
- Fréchet distance is a *shape similarity* metric between two polylines.
  It is the minimum "leash length" needed to walk along both curves in order.
- Lower is better (0 means identical curves).
- It is scale-dependent: if your curves are on different scales (e.g., raw vs normalized),
  distances are not comparable across datasets. Standardize first if needed.
- If x-coordinates (time) are not provided and you only pass y-values, this script uses
  x = 0..T-1. That is fine if sampling is uniform; if time points are uneven, pass actual time.
- Large distances often come from:
  (1) vertical offsets (systematic bias),
  (2) peak timing shifts (phase lag),
  (3) one curve having extra wiggles / noise.
- Use this for ranking (best/worst fits) rather than as an absolute “good/bad” without context.
  Establish a baseline distribution: e.g., median and 90th percentile across rows.

Outputs:
- frechet_tfopt.csv
- frechet_kinopt.csv

Each output includes per-row distance, validity flags, ranks, and basic metadata columns (if present).

Usage: curve_similarity.py [--tfopt-xlsx=<path>] [--kinopt-xlsx=<path>] [--out-dir=<path>]

License: BSD-3-Clause
Author: Abhinav Mishra
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from frechet import frechet_distance


def _find_common_id_cols(obs: pd.DataFrame, est: pd.DataFrame) -> list[str]:
    """
    Identifies shared non-numeric columns to treat as metadata (identifiers).

    This function ensures that key identifiers like 'Gene' and 'Psite' appear
    first in the returned list if they exist, followed by any other common
    metadata columns.

    Args:
        obs (pd.DataFrame): DataFrame containing observed data.
        est (pd.DataFrame): DataFrame containing estimated data.

    Returns:
        list[str]: A list of column names to be treated as identifiers.
    """
    # Identify numeric columns to exclude
    obs_num = obs.select_dtypes(include="number").columns
    est_num = est.select_dtypes(include="number").columns

    # Identify metadata columns (non-numeric)
    obs_meta = [c for c in obs.columns if c not in obs_num]
    est_meta = [c for c in est.columns if c not in est_num]

    # Find intersection of metadata columns
    common = [c for c in obs_meta if c in est_meta]

    # enforce desired order
    ordered = []
    for key in ["Gene", "Psite", "PSite"]:
        if key in common:
            ordered.append(key)

    for c in common:
        if c not in ordered:
            ordered.append(c)

    return ordered


def _extract_numeric_matrix(df: pd.DataFrame, exclude_cols: list[str]) -> np.ndarray:
    """
    Extracts the numeric curve data from a DataFrame as a float64 matrix.

    Args:
        df (pd.DataFrame): The input DataFrame.
        exclude_cols (list[str]): List of columns to exclude (usually identifiers).

    Returns:
        np.ndarray: A 2D numpy array of shape (n_rows, n_timepoints).

    Raises:
        ValueError: If no numeric columns remain after exclusion.
    """
    # Filter columns: must not be in exclude list AND must be numeric type
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric curve columns found after excluding metadata columns.")

    # Convert to numpy array ensuring float64 precision
    mat = df[numeric_cols].to_numpy(dtype=np.float64)
    return mat


def _row_coords_from_y(y: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    """
    Convert a 1D y-array (values) into Nx2 coordinates (x, y) for frechet_distance.

    The Fréchet distance algorithm requires 2D coordinates (like a path on a 2D plane).
    This function pairs the y-values (curve amplitude) with x-values (time).

    Normalization:
        If 'x' is provided, it is normalized by dividing by max(x). This prevents
        the time scale (e.g., 0-960 minutes) from overwhelming the value scale
        (e.g., 0-1 intensity) in the distance calculation.

    Args:
        y (np.ndarray): 1D array of curve values.
        x (np.ndarray | None): 1D array of time points. If None, indices 0..T-1 are used.

    Returns:
        np.ndarray: Nx2 array where col 0 is x (normalized) and col 1 is y.

    Note:
    - If your time points are uneven, pass x explicitly (same length as y).
    - If y contains NaNs, frechet_distance will produce NaNs/inf behavior; handle upstream.
    """
    if x is None:
        x = np.arange(y.shape[0], dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y length mismatch")

        # Normalize x to [0, 1] range relative to the maximum time point
        # This keeps the 'x' contribution to the Euclidean distance proportional.
        x = x / x.max()

    return np.column_stack([x, y.astype(np.float64)])


def compute_frechet_per_row(
        obs: pd.DataFrame,
        est: pd.DataFrame,
        timepoints: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute Fréchet distance for each row between obs and est curves.

    This function aligns rows between the observed and estimated DataFrames,
    extracts the numeric curve data, and computes the discrete Fréchet distance
    for every pair of curves.

    Interpretation of columns in result:
    - frechet: lower means closer match in overall path/shape.
    - frechet_rank: 1 = best fit (smallest distance).

    Args:
        obs (pd.DataFrame): Observed data.
        est (pd.DataFrame): Estimated data.
        timepoints (np.ndarray | None): Array of time points corresponding to columns.

    Returns:
        pd.DataFrame: A DataFrame containing the distance, validity flags, ranks,
                      and original metadata.
    """
    if obs.shape[0] != est.shape[0]:
        raise ValueError(f"Row count mismatch: observed={obs.shape[0]} estimated={est.shape[0]}")

    # Find columns that are just labels/IDs to exclude from calculation but keep in output
    id_cols = _find_common_id_cols(obs, est)

    y_obs = _extract_numeric_matrix(obs, exclude_cols=id_cols)
    y_est = _extract_numeric_matrix(est, exclude_cols=id_cols)

    if y_obs.shape != y_est.shape:
        raise ValueError(f"Curve shape mismatch: observed={y_obs.shape} estimated={y_est.shape}")

    # If you have missing values, you need a policy. Minimal policy here: drop rows with any NaN.
    # Rationale: Fréchet distance expects complete polylines; imputing can be misleading.
    nan_mask = np.isnan(y_obs).any(axis=1) | np.isnan(y_est).any(axis=1)

    # We compute only on complete rows; keep a flag for transparency.
    distances = np.full((obs.shape[0],), np.nan, dtype=np.float64)

    for i in range(obs.shape[0]):
        if nan_mask[i]:
            continue

        # Prepare 2D coordinates (Time, Value) for both curves
        coords_obs = _row_coords_from_y(y_obs[i], x=timepoints)
        coords_est = _row_coords_from_y(y_est[i], x=timepoints)

        # Calculate the distance
        distances[i] = frechet_distance(coords_obs, coords_est)

    out = pd.DataFrame({"row_index": np.arange(obs.shape[0]), "frechet": distances, "has_nan": nan_mask})

    # carry metadata if any
    if id_cols:
        out = pd.concat([obs[id_cols].reset_index(drop=True), out], axis=1)

    # rank (NaNs go to bottom)
    out["frechet_rank"] = out["frechet"].rank(method="min", ascending=True, na_option="bottom").astype("Int64")

    # quick diagnostic bins (helps interpret without staring at raw numbers)
    # If scale varies, these bins are only meaningful *within* a file.
    finite = out["frechet"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) > 0:
        # Calculate quantiles to categorize fits
        q50 = float(finite.quantile(0.50))
        q90 = float(finite.quantile(0.90))
        q99 = float(finite.quantile(0.99))

        def bucket(v: float) -> str:
            """ Categorize distance into descriptive buckets. """
            if np.isnan(v):
                return "nan"
            if v <= q50:
                return "best_half"
            if v <= q90:
                return "mid"
            if v <= q99:
                return "poor"
            return "worst_1pct"

        out["frechet_bucket"] = out["frechet"].map(bucket)
        # Store quantiles in DataFrame attributes for reference
        out.attrs["frechet_quantiles"] = {"q50": q50, "q90": q90, "q99": q99}

    return out


def load_observed_estimated(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper to load the specific 'Observed' and 'Estimated' sheets from an Excel file.

    Args:
        xlsx_path (str): Path to the Excel file.

    Returns:
        tuple: (observed_df, estimated_df)
    """
    obs = pd.read_excel(xlsx_path, sheet_name="Observed")
    est = pd.read_excel(xlsx_path, sheet_name="Estimated")
    return obs, est


def main(
        tfopt_xlsx: str = "data/tfopt_results.xlsx",
        kinopt_xlsx: str = "data/kinopt_results.xlsx",
        out_dir: str = "results_scripts",
):
    """
    Main entry point for the Fréchet distance analysis.

    1. Defines timepoints for normalization.
    2. Loads TF and Kinase data.
    3. Computes distances.
    4. Saves results to CSV.
    5. Prints a summary to the console.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Specific timepoints for the experimental design
    TFOPT_TIMEPOINTS = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960], dtype=np.float64)

    KINOPT_TIMEPOINTS = np.array(
        [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
        dtype=np.float64
    )

    # --- TFopt ---
    # Load and process TF optimization results
    tf_obs, tf_est = load_observed_estimated(tfopt_xlsx)
    tf_res = compute_frechet_per_row(tf_obs, tf_est, timepoints=TFOPT_TIMEPOINTS)
    tf_res.to_csv(out_dir / "frechet_tfopt.csv", index=False)

    # --- Kinopt ---
    # Load and process Kinase optimization results
    kin_obs, kin_est = load_observed_estimated(kinopt_xlsx)
    # if 'GeneID' in obs.columns present, change it to 'Gene'
    if 'GeneID' in kin_obs.columns:
        kin_obs = kin_obs.rename(columns={'GeneID': 'Gene'})
    kin_res = compute_frechet_per_row(kin_obs, kin_est, timepoints=KINOPT_TIMEPOINTS)
    kin_res.to_csv(out_dir / "frechet_kinopt.csv", index=False)

    # Minimal console summaries to help interpretation
    def summarize(name: str, df: pd.DataFrame):
        finite = df["frechet"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"\n{name}:")
        print(f"  rows={len(df)}  computed={len(finite)}  skipped_nan={int(df['has_nan'].sum())}")
        if len(finite):
            print(f"  frechet median={finite.median():.6g}  p90={finite.quantile(0.9):.6g}  max={finite.max():.6g}")
            print("  best 5 rows (smallest distance):")
            print(df.nsmallest(5, "frechet")[df.columns[: min(8, len(df.columns))]].to_string(index=False))

    summarize("TFopt", tf_res)
    summarize("Kinopt", kin_res)

    print("\nWrote:")
    print(" -", out_dir / "frechet_tfopt.csv")
    print(" -", out_dir / "frechet_kinopt.csv")


if __name__ == "__main__":
    main()
