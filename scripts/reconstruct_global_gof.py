import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Add current directory to path to import config
sys.path.append(os.getcwd())

try:
    from global_model.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO
except ImportError:
    print("[Warning] Could not import TIME_POINTS from config. Using defaults.")
    TIME_POINTS_PROTEIN = np.array([0, 5, 10, 20, 30, 60, 120, 240])  # Example default
    TIME_POINTS_RNA = np.array([0, 15, 30, 60, 120, 240])
    TIME_POINTS_PHOSPHO = np.array([0, 1, 5, 10, 20, 60])


def clean_and_melt(df, time_points, type_label):
    """
    Converts raw wide input (GeneID, x1, x2...) into clean long format (protein, time, fc).
    Handles filtering for Protein vs Phospho.
    """
    if df is None or df.empty:
        return None

    # 1. Rename GeneID -> protein
    if "GeneID" in df.columns:
        df = df.rename(columns={"GeneID": "protein"})

    # 2. Rename Psite -> psite (if exists)
    if "Psite" in df.columns:
        df = df.rename(columns={"Psite": "psite"})

    # 3. Filter Protein vs Phospho
    # If type_label is "Protein", we want rows where psite is NaN or empty
    # If type_label is "Phospho", we want rows where psite is valid
    if "psite" in df.columns:
        if type_label == "Protein":
            # Keep only rows where psite is NaN
            df = df[df["psite"].isna() | (df["psite"] == "")]
            # Drop the psite column as it's useless for protein
            df = df.drop(columns=["psite"], errors="ignore")
        elif type_label == "Phospho":
            # Keep rows where psite is NOT NaN
            df = df[df["psite"].notna() & (df["psite"] != "")]

    # 4. Identify Value Columns (x1, x2, ... or actual numbers)
    # We assume columns starting with 'x' are time points in order
    x_cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]

    # Sort them nicely (x1, x2, x10...)
    x_cols.sort(key=lambda x: int(x[1:]))

    if not x_cols:
        print(f"[{type_label}] No 'x' columns found. Assuming file is already long format?")
        return df

    # 5. Melt to Long Format
    id_vars = ["protein"]
    if "psite" in df.columns:
        id_vars.append("psite")

    df_long = df.melt(id_vars=id_vars, value_vars=x_cols, var_name="time_idx", value_name="fc")

    # 6. Map 'x1' -> Time Value
    # Map x1 -> time_points[0], x2 -> time_points[1]
    time_map = {f"x{i + 1}": t for i, t in enumerate(time_points)}
    df_long["time"] = df_long["time_idx"].map(time_map)

    # Drop rows where time didn't map (e.g. x14 but only 10 timepoints defined)
    df_long = df_long.dropna(subset=["time"])

    return df_long[["protein", "time", "fc"] + (["psite"] if "psite" in df_long.columns else [])]


def plot_gof(name, df_obs, df_pred, merge_keys, output_path, color):
    """
    Generates a Goodness of Fit Scatter Plot.
    """
    if df_obs is None or df_obs.empty:
        print(f"[{name}] Missing observed data. Skipping.")
        return
    if df_pred is None or df_pred.empty:
        print(f"[{name}] Missing predicted data. Skipping.")
        return

    # Merge Data
    merged = pd.merge(df_obs, df_pred, on=merge_keys, how="inner", suffixes=("_obs", "_pred"))

    col_obs = "fc_obs" if "fc_obs" in merged.columns else "fc"
    col_pred = "pred_fc"

    merged = merged.dropna(subset=[col_obs, col_pred])

    if merged.empty:
        print(f"[{name}] No overlap between Obs and Pred timepoints/proteins.")
        return

    x = merged[col_obs]
    y = merged[col_pred]

    # Metrics
    if len(x) > 1:
        r_val, _ = pearsonr(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
    else:
        r_val, rmse = 0, 0

    # Plot
    plt.figure(figsize=(6, 6))
    lims = [np.min([x.min(), y.min()]), np.max([x.max(), y.max()])]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.scatter(x, y, c=color, alpha=0.6, edgecolors='w', s=60)

    stats_text = f"Pearson R: {r_val:.3f}\nRMSE: {rmse:.3f}\nN: {len(x)}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

    plt.title(f"{name}: Goodness of Fit", fontsize=15)
    plt.xlabel("Observed Fold Change", fontsize=12)
    plt.ylabel("Predicted Fold Change", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[{name}] Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--prot")
    parser.add_argument("--rna")
    parser.add_argument("--phos")
    args = parser.parse_args()

    # 1. Load Predictions
    df_pred_p = pd.read_csv(os.path.join(args.results, "pred_prot_picked.csv"))
    df_pred_r = pd.read_csv(os.path.join(args.results, "pred_rna_picked.csv"))
    df_pred_ph = pd.read_csv(os.path.join(args.results, "pred_phospho_picked.csv"))

    # 2. Load & Clean Observations
    # Note: We read raw CSV then clean it
    raw_prot = pd.read_csv(args.prot) if args.prot else None
    raw_rna = pd.read_csv(args.rna) if args.rna else None
    raw_phos = pd.read_csv(args.phos) if args.phos else None

    # Apply cleaning (Melting x1..xN -> time, renaming GeneID -> protein)
    # We filter input1.csv twice: once for Protein, once for Phospho
    df_obs_p = clean_and_melt(raw_prot, TIME_POINTS_PROTEIN, "Protein")
    df_obs_r = clean_and_melt(raw_rna, TIME_POINTS_RNA, "RNA")
    df_obs_ph = clean_and_melt(raw_phos, TIME_POINTS_PHOSPHO, "Phospho")

    # 3. Generate Plots
    plot_gof("Protein", df_obs_p, df_pred_p, ["protein", "time"],
             os.path.join(args.results, "gof_protein_reconstructed.png"), "#2ca02c")

    plot_gof("RNA", df_obs_r, df_pred_r, ["protein", "time"],
             os.path.join(args.results, "gof_rna_reconstructed.png"), "#1f77b4")

    plot_gof("Phospho", df_obs_ph, df_pred_ph, ["protein", "psite", "time"],
             os.path.join(args.results, "gof_phospho_reconstructed.png"), "#d62728")


if __name__ == "__main__":
    main()