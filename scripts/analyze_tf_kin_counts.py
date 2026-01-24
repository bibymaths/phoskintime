#!/usr/bin/env python3
"""
Python script for analyzing TF and kinase counts in beta values and target gene data.

This script processes optimization results for Transcription Factors (TF) and Kinases,
calculating statistics on phosphosites (PSites) and aggregating counts per target gene.

Usage: analyze_tf_kin_counts.py [--tfopt-xlsx=<path>] [--kinopt-xlsx=<path>] [--out-dir=<path>]

License: BSD-3-Clause
Author: Abhinav Mishra
"""

from pathlib import Path
import pandas as pd


def psite_counts_beta(df: pd.DataFrame, entity_col: str, psite_col: str) -> pd.DataFrame:
    """
    Calculates statistics for phosphosites (PSites) associated with a given entity.

    It groups the input DataFrame by the specified entity column and computes:
    - Total number of rows per entity.
    - Number of non-null PSite entries.
    - Number of unique PSites.
    - Whether the entity has any PSite (boolean).
    - Number of rows where PSite information is missing.

    Args:
        df (pd.DataFrame): The input DataFrame containing beta values.
        entity_col (str): The column name representing the entity (e.g., 'TF' or 'Kinase').
        psite_col (str): The column name representing the phosphosite (e.g., 'PSite').

    Returns:
        pd.DataFrame: A DataFrame with the calculated statistics, sorted by
                      presence of PSites, count of unique PSites, and entity name.
    """
    x = df.copy()
    # Ensure the psite column is treated as string to handle mixed types safely
    x[psite_col] = x[psite_col].astype("string")

    # Group by the entity, preserving NA values in the grouping key if any exist
    g = x.groupby(entity_col, dropna=False)

    # aggregate statistics
    out = pd.DataFrame(
        {
            "n_rows": g.size(),
            "n_psites_nonnull": g[psite_col].apply(lambda s: s.notna().sum()),
            "n_unique_psites": g[psite_col].nunique(dropna=True),
            "has_any_psite": g[psite_col].apply(lambda s: bool(s.notna().any())),
            "n_rows_psite_missing": g[psite_col].apply(lambda s: int(s.isna().sum())),
        }
    ).reset_index()

    # Sort: entities with psites first, then by count of unique psites (desc), then alphabetically
    return out.sort_values(["has_any_psite", "n_unique_psites", entity_col], ascending=[False, False, True])


def main(
        tfopt_xlsx: str = "data/tfopt_results.xlsx",
        kinopt_xlsx: str = "data/kinopt_results.xlsx",
        out_dir: str = "results_scripts",
):
    """
    Main function to execute the analysis pipeline.

    Steps:
    1. Loads TF and Kinase optimization data (Alpha and Beta values) from Excel files.
    2. Computes PSite statistics for TFs and Kinases using Beta values.
    3. Aggregates the number of TFs and Kinases targeting each gene (using Alpha values).
    4. Saves the results to CSV files in the specified output directory.
    5. Prints summary statistics to the console.

    Args:
        tfopt_xlsx (str): Path to the Excel file containing TF optimization results.
        kinopt_xlsx (str): Path to the Excel file containing Kinase optimization results.
        out_dir (str): Directory where output CSV files will be saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load TFopt ---
    # Reading Alpha values for mRNA-TF relationships and Beta values for TF-PSite relationships
    tf_alpha = pd.read_excel(tfopt_xlsx, sheet_name="Alpha Values")  # cols: mRNA, TF, Value
    tf_beta = pd.read_excel(tfopt_xlsx, sheet_name="Beta Values")  # cols: TF, PSite, Value

    # --- Load Kinopt ---
    # Reading Alpha values for Gene-Kinase relationships and Beta values for Kinase-PSite relationships
    kin_alpha = pd.read_excel(kinopt_xlsx, sheet_name="Alpha Values")  # cols: Gene, Psite, Kinase, Alpha
    kin_beta = pd.read_excel(kinopt_xlsx, sheet_name="Beta Values")  # cols: Kinase, Psite, Beta

    # 1) Beta psite counts
    # Calculate stats for TFs and Kinases based on their Beta value sheets
    tf_beta_stats = psite_counts_beta(tf_beta, entity_col="TF", psite_col="PSite")
    kin_beta_stats = psite_counts_beta(kin_beta, entity_col="Kinase", psite_col="Psite")

    # Save the beta statistics to CSV
    tf_beta_stats.to_csv(out_dir / "tf_beta_psite_counts.csv", index=False)
    kin_beta_stats.to_csv(out_dir / "kin_beta_psite_counts.csv", index=False)

    # 2) For each target gene (mRNA/protein): #TFs and #Kinases
    # Group TF alpha data by mRNA to count unique TFs targeting each gene
    tf_per_gene = (
        tf_alpha.groupby("mRNA")["TF"]
        .nunique()
        .reset_index(name="n_tfs")
        .rename(columns={"mRNA": "Gene"})
    )

    # Group Kinase alpha data by Gene to count unique Kinases targeting each gene
    kin_per_gene = (
        kin_alpha.groupby("Gene")["Kinase"]
        .nunique()
        .reset_index(name="n_kinases")
    )

    # Merge TF and Kinase counts; use outer join to include genes present in either dataset
    per_gene = tf_per_gene.merge(kin_per_gene, on="Gene", how="outer")

    # Fill missing values with 0 (e.g., a gene might have TFs but no Kinases)
    per_gene["n_tfs"] = per_gene["n_tfs"].fillna(0).astype(int)
    per_gene["n_kinases"] = per_gene["n_kinases"].fillna(0).astype(int)

    # Create boolean flags for presence in the respective optimization results
    per_gene["in_tfopt"] = per_gene["n_tfs"] > 0
    per_gene["in_kinopt"] = per_gene["n_kinases"] > 0

    # Sort by number of kinases (desc), then number of TFs (desc), then Gene name (asc)
    per_gene = per_gene.sort_values(["n_kinases", "n_tfs", "Gene"], ascending=[False, False, True])

    # Save the per-gene aggregated counts
    per_gene.to_csv(out_dir / "per_gene_num_tfs_num_kinases.csv", index=False)

    # Minimal console output
    print("Wrote:")
    print(" -", out_dir / "tf_beta_psite_counts.csv")
    print(" -", out_dir / "kin_beta_psite_counts.csv")
    print(" -", out_dir / "per_gene_num_tfs_num_kinases.csv")
    print()
    print("Top TFs by #unique PSites:")
    print(tf_beta_stats.head(10).to_string(index=False))
    print()
    print("Top Kinases by #unique PSites:")
    print(kin_beta_stats.head(10).to_string(index=False))
    print()
    print("Top genes by #kinases then #TFs:")
    print(per_gene.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
