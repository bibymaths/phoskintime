#!/usr/bin/env python3
"""
Minimal analysis for:
- # psites per TF / per Kinase in Beta Values (+ flag if any PSite exists)
- per target gene (mRNA/protein): #TFs (from tfopt Alpha) and #Kinases (from kinopt Alpha)
"""

from pathlib import Path
import pandas as pd


def psite_counts_beta(df: pd.DataFrame, entity_col: str, psite_col: str) -> pd.DataFrame:
    x = df.copy()
    x[psite_col] = x[psite_col].astype("string")

    g = x.groupby(entity_col, dropna=False)
    out = pd.DataFrame(
        {
            "n_rows": g.size(),
            "n_psites_nonnull": g[psite_col].apply(lambda s: s.notna().sum()),
            "n_unique_psites": g[psite_col].nunique(dropna=True),
            "has_any_psite": g[psite_col].apply(lambda s: bool(s.notna().any())),
            "n_rows_psite_missing": g[psite_col].apply(lambda s: int(s.isna().sum())),
        }
    ).reset_index()

    return out.sort_values(["has_any_psite", "n_unique_psites", entity_col], ascending=[False, False, True])


def main(
    tfopt_xlsx: str = "data/tfopt_results.xlsx",
    kinopt_xlsx: str = "data/kinopt_results.xlsx",
    out_dir: str = "results_scripts",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load TFopt ---
    tf_alpha = pd.read_excel(tfopt_xlsx, sheet_name="Alpha Values")  # cols: mRNA, TF, Value
    tf_beta  = pd.read_excel(tfopt_xlsx, sheet_name="Beta Values")   # cols: TF, PSite, Value

    # --- Load Kinopt ---
    kin_alpha = pd.read_excel(kinopt_xlsx, sheet_name="Alpha Values") # cols: Gene, Psite, Kinase, Alpha
    kin_beta  = pd.read_excel(kinopt_xlsx, sheet_name="Beta Values")  # cols: Kinase, Psite, Beta

    # 1) Beta psite counts
    tf_beta_stats  = psite_counts_beta(tf_beta,  entity_col="TF",     psite_col="PSite")
    kin_beta_stats = psite_counts_beta(kin_beta, entity_col="Kinase", psite_col="Psite")

    tf_beta_stats.to_csv(out_dir / "tf_beta_psite_counts.csv", index=False)
    kin_beta_stats.to_csv(out_dir / "kin_beta_psite_counts.csv", index=False)

    # 2) For each target gene (mRNA/protein): #TFs and #Kinases
    tf_per_gene = (
        tf_alpha.groupby("mRNA")["TF"]
        .nunique()
        .reset_index(name="n_tfs")
        .rename(columns={"mRNA": "Gene"})
    )

    kin_per_gene = (
        kin_alpha.groupby("Gene")["Kinase"]
        .nunique()
        .reset_index(name="n_kinases")
    )

    per_gene = tf_per_gene.merge(kin_per_gene, on="Gene", how="outer")
    per_gene["n_tfs"] = per_gene["n_tfs"].fillna(0).astype(int)
    per_gene["n_kinases"] = per_gene["n_kinases"].fillna(0).astype(int)
    per_gene["in_tfopt"] = per_gene["n_tfs"] > 0
    per_gene["in_kinopt"] = per_gene["n_kinases"] > 0
    per_gene = per_gene.sort_values(["n_kinases", "n_tfs", "Gene"], ascending=[False, False, True])

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
