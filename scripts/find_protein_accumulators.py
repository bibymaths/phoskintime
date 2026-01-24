#!/usr/bin/env python3
"""
Identify 'Accumulator' proteins where predicted protein fold-change significantly
exceeds predicted mRNA fold-change.

This script loads protein and RNA prediction files, calculates the maximum
predicted fold-change (pred_fc) for each protein, and computes a coupling ratio:
    Ratio = Max_Protein_FC / (Max_RNA_FC + epsilon)

Proteins with a ratio > 100 are flagged as 'Accumulators', suggesting massive
protein abundance changes despite flat or low mRNA changes (e.g., due to
post-transcriptional regulation or high stability).

Usage:
    python identify_accumulators.py --prot pred_prot_picked.csv --rna pred_rna_picked.csv

License: BSD-3-Clause
Author: Abhinav Mishra
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify 'Accumulator' proteins (High Protein FC / Low RNA FC)."
    )
    parser.add_argument(
        "--prot",
        type=str,
        required=True,
        help="Path to the protein predictions CSV file (must contain 'protein' and 'pred_fc' columns)."
    )
    parser.add_argument(
        "--rna",
        type=str,
        required=True,
        help="Path to the RNA predictions CSV file (must contain 'protein' and 'pred_fc' columns)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="The ratio threshold to define an accumulator (default: 100.0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate file paths
    prot_path = Path(args.prot)
    rna_path = Path(args.rna)

    if not prot_path.exists():
        print(f"Error: Protein file not found at {prot_path}", file=sys.stderr)
        sys.exit(1)
    if not rna_path.exists():
        print(f"Error: RNA file not found at {rna_path}", file=sys.stderr)
        sys.exit(1)

    # Load predictions
    print(f"Loading protein predictions from: {prot_path}")
    dfp = pd.read_csv(prot_path)

    print(f"Loading RNA predictions from: {rna_path}")
    dfr = pd.read_csv(rna_path)

    # Validate required columns
    required_cols = {'protein', 'pred_fc'}
    if not required_cols.issubset(dfp.columns):
        print(f"Error: Protein file missing columns. Required: {required_cols}", file=sys.stderr)
        sys.exit(1)
    if not required_cols.issubset(dfr.columns):
        print(f"Error: RNA file missing columns. Required: {required_cols}", file=sys.stderr)
        sys.exit(1)

    # Find max values for each protein
    # We group by 'protein' to handle cases where there might be multiple predictions per protein
    # (e.g., different time points or conditions) and take the peak fold-change.
    max_p = dfp.groupby('protein')['pred_fc'].max()
    max_r = dfr.groupby('protein')['pred_fc'].max()

    # Align the two series to ensure we only compare proteins present in both datasets
    # inner join by default on the index (protein name)
    combined = pd.concat([max_p, max_r], axis=1, keys=['prot_max', 'rna_max'], join='inner')

    # Calculate the coupling ratio
    # Adding a small epsilon (1e-6) to the denominator to prevent DivisionByZero errors
    # if the max RNA fold-change is 0.
    combined['ratio'] = combined['prot_max'] / (combined['rna_max'] + 1e-6)

    # Filter for 'Accumulators' based on the threshold
    accumulators = combined[combined['ratio'] > args.threshold].sort_values(by='ratio', ascending=False)

    print("\n" + "=" * 60)
    print(f"🚨 Top 'Accumulator' Proteins (Massive Protein vs Flat mRNA)")
    print(f"   Threshold: Ratio > {args.threshold}")
    print("=" * 60)

    if accumulators.empty:
        print("No accumulators found exceeding the threshold.")
    else:
        print(accumulators[['prot_max', 'rna_max', 'ratio']].head(10).to_string())
        print("-" * 60)
        print(f"Total accumulators found: {len(accumulators)}")


if __name__ == "__main__":
    main()
