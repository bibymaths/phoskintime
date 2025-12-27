#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd

def pick_best(F, weights):
    """
    F: (n,3) objectives [prot_mse, rna_mse, reg_loss]
    weights: (3,) [w_prot, w_rna, w_reg]
    Uses min-max normalization per objective, then weighted sum.
    """
    F = np.asarray(F, dtype=float)
    F_min = F.min(axis=0)
    F_ptp = np.ptp(F, axis=0) + 1e-12
    F_norm = (F - F_min) / F_ptp
    scores = (F_norm * weights).sum(axis=1)
    i = int(np.argmin(scores))
    return i, float(scores[i])

def main(out_dir="out_plain"):
    F = np.load(os.path.join(out_dir, "pareto_F.npy"))
    X = np.load(os.path.join(out_dir, "pareto_X.npy"))

    # ---- define scan grid (simple log grids) ----
    lambda_rna_grid   = np.logspace(-2, 2, 9)     # 0.01 .. 100
    lambda_prior_grid = np.logspace(-4, 0, 9)     # 1e-4 .. 1

    rows = []
    for lr in lambda_rna_grid:
        for lp in lambda_prior_grid:
            weights = np.array([1.0, lr, lp], dtype=float)
            best_i, best_score = pick_best(F, weights)

            rows.append({
                "lambda_rna": lr,
                "lambda_prior": lp,
                "best_i": best_i,
                "best_score": best_score,
                "prot_mse": float(F[best_i, 0]),
                "rna_mse": float(F[best_i, 1]),
                "reg_loss": float(F[best_i, 2]),
            })

    df = pd.DataFrame(rows).sort_values(["lambda_rna", "lambda_prior"])
    df.to_csv(os.path.join(out_dir, "lambda_scan.csv"), index=False)

    # also save the unique picked solutions (often repeats)
    uniq = df.drop_duplicates("best_i").copy()
    uniq.to_csv(os.path.join(out_dir, "lambda_scan_unique_picks.csv"), index=False)

    # write one “recommended” choice: lowest prot_mse among solutions with rna_mse not crazy
    # (adjust this rule to taste)
    cand = uniq.sort_values(["prot_mse", "rna_mse", "reg_loss"]).head(1).iloc[0]
    rec = {
        "lambda_rna": float(cand["lambda_rna"]),
        "lambda_prior": float(cand["lambda_prior"]),
        "best_i": int(cand["best_i"]),
        "objectives": {
            "prot_mse": float(cand["prot_mse"]),
            "rna_mse": float(cand["rna_mse"]),
            "reg_loss": float(cand["reg_loss"]),
        }
    }
    with open(os.path.join(out_dir, "lambda_scan_recommended.json"), "w") as f:
        json.dump(rec, f, indent=2)

    print("Wrote:")
    print(" - lambda_scan.csv")
    print(" - lambda_scan_unique_picks.csv")
    print(" - lambda_scan_recommended.json")

if __name__ == "__main__":
    main()
