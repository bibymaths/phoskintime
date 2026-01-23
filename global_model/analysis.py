import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from global_model.config import RESULTS_DIR
from global_model.simulate import simulate_odeint
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


def simulate_until_steady(sys, t_max=1440.0, n_points=1000):
    """
    Simulates the system from t=0 to t_max (default 24h) to observe convergence.
    Uses log-spacing to capture fast initial phosphorylation and slow transcription.
    """
    # Log-space time grid (0, 0.001 ... t_max)
    t_log = np.logspace(np.log10(1e-3), np.log10(t_max), n_points - 1)
    t_eval = np.concatenate(([0.0], t_log))

    logger.info(f"[SteadyState] Simulating for {t_max} minutes...")

    # Run simulation (Tight tolerances for accuracy)
    Y = simulate_odeint(sys, t_eval, rtol=1e-6, atol=1e-8, mxstep=50000)

    # Check rate of change at the end
    dt = t_eval[-1] - t_eval[-2]
    dist = np.linalg.norm(Y[-1] - Y[-2])
    rate = dist / dt

    logger.info(f"[SteadyState] Final rate of change: {rate:.2e}")
    logger.info("[SteadyState] Simulate until steady-state [Done].")

    return t_eval, Y


def plot_steady_state_all(t, Y, sys, idx, output_dir):
    """
    Plots RNA, Protein, and Phospho trajectories for EVERY protein.
    Saves one PNG per protein.
    """
    save_dir = os.path.join(output_dir, "steady_state_plots")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"[Plot] Saving plots for {len(idx.proteins)} proteins to: {save_dir}/")

    for i, p_name in enumerate(idx.proteins):
        st = idx.offset_y[i]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax_rna, ax_prot, ax_phos = axes

        # 1. RNA
        ax_rna.plot(t, Y[:, st], color='#1f77b4', linewidth=2)
        ax_rna.set_title(f"{p_name} RNA")
        ax_rna.set_ylabel("Conc (a.u.)")

        # 2. Protein & Phospho
        p_unphos = Y[:, st + 1]
        ns = idx.n_sites[i]

        if ns > 0:
            # Sum phospho states
            p_phos_states = Y[:, st + 2: st + 2 + ns]
            total_phos = np.sum(p_phos_states, axis=1)
            total_prot = p_unphos + total_phos

            # Plot Phospho
            ax_phos.plot(t, total_phos, color='#d62728', linewidth=2, label="Total Phospho")
            for j in range(ns):
                site_name = idx.sites[i][j]
                ax_phos.plot(t, p_phos_states[:, j], linestyle="--", alpha=0.6, label=site_name)
            ax_phos.legend(fontsize=6)
        else:
            total_prot = p_unphos
            ax_phos.text(0.5, 0.5, "No Phospho Sites", ha='center', transform=ax_phos.transAxes, fontsize=10)

        # 3. Total Protein
        ax_prot.plot(t, total_prot, color='#2ca02c', linewidth=2)
        ax_prot.set_title(f"{p_name} Protein")

        # Styling
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (min)")
            if t[-1] > 100:  # Log scale for long times
                ax.set_xscale("symlog", linthresh=10.0)

        plt.suptitle(f"Dynamics: {p_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{p_name}_dynamics.png"), dpi=300)
        plt.close(fig)
        logger.info(f"[Plot] Steady-state plot for {p_name} saved to: {save_dir}/")

    # ============================================================
    # EXTRA STEADY-STATE SUMMARIES (uses sys)
    # ============================================================
    summary_dir = os.path.join(output_dir, "steady_state_summary")
    os.makedirs(summary_dir, exist_ok=True)

    y_last = Y[-1].astype(float, copy=False)
    t_last = float(t[-1])

    # ---- (A) convergence diagnostic: dy at final time
    dy_last = sys.rhs(t_last, y_last)
    abs_dy = np.abs(dy_last)

    plt.figure(figsize=(10, 5))
    plt.hist(abs_dy[np.isfinite(abs_dy)], bins=100)
    plt.xlabel("|dy/dt| at final time")
    plt.ylabel("Count")
    plt.title("Steady-state convergence diagnostic (all states)")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "steady_state_dy_hist.png"), dpi=300)
    plt.close()

    # top 200 largest derivatives
    topk = 200 if abs_dy.size > 200 else abs_dy.size
    top_idx = np.argsort(abs_dy)[-topk:][::-1]
    pd.DataFrame({"state_index": top_idx, "abs_dy": abs_dy[top_idx], "dy": dy_last[top_idx]}).to_csv(
        os.path.join(summary_dir, "steady_state_top_dy.csv"), index=False
    )

    # ---- (B) per-protein phospho fraction at steady state
    rows = []
    eps = 1e-12
    for i, p in enumerate(idx.proteins):
        st = idx.offset_y[i]
        R = y_last[st]
        P0 = y_last[st + 1]
        ns = int(idx.n_sites[i]) if hasattr(idx, "n_sites") else 0
        if ns > 0:
            Pph = float(np.sum(y_last[st + 2: st + 2 + ns]))
        else:
            Pph = 0.0
        Ptot = float(P0 + Pph)
        frac = Pph / (Ptot + eps)
        rows.append((p, R, P0, Pph, Ptot, frac, ns))

    df_ss = pd.DataFrame(rows, columns=["protein", "R_ss", "P_unphos_ss", "P_phos_ss", "P_total_ss", "phos_fraction", "n_sites"])
    df_ss.to_csv(os.path.join(summary_dir, "steady_state_protein_summary.csv"), index=False)

    # plot top phospho-fractions
    df_top = df_ss.sort_values("phos_fraction", ascending=False).head(50)
    plt.figure(figsize=(12, 6))
    plt.bar(df_top["protein"].astype(str), df_top["phos_fraction"].values)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Phospho fraction at steady state")
    plt.title("Top 50 proteins by phospho fraction (steady state)")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "steady_state_top_phos_fraction.png"), dpi=300)
    plt.close()

    # ---- (C) kinase -> phosphorylation drive at steady state
    # Kt used by the model at final time
    Kt = sys.kin.eval(t_last) * sys.c_k  # (n_kinases,)
    W = sys.W_global.tocoo()

    # contribution per edge = W.data * Kt[col]
    edge_contrib = W.data * Kt[W.col]

    # sum contribution by kinase
    nK = len(idx.kinases)
    kin_sum = np.zeros(nK, dtype=float)
    np.add.at(kin_sum, W.col, edge_contrib)

    df_kin = pd.DataFrame({"kinase": idx.kinases, "Kt": Kt, "phospho_drive_sum": kin_sum})
    df_kin = df_kin.sort_values("phospho_drive_sum", ascending=False)
    df_kin.to_csv(os.path.join(summary_dir, "steady_state_kinase_drive.csv"), index=False)

    # plot top 30 kinases by drive
    dfk_top = df_kin.head(30)
    plt.figure(figsize=(12, 6))
    plt.bar(dfk_top["kinase"].astype(str), dfk_top["phospho_drive_sum"].values)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Σ (W_ik * Kt_k) across all sites")
    plt.title("Top 30 kinases by global phosphorylation drive (steady state)")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "steady_state_top_kinase_drive.png"), dpi=300)
    plt.close()

    # ---- (C2) Dominant kinase per site (argmax of W_ik * Kt_k)
    # For each site row, find which kinase contributes the most at steady state.

    n_sites_total = sys.W_global.shape[0]
    best_val = np.full(n_sites_total, -np.inf, dtype=float)
    best_k = np.full(n_sites_total, -1, dtype=np.int32)

    # track 2nd best for dominance ratio (optional but very useful)
    second_val = np.full(n_sites_total, -np.inf, dtype=float)

    # Iterate COO edges once; update best/second best per row
    for r, c, v in zip(W.row, W.col, edge_contrib):
        if v > best_val[r]:
            second_val[r] = best_val[r]
            best_val[r] = v
            best_k[r] = c
        elif v > second_val[r]:
            second_val[r] = v

    # Build site labels in the same order as W rows (site index in idx.offset_s space)
    # Your W rows correspond to "sites" flattened in idx order.
    # We create labels: PROTEIN_PSITE
    site_labels = []
    for i, p in enumerate(idx.proteins):
        for s in idx.sites[i]:
            site_labels.append(f"{p}_{s}")
    site_labels = np.asarray(site_labels, dtype=object)

    # Guard: if W has extra rows (should not), pad labels
    if site_labels.size < n_sites_total:
        pad = np.array([f"site_{i}" for i in range(site_labels.size, n_sites_total)], dtype=object)
        site_labels = np.concatenate([site_labels, pad], axis=0)

    # Dominance ratio: best / (second best + eps)
    eps = 1e-12
    dom_ratio = best_val / (second_val + eps)

    # Create table
    df_dom = pd.DataFrame({
        "site": site_labels[:n_sites_total],
        "dominant_kinase": np.where(best_k >= 0, np.asarray(idx.kinases, dtype=object)[best_k], None),
        "dominant_contrib": best_val,
        "second_contrib": second_val,
        "dominance_ratio": dom_ratio
    })

    # Rows with no incoming edges (best_k = -1) will have -inf; drop them
    df_dom = df_dom.replace([np.inf, -np.inf], np.nan).dropna(subset=["dominant_kinase", "dominant_contrib"])
    df_dom.to_csv(os.path.join(summary_dir, "steady_state_dominant_kinase_per_site.csv"), index=False)

    # Count: how many sites each kinase dominates
    dom_counts = df_dom["dominant_kinase"].value_counts().reset_index()
    dom_counts.columns = ["kinase", "n_sites_dominated"]
    dom_counts.to_csv(os.path.join(summary_dir, "steady_state_dominant_kinase_counts.csv"), index=False)

    # Plot top 30 by dominated-site count
    topN = dom_counts.head(30)
    plt.figure(figsize=(12, 6))
    plt.bar(topN["kinase"].astype(str), topN["n_sites_dominated"].values)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("# sites where kinase is dominant")
    plt.title("Top 30 kinases by dominant-site count (steady state)")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "steady_state_top_dominant_kinase_counts.png"), dpi=300)
    plt.close()

    # Optional: most strongly dominated sites (largest dominance ratio)
    df_dom_strong = df_dom.sort_values("dominance_ratio", ascending=False).head(100)
    df_dom_strong.to_csv(os.path.join(summary_dir, "steady_state_most_dominated_sites_top100.csv"), index=False)

    logger.info(
        f"[Plot] Dominant-kinase-per-site exported: "
        f"{len(df_dom)} sites with edges, {dom_counts.shape[0]} kinases represented."
    )

    # ---- (C3) Kinase activity vs global phosphorylation drive (scatter + labels)
    # Requires df_kin with columns: kinase, Kt, phospho_drive_sum

    df_sc = df_kin.copy()
    df_sc["kinase"] = df_sc["kinase"].astype(str)

    eps = 1e-12
    df_sc["mismatch"] = df_sc["Kt"] / (df_sc["phospho_drive_sum"] + eps)

    top_n_drive = 15
    top_n_kt = 15
    top_n_mismatch = 10  # set 0 to disable

    drive_set = set(df_sc.nlargest(top_n_drive, "phospho_drive_sum")["kinase"].tolist())
    kt_set = set(df_sc.nlargest(top_n_kt, "Kt")["kinase"].tolist())
    mismatch_set = set()
    if top_n_mismatch > 0:
        # only consider kinases with non-trivial Kt to avoid noise
        mismatch_set = set(df_sc[df_sc["Kt"] > 1e-6].nlargest(top_n_mismatch, "mismatch")["kinase"].tolist())

    label_set = drive_set | kt_set | mismatch_set

    plt.figure(figsize=(9, 7))
    plt.scatter(df_sc["Kt"].values, df_sc["phospho_drive_sum"].values, alpha=0.7, edgecolors="black", linewidths=0.3)

    plt.xlabel("Kinase activity Kt at steady state")
    plt.ylabel("Global phosphorylation drive Σ(W_ik * Kt_k)")
    plt.title("Kinase activity vs phosphorylation drive (steady state)")

    # annotate with de-duplication
    for _, r in df_sc[df_sc["kinase"].isin(label_set)].iterrows():
        x = float(r["Kt"])
        y = float(r["phospho_drive_sum"])
        lab = r["kinase"]

        plt.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8)
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "steady_state_kinase_Kt_vs_drive.png"), dpi=300)
    plt.close()

    # Save the labeled sets for traceability
    pd.DataFrame({"kinase": sorted(list(drive_set))}).to_csv(
        os.path.join(summary_dir, "steady_state_labels_top_drive.csv"), index=False
    )
    pd.DataFrame({"kinase": sorted(list(kt_set))}).to_csv(
        os.path.join(summary_dir, "steady_state_labels_top_Kt.csv"), index=False
    )
    if top_n_mismatch > 0:
        pd.DataFrame({"kinase": sorted(list(mismatch_set))}).to_csv(
            os.path.join(summary_dir, "steady_state_labels_top_mismatch.csv"), index=False
        )

    logger.info(
        f"[Plot] Saved kinase Kt vs drive scatter with labels: "
        f"{len(drive_set)} (top drive) + {len(kt_set)} (top Kt) + {len(mismatch_set)} (mismatch) "
        f"-> {len(label_set)} unique labels."
    )

    # ---- (D) S_all distribution at steady state (site-wise phosphorylation propensity)
    S_all = sys.W_global.dot(Kt)  # (total_sites,)
    pd.DataFrame({"S_all": S_all}).to_csv(os.path.join(summary_dir, "steady_state_S_all.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.hist(S_all[np.isfinite(S_all)], bins=100)
    plt.xlabel("S_all (site phosphorylation rate constant)")
    plt.ylabel("Count")
    plt.title("Distribution of site phosphorylation drive S_all (steady state)")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "steady_state_S_all_hist.png"), dpi=300)
    plt.close()

    logger.info(f"[Plot] Steady-state summary saved to: {summary_dir}")

    logger.info("[Plot] Simulate until steady-state [Done].")
