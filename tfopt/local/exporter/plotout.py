import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from tfopt.local.config.constants import OUT_DIR
import matplotlib

matplotlib.use('Agg')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)


def plot_estimated_vs_observed(predictions, expression_matrix, gene_ids, time_points, regulators, tf_protein_matrix,
                               tf_ids, num_targets, save_path=OUT_DIR):
    """
    Plots the estimated vs observed values for a given set of genes and their corresponding TFs.

    Args:
        predictions (np.ndarray): Predicted expression levels.
        expression_matrix (np.ndarray): Observed expression levels.
        gene_ids (list): List of gene identifiers.
        time_points (np.ndarray): Time points for the experiments.
        regulators (np.ndarray): Matrix of regulators for each gene.
        tf_protein_matrix (np.ndarray): Matrix of TF protein levels.
        tf_ids (list): List of TF identifiers.
        num_targets (int): Number of target genes to plot.
        save_path (str): Directory to save the plots.
    """
    T = len(time_points)
    time_vals_expr = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    time_vals_tf = np.array([4, 8, 16, 30, 60, 120, 240, 480, 960])
    combined_ticks = np.unique(np.concatenate((time_vals_expr, time_vals_tf)))
    num_targets = min(num_targets, predictions.shape[0])

    for i in range(num_targets):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # --- Full time series plot ---
        ax = axes[1]
        ax.plot(time_vals_expr, expression_matrix[i, :], 's-', label='Observed', color='black')
        ax.plot(time_vals_expr, predictions[i, :], '-', label='Estimated', color='red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :T]
                ax.plot(time_vals_tf, protein_signal, ':', label=f"{tf_name}", alpha=0.3)
                plotted_tfs.add(tf_name)
        ax.set_title(f"mRNA: {gene_ids[i]}")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fold Changes")
        ax.set_xticks(combined_ticks[4:])
        ax.set_xticklabels(combined_ticks[4:])
        ax.grid(True, alpha=0.3)

        # --- First 5 time points plot ---
        ax = axes[0]
        ax.plot(time_vals_expr[:5], expression_matrix[i, :5], 's-', label='Observed', color='black')
        ax.plot(time_vals_expr[:5], predictions[i, :5], '-', label='Estimated', color='red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :5]
                ax.plot(time_vals_tf[:5], protein_signal, ':', label=f"{tf_name}", alpha=0.3)
                plotted_tfs.add(tf_name)
        ax.set_xlabel("Time (minutes)")
        ax.set_xticks(time_vals_expr[:5])
        ax.set_xticklabels(time_vals_expr[:5])
        ax.legend(
            title="TFs",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=10,
            frameon=True,
            fontsize=8,
            title_fontsize=9
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{gene_ids[i]}_model_fit_.png", dpi=300)
        plt.close()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_vals_expr,
            y=expression_matrix[i, :],
            mode='markers+lines',
            name='Observed',
            marker=dict(symbol='square')
        ))
        fig.add_trace(go.Scatter(
            x=time_vals_expr,
            y=predictions[i, :],
            mode='lines+markers',
            name='Estimated'
        ))
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :len(time_vals_tf)]
                fig.add_trace(go.Scatter(
                    x=time_vals_tf,
                    y=protein_signal,
                    mode='lines',
                    name=f"TF: {tf_name}",
                    line=dict(dash='dot')
                ))
                plotted_tfs.add(tf_name)
        fig.update_layout(
            title=f"mRNA: {gene_ids[i]}",
            xaxis_title="Time (minutes)",
            yaxis_title="Fold Changes",
            xaxis=dict(
                tickmode='array',
                tickvals=combined_ticks,
                ticktext=[str(t) for t in combined_ticks]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            width=900, height=900
        )
        fig.write_html(f"{save_path}/{gene_ids[i]}_model_fit_.html")

def plot_multistart_summary_runtime_overlay(
    summary_csv,
    out_path=None,
    figsize=(8, 8),
    x_col="rank",
    y_col="fun",
    c_col="runtime_s",
    success_col="success",
    cv_col="constr_violation",
    annotate_best=True,
):
    """
    Read a multistart summary CSV and plot objective vs rank with point color = runtime.

    Minimal, information-dense conventions:
      - x: rank (best -> worst)
      - y: final objective (fun)
      - color: runtime in seconds
      - optional: de-emphasize non-success / infeasible points (if columns exist)

    Args:
        summary_csv (str | Path): Path to the multistart_summary.csv
        out_path (str | Path | None): If provided, saves the figure (e.g. .png)
        figsize (tuple): Figure size in inches
        x_col, y_col, c_col: Column names
        success_col, cv_col: Optional columns for styling (used if present)
        annotate_best (bool): Annotate the best run (rank=1 or min fun)

    Returns:
        (fig, ax, df): Matplotlib figure/axis and the loaded DataFrame
    """
    summary_csv = Path(summary_csv)
    df = pd.read_csv(summary_csv)

    # Basic sanitization / sorting
    df = df.sort_values(by=x_col, ascending=True).reset_index(drop=True)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    c = df[c_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # If success/feasibility exists, de-emphasize problematic points without changing structure
    has_success = success_col in df.columns
    has_cv = cv_col in df.columns

    if has_success or has_cv:
        ok = np.ones(len(df), dtype=bool)
        if has_success:
            ok &= df[success_col].astype(int).to_numpy() == 1
        if has_cv:
            ok &= df[cv_col].to_numpy(dtype=float) <= 1e-8

        # Plot "ok" points normally, and "not ok" faintly
        sc_ok = ax.scatter(x[ok], y[ok], c=c[ok], s=28, linewidths=0)
        if (~ok).any():
            ax.scatter(x[~ok], y[~ok], c=c[~ok], s=28, linewidths=0, alpha=0.25)
    else:
        sc_ok = ax.scatter(x, y, c=c, s=28, linewidths=0)

    # Colorbar (runtime overlay)
    cbar = fig.colorbar(sc_ok, ax=ax)
    cbar.set_label("Runtime (s)")

    # Labels: minimal but explicit
    ax.set_xlabel("Run rank")
    ax.set_ylabel("Final objective")

    # Tick strategy: keep it readable for 64+ points
    n = len(df)
    if n >= 16:
        step = max(1, n // 8)
        ax.set_xticks(np.arange(1, n + 1, step))
    ax.set_xlim(0.5, n + 0.5)

    # y-limits with small padding
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = 0.03 * (y_max - y_min) if y_max > y_min else 0.01
    ax.set_ylim(y_min - pad, y_max + pad)

    # Subtle grid for reading slopes
    ax.grid(True, alpha=0.25)

    # Optional annotation of best
    if annotate_best:
        # Prefer rank==1; fallback to min fun
        if (df[x_col] == 1).any():
            i_best = int(df.index[df[x_col] == 1][0])
        else:
            i_best = int(np.argmin(y))

        ax.annotate(
            f"best: {y[i_best]:.6g}",
            xy=(x[i_best], y[i_best]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, ax, df