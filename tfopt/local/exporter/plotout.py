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
    Creates a scatter plot visualizing multi-start optimization results with runtime overlay.

    This function reads a CSV summary of multiple optimization runs and generates a scatter plot
    showing the relationship between run rank and final objective value, with runtime (or iterations)
    represented as color intensity. Successful and feasible runs are emphasized while unsuccessful
    or infeasible runs are shown with reduced opacity.

    Args:
        summary_csv (str or Path): Path to the CSV file containing multi-start optimization results.
        out_path (str or Path, optional): Path to save the output figure. If None, figure is not saved.
            Defaults to None.
        figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (8, 8).
        x_col (str, optional): Column name for x-axis (run rank). If missing, will be created from
            y_col ranking. Defaults to "rank".
        y_col (str, optional): Column name for y-axis (final objective value). Defaults to "fun".
        c_col (str, optional): Column name for color mapping (typically runtime). Falls back to "nit"
            (iterations) if not found. Defaults to "runtime_s".
        success_col (str, optional): Column name indicating optimization success status.
            Defaults to "success".
        cv_col (str, optional): Column name for constraint violation values. Falls back to common
            alternatives if not found. Defaults to "constr_violation".
        annotate_best (bool, optional): Whether to annotate the best (rank 1) point on the plot.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The generated figure object.
            - ax (matplotlib.axes.Axes): The axes object of the plot.
            - df (pd.DataFrame): The processed DataFrame with sorted results.

    Notes:
        - Points are considered feasible if constraint violation <= 1e-8
        - Infeasible or unsuccessful runs are plotted with reduced opacity (0.25)
        - If rank column is missing, it's automatically generated from objective values
        - If runtime column is missing, falls back to iteration count or constant color
    """

    summary_csv = Path(summary_csv)
    df = pd.read_csv(summary_csv)
    # ---- Column handling ----
    # 1) Constraint violation column name
    if cv_col not in df.columns:
        for alt in ("constraint_violation", "constr_violation", "constraintViolation", "cv"):
            if alt in df.columns:
                cv_col = alt
                break

    # 2) Rank column (create from objective if missing)
    if x_col not in df.columns:
        # best objective gets rank 1
        df[x_col] = df[y_col].rank(method="first", ascending=True).astype(int)

    # 3) Runtime column (fallback to iterations if missing)
    c_label = "Runtime (s)"
    if c_col not in df.columns:
        if "nit" in df.columns:
            c_col = "nit"
            c_label = "Iterations (nit)"
        else:
            # constant color if nothing usable exists
            df[c_col] = 0.0
            c_label = c_col

    # 4) Parse/clean success
    has_success = success_col in df.columns
    if has_success:
        s = df[success_col]
        if s.dtype == bool:
            success_ok = s.to_numpy()
        else:
            # handles "True"/"False", "1"/"0", etc.
            success_ok = s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    else:
        success_ok = np.ones(len(df), dtype=bool)

    # 5) Parse/clean constraint violation (blank -> 0.0)
    has_cv = cv_col in df.columns
    if has_cv:
        cv = pd.to_numeric(df[cv_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        cv = np.zeros(len(df), dtype=float)

    # ---- Sorting / arrays ----
    df = df.sort_values(by=x_col, ascending=True).reset_index(drop=True)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    c = pd.to_numeric(df[c_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # Feasible + successful points emphasized
    ok = success_ok
    if has_cv:
        ok = ok & (cv <= 1e-8)

    sc_ok = ax.scatter(x[ok], y[ok], c=c[ok], s=28, linewidths=0)
    if (~ok).any():
        ax.scatter(x[~ok], y[~ok], c=c[~ok], s=28, linewidths=0, alpha=0.25)

    cbar = fig.colorbar(sc_ok, ax=ax)
    cbar.set_label(c_label)

    ax.set_xlabel("Run rank")
    ax.set_ylabel("Final objective")

    n = len(df)
    if n >= 16:
        step = max(1, n // 8)
        ax.set_xticks(np.arange(1, n + 1, step))
    ax.set_xlim(0.5, n + 0.5)

    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = 0.03 * (y_max - y_min) if y_max > y_min else 0.01
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.grid(True, alpha=0.25)

    if annotate_best:
        # With rank created from fun, rank==1 always exists
        i_best = int(df.index[df[x_col] == 1][0])
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