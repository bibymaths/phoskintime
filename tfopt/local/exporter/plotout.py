import os

import numpy as np
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
    The function generates two types of plots:
    1. A full time series plot showing the observed and estimated values for each gene.
    2. A plot showing the first 5 time points for each gene.
    Additionally, it generates an interactive plot using Plotly for each gene.

    :param predictions: 2D numpy array of predicted values.
    :param expression_matrix: 2D numpy array of observed values.
    :param gene_ids: List of gene identifiers.
    :param time_points: 1D numpy array of time points.
    :param regulators: 2D numpy array of regulators for each gene.
    :param tf_protein_matrix: 2D numpy array of TF protein signals.
    :param tf_ids: List of TF identifiers.
    :param num_targets: Number of target genes to plot.
    :param save_path: Directory path to save the plots.
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
                ax.plot(time_vals_tf[:5], protein_signal, ':', label=f"{tf_name}", alpha = 0.3)
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