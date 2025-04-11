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
    Plots observed and estimated gene expression time series for the first num_targets genes,
    overlaying the TF protein time series for all TFs regulating each gene.
    """
    T = len(time_points)
    time_vals_expr = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    time_vals_tf = np.array([4, 8, 16, 30, 60, 120, 240, 480, 960])
    combined_ticks = np.unique(np.concatenate((time_vals_expr, time_vals_tf)))
    num_targets = min(num_targets, predictions.shape[0])

    for i in range(num_targets):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # --- Full time series plot ---
        ax = axes[0]
        ax.plot(time_vals_expr, expression_matrix[i, :], 's-', label='Observed', color='black')
        ax.plot(time_vals_expr, predictions[i, :], '-', label='Estimated', color='red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :T]
                ax.plot(time_vals_tf, protein_signal, ':', label=f"{tf_name}")
                plotted_tfs.add(tf_name)
        ax.set_title(f"mRNA: {gene_ids[i]}")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fold Changes")
        ax.set_xticks(combined_ticks)
        ax.set_xticklabels(combined_ticks, rotation=45)
        ax.grid(True, alpha=0.3)

        # --- First 5 time points plot ---
        ax = axes[1]
        ax.plot(time_vals_expr[:5], expression_matrix[i, :5], 's-', label='Observed', color='black')
        ax.plot(time_vals_expr[:5], predictions[i, :5], '-', label='Estimated', color='red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :5]
                ax.plot(time_vals_tf[:5], protein_signal, ':', label=f"{tf_name}")
                plotted_tfs.add(tf_name)
        ax.set_xlabel("Time (minutes)")
        ax.set_xticks(time_vals_expr[:5])
        ax.set_xticklabels(time_vals_expr[:5], rotation=45)
        ax.legend(title="TFs")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{gene_ids[i]}_model_fit_.png", dpi=300)
        plt.close()

        # This block is for saving two plots for one TF
        # One for full time series and one for first 5 time points
        # To see clearly the dynamics early on

        # plt.figure(figsize=(8, 8))
        # plt.plot(time_vals_expr, expression_matrix[i, :], 's-', label='Observed')
        # plt.plot(time_vals_expr, predictions[i, :], '-', label='Estimated')
        # # Plot protein time series for each regulator of mRNA i (only unique TFs)
        # plotted_tfs = set()
        # for r in regulators[i, :]:
        #     if r == -1:  # Skip invalid TF
        #         continue
        #     tf_name = tf_ids[r]
        #     if tf_name not in plotted_tfs:
        #         protein_signal = tf_protein_matrix[r, :T]
        #         plt.plot(time_vals_tf, protein_signal, ':', label=f"mRNA: {tf_name}")
        #         plotted_tfs.add(tf_name)
        # plt.title(f"mRNA: {gene_ids[i]}")
        # plt.xlabel("Time (minutes)")
        # plt.ylabel("Fold Changes")
        # plt.xticks(combined_ticks, combined_ticks, rotation=45)
        # plt.legend()
        # plt.tight_layout()
        # plt.grid(True, alpha=0.3)
        # plt.show()
        #
        # plt.figure(figsize=(8, 8))
        # plt.plot(time_vals_expr[:5], expression_matrix[i, :5], 's-', label='Observed')
        # plt.plot(time_vals_expr[:5], predictions[i, :5], '-', label='Estimated')
        # # Plot protein time series for each regulator of mRNA i (only unique TFs)
        # plotted_tfs = set()
        # for r in regulators[i, :]:
        #     if r == -1:  # Skip invalid TF
        #         continue
        #     tf_name = tf_ids[r]
        #     if tf_name not in plotted_tfs:
        #         protein_signal = tf_protein_matrix[r, :5]
        #         plt.plot(time_vals_tf[:5], protein_signal, ':', label=f"mRNA: {tf_name}")
        #         plotted_tfs.add(tf_name)
        # plt.title(f"mRNA: {gene_ids[i]}")
        # plt.xlabel("Time (minutes)")
        # plt.ylabel("Fold Changes")
        # plt.xticks(time_vals_expr[:5], time_vals_expr[:5], rotation=45)
        # plt.legend()
        # plt.tight_layout()
        # plt.grid(True, alpha=0.3)
        # plt.show()

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
                    name=f"mRNA: {tf_name}",
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.write_html(f"{save_path}/{gene_ids[i]}_model_fit_.html")