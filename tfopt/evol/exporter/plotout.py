import os

import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from tfopt.evol.config.constants import OUT_DIR
import matplotlib
matplotlib.use('Agg')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

def plot_estimated_vs_observed(predictions, expression_matrix, gene_ids, time_points, regulators, tf_protein_matrix,
                               tf_ids, num_targets, save_path=OUT_DIR):
    """
    Plot the estimated vs observed expression levels for a set of genes.
    This function generates two plots for each gene:
    1. A full time series plot showing the observed and estimated expression levels.
    2. A plot showing the observed and estimated expression levels for the first 5 time points.
    Additionally, it plots the protein signals for the regulators of each gene.
    The plots are saved as PNG files in the specified save path.

    :param predictions: Predicted expression levels.
    :param expression_matrix: Observed expression levels.
    :param gene_ids: List of gene identifiers.
    :param time_points: Time points for the expression data.
    :param regulators: Matrix of regulators for each gene.
    :param tf_protein_matrix: Matrix of protein signals for each transcription factor.
    :param tf_ids: List of transcription factor identifiers.
    :param num_targets: Number of target genes to plot.
    :param save_path: Path to save the plots.
    :type predictions: np.ndarray
    :type expression_matrix: np.ndarray
    :type gene_ids: list
    :type time_points: np.ndarray
    :type regulators: np.ndarray
    :type tf_protein_matrix: np.ndarray
    :type tf_ids: list
    :type num_targets: int
    :type save_path: str
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

def compute_predictions(x, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA, beta_start_indices, num_psites):
    """
    Compute the predicted expression levels based on the optimization variables.
    This function calculates the predicted expression levels for each mRNA based on the optimization variables,
    the regulators, and the protein and phosphorylation site tensors.

    :param x:
    :param regulators:
    :param protein_mat:
    :param psite_tensor:
    :param n_reg:
    :param T_use:
    :param n_mRNA:
    :param beta_start_indices:
    :param num_psites:
    :return: predictions
    """
    n_alpha = n_mRNA * n_reg
    predictions = np.zeros((n_mRNA, T_use))
    for i in range(n_mRNA):
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1: # No valid TF for this regulator
                continue
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]
            beta_vec = x[n_alpha + beta_start : n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        predictions[i, :] = R_pred
    return predictions