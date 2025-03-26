
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from config.constants import COLOR_PALETTE, OUT_DIR, CONTOUR_LEVELS, available_markers
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Parallel Coordinates Plot
# -----------------------------
# - Purpose: Visualizes the evolution of system state values (e.g., R, P, P1, P2, ...) across multiple time points.
# - Time/Dynamics Assumptions:
#   - The system follows a dynamic trajectory over time, possibly reaching equilibrium.
#   - Time points are ordered and unevenly spaced; early dynamics are faster.
# - Interpretation:
#   - Reveals relative activation/inhibition of different states across time.
#   - Decline in R (mRNA) and rise in P and phosphorylated states (P1, P2, ...) indicates cascade activation.
#   - Useful for spotting saturation or feedback effects in the system’s progression.
def plot_parallel(solution, labels, gene, out_dir):
    df = pd.DataFrame(solution, columns=labels)
    df['Time'] = range(1, len(df) + 1)
    plt.figure(figsize=(8, 8))
    parallel_coordinates(df, class_column='Time', colormap=plt.get_cmap("tab20"))
    plt.title(gene)
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.legend(title="Time Points", loc="upper right", labels=df['Time'].astype(str).tolist())
    plt.savefig(os.path.join(out_dir, f"{gene}_parallel_coordinates_.png"), dpi=300)
    plt.close()

# -----------------------------
# PCA Plot (3D)
# -----------------------------
# - Purpose: Reduces high-dimensional ODE outputs into 3 orthogonal axes that explain most variance.
# - Time/Dynamics Assumptions:
#   - Early time points should be more dynamic; later ones may converge or stabilize.
#   - PCA assumes linear separability of variance.
# - Interpretation:
#   - PC1 generally represents the major mode of change across the time course.
#   - Curved or looping paths in PCA space suggest transitions between dynamic regimes.
#   - Convergence or clustering of late points implies approach to steady-state.
def pca_components(solution, gene, target_variance=0.99, out_dir=OUT_DIR):
    pca = PCA(n_components=min(solution.shape))
    pca.fit(solution)
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)
    required_components = np.argmax(cumulative_explained_variance >= target_variance) + 1
    plt.figure(figsize=(8, 8))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100,
            alpha=0.6, color='b', label='Individual')
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance * 100,
             marker='o', color='r', label='Cumulative')
    plt.axvline(x=required_components, color='g', linestyle='--',
                label=f'{required_components} Components')
    plt.title(gene)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{gene}_scree_plot_.png"), dpi=300)
    plt.close()

def plot_pca(solution, gene, out_dir, components=3):
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(solution)
    ev = pca.explained_variance_ratio_ * 100
    indices = np.arange(len(solution))
    if components == 3:
        x, y, z = pca_result[:, 0], pca_result[:, 1], pca_result[:, 2]
        cs_x, cs_y, cs_z = CubicSpline(indices, x), CubicSpline(indices, y), CubicSpline(indices, z)
        si = np.linspace(0, len(solution) - 1, 1000)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=indices, cmap='tab20')
        fig.colorbar(sc, label="Time Index")
        ax.plot(cs_x(si), cs_y(si), cs_z(si), color='blue', alpha=0.7, label='Temporal Path')
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ax.text(xi, yi, zi, str(i + 1), fontsize=10, color="black")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)")
        ax.set_title(gene)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{gene}_pca_plot_.png"), dpi=300)
        plt.close()

# -----------------------------
# 3. t-SNE
# -----------------------------
# - Purpose: Projects high-dimensional state space to 2D while preserving local distances (nonlinear structure).
# - Time/Dynamics Assumptions:
#   - Emphasizes local similarity; less emphasis on global distances.
#   - Assumes short time intervals reflect similar biological states.
# - Interpretation:
#   - Curved trajectories indicate progressive changes; abrupt turns highlight possible phase transitions.
#   - Clusters suggest metastable or recurring system states.
#   - Temporal coloring helps trace how states evolve and diverge nonlinearly over time.
def plot_tsne(solution, gene, out_dir, perplexity=30):
    perplexity = min(perplexity, len(solution) - 1)
    tsne_result = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(solution)
    x, y = tsne_result[:, 0], tsne_result[:, 1]
    indices = np.arange(len(solution))
    cs_x, cs_y = CubicSpline(indices, x), CubicSpline(indices, y)
    si = np.linspace(0, len(solution) - 1, 1000)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=indices, cmap='tab20')
    plt.colorbar(label="Time Index")
    plt.plot(cs_x(si), cs_y(si), color='blue', alpha=0.7, label='Temporal Path')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i + 1), fontsize=10, color="black")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(gene)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{gene}_tsne_plot_.png"), dpi=300)
    plt.close()


def plot_param_series(gene, estimated_params, param_names, time_points, out_dir):
    arr = np.array(estimated_params)
    plt.figure(figsize=(8, 8))
    for i in range(arr.shape[1]):
        plt.plot(time_points, arr[:, i], label=param_names[i])
    plt.title(f"{gene}")
    plt.xlabel("Time")
    plt.ylabel("Kinetic Rates")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{gene}_params_.png"), dpi=300)
    plt.close()

def plot_profiles(gene, data, out_dir):
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 8))
    for col in df.columns:
        if col != "Time (min)":
            plt.plot(df["Time (min)"], df[col], marker='o', label=col)
    plt.xlabel("Time (min)")
    plt.ylabel("Kinetic Rates")
    plt.title(f"{gene}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(out_dir, f"{gene}_profiles.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()

def plot_model_fit(gene, model_fit, P_data, sol, num_psites, psite_labels, time_points, out_dir):
    plt.figure(figsize=(8, 8))
    plt.plot(time_points, sol[:, 0], '-', color='black', alpha=0.7, label='mRNA (R)')
    plt.plot(time_points, sol[:, 1], '-', color='red', alpha=0.7, label='Protein (P)')
    for i in range(num_psites):
        plt.plot(time_points, P_data[i, :], '-', marker='s', color=COLOR_PALETTE[i], label=f'P+{psite_labels[i]}')
        plt.plot(time_points, model_fit[i, :], '-', color=COLOR_PALETTE[i], label=f'P+{psite_labels[i]}(model)')
    plt.xlabel("Time (minutes)")
    plt.ylabel("Phosphorylation level (FC)")
    plt.title(gene)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{gene}_model_fit_.png"), dpi=300)
    plt.close()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=sol[:,0],
        mode='lines+markers',
        name='mRNA(R)(model)',
        line=dict(color='black')
    ))
    fig.add_trace(go.Scatter(
        x=time_points,
        y=sol[:,1],
        mode='lines+markers',
        name='Protein(P)(model)',
        line=dict(color='red')
    ))
    for i in range(num_psites):
        fig.add_trace(go.Scatter(
            x=time_points,
            y=P_data[i, :] if num_psites > 1 else P_data.flatten(),
            mode='lines+markers',
            name=f'P+{psite_labels[i]}',
            line=dict(dash='dash', color=COLOR_PALETTE[i])
        ))
        fig.add_trace(go.Scatter(
            x=time_points,
            y=model_fit[i, :],
            mode='lines+markers',
            name=f'P+{psite_labels[i]}(model)',
            line=dict(color=COLOR_PALETTE[i])
        ))
    fig.update_layout(title=f'{gene}',
                      xaxis_title="Time (minutes)", yaxis_title="Phosphorylation level (FC)",
                      template="plotly_white", width=900, height=900)
    fig.write_html(os.path.join(out_dir, f"{gene}_model_fit_.html"))

# ------------------------------------------------------------
# A–S Density Contour Plot
# ------------------------------------------------------------
# This plot visualizes the joint distribution of estimated parameters A (mRNA production rate)
# and S (phosphorylation rate) across all time points and phosphorylation sites for a given protein.
#
# - The background shows density contours based on a kernel density estimate (KDE).
# - Warmer colors (e.g., yellow, orange) indicate regions where A–S parameter combinations occur more frequently.
# - Overlaid black dots represent the actual estimated parameter values.
#
# Interpretation:
# - Reveals clusters or dense regions in parameter space.
# - Highlights typical regions of activity and detects outliers.
# - Ignores time progression—focuses only on spatial frequency in A–S space.

# ------------------------------------------------------------
# A–S Scatter Plot Colored by Time with Linear Regression
# ------------------------------------------------------------
# This plot shows the temporal evolution of phosphorylation rates (S) against mRNA production rates (A)
# for each phosphorylation site.
#
# - Each phosphorylation site (e.g., S1, S2, S3, ...) is plotted with a unique marker.
# - Points are color-coded by time (from early to late), using a continuous colormap.
# - A linear regression line is fitted for each site's S vs. A values.
#
# Interpretation:
# - Illustrates how phosphorylation responds to changes in mRNA production across time.
# - Site-specific slopes reveal coupling or independence between A and S.
# - Useful for identifying regulatory trends and time-resolved behaviors across phosphorylation sites.
def plot_A_S(gene, est_arr, num_psites, time_vals, out_dir):
    est_arr = np.array(est_arr)
    A_vals = est_arr[:, 0]
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(time_vals), vmax=max(time_vals))
    plt.figure(figsize=(8, 8))
    legend_handles = []

    for i in range(num_psites):
        S_vals = est_arr[:, 4 + i]
        sc = plt.scatter(A_vals, S_vals, c=time_vals, cmap=cmap, norm=norm,
                         s=50, alpha=0.8, marker=available_markers[i])
        slope, intercept = np.polyfit(A_vals, S_vals, 1)
        x_fit = np.linspace(A_vals.min(), A_vals.max(), 100)
        y_fit = slope * x_fit + intercept
        line_color = f"C{i}"
        plt.plot(x_fit, y_fit, color=line_color, lw=1)
        legend_handles.append(Line2D([0], [0],
                                     marker=available_markers[i],
                                     color='w',
                                     markerfacecolor=line_color,
                                     markeredgecolor='k',
                                     markersize=8,
                                     label=f"S{i + 1}"))
    plt.xlabel("A (mRNA production rate)")
    plt.ylabel("S (Phosphorylation rate)")
    plt.title(gene)
    cbar = plt.colorbar(sc)
    cbar.set_label("Time (min)")
    plt.grid(True)
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{gene}_scatter_A_S_.png"), dpi=300)
    plt.close()

    all_points = np.vstack([np.column_stack((A_vals, est_arr[:, 4 + i])) for i in range(num_psites)])
    kde = gaussian_kde(all_points.T)
    A_lin = np.linspace(A_vals.min(), A_vals.max(), 100)
    all_S = all_points[:, 1]
    S_lin = np.linspace(all_S.min(), all_S.max(), 100)
    A_grid, S_grid = np.meshgrid(A_lin, S_lin)
    grid_coords = np.vstack([A_grid.ravel(), S_grid.ravel()])
    density = kde(grid_coords).reshape(A_grid.shape)
    plt.figure(figsize=(8, 8))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='black', s=30, alpha=0.5)
    contourf = plt.contourf(A_grid, S_grid, density, levels=10, cmap="inferno", alpha=0.7)
    plt.contour(A_grid, S_grid, density, levels=CONTOUR_LEVELS, colors='white', linewidths=0.5)
    plt.xlabel("A")
    plt.ylabel("S")
    plt.title(gene)
    cbar = plt.colorbar(contourf)
    cbar.set_label("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{gene}_density_A_S_.png"), dpi=300)
    plt.close()