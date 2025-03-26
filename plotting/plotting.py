

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config.constants import COLOR_PALETTE, OUT_DIR, CONTOUR_LEVELS, available_markers
from pandas.plotting import parallel_coordinates
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_parallel(solution, labels, gene, out_dir):
    df = pd.DataFrame(solution, columns=labels)
    df['Time'] = range(1, len(df) + 1)
    plt.figure(figsize=(8, 8))
    parallel_coordinates(df, class_column='Time', colormap=plt.get_cmap("tab20"))
    plt.title(gene)
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.legend(title="Time Points", loc="upper right", labels=df['Time'].astype(str).tolist())
    plt.savefig(os.path.join(out_dir, f"parallel_coordinates_{gene}.png"), dpi=300)
    plt.close()

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
    plt.savefig(os.path.join(out_dir, f"scree_plot_{gene}.png"), dpi=300)
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
        plt.savefig(os.path.join(out_dir, f"pca_plot_{gene}.png"), dpi=300)
        plt.close()


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
    plt.savefig(os.path.join(out_dir, f"tsne_plot_{gene}.png"), dpi=300)
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
    plt.savefig(os.path.join(out_dir, f"param_series_{gene}.png"), dpi=300)
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
    plt.savefig(os.path.join(out_dir, f"model_fit_{gene}.png"), dpi=300)
    plt.close()


def plot_A_S(gene, est_arr, num_psites, time_vals, out_dir):
    est_arr = np.array(est_arr)
    A_vals = est_arr[:, 0]
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(time_vals), vmax=max(time_vals))

    plt.figure(figsize=(8, 8))
    for i in range(num_psites):
        S_vals = est_arr[:, 4 + i]
        sc = plt.scatter(A_vals, S_vals, c=time_vals, cmap=cmap, norm=norm,
                         s=50, alpha=0.8, marker=available_markers[i])
        slope, intercept = np.polyfit(A_vals, S_vals, 1)
        x_fit = np.linspace(A_vals.min(), A_vals.max(), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color=f"C{i}", lw=1, label=f"S{i+1}")

    plt.xlabel("A (mRNA production rate)")
    plt.ylabel("S (Phosphorylation rate)")
    plt.title(gene)
    cbar = plt.colorbar(sc)
    cbar.set_label("Time (min)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"A_S_{gene}_scatter.png"), dpi=300)
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
    plt.savefig(os.path.join(out_dir, f"A_S_{gene}_density.png"), dpi=300)
    plt.close()