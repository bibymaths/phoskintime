import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config.constants import COLOR_PALETTE, OUT_DIR, CONTOUR_LEVELS, available_markers


class Plotter:
    """
    A class to encapsulate plotting functionalities for ODE model analysis.

    Attributes:
        gene (str): The gene or experiment name.
        out_dir (str): The directory where plots will be saved.
        color_palette (list): List of color codes used for plotting.
    """

    def __init__(self, gene: str, out_dir: str = OUT_DIR):
        self.gene = gene
        self.out_dir = out_dir
        self.color_palette = COLOR_PALETTE

    def _save_fig(self, fig, filename: str, dpi: int = 300):
        """
        Saves and closes the given matplotlib figure.
        """
        path = os.path.join(self.out_dir, filename)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    # -----------------------------
    # Parallel Coordinates Plot
    # -----------------------------
    def plot_parallel(self, solution: np.ndarray, labels: list):
        df = pd.DataFrame(solution, columns=labels)
        df['Time'] = range(1, len(df) + 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        parallel_coordinates(df, class_column='Time', colormap=plt.get_cmap("tab20"), ax=ax)
        ax.set_title(self.gene)
        ax.set_xlabel("States")
        ax.set_ylabel("Values")
        ax.legend(title="Time Points", loc="upper right", labels=df['Time'].astype(str).tolist())
        self._save_fig(fig, f"{self.gene}_parallel_coordinates_.png")

    # -----------------------------
    # PCA Components / Scree Plot
    # -----------------------------
    def pca_components(self, solution: np.ndarray, target_variance: float = 0.99):
        pca = PCA(n_components=min(solution.shape))
        pca.fit(solution)
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)
        required_components = np.argmax(cumulative_explained_variance >= target_variance) + 1

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance * 100,
               alpha=0.6, color='b', label='Individual')
        ax.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance * 100,
                marker='o', color='r', label='Cumulative')
        ax.axvline(x=required_components, color='g', linestyle='--',
                   label=f'{required_components} Components')
        ax.set_title(self.gene)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance (%)')
        ax.legend()
        ax.grid(True)
        self._save_fig(fig, f"{self.gene}_scree_plot_.png")
        return required_components, explained_variance

    # -----------------------------
    # 3D PCA Plot
    # -----------------------------
    def plot_pca(self, solution: np.ndarray, components: int = 3):
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
            ax.set_title(self.gene)
            ax.legend()
            self._save_fig(fig, f"{self.gene}_pca_plot_.png")
        else:
            # Optionally handle non-3D cases here
            pass

    # -----------------------------
    # t-SNE Plot
    # -----------------------------
    def plot_tsne(self, solution: np.ndarray, perplexity: int = 30):
        perplexity = min(perplexity, len(solution) - 1)
        tsne_result = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(solution)
        x, y = tsne_result[:, 0], tsne_result[:, 1]
        indices = np.arange(len(solution))
        cs_x, cs_y = CubicSpline(indices, x), CubicSpline(indices, y)
        si = np.linspace(0, len(solution) - 1, 1000)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, c=indices, cmap='tab20')
        ax.plot(cs_x(si), cs_y(si), color='blue', alpha=0.7, label='Temporal Path')
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi, str(i + 1), fontsize=10, color="black")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(self.gene)
        ax.grid(True)
        ax.legend()
        self._save_fig(fig, f"{self.gene}_tsne_plot_.png")

    # -----------------------------
    # Parameter Series Plot
    # -----------------------------
    def plot_param_series(self, estimated_params: list, param_names: list, time_points: np.ndarray):
        arr = np.array(estimated_params)
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(arr.shape[1]):
            ax.plot(time_points, arr[:, i], label=param_names[i])
        ax.set_title(self.gene)
        ax.set_xlabel("Time")
        ax.set_ylabel("Kinetic Rates")
        ax.grid(True)
        ax.legend(loc="best")
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_params_.png")

    # -----------------------------
    # Profile Plot
    # -----------------------------
    def plot_profiles(self, data: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 8))
        for col in data.columns:
            if col != "Time (min)":
                ax.plot(data["Time (min)"], data[col], marker='o', label=col)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Kinetic Rates")
        ax.set_title(self.gene)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_profiles.png")

    # -----------------------------
    # Model Fit Plot (Matplotlib & Plotly)
    # -----------------------------
    def plot_model_fit(self, model_fit: np.ndarray, P_data: np.ndarray, sol: np.ndarray,
                       num_psites: int, psite_labels: list, time_points: np.ndarray):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(time_points, sol[:, 0], '-', color='black', alpha=0.7, label='mRNA (R)')
        ax.plot(time_points, sol[:, 1], '-', color='red', alpha=0.7, label='Protein (P)')
        for i in range(num_psites):
            ax.plot(time_points, P_data[i, :], '-', marker='s',
                    color=self.color_palette[i], label=f'P+{psite_labels[i]}')
            ax.plot(time_points, model_fit[i, :], '-', color=self.color_palette[i],
                    label=f'P+{psite_labels[i]} (model)')
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Phosphorylation level (FC)")
        ax.set_title(self.gene)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_model_fit_.png")

        # Plot using Plotly for an interactive version.
        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(
            x=time_points,
            y=sol[:, 0],
            mode='lines+markers',
            name='mRNA(R)(model)',
            line=dict(color='black')
        ))
        fig_plotly.add_trace(go.Scatter(
            x=time_points,
            y=sol[:, 1],
            mode='lines+markers',
            name='Protein(P)(model)',
            line=dict(color='red')
        ))
        for i in range(num_psites):
            fig_plotly.add_trace(go.Scatter(
                x=time_points,
                y=P_data[i, :] if num_psites > 1 else P_data.flatten(),
                mode='lines+markers',
                name=f'P+{psite_labels[i]}',
                line=dict(dash='dash', color=self.color_palette[i])
            ))
            fig_plotly.add_trace(go.Scatter(
                x=time_points,
                y=model_fit[i, :],
                mode='lines+markers',
                name=f'P+{psite_labels[i]} (model)',
                line=dict(color=self.color_palette[i])
            ))
        fig_plotly.update_layout(title=self.gene,
                                 xaxis_title="Time (minutes)",
                                 yaxis_title="Phosphorylation level (FC)",
                                 template="plotly_white",
                                 width=900, height=900)
        fig_plotly.write_html(os.path.join(self.out_dir, f"{self.gene}_model_fit_.html"))

    # -----------------------------
    # A–S Scatter and Density Contour Plot
    # -----------------------------
    def plot_A_S(self, est_arr: np.ndarray, num_psites: int, time_vals: np.ndarray):
        est_arr = np.array(est_arr)
        A_vals = est_arr[:, 0]
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=min(time_vals), vmax=max(time_vals))
        fig, ax = plt.subplots(figsize=(8, 8))
        legend_handles = []
        for i in range(num_psites):
            S_vals = est_arr[:, 4 + i]
            sc = ax.scatter(A_vals, S_vals, c=time_vals, cmap=cmap, norm=norm,
                            s=50, alpha=0.8, marker=available_markers[i])
            slope, intercept = np.polyfit(A_vals, S_vals, 1)
            x_fit = np.linspace(A_vals.min(), A_vals.max(), 100)
            y_fit = slope * x_fit + intercept
            line_color = f"C{i}"
            ax.plot(x_fit, y_fit, color=line_color, lw=1)
            legend_handles.append(Line2D([0], [0],
                                         marker=available_markers[i],
                                         color='w',
                                         markerfacecolor=line_color,
                                         markeredgecolor='k',
                                         markersize=8,
                                         label=f"S{i + 1}"))
        ax.set_xlabel("A (mRNA production rate)")
        ax.set_ylabel("S (Phosphorylation rate)")
        ax.set_title(self.gene)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Time (min)")
        ax.grid(True)
        ax.legend(handles=legend_handles)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_scatter_A_S_.png")

        # Density contour plot for A and S.
        all_points = np.vstack([np.column_stack((A_vals, est_arr[:, 4 + i])) for i in range(num_psites)])
        kde = gaussian_kde(all_points.T)
        A_lin = np.linspace(A_vals.min(), A_vals.max(), 100)
        all_S = all_points[:, 1]
        S_lin = np.linspace(all_S.min(), all_S.max(), 100)
        A_grid, S_grid = np.meshgrid(A_lin, S_lin)
        grid_coords = np.vstack([A_grid.ravel(), S_grid.ravel()])
        density = kde(grid_coords).reshape(A_grid.shape)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(all_points[:, 0], all_points[:, 1], c='black', s=30, alpha=0.5)
        contourf = ax.contourf(A_grid, S_grid, density, levels=10, cmap="inferno", alpha=0.7)
        ax.contour(A_grid, S_grid, density, levels=CONTOUR_LEVELS, colors='white', linewidths=0.5)
        ax.set_xlabel("A")
        ax.set_ylabel("S")
        ax.set_title(self.gene)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label("Density")
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_density_A_S_.png")

    def plot_all(self, solution: np.ndarray, labels: list, estimated_params: list,
                 time_points: np.ndarray, P_data: np.ndarray, seq_model_fit: np.ndarray,
                 psite_labels: list, perplexity: int = 5, components: int = 3, target_variance: float = 0.99):
        """
        A single method that calls all plotting functions.
        """
        self.plot_parallel(solution, labels)
        self.plot_tsne(solution, perplexity=perplexity)
        self.plot_pca(solution, components=components)
        self.pca_components(solution, target_variance=target_variance)
        self.plot_param_series(estimated_params, get_param_names(len(psite_labels)), time_points)
        self.plot_model_fit(seq_model_fit, P_data, solution, len(psite_labels), psite_labels, time_points)
        self.plot_A_S(estimated_params, len(psite_labels), time_points)