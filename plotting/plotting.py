import os
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config.constants import get_param_names, COLOR_PALETTE, OUT_DIR, CONTOUR_LEVELS, available_markers, model_type, \
    ESTIMATION_MODE


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

    def plot_parallel(self, solution: np.ndarray, labels: list):
        """
        Plots a parallel coordinates plot for the given solution.

        :param solution: 2D numpy array of shape (sampels, features)
        :param labels: list of labels
        """
        df = pd.DataFrame(solution, columns=labels)
        df['Time'] = range(1, len(df) + 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        parallel_coordinates(df, class_column='Time',
                             colormap=plt.get_cmap("tab20"),
                             ax=ax)
        ax.set_title(self.gene)
        ax.set_xlabel("States")
        ax.set_ylabel("Values")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth=0.8,
                color='gray', alpha=0.5)
        ax.grid(which='minor', linestyle=':', linewidth=0.5,
                color='gray', alpha=0.2)
        ax.legend(title="Time Points",
                  loc="upper right",
                  labels=df['Time'].astype(str).tolist())
        self._save_fig(fig, f"{self.gene}_parallel_coordinates_.png")

    def pca_components(self, solution: np.ndarray, target_variance: float = 0.99):
        """
        Plots a scree plot showing the explained variance ratio for PCA components.

        :param solution: 2D numpy array of shape (samples, features) representing the data.
        :param target_variance: The target cumulative explained variance to determine the required number of components.
        :return: A tuple containing the number of required components and the explained variance ratio.
        """

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
        ax.grid(True, alpha=0.2)
        self._save_fig(fig, f"{self.gene}_scree_plot_.png")
        return required_components, explained_variance

    def plot_pca(self, solution: np.ndarray, components: int = 3):
        """
        Plots the PCA results for the given solution.

        :param solution: 2D numpy array of shape (samples, features) representing the data.
        :param components: Number of PCA components to plot. Defaults to 3.
        """
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
            sc = ax.scatter(x, y, z, c=indices, cmap='tab20', marker='o', edgecolor='black', alpha=0.7)
            fig.colorbar(sc, label="Time Index")
            ax.plot(cs_x(si), cs_y(si), cs_z(si), color='red', alpha=0.3, label='Temporal Path')
            for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
                ax.text(xi, yi, zi, str(i + 1), fontsize=10, color="black")
            ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
            ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)")
            ax.set_title(self.gene)
            ax.legend()
            ax.grid(True, alpha=0.2)
            self._save_fig(fig, f"{self.gene}_pca_plot_.png")
        else:
            # Optionally handle non-3D cases here
            pass

    def plot_tsne(self, solution: np.ndarray, perplexity: int = 30):
        """
        Plots a t-SNE visualization of the given solution.

        :param solution: 2D numpy array of shape (samples, features) representing the data.
        :param perplexity: Perplexity parameter for t-SNE. Defaults to 30.
        """
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
        ax.grid(True, alpha=0.2)
        ax.legend()
        self._save_fig(fig, f"{self.gene}_tsne_plot_.png")

    def plot_param_bar(self, params_df: pd.DataFrame, s_df: pd.DataFrame):
        """
        Plots a bar chart of parameter values for the given gene.

        This method visualizes the estimated parameter values for a specific gene
        and its phosphorylation sites. It uses color coding to distinguish between
        different phosphorylation sites and other parameters.

        :param params_df: DataFrame containing parameter values.
        :param s_df: DataFrame containing phosphorylation site information.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        unique_psites = s_df.loc[s_df['GeneID'] == self.gene, 'Psite'].tolist()
        color_map = {psite: plt.cm.tab20(i / len(unique_psites)) for i, psite in enumerate(unique_psites)}

        if len(unique_psites) == 1:
            single_psite = unique_psites[0]
            color = color_map[single_psite]
            if 'S' in params_df.columns and not params_df['S'].isna().all():
                ax.bar('S', params_df['S'].mean(), color=color, label=f"{single_psite}")

        for i, psite in enumerate(unique_psites):
            color = color_map[psite]
            for param in [f"S{i + 1}", f"D{i + 1}"]:
                if param in params_df.columns and not params_df[param].isna().all():
                    ax.bar(param, params_df[param].mean(), color=color,
                           label=f"{psite}" if psite not in [h.get_label() for h in
                                                             ax.get_legend_handles_labels()[0]] else None)

        other_params = [col for col in params_df.columns
                        if col not in ['Protein', 'S'] + [f"S{i + 1}" for i in range(len(unique_psites))] + [f"D{i + 1}"
                                                                                                             for i in
                                                                                                             range(
                                                                                                                 len(unique_psites))]]
        for i, param in enumerate(other_params):
            if param in params_df.columns and not params_df[param].isna().all():
                ax.bar(param, params_df[param].mean(), color=plt.cm.Paired(i / len(other_params)), alpha=0.6)

        ax.set_title(self.gene)
        ax.set_ylabel('Estimated Values')
        ax.set_xlabel('Parameters')
        plt.xticks(rotation=45)
        ax.legend(title="Residue_Position", loc='upper right', ncol=2)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_params_bar_.png")


    def plot_param_series(self, estimated_params: list, param_names: list, time_points: np.ndarray):
        """
        Plots the time series of estimated parameters over the given time points.

        This method visualizes the evolution of kinetic rates or parameters
        over time for a specific gene.

        :param estimated_params: List of estimated parameter values at each time point.
        :param param_names: List of parameter names corresponding to the estimated parameters.
        :param time_points: 1D numpy array of time points.
        """
        arr = np.array(estimated_params)
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(arr.shape[1]):
            ax.plot(time_points, arr[:, i], label=param_names[i])
        ax.set_title(self.gene)
        ax.set_xlabel("Time")
        ax.set_ylabel("Kinetic Rates")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_params_series_.png")


    def plot_profiles(self, data: pd.DataFrame):
        """
        Plots the profiles of estimated parameters over time.

        :param data: DataFrame containing the estimated parameters and time points.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        for col in data.columns:
            if col != "Time":
                ax.plot(data["Time"], data[col], marker='o', label=col)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Kinetic Rates")
        ax.set_title(self.gene)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_params_profiles.png")

    def plot_model_fit(self, model_fit: np.ndarray, P_data: np.ndarray, sol: np.ndarray,
                       num_psites: int, psite_labels: list, time_points: np.ndarray):
        """
        Plots the model fit for the given data.

        :param model_fit: Estimated model fit values.
        :param P_data: Observed data for phosphorylation levels.
        :param sol: ODE solution for mRNA and protein levels.
        :param num_psites: number of phosphorylation sites.
        :param psite_labels: labels for the phosphorylation sites.
        :param time_points: time points for the data.
        :return:
        """
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
        ax.grid(True, alpha=0.2)
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

    def plot_A_S(self, est_arr: np.ndarray, num_psites: int, time_vals: np.ndarray):
        """
        Plots the scatter plot of A vs S and the density contour plot.

        :param est_arr: Estimated parameters array.
        :param num_psites: Number of phosphorylation sites.
        :param time_vals: Time values for the data.
        :return:
        """
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
        ax.grid(True, alpha=0.2)
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
        Function that calls parallel, t-SNE, PCA, and model fit plots.
        If mode is sequential, it also calls parameter series and A-S plots.

        :param solution: 2D numpy array of shape (samples, features) representing the data.
        :param labels: List of labels for the solution.
        :param estimated_params: List of estimated parameter values.
        :param time_points: 1D numpy array of time points.
        :param P_data: Observed data for phosphorylation levels.
        :param seq_model_fit: Estimated model fit values.
        :param psite_labels: Labels for the phosphorylation sites.
        :param perplexity: Perplexity parameter for t-SNE.
        :param components: Number of PCA components to plot.
        :param target_variance: The target cumulative explained variance to determine the required number of components.
        """
        self.plot_parallel(solution, labels)
        self.plot_tsne(solution, perplexity=perplexity)
        self.plot_pca(solution, components=components)
        self.pca_components(solution, target_variance=target_variance)
        self.plot_model_fit(seq_model_fit, P_data, solution, len(psite_labels), psite_labels, time_points)
        if ESTIMATION_MODE == 'sequential':
            self.plot_param_series(estimated_params, get_param_names(len(psite_labels)), time_points)
            self.plot_A_S(estimated_params, len(psite_labels), time_points)

    def plot_clusters(self, s_values_df: pd.DataFrame, cluster_labels):
        """
        Plots the clusters of S values for the given gene.
        Expects s_values_df to have columns 'S_value', 'GeneID', and 'Psite'.

        :param s_values_df: DataFrame containing S values and gene information.
        :param cluster_labels: Cluster labels for each S value.
        """
        df = s_values_df.copy()
        df['Cluster'] = cluster_labels
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(x=df.index, y=df['S_value'], hue=cluster_labels, palette="viridis", s=100, ax=ax)
        for i, row in df.iterrows():
            ax.text(i, row['S_value'], f"{row['GeneID']}-{row['Psite']}", fontsize=9, ha='right')
        ax.set_title('')
        ax.set_ylabel('S', fontstyle='italic')
        ax.set_xticks([])
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_protein_clusters.png")

    def plot_heatmap(self, param_value_df: pd.DataFrame):
        """
        Expects param_value_df to have a 'Protein' column.
        """
        df = param_value_df.copy()
        if 'Protein' in df.columns:
            df.set_index('Protein', inplace=True)
        correlation_matrix = df.T.corr()
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax)
        ax.set_title('')
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_heatmap_protein.png")

    def plot_error_distribution(self, error_df: pd.DataFrame):
        """
        Expects error_df to have a 'MAE' column.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.histplot(error_df['MAE'], kde=True, color='blue', label='MSE', ax=ax)
        sns.histplot(error_df['MAE'], kde=True, color='orange', label='MAE', ax=ax)
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.set_title('')
        ax.legend()
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_model_error.png")

    def plot_gof_1(self, merged_data: pd.DataFrame):
        """
        Expects merged_data to contain 'GeneID', 'Psite', and columns 'x1_obs' to 'x14_obs' and 'x1_est' to 'x14_est'.
        """
        overall_std = merged_data.loc[:, 'x1_obs':'x14_obs'].values.std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = merged_data['GeneID'].unique()
        palette = sns.color_palette("tab20", len(unique_genes))
        gene_color_map = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(8, 8))
        gene_handles = []
        obs_array = merged_data.loc[:, 'x1_obs':'x14_obs'].values
        est_array = merged_data.loc[:, 'x1_est':'x14_est'].values
        for gene, psite, obs_vals, est_vals in zip(merged_data['GeneID'],
                                                   merged_data['Psite'],
                                                   obs_array, est_array):
            sorted_indices = np.argsort(obs_vals)
            obs_vals_sorted = obs_vals[sorted_indices]
            est_vals_sorted = est_vals[sorted_indices]
            ax.scatter(obs_vals_sorted, est_vals_sorted, color=gene_color_map[gene],
                       edgecolor='black', s=50)
            if gene not in [handle.get_label() for handle in gene_handles]:
                handle = plt.Line2D([], [], color=gene_color_map[gene],
                                    marker='o', linestyle='', markersize=8, label=gene)
                gene_handles.append(handle)
        min_val = min(obs_array.min(), est_array.min())
        max_val = max(obs_array.max(), est_array.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color='gray', linestyle='-', linewidth=1.5)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_95, max_val + ci_offset_95],
                color='red', linestyle='--', linewidth=1, label='95% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_95, max_val - ci_offset_95],
                color='red', linestyle='--', linewidth=1)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_99, max_val + ci_offset_99],
                color='gray', linestyle='--', linewidth=1, label='99% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_99, max_val - ci_offset_99],
                color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Fitted")
        ax.set_title(f"{model_type} model")
        ax.legend(handles=gene_handles, loc='upper left', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"_gof_1.png")

    def plot_gof_2(self, merged_data: pd.DataFrame):
        overall_std = merged_data.loc[:, 'x1_obs':'x14_obs'].values.std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = merged_data['GeneID'].unique()
        palette = sns.color_palette("tab20", len(unique_genes))
        gene_color_map = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(8, 8))
        gene_handles = []
        obs_array = merged_data.loc[:, 'x1_obs':'x14_obs'].values
        est_array = merged_data.loc[:, 'x1_est':'x14_est'].values
        for gene, psite, obs_vals, est_vals in zip(merged_data['GeneID'],
                                                   merged_data['Psite'],
                                                   obs_array, est_array):
            sorted_indices = np.argsort(obs_vals)
            obs_vals_sorted = obs_vals[sorted_indices]
            est_vals_sorted = est_vals[sorted_indices]
            ax.scatter(obs_vals_sorted, est_vals_sorted, color=gene_color_map[gene],
                       edgecolor='black', s=50)
            if gene not in [handle.get_label() for handle in gene_handles]:
                handle = plt.Line2D([], [], color=gene_color_map[gene],
                                    marker='o', linestyle='', markersize=8, label=gene)
                gene_handles.append(handle)
        min_val = min(obs_array.min(), est_array.min())
        max_val = max(obs_array.max(), est_array.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color='gray', linestyle='-', linewidth=1.5)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_95, max_val + ci_offset_95],
                color='red', linestyle='--', linewidth=1, label='95% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_95, max_val - ci_offset_95],
                color='red', linestyle='--', linewidth=1)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_99, max_val + ci_offset_99],
                color='gray', linestyle='--', linewidth=1, label='99% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_99, max_val - ci_offset_99],
                color='gray', linestyle='--', linewidth=1)
        # Expand axis limits slightly
        x_min = obs_array.min() - 0.1 * (obs_array.max() - obs_array.min())
        x_max = obs_array.max() + 0.1 * (obs_array.max() - obs_array.min())
        y_min = est_array.min() - 0.1 * (est_array.max() - est_array.min())
        y_max = est_array.max() + 0.1 * (est_array.max() - est_array.min())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Fitted")
        ax.set_title(f"{model_type} model")
        ax.legend(handles=gene_handles, loc='upper left', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"_gof_2.png")

    def plot_gof_3(self, merged_data: pd.DataFrame):
        overall_std = merged_data.loc[:, 'x1_obs':'x14_obs'].values.std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = merged_data['GeneID'].unique()
        palette = sns.color_palette("tab20", len(unique_genes))
        gene_color_map = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(8, 8))
        plotted_genes = set()
        text_annotations = []
        obs_array = merged_data.loc[:, 'x1_obs':'x14_obs'].values
        est_array = merged_data.loc[:, 'x1_est':'x14_est'].values
        for gene, psite, obs_vals, est_vals in zip(merged_data['GeneID'],
                                                   merged_data['Psite'],
                                                   obs_array, est_array):
            sorted_indices = np.argsort(obs_vals)
            obs_vals_sorted = obs_vals[sorted_indices]
            est_vals_sorted = est_vals[sorted_indices]
            ax.scatter(obs_vals_sorted, est_vals_sorted, color=gene_color_map[gene],
                       edgecolor='black', s=50)
            for obs, est in zip(obs_vals_sorted, est_vals_sorted):
                if gene not in plotted_genes and (est > obs + ci_offset_95 or est < obs - ci_offset_95):
                    txt = ax.text(obs, est, gene, fontsize=10, color=gene_color_map[gene],
                                  fontweight='bold', ha='center', va='center',
                                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                    text_annotations.append(txt)
                    plotted_genes.add(gene)
        min_val = min(obs_array.min(), est_array.min())
        max_val = max(obs_array.max(), est_array.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color='gray', linestyle='-', linewidth=1.5)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_95, max_val + ci_offset_95],
                color='red', linestyle='--', linewidth=1, label='95% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_95, max_val - ci_offset_95],
                color='red', linestyle='--', linewidth=1)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_99, max_val + ci_offset_99],
                color='gray', linestyle='--', linewidth=1, label='99% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_99, max_val - ci_offset_99],
                color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Fitted")
        ax.set_title(f"{model_type} model")
        ax.legend(loc='upper left', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        adjust_text(text_annotations, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        self._save_fig(fig, f"gof_3.png")

    def plot_gof_4(self, merged_data: pd.DataFrame):
        overall_std = merged_data.loc[:, 'x1_obs':'x14_obs'].values.std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = merged_data['GeneID'].unique()
        palette = sns.color_palette("tab20", len(unique_genes))
        gene_color_map = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(8, 8))
        plotted_genes = set()
        text_annotations = []
        obs_array = merged_data.loc[:, 'x1_obs':'x14_obs'].values
        est_array = merged_data.loc[:, 'x1_est':'x14_est'].values
        for gene, psite, obs_vals, est_vals in zip(merged_data['GeneID'],
                                                   merged_data['Psite'],
                                                   obs_array, est_array):
            sorted_indices = np.argsort(obs_vals)
            obs_vals_sorted = obs_vals[sorted_indices]
            est_vals_sorted = est_vals[sorted_indices]
            ax.scatter(obs_vals_sorted, est_vals_sorted, color=gene_color_map[gene],
                       edgecolor='black', s=50)
            for obs, est in zip(obs_vals_sorted, est_vals_sorted):
                if gene not in plotted_genes and (est > obs + ci_offset_95 or est < obs - ci_offset_95):
                    txt = ax.text(obs, est, gene, fontsize=10, color=gene_color_map[gene],
                                  fontweight='bold', ha='center', va='center',
                                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                    text_annotations.append(txt)
                    plotted_genes.add(gene)
        min_val = min(obs_array.min(), est_array.min())
        max_val = max(obs_array.max(), est_array.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color='gray', linestyle='-', linewidth=1.5)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_95, max_val + ci_offset_95],
                color='red', linestyle='--', linewidth=1, label='95% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_95, max_val - ci_offset_95],
                color='red', linestyle='--', linewidth=1)
        ax.plot([min_val, max_val],
                [min_val + ci_offset_99, max_val + ci_offset_99],
                color='gray', linestyle='--', linewidth=1, label='99% CI')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_99, max_val - ci_offset_99],
                color='gray', linestyle='--', linewidth=1)
        # Expand axis limits
        x_min = obs_array.min() - 0.1 * (obs_array.max() - obs_array.min())
        x_max = obs_array.max() + 0.1 * (obs_array.max() - obs_array.min())
        y_min = est_array.min() - 0.1 * (est_array.max() - est_array.min())
        y_max = est_array.max() + 0.1 * (est_array.max() - est_array.min())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Fitted")
        ax.set_title(f"{model_type} model")
        ax.legend(loc='upper left', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        adjust_text(text_annotations, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        self._save_fig(fig, f"_gof_4.png")

    def plot_gof_5(self, merged_data: pd.DataFrame):
        """
        Uses the row means of observed (x1_obs:x14_obs) and estimated (x1_est:x14_est) values.
        """
        df = merged_data.copy()
        if 'Observed_Mean' not in df.columns or 'Estimated_Mean' not in df.columns:
            df['Observed_Mean'] = df.loc[:, 'x1_obs':'x14_obs'].mean(axis=1)
            df['Estimated_Mean'] = df.loc[:, 'x1_est':'x14_est'].mean(axis=1)
        overall_std = df['Observed_Mean'].std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = df['GeneID'].unique()
        palette = sns.color_palette("tab20", len(unique_genes))
        colors = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(8, 8))
        plotted_genes = set()
        for obs, est, gene in zip(df['Observed_Mean'], df['Estimated_Mean'], df['GeneID']):
            ax.scatter(obs, est, color=colors[gene], edgecolor='black', s=100, marker='o')
            if gene not in plotted_genes and (est > obs + ci_offset_95 or est < obs - ci_offset_95):
                ax.text(obs, est, gene, fontsize=10, color=colors[gene],
                        fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                plotted_genes.add(gene)
        x_vals = [min(df['Observed_Mean'].min(), df['Estimated_Mean'].min()),
                  max(df['Observed_Mean'].max(), df['Estimated_Mean'].max())]
        ax.plot(x_vals, x_vals, color='grey', linestyle='-', linewidth=1.5)
        ax.plot(x_vals, [x + ci_offset_95 for x in x_vals],
                color='red', linestyle='--', linewidth=1, label='95% CI')
        ax.plot(x_vals, [x - ci_offset_95 for x in x_vals],
                color='red', linestyle='--', linewidth=1)
        ax.plot(x_vals, [x + ci_offset_99 for x in x_vals],
                color='gray', linestyle='--', linewidth=1, label='99% CI')
        ax.plot(x_vals, [x - ci_offset_99 for x in x_vals],
                color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Fitted")
        ax.set_title(f"{model_type} model")
        ax.legend(loc='upper left', fontsize='small', ncol=2)
        ax.grid(True)
        plt.tight_layout()
        self._save_fig(fig, f"_gof_5.png")

    def plot_gof_6(self, merged_data: pd.DataFrame):
        df = merged_data.copy()
        if 'Observed_Mean' not in df.columns or 'Estimated_Mean' not in df.columns:
            df['Observed_Mean'] = df.loc[:, 'x1_obs':'x14_obs'].mean(axis=1)
            df['Estimated_Mean'] = df.loc[:, 'x1_est':'x14_est'].mean(axis=1)
        overall_std = df['Observed_Mean'].std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = df['GeneID'].unique()
        palette = sns.color_palette("tab20", len(unique_genes))
        colors = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(8, 8))
        plotted_genes = set()
        sorted_indices = np.argsort(df['Observed_Mean'].values)
        for idx in sorted_indices:
            obs = df['Observed_Mean'].iloc[idx]
            est = df['Estimated_Mean'].iloc[idx]
            gene = df['GeneID'].iloc[idx]
            ax.scatter(obs, est, color=colors[gene], edgecolor='black', s=100, marker='o')
            if gene not in plotted_genes and (est > obs + ci_offset_95 or est < obs - ci_offset_95):
                ax.text(obs, est, gene, fontsize=10, color=colors[gene],
                        fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                plotted_genes.add(gene)
        x_min = df['Observed_Mean'].min() - 0.1 * (df['Observed_Mean'].max() - df['Observed_Mean'].min())
        x_max = df['Observed_Mean'].max() + 0.1 * (df['Observed_Mean'].max() - df['Observed_Mean'].min())
        y_min = df['Estimated_Mean'].min() - 0.1 * (df['Estimated_Mean'].max() - df['Estimated_Mean'].min())
        y_max = df['Estimated_Mean'].max() + 0.1 * (df['Estimated_Mean'].max() - df['Estimated_Mean'].min())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        x_vals = [df['Observed_Mean'].min(), df['Observed_Mean'].max()]
        ax.plot(x_vals, x_vals, color='grey', linestyle='-', linewidth=1.5)
        ax.plot(x_vals, [x + ci_offset_95 for x in x_vals],
                color='red', linestyle='--', linewidth=1, label='95% CI')
        ax.plot(x_vals, [x - ci_offset_95 for x in x_vals],
                color='red', linestyle='--', linewidth=1)
        ax.plot(x_vals, [x + ci_offset_99 for x in x_vals],
                color='gray', linestyle='--', linewidth=1, label='99% CI')
        ax.plot(x_vals, [x - ci_offset_99 for x in x_vals],
                color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Fitted")
        ax.set_title(f"{model_type} model")
        ax.legend(loc='upper left', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"_gof_6.png")

    def plot_kld(self, merged_data: pd.DataFrame):
        """
        Expects merged_data to have columns 'x1_obs' to 'x14_obs' and 'x1_est' to 'x14_est',
        as well as 'GeneID' and 'Psite'.
        """
        obs_data = merged_data.loc[:, 'x1_obs':'x14_obs']
        est_data = merged_data.loc[:, 'x1_est':'x14_est']
        normalized_obs = obs_data.div(obs_data.sum(axis=1), axis=0)
        normalized_est = est_data.div(est_data.sum(axis=1), axis=0)
        kl_div = normalized_obs.apply(lambda row: entropy(row, normalized_est.loc[row.name]), axis=1)
        kl_df = merged_data[['GeneID', 'Psite']].copy()
        kl_df['KL'] = kl_div.values
        kl_by_gene = kl_df.groupby('GeneID')['KL'].mean().sort_values()

        fig, ax = plt.subplots(figsize=(8, 8))
        indices = kl_by_gene.index.tolist()
        values = kl_by_gene.values
        ax.scatter(indices, values, marker='s', color='blue', label='Mean Normalized')
        ax.set_xticklabels(indices, rotation=45, ha='right')
        ax.set_ylabel("Entropy")
        ax.set_title("Kullback-Liebler Divergence")
        ax.legend()
        plt.tight_layout()
        self._save_fig(fig, f"_kld.png")