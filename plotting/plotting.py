import itertools
import matplotlib.cm as cm
import os, re
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from itertools import combinations
from adjustText import adjust_text
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config.constants import COLOR_PALETTE, OUT_DIR, available_markers, model_type, TIME_POINTS_RNA, \
    PERTURBATIONS_TRACE_OPACITY


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

        n_states = len(labels)

        fig_width = min(max(1.2 * n_states, 8), 24)
        fig_height = 6 if n_states <= 20 else 8

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        parallel_coordinates(
            df, class_column='Time',
            colormap=plt.get_cmap("tab20"),
            ax=ax
        )

        ax.set_title(self.gene)
        ax.set_xlabel("States")
        ax.set_ylabel("Values")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth=0.8, color='gray', alpha=0.5)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.2)

        # Hide or rotate x-tick labels if too many
        if len(labels) > 40:
            ax.set_xticks([])  # hide ticks
        else:
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        plt.tight_layout()
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
        return pca_result, ev

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
        return tsne_result

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

    def plot_model_fit(self, model_fit: np.ndarray, P_data: np.ndarray, R_data: np.ndarray, sol: np.ndarray,
                       num_psites: int, psite_labels: list, time_points: np.ndarray):
        """
        Plots the model fit for the given data.

        :param model_fit: Estimated model fit values.
        :param P_data: Observed data for phosphorylation levels.
        :param R_data: Observed data for mRNA levels.
        :param sol: ODE solution for mRNA and protein levels.
        :param num_psites: number of phosphorylation sites.
        :param psite_labels: labels for the phosphorylation sites.
        :param time_points: time points for the data.
        :return:
        """
        model_fit = model_fit[9:].reshape(num_psites, 14)
        cutoff_idx = 8
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        ax = axes[0]
        ax.plot(time_points[:cutoff_idx], sol[:cutoff_idx, 0], '-', color='black', alpha=0.7, linewidth=1)
        ax.plot(TIME_POINTS_RNA[:3], R_data[:3], '--', marker='s', markersize=5, mew=0.5, mec='black',
                color='black', alpha=0.7, linewidth=0.75)
        ax.plot(time_points[:cutoff_idx], sol[:cutoff_idx, 1], '-', color='red', alpha=0.7, linewidth=1)
        for i in range(num_psites):
            ax.plot(time_points[:cutoff_idx], P_data[i, :cutoff_idx], '--', marker='s', markersize=5, mew=0.5,
                    mec='black',
                    color=self.color_palette[i], linewidth=0.75)
            ax.plot(time_points[:cutoff_idx], model_fit[i, :cutoff_idx], '-', color=self.color_palette[i], linewidth=1)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("FC")
        ax.set_xticks(time_points[:cutoff_idx])
        ax.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[:cutoff_idx]],
            rotation=45,
            fontsize=6
        )
        ax.grid(True, alpha=0.05)
        ax = axes[1]
        ax.plot(time_points[cutoff_idx:], sol[cutoff_idx:, 0], '-', color='black', alpha=0.7, label='mRNA (R)', linewidth=1)
        ax.plot(TIME_POINTS_RNA[3:], R_data[3:], '--', marker='s', markersize=5, mew=0.5, mec='black',
                color='black', alpha=0.7, linewidth=0.75)
        ax.plot(time_points[cutoff_idx:], sol[cutoff_idx:, 1], '-', color='red', alpha=0.7, label='Protein (P)', linewidth=1)
        for i in range(num_psites):
            ax.plot(time_points[cutoff_idx:], P_data[i, cutoff_idx:], '--', marker='s', markersize=5, mew=0.5, mec='black',
                    color=self.color_palette[i], label=f'{psite_labels[i]}', linewidth=0.75)
            ax.plot(time_points[cutoff_idx:], model_fit[i, cutoff_idx:], '-', color=self.color_palette[i], linewidth=1)
        ax.set_xlabel("Time (minutes)")
        ax.set_xticks(time_points[cutoff_idx:])
        ax.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[cutoff_idx:]],
            rotation=45,
            fontsize=6
        )
        ax.legend()
        ax.grid(True, alpha=0.05)
        plt.suptitle(f'{self.gene}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(fig, f"{self.gene}_model_fit_.png")

        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(
            x=time_points,
            y=sol[:, 0],
            mode='lines+markers',
            line=dict(color='black'),
            showlegend=False
        ))
        fig_plotly.add_trace(go.Scatter(
            x=TIME_POINTS_RNA,
            y=R_data,
            mode='lines+markers',
            name='mRNA(R)',
            line=dict(dash='dash', color='black')
        ))
        fig_plotly.add_trace(go.Scatter(
            x=time_points,
            y=sol[:, 1],
            mode='lines+markers',
            name='Protein(P)',
            line=dict(color='red')
        ))
        for i in range(num_psites):
            fig_plotly.add_trace(go.Scatter(
                x=time_points,
                y=P_data[i, :] if num_psites > 1 else P_data.flatten(),
                mode='lines+markers',
                name=f'{psite_labels[i]}',
                line=dict(dash='dash', color=self.color_palette[i])
            ))
            fig_plotly.add_trace(go.Scatter(
                x=time_points,
                y=model_fit[i, :],
                mode='lines+markers',
                line=dict(color=self.color_palette[i]),
                showlegend=False
            ))
        fig_plotly.update_layout(title=self.gene,
                                 xaxis_title="Time (minutes)",
                                 yaxis_title="Fold Changes",
                                 template="plotly_white",
                                 width=900, height=900)
        fig_plotly.write_html(os.path.join(self.out_dir, f"{self.gene}_model_fit_.html"))

    def plot_param_scatter(self, est_arr: np.ndarray, num_psites: int, time_vals: np.ndarray):
        """
        Plots scatter and density plots for (A, S), (B, S), (C, S), (D, S).

        :param est_arr: Estimated parameters array.
        :param num_psites: Number of phosphorylation sites.
        :param time_vals: Time values for the data.
        """
        est_arr = np.array(est_arr)
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=min(time_vals), vmax=max(time_vals))

        param_labels = ["A", "B", "C", "D"]

        for idx, label in enumerate(param_labels):
            param_vals = est_arr[:, idx]

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            legend_handles = []
            for i in range(num_psites):
                S_vals = est_arr[:, 4 + i]
                sc = ax.scatter(param_vals, S_vals, c=time_vals, cmap=cmap, norm=norm,
                                s=50, alpha=0.8, marker=available_markers[i])
                slope, intercept = np.polyfit(param_vals, S_vals, 1)
                x_fit = np.linspace(param_vals.min(), param_vals.max(), 100)
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
            ax.set_xlabel(f"{label} (rate)")
            ax.set_ylabel("S (Phosphorylation rate)")
            ax.set_title(self.gene)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Time (min)")
            ax.grid(True, alpha=0.2)
            ax.legend(handles=legend_handles)
            plt.tight_layout()
            self._save_fig(fig, f"{self.gene}_scatter_{label}_S_.png")

            # Density contour plot
            all_points = np.vstack([np.column_stack((param_vals, est_arr[:, 4 + i])) for i in range(num_psites)])
            kde = gaussian_kde(all_points.T)
            param_lin = np.linspace(param_vals.min(), param_vals.max(), 100)
            all_S = all_points[:, 1]
            S_lin = np.linspace(all_S.min(), all_S.max(), 100)
            param_grid, S_grid = np.meshgrid(param_lin, S_lin)
            grid_coords = np.vstack([param_grid.ravel(), S_grid.ravel()])
            density = kde(grid_coords).reshape(param_grid.shape)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(all_points[:, 0], all_points[:, 1], c='black', s=30, alpha=0.5)
            contourf = ax.contourf(param_grid, S_grid, density, levels=10, cmap="inferno", alpha=0.7)
            # Use a fixed value of 100 for contour levels to maintain consistency in density visualization.
            ax.contour(param_grid, S_grid, density, levels=100, colors='white', linewidths=0.5)
            ax.set_xlabel(f"{label}")
            ax.set_ylabel("S")
            ax.set_title(self.gene)
            cbar = plt.colorbar(contourf, ax=ax)
            cbar.set_label("Density")
            plt.tight_layout()
            self._save_fig(fig, f"{self.gene}_density_{label}_S_.png")

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

    def plot_gof(self, merged_data: pd.DataFrame):
        """
        Plot the goodness of fit for the model.
        """
        overall_std = merged_data.loc[:, 'x1_obs':'x14_obs'].values.std()
        ci_offset_95 = 1.96 * overall_std
        ci_offset_99 = 2.576 * overall_std

        unique_genes = merged_data['Gene'].unique()
        palette = sns.color_palette("husl", len(unique_genes))
        gene_color_map = {gene: palette[i] for i, gene in enumerate(unique_genes)}

        fig, ax = plt.subplots(figsize=(10, 10))
        plotted_genes = set()
        text_annotations = []
        obs_array = merged_data.loc[:, 'x1_obs':'x14_obs'].values
        est_array = merged_data.loc[:, 'x1_est':'x14_est'].values
        for gene, psite, obs_vals, est_vals in zip(merged_data['Gene'],
                                                   merged_data['Psite'],
                                                   obs_array, est_array):
            sorted_indices = np.argsort(obs_vals)
            obs_vals_sorted = obs_vals[sorted_indices]
            est_vals_sorted = est_vals[sorted_indices]
            ax.scatter(obs_vals_sorted, est_vals_sorted, color=gene_color_map[gene],
                       edgecolor='black', s=100, alpha=0.5)
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
        ax.set_title(f"{model_type} Model")
        ax.legend(loc='upper left', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        adjust_text(text_annotations, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        self._save_fig(fig, f"{model_type}_Goodness_of_Fit_.png")

    def plot_kld(self, merged_data: pd.DataFrame):
        """
        Plots the Kullback-Divergence for the model.
        """
        obs_data = merged_data.loc[:, 'x1_obs':'x14_obs']
        est_data = merged_data.loc[:, 'x1_est':'x14_est']
        normalized_obs = obs_data.div(obs_data.sum(axis=1), axis=0)
        normalized_est = est_data.div(est_data.sum(axis=1), axis=0)
        kl_div = normalized_obs.apply(lambda row: entropy(row, normalized_est.loc[row.name]), axis=1)
        kl_df = merged_data[['Gene', 'Psite']].copy()
        kl_df['KL'] = kl_div.values
        kl_by_gene = kl_df.groupby('Gene')['KL'].mean().sort_values()

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['lightcoral' if val > 0.03 else 'dodgerblue' for val in kl_by_gene.values]
        ax.barh(kl_by_gene.index, kl_by_gene.values, color=colors)
        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("Protein")
        ax.set_title(f"{model_type} Model")
        plt.tight_layout()
        self._save_fig(fig, f"{model_type}_kld_.png")

    def plot_params_bar(self, ci_results: dict, param_labels: list = None, time: str = None):
        """
        Plots bar plot for estimated parameter with 95% Confidence Interval.
        """
        beta_hat = ci_results['beta_hat']
        p_values = ci_results['pval']
        lwr_ci = ci_results['lwr_ci']
        upr_ci = ci_results['upr_ci']
        num_params = len(beta_hat)
        if param_labels is None:
            param_labels = [f"{i + 1}" for i in range(num_params)]

        valid_indices = [
            i for i, label in enumerate(param_labels)
            if re.fullmatch(r"[A-Za-z][0-9]?$", label)
        ]

        if not valid_indices:
            return

        beta_hat = beta_hat[valid_indices]
        p_values = p_values[valid_indices]
        lwr_ci = lwr_ci[valid_indices]
        upr_ci = upr_ci[valid_indices]
        param_labels = [param_labels[i] for i in valid_indices]

        lower_error = beta_hat - lwr_ci
        upper_error = upr_ci - beta_hat
        errors = np.vstack((lower_error, upper_error))

        x = np.arange(len(beta_hat))

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = []
        s_counter = 0
        for label in param_labels:
            if re.fullmatch(r"S\d+", label):
                color = self.color_palette[s_counter]
                colors.append(color)
                s_counter += 1
            else:
                colors.append('lightgray')

        ax.bar(x, beta_hat, yerr=errors, capsize=5, align='center',
               alpha=0.7, edgecolor='black', color=colors)

        for i, (xi, yi, pval) in enumerate(zip(x, beta_hat, p_values)):
            ax.text(xi, yi + upper_error[i] + 0.01 * np.max(beta_hat),
                    f"p={pval:.1e}", ha='center', va='bottom',
                    fontsize=6, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, ha='right')
        ax.set_ylabel('Estimate')
        ax.set_title(self.gene)
        ax.grid(True, axis='y', linestyle='--', alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_params_bar_.png")

    def plot_knockouts(self, results_dict: dict, num_psites: int, psite_labels: list):
        """
        Plot wild-type and knockout simulation results for comparison.
        """
        marker_cycle = itertools.cycle(available_markers)
        time_points = results_dict['WT'][0]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
        (ax_rp_zoom, ax_rp_full), (ax_ph_zoom, ax_ph_full) = axes
        time_cutoff = 8
        for label, (t, sol, p_fit) in results_dict.items():
            p_fit = p_fit[9:].reshape(num_psites, 14)
            marker = next(marker_cycle)
            # -- Last 'n' time range plots
            ax_rp_full.plot(t[time_cutoff:], sol[time_cutoff:, 0], label=f"{label} (mRNA)", linewidth=0.5, marker=marker,
                            markeredgecolor='black', markersize=6, mew=0.5)
            ax_rp_full.plot(t[time_cutoff:], sol[time_cutoff:, 1], label=f"{label} (Protein)", linewidth=0.5, marker=marker,
                            markeredgecolor='black', markersize=6, mew=0.5)
            for i in range(num_psites):
                ax_ph_full.plot(t[time_cutoff:], p_fit[i, time_cutoff:], label=f"{label} ({psite_labels[i]})", linewidth=0.5, marker=marker,
                                markeredgecolor='black', markersize=6, mew=0.5)

            # -- First 'n' points
            t_early = t[:time_cutoff]
            sol_early = sol[:time_cutoff]
            p_fit_early = p_fit[:, :time_cutoff]

            ax_rp_zoom.plot(t_early, sol_early[:, 0], linewidth=0.5, marker=marker,
                            markeredgecolor='black', markersize=6, mew=0.5)
            ax_rp_zoom.plot(t_early, sol_early[:, 1], linewidth=0.5, marker=marker,
                            markeredgecolor='black', markersize=6, mew=0.5)
            for i in range(num_psites):
                ax_ph_zoom.plot(t_early, p_fit_early[i, :], linewidth=0.5, marker=marker,
                                markeredgecolor='black', markersize=6, mew=0.5)

        ax_rp_full.legend(loc='upper right', fontsize=8)
        ax_rp_full.grid(True, alpha=0.1)

        ax_ph_full.set_xlabel("Time (min)")
        ax_ph_full.legend(loc='upper right', fontsize=8)
        ax_ph_full.grid(True, alpha=0.2)

        ax_ph_full.set_xticks(time_points[time_cutoff:])
        ax_ph_full.set_xticklabels([f"{int(tp)}" for tp in time_points][time_cutoff:], rotation=45, fontsize=6)

        ax_rp_zoom.grid(True, alpha=0.2)
        ax_rp_zoom.set_ylabel("FC")
        ax_rp_zoom.set_title("Transcription and Translation")

        ax_ph_zoom.set_xlabel("Time (min)")
        ax_ph_zoom.set_ylabel("FC")
        ax_ph_zoom.set_title("Phosphorylation")
        ax_ph_zoom.grid(True, alpha=0.1)

        ax_ph_zoom.set_xticks(time_points[:time_cutoff])
        ax_ph_zoom.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[:time_cutoff]],
            rotation=45,
            fontsize=6
        )
        plt.suptitle(f"{self.gene}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(fig, f"{self.gene}_.png")

    def plot_top_param_pairs(self, excel_path: str):
        """
        For each gene's '_perturbations' sheet in the Excel file,
        plot scatter plots for the parameter pairs with correlation.

        Args:
            excel_path (str): Path to the Excel file.
        """
        xls = pd.ExcelFile(excel_path)

        for sheet in xls.sheet_names:
            if not sheet.endswith("_perturbations"):
                continue

            self.gene = sheet.replace("_perturbations", "")
            df = pd.read_excel(xls, sheet_name=sheet)
            param_cols = [
                col for col in df.columns
                if isinstance(col, str)
                   and col != "RMSE"
                   and not col.startswith("(")
                   and re.fullmatch(r"[A-Za-z]\d?$", col)
            ]

            if len(param_cols) < 2:
                continue

            df_clean = df[param_cols].dropna().drop_duplicates()
            if df_clean.empty:
                continue

            corr = df_clean.corr().abs()
            pairs = list(combinations(corr.columns, 2))
            pair_scores = [(a, b, corr.loc[a, b]) for a, b in pairs]

            for a, b, score in pair_scores:
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.regplot(
                    x=df_clean[a],
                    y=df_clean[b],
                    ax=ax,
                    scatter_kws={'alpha': 0.7, 's': 10},
                    line_kws={'color': 'red', 'alpha': 0.5},
                    ci=95
                )
                ax.set_xlabel(a, fontsize=10)
                ax.set_ylabel(b, fontsize=10)
                ax.set_title(f"{self.gene}: {a} vs {b}\nCorrelation = {score:.2f}", fontsize=11)
                plt.tight_layout()
                self._save_fig(fig, f"{self.gene}_param_scatter_{a}_vs_{b}.png")

    def plot_model_perturbations(self, problem: dict, Si: dict, cutoff_idx: int, time_points: np.ndarray, n_sites: int,
                                 best_model_psite_solutions: np.ndarray, best_mrna_solutions: np.ndarray,
                                 best_protein_solutions: np.ndarray, psite_labels: list[str],
                                 psite_data_ref: np.ndarray, rna_ref: np.ndarray, model_fit_sol: np.ndarray) -> None:
        """
        Plot the best model perturbations for the given data.

        Args:
            problem: The optimization problem.
            Si: The simulation index.
            cutoff_idx: The cutoff index for the time points.
            time_points: The time points for the data.
            n_sites: The number of phosphorylation sites.
            best_model_psite_solutions: The best model phosphorylation site solutions.
            best_mrna_solutions: The best model mRNA solutions.
            best_protein_solutions: The best model protein solutions.
            psite_labels: The labels for the phosphorylation sites.
            psite_data_ref: The reference data for the phosphorylation sites.
            rna_ref: The reference data for mRNA.
        """
        simulations_trace_intensity = PERTURBATIONS_TRACE_OPACITY

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

        # --- Left plot: Until 9th time point ---
        ax = axes[0]

        # Plot the true psite curve with all simulations
        for site_idx in range(n_sites):
            color = COLOR_PALETTE[site_idx]
            for sim_idx in range(best_model_psite_solutions.shape[0]):
                ax.plot(
                    time_points[:cutoff_idx],
                    best_model_psite_solutions[sim_idx, :cutoff_idx, site_idx],
                    color=color,
                    alpha=simulations_trace_intensity,
                    linewidth=0.5
                )
            ax.plot(
                time_points[:cutoff_idx],
                model_fit_sol[:cutoff_idx, 2+site_idx],
                color=color,
                linewidth=1,
            )
            ax.plot(
                time_points[:cutoff_idx],
                psite_data_ref[site_idx, :cutoff_idx],
                marker='s',
                linestyle='--',
                color=color,
                markersize=5,
                linewidth=0.75,
                mew=0.5, mec='black',
            )

        # Plot the true mRNA curve with all simulations
        for sim_idx in range(best_mrna_solutions.shape[0]):
            ax.plot(
                time_points[:cutoff_idx],
                best_mrna_solutions[sim_idx, :cutoff_idx],
                color='gray',
                alpha=simulations_trace_intensity,
                linewidth=0.5
            )
        ax.plot(
            time_points[:cutoff_idx],
            model_fit_sol[:cutoff_idx, 0],
            color='black',
            linewidth=1,
        )
        ax.plot(
            TIME_POINTS_RNA[:3],
            rna_ref[:3],
            marker='s',
            linestyle='--',
            color='black',
            markersize=5,
            linewidth=0.75,
            mew=0.5, mec='black',
        )

        # Plot the true protein curve with all simulations
        for sim_idx in range(best_protein_solutions.shape[0]):
            ax.plot(
                time_points[:cutoff_idx],
                best_protein_solutions[sim_idx, :cutoff_idx],
                color='red',
                alpha=simulations_trace_intensity,
                linewidth=0.5
            )
        ax.plot(
            time_points[:cutoff_idx],
            model_fit_sol[:cutoff_idx, 1],
            color='red',
            linewidth=1,
        )

        ax.set_xlabel('Time (min)')
        ax.set_xticks(time_points[:cutoff_idx])
        ax.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[:cutoff_idx]],
            rotation=45,
            fontsize=6
        )
        ax.set_ylabel('FC')
        ax.grid(True, alpha=0.05)

        # --- Right plot: From 9th time point onwards ---
        ax = axes[1]

        # Plot the true psite curve with all simulations
        for site_idx in range(n_sites):
            color = COLOR_PALETTE[site_idx]
            for sim_idx in range(best_model_psite_solutions.shape[0]):
                ax.plot(
                    time_points[cutoff_idx:],
                    best_model_psite_solutions[sim_idx, cutoff_idx:, site_idx],
                    color=color,
                    alpha=simulations_trace_intensity,
                    linewidth=0.5
                )
            ax.plot(
                time_points[cutoff_idx:],
                model_fit_sol[cutoff_idx:, 2+site_idx],
                color=color,
                linewidth=1,
                label=psite_labels[site_idx]
            )
            ax.plot(
                time_points[cutoff_idx:],
                psite_data_ref[site_idx, cutoff_idx:],
                marker='s',
                linestyle='--',
                color=color,
                markersize=5,
                linewidth=0.75,
                mew=0.5, mec='black'
            )

        # Plot the true mRNA curve with all simulations
        for sim_idx in range(best_mrna_solutions.shape[0]):
            ax.plot(
                time_points[cutoff_idx:],
                best_mrna_solutions[sim_idx, cutoff_idx:],
                color='gray',
                alpha=simulations_trace_intensity,
                linewidth=0.5
            )
        ax.plot(
            time_points[cutoff_idx:],
            model_fit_sol[cutoff_idx:, 0],
            color='black',
            linewidth=1,
            label='mRNA (R)'
        )
        ax.plot(
            TIME_POINTS_RNA[3:],
            rna_ref[3:],
            marker='s',
            linestyle='--',
            color='black',
            markersize=5,
            linewidth=0.75,
            mew=0.5, mec='black',
        )

        # Plot the true protein curve with all simulations
        for sim_idx in range(best_protein_solutions.shape[0]):
            ax.plot(
                time_points[cutoff_idx:],
                best_protein_solutions[sim_idx, cutoff_idx:],
                color='red',
                alpha=simulations_trace_intensity,
                linewidth=0.5
            )

        ax.plot(
            time_points[cutoff_idx:],
            model_fit_sol[cutoff_idx:, 1],
            color='red',
            linewidth=1,
            label='Protein (P)'
        )

        ax.set_xlabel('Time (min)')
        ax.set_xticks(time_points[cutoff_idx:])
        ax.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[cutoff_idx:]],
            rotation=45,
            fontsize=6
        )
        ax.grid(True, alpha=0.05)
        ax.legend()

        plt.suptitle(f'{self.gene}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(fig, f"{self.gene}_sensitivity_perturbations.png")

        # Absolute Mean of Elementary Effects : represents the overall importance
        # of each parameter, reflecting its sensitivity
        ## Bar Plot of mu* ##
        # Standard Deviation of Elementary Effects: High standard deviation suggests
        # that the parameter has nonlinear effects or is involved in interactions
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.bar(problem['names'], Si['mu_star'], yerr=Si['mu_star_conf'], color='skyblue')
        ax.set_title(f'{self.gene}')
        ax.set_ylabel('mu* (Importance)')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_sensitivity_bar_plot_mu.png")
  

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.bar(problem['names'], Si['sigma'], color='orange')
        ax.set_title(f'{self.gene}')
        ax.set_ylabel('σ (Standard Deviation)')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_sensitivity_bar_plot_sigma.png")
        

        ## Bar Plot of sigma ##
        # Distinguish between parameters with purely linear effects (low sigma) and
        # those with nonlinear or interaction effects (high sigma).
        # **--- Parameters with high mu* and high sigma ---**
        #           <particularly important to watch>
        ## Scatter Plot of mu* vs sigma ##
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(Si['mu_star'], Si['sigma'], color='green', s=100)
        for i, param in enumerate(problem['names']):
            ax.text(Si['mu_star'][i], Si['sigma'][i], param, fontsize=12, ha='right', va='bottom')
        ax.set_title(f'{self.gene}')
        ax.set_xlabel('mu* (Mean Absolute Effect)')
        ax.set_ylabel('σ (Standard Deviation)')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_sensitivity_scatter_plot_musigma.png")
        

        # A radial plot (also known as a spider or radar plot) can give a visual
        # overview of multiple sensitivity metrics (e.g., mu*, sigma, etc.) for
        # each parameter in a circular format.

        # Each parameter gets a spoke, and the distance from the center represents
        # the sensitivity for a given metric.
        ## Radial Plot (Spider Plot) of Sensitivity Metrics ##
        categories = problem['names']
        N_cat = len(categories)
        mu_star = Si['mu_star']
        sigma = Si['sigma']
        angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
        mu_star = np.concatenate((mu_star, [mu_star[0]]))
        sigma = np.concatenate((sigma, [sigma[0]]))
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, mu_star, color='skyblue', alpha=0.4, label='Mu*')
        ax.fill(angles, sigma, color='orange', alpha=0.4, label='Sigma')
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(f'{self.gene}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2)
        self._save_fig(fig, f"{self.gene}_sensitivity_radial_plot.png")
        

        # Visualize the proportion of total sensitivity contributed by each
        # parameter using a pie chart, showing the relative importance of each
        # parameter's contribution to sensitivity.
        ## Pie Chart for Sensitivity Contribution ##
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(Si['mu_star'], labels=problem['names'], autopct='%1.1f%%', startangle=90,
               colors=plt.cm.Paired.colors,
               textprops={'fontsize': 6})
        ax.set_title(f'{self.gene}')
        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_sensitivity_pie_chart.png")

    def plot_time_state_grid(self, samples: np.ndarray, time_points: np.ndarray, state_names: list):
        """
        Grid of strip plots per state showing variability across time.

        Args:
            samples: shape (n_samples, n_timepoints, n_states)
            time_points: array of time points
            state_names: list of state names
        """
        # Identify valid state indices (e.g., 'P', 'P1', 'X2')
        valid_indices = [
            idx for idx, name in enumerate(state_names)
            if re.fullmatch(r"[A-Za-z][0-9]?$", name)
        ]

        # Skip plotting if no valid states remain
        if len(valid_indices) == 0:
            print("Skipping time-state grid: no valid state names.")
            return

        # Filter state names and corresponding sample data
        filtered_names = [state_names[i] for i in valid_indices]
        filtered_samples = samples[:, :, valid_indices]

        # Prepare time labels
        time_labels = [f"{int(t)}" if t > 1 else f"{t}" for t in time_points]

        # Build long-form DataFrame
        data = []
        for state_idx, state in enumerate(filtered_names):
            for t_idx, label in enumerate(time_labels):
                for v in filtered_samples[:, t_idx, state_idx]:
                    data.append({'Value': v, 'Time': label, 'State': state})
        df = pd.DataFrame(data)

        # Ensure time is treated as categorical for consistent ordering
        df['Time'] = pd.Categorical(df['Time'], categories=time_labels, ordered=True)

        # Assign consistent colors to each state
        full_palette = list(itertools.islice(itertools.cycle(self.color_palette), len(filtered_names)))
        palette_dict = dict(zip(filtered_names, full_palette))

        # Create strip plots for each state across time
        g = sns.catplot(
            data=df,
            x="Time",
            y="Value",
            hue="State",
            col="State",
            kind="strip",
            col_wrap=4,
            height=3,
            sharey=False,
            alpha=0.3,
            jitter=0.2,
            size=2,
            palette=palette_dict,
            legend=False
        )

        g.set_axis_labels("", "")
        g.set_titles("{col_name}")

        for ax in g.axes.flatten():
            ax.tick_params(axis='y', labelsize=6)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_fontsize(6)

        g.fig.suptitle(f"{self.gene}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(g.fig, f"{self.gene}_sensitivity_state_grid.png")

    def plot_phase_space(self, samples: np.ndarray, state_names: list):
        """
        Phase space plots: one state vs another for each simulation.

        Args:
            samples (np.ndarray): Shape (n_samples, n_timepoints, n_states)
            state_names (list): List of state names (length = num_states)
        """
        n_samples, n_timepoints, n_states = samples.shape
        assert n_states >= 2, "Need at least two states for phase space plots."

        # Identify valid state name indices
        valid_indices = [
            idx for idx, name in enumerate(state_names)
            if re.fullmatch(r"[A-Za-z][0-9]?$", name)
        ]

        # If fewer than two valid states, skip plotting
        if len(valid_indices) < 2:
            return

        # Filter state names and sample dimensions
        filtered_names = [state_names[i] for i in valid_indices]
        filtered_samples = samples[:, :, valid_indices]

        # Assign one color per simulation
        cmap = cm.get_cmap("tab20", n_samples)
        sim_colors = [cmap(i) for i in range(n_samples)]

        # Generate pairwise plots
        for i, j in combinations(range(len(filtered_names)), 2):
            x_state = filtered_names[i]
            y_state = filtered_names[j]

            fig, ax = plt.subplots(figsize=(6, 6))
            for sim in range(n_samples):
                ax.plot(
                    filtered_samples[sim, :, i],
                    filtered_samples[sim, :, j],
                    alpha=0.5,
                    linewidth=0.5,
                    color=sim_colors[sim]
                )

            ax.set_xlabel(x_state)
            ax.set_ylabel(y_state)
            ax.set_title(f"{self.gene}", fontsize=10)
            ax.grid(True, alpha=0.05)
            plt.tight_layout()
            self._save_fig(fig, f"{self.gene}_sensitivity_phase_space_{x_state}_vs_{y_state}.png")

    def plot_future_fit(self, P_data: np.ndarray, R_data: np.ndarray, sol: np.ndarray,
                       num_psites: int, psite_labels: list, time_points: np.ndarray):
        """
        Plots the model fit for the future time points.

        :param model_fit: Estimated model fit values.
        :param P_data: Observed data for phosphorylation levels.
        :param R_data: Observed data for mRNA levels.
        :param sol: ODE solution for mRNA and protein levels.
        :param num_psites: number of phosphorylation sites.
        :param psite_labels: labels for the phosphorylation sites.
        :param time_points: time points for the data.
        :return:
        """
        cutoff_idx = 8
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        ax = axes[0]
        ax.plot(time_points[:cutoff_idx], sol[:cutoff_idx, 0], '-', color='black', alpha=0.7, linewidth=1)
        ax.plot(TIME_POINTS_RNA[:3], R_data[:3], '--', marker='s', markersize=5, mew=0.5, mec='black',
                color='black', alpha=0.7, linewidth=0.75)
        ax.plot(time_points[:cutoff_idx], sol[:cutoff_idx, 1], '-', color='red', alpha=0.7, linewidth=1)
        for i in range(num_psites):
            ax.plot(time_points[:cutoff_idx], P_data[i, :cutoff_idx], '--', marker='s', markersize=5, mew=0.5,
                    mec='black',
                    color=self.color_palette[i], linewidth=0.75)
            ax.plot(time_points[:cutoff_idx], sol[:cutoff_idx, i+1], '-', color=self.color_palette[i], linewidth=1)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("FC")
        ax.set_xticks(time_points[:cutoff_idx])
        ax.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[:cutoff_idx]],
            rotation=45,
            fontsize=6
        )
        ax.grid(True, alpha=0.05)
        ax = axes[1]
        ax.plot(time_points[cutoff_idx:], sol[cutoff_idx:, 0], '-', color='black', alpha=0.7, label='mRNA (R)', linewidth=1)
        ax.plot(TIME_POINTS_RNA[3:], R_data[3:], '--', marker='s', markersize=5, mew=0.5, mec='black',
                color='black', alpha=0.7, linewidth=0.75)
        ax.plot(time_points[cutoff_idx:], sol[cutoff_idx:, 1], '-', color='red', alpha=0.7, label='Protein (P)', linewidth=1)
        for i in range(num_psites):
            ax.plot(time_points[cutoff_idx:cutoff_idx+6], P_data[i, cutoff_idx:], '--', marker='s', markersize=5, mew=0.5, mec='black',
                    color=self.color_palette[i], label=f'{psite_labels[i]}', linewidth=0.75)
            ax.plot(time_points[cutoff_idx:], sol[cutoff_idx:, i+1], '-', color=self.color_palette[i], linewidth=1)
        ax.set_xlabel("Time (minutes)")
        ax.set_xticks(time_points[cutoff_idx:])
        ax.set_xticklabels(
            [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[cutoff_idx:]],
            rotation=45,
            fontsize=6
        )
        ax.legend()
        ax.grid(True, alpha=0.05)
        plt.suptitle(f'{self.gene}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(fig, f"{self.gene}_future_fit_.png")

    def plot_regularization(self, excel_path: str):
        """
        Read every '<gene>_params' sheet in the Excel file, pull the Regularization value,
        and plot a horizontal bar chart of regularization vs. gene.
        """
        xls = pd.ExcelFile(excel_path)
        genes = []
        regs = []

        for sheet in xls.sheet_names:
            if not sheet.endswith("_params"):
                continue  # Only process *_params sheets
            gene = sheet[:-7]
            df = pd.read_excel(xls, sheet_name=sheet)
            if 'Regularization' not in df.columns:
                continue
            reg_value = df['Regularization'].iloc[0]
            genes.append(gene)
            regs.append(reg_value)

        if not genes:
            return

        sorted_pairs = sorted(zip(regs, genes), reverse=True)
        regs, genes = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(8, max(8, len(genes) * 0.5)))
        bars = ax.barh(genes, regs, color='skyblue', edgecolor='black', height=0.6)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Protein', fontsize=9)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(
            f"{model_type} Model  "
            r"Tikhnov Regularization (L2): "
            r"$R = \frac{\lambda_{\mathrm{reg}}}{m}\sum_{j=1}^m p_j^2$",
            fontsize=8
        )
        for bar in bars:
            ax.text(bar.get_width() + max(regs) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}", va='center', ha='left', fontsize=6)
        plt.tight_layout()
        self._save_fig(fig, f"{model_type}_model_regularization.png")

    def plot_model_error(self, excel_path: str):
        """
        Read every '<gene>_params' sheet in the Excel file, pull the RMSE value,
        and plot a horizontal bar chart of RMSE vs. gene.
        """
        xls = pd.ExcelFile(excel_path)
        genes = []
        mses = []

        for sheet in xls.sheet_names:
            if not sheet.endswith("_params"):
                continue
            gene = sheet[:-7]
            df = pd.read_excel(xls, sheet_name=sheet)
            if 'MSE' not in df.columns:
                continue
            error_val = df['MSE'].iloc[0]
            genes.append(gene)
            mses.append(error_val)

        if not genes:
            return

        sorted_pairs = sorted(zip(mses, genes), reverse=True)
        mses, genes = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(8, max(8, len(genes) * 0.5)))
        bars = ax.barh(genes, mses, color='coral', edgecolor='black', height=0.6)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Protein', fontsize=9)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(
            f"{model_type} Model  "
            r"$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$",
            fontsize=8
        )

        for bar in bars:
            ax.text(bar.get_width() + max(mses) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}", va='center', ha='left', fontsize=6)

        plt.tight_layout()
        self._save_fig(fig, f"{model_type}_model_error.png")