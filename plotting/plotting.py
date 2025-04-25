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
            ax.contour(param_grid, S_grid, density, levels=CONTOUR_LEVELS, colors='white', linewidths=0.5)
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
        self._save_fig(fig, f"_Goodness_of_Fit_.png")

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
        ax.set_title("")
        plt.tight_layout()
        self._save_fig(fig, f"_kld_.png")

    def plot_params_bar(self, ci_results: dict, param_labels: list = None,
                        title: str = ''):
        """
        Plots parameter estimates as a bar plot with confidence intervals.

        Parameters:
          - ci_results: Output dictionary from confidence_intervals() function.
          - param_labels: Optional list of labels for each parameter.
          - title: Title for the plot.
        """
        beta_hat = ci_results['beta_hat']
        lwr_ci = ci_results['lwr_ci']
        upr_ci = ci_results['upr_ci']

        num_params = len(beta_hat)
        x = np.arange(num_params)

        if param_labels is None:
            param_labels = [f"Rate{i + 1}" for i in range(num_params)]

        lower_error = beta_hat - lwr_ci
        upper_error = upr_ci - beta_hat
        errors = np.vstack((lower_error, upper_error))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x, beta_hat, yerr=errors, capsize=5, align='center', alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=45, ha='right')
        ax.set_ylabel('Estimate')
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        self._save_fig(fig, f"{self.gene}_params_bar_.png")
