import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')

class Plotter:
    """
    A class to plot various analysis results from an Excel file.
    The class provides methods to visualize the alpha and beta values,
    residuals, observed and estimated values, and other metrics.

    The plots include:
    - Alpha distribution
    - Beta bar plots
    - Heatmap of absolute residuals
    - Goodness of fit
    - Kullback-Leibler divergence
    - PCA
    - Box plots for alpha and beta values
    - CDF for alpha and beta values
    - Time-wise residuals
    """
    def __init__(self, filepath, savepath):
        """
        Initializes the Plotter instance by loading data from the Excel file.
        """
        self.filepath = filepath
        self.savepath = savepath
        self.load_data()

    def load_data(self):
        """
        Loads data from the specified Excel file.
        The data includes residuals, observed values, estimated values,
        alpha values, and beta values.
        """
        self.df = pd.read_excel(self.filepath, sheet_name='Residuals', index_col=0)
        self.df_obs = pd.read_excel(self.filepath, sheet_name='Observed', index_col=0)
        self.df_est = pd.read_excel(self.filepath, sheet_name='Estimated', index_col=0)
        self.df_alpha = pd.read_excel(self.filepath, sheet_name='Alpha Values', index_col=(0,1))
        self.df_beta = pd.read_excel(self.filepath, sheet_name='Beta Values')

    def plot_alpha_distribution(self):
        """
        Plots the distribution of alpha parameter values grouped by transcription factors (TFs)
        using a strip plot.
        """
        unique_mrnas = self.df_alpha.index.get_level_values(0).unique()
        all_markers = [m for m in mmarkers.MarkerStyle.markers
                       if isinstance(m, str) and len(m) == 1 and m not in {' ', ''}]

        for mrna in unique_mrnas:
            plt.figure(figsize=(8, 8))

            # Get alpha values for this mRNA
            df_subset = self.df_alpha.loc[mrna].copy()
            df_subset.columns = ['Value']
            df_subset = df_subset.sort_values(by='Value')  # sort TFs by alpha

            tf_list = df_subset.index.tolist()
            alpha_values = df_subset['Value'].values

            marker_map = {tf: all_markers[i % len(all_markers)] for i, tf in enumerate(tf_list)}

            # Plot each TF individually
            for i, (tf, alpha) in enumerate(zip(tf_list, alpha_values)):
                plt.scatter(
                    alpha, 0,  # constant y for strip plot effect
                    label=tf,
                    marker=marker_map[tf],
                    edgecolor='black',
                    s=80  # marker size
                )

            # Create custom legend
            legend_handles = [
                mlines.Line2D([], [], color='black', marker=marker_map[tf],
                              linestyle='None', markersize=7, label=tf)
                for tf in tf_list
            ]
            plt.legend(
                handles=legend_handles,
                title='TFs',
                fontsize='7',
                title_fontsize='9',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True
            )

            plt.title(f'mRNA: {mrna}')
            plt.xlabel('Alpha Value')
            plt.yticks([])  # remove y-axis ticks
            plt.tight_layout()
            plt.savefig(f"{self.savepath}/alpha_distribution_{mrna}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_beta_barplots(self):
        """
        Processes the beta values DataFrame and creates a separate bar plot
        for each unique transcription factor (TF).
        """
        # Data cleaning for df_beta
        self.df_beta['PSite'] = self.df_beta['PSite'].fillna(self.df_beta['TF'])
        self.df_beta['Value'] = self.df_beta.apply(
            lambda row: 0 if row['PSite'] == row['TF'] and pd.isna(row['Value']) else row['Value'], axis=1
        )
        self.df_beta['PSite'] = self.df_beta.apply(
            lambda row: 'β₀' if row['PSite'] == row['TF'] else row['PSite'], axis=1
        )

        unique_tfs = self.df_beta['TF'].unique()
        # Plot bar plot for each TF
        for tf in unique_tfs:
            tf_data = self.df_beta[self.df_beta['TF'] == tf]
            plt.figure(figsize=(6, 6))
            sns.barplot(
                data=tf_data,
                x='PSite',
                y='Value',
                palette='Dark2',
                edgecolor='black',
                linewidth=0.5
            )
            plt.xlabel("Phosphorylation - Residue Position", fontsize = 7)
            plt.ylabel("β", fontsize = 7)
            plt.title(f"Effect of Phosphorylation on Transcription Factor {tf} Activity", fontsize=7)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(f'{self.savepath}/TF_{tf}_beta_group.png', dpi=300)
            plt.close()

    def plot_heatmap_abs_residuals(self):
        """
        Plots a heatmap of the absolute values of the residuals.
        """
        plt.figure(figsize=(12, 12))
        abs_df = self.df.abs()
        # Use fixed x tick labels as given in the original code
        sns.heatmap(
            abs_df,
            xticklabels=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
            yticklabels=abs_df.index,
            cmap='viridis'
        )
        plt.title('Absolute Residuals')
        plt.xlabel('Time Points')
        plt.ylabel('mRNA')
        plt.yticks(fontsize=6, rotation=0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/Residual_Heatmap.png', dpi=300)
        plt.close()

    def plot_goodness_of_fit(self):
        """
        Creates a scatter plot comparing observed vs. estimated values,
        fits a linear regression model, plots the 95% confidence interval,
        and labels points outside the confidence interval.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Flatten arrays
        obs_flat = self.df_obs.values.flatten()
        est_flat = self.df_est.values.flatten()
        mRNAs    = self.df.index

        ax.scatter(self.df_obs, self.df_est, alpha=0.5)

        for i, mRNA in enumerate(mRNAs):
            ax.scatter(
                self.df_obs.iloc[i],
                self.df_est.iloc[i],
                label=mRNA,
                alpha=0.5,
                s=100,
                edgecolor='black'
            )

        # compute residuals’ std and CI offsets
        diffs       = est_flat - obs_flat
        std_diff    = np.std(diffs, ddof=1)
        ci_offset_95 = 1.96 * std_diff
        ci_offset_99 = 2.576 * std_diff

        # define plotting range
        min_val = min(obs_flat.min(), est_flat.min())
        max_val = max(obs_flat.max(), est_flat.max())

        # 45° diagonal
        ax.plot([min_val, max_val], [min_val, max_val],
                color='gray', linestyle='-', linewidth=1.5, label='y = x')

        # ±95% band
        ax.plot([min_val, max_val],
                [min_val + ci_offset_95, max_val + ci_offset_95],
                color='red', linestyle='--', linewidth=1, label='±95%')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_95, max_val - ci_offset_95],
                color='red', linestyle='--', linewidth=1)

        # ±99% band
        ax.plot([min_val, max_val],
                [min_val + ci_offset_99, max_val + ci_offset_99],
                color='gray', linestyle='--', linewidth=1, label='±99%')
        ax.plot([min_val, max_val],
                [min_val - ci_offset_99, max_val - ci_offset_99],
                color='gray', linestyle='--', linewidth=1)

        # label points outside the 95% band
        for x_val, y_val, mRNA in zip(obs_flat, est_flat, mRNAs):
            if abs(y_val - x_val) > ci_offset_95:
                ax.text(x_val, y_val, mRNA,
                        fontsize=8, ha='right', va='bottom', fontweight='semibold', fontstyle='normal')

        ax.set_xlabel('Observed Values')
        ax.set_ylabel('Estimated Values')
        ax.set_title('Goodness of Fit')
        ax.grid(True, alpha=0.1)
        ax.legend(loc='best', fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/Goodness_of_Fit.png', dpi=300)
        plt.close(fig)

    def plot_kld(self):
        """
        Plots the Kullback-Leibler Divergence (KLD) for each mRNA.
        The KLD is calculated between the observed and estimated distributions
        of the mRNA expression levels.
        """

        # Normalize observed and estimated values
        normalized_obs = self.df_obs.loc[:, 'x1':'x9'].div(self.df_obs.loc[:, 'x1':'x9'].sum(axis=1), axis=0)
        normalized_est = self.df_est.loc[:, 'x1':'x9'].div(self.df_est.loc[:, 'x1':'x9'].sum(axis=1), axis=0)
        # Calculate KLD for each mRNA
        kld = normalized_obs.apply(lambda row: entropy(row, normalized_est.loc[row.name]), axis=1)
        # Create a DataFrame for KLD values
        kld_df = pd.DataFrame({'mRNA': self.df_obs.index, 'KL': kld.values}).set_index('mRNA')
        # Sort KLD values by mRNA
        kld_by_gene = kld_df.sort_values(by='KL', ascending=False)
        # Plot the KLD values
        plt.figure(figsize=(12, 12))
        # Add horizontal bar plot for KLD values with different color for bars above 0.03
        plt.barh(kld_by_gene.index[kld_by_gene['KL'] > 0.03], kld_by_gene['KL'][kld_by_gene['KL'] > 0.03],
                 color='coral', alpha=0.6)
        plt.barh(kld_by_gene.index[kld_by_gene['KL'] <= 0.03], kld_by_gene['KL'][kld_by_gene['KL'] <= 0.03],
                 color='cornflowerblue', alpha=0.6)
        plt.ylabel("mRNA", fontsize=7)
        plt.yticks(fontsize=6)
        plt.xticks(fontsize=9)
        plt.xlabel("Kullback-Leibler Divergence", fontsize=7)
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/KLD.png', dpi=300)
        plt.close()

    def plot_pca(self):
        """
        Plots a PCA (Principal Component Analysis) of the observed and estimated values.
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_est)
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        # Create a DataFrame for PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=self.df_est.index)
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(pca_df)
        pca_df['Cluster'] = kmeans.labels_
        # Add clusters to the PCA plot
        plt.figure(figsize=(12, 12))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
        # Plot clusters
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', alpha=0.5)
        for i, mRNA in enumerate(pca_df.index):
            plt.annotate(mRNA, (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]), fontsize=6,
                         ha='right', va='bottom', annotation_clip=True)
        plt.title('Principal Component Analysis')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/PCA.png', dpi=300)
        plt.close()

    def plot_boxplot_alpha(self):
        """
        Plots a boxplot of the alpha values.
        """
        plt.figure(figsize=(6, 6))
        sns.boxplot(data=self.df_alpha.iloc[:, 0], palette='Dark2')
        plt.title('Distribution of Alpha Values')
        plt.ylabel('Alpha Value')
        plt.xlabel('mRNA')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/Box_Plot_Alpha.png', dpi=300)
        plt.close()

    def plot_boxplot_beta(self):
        """
        Plots a boxplot of the beta values.
        """
        plt.figure(figsize=(6, 6))
        sns.boxplot(data=self.df_beta['Value'], palette='Dark2')
        plt.title('Distribution of Beta Values')
        plt.ylabel('Beta Value')
        plt.xlabel('Phosphorylation - Residue Position')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/Box_Plot_Beta.png', dpi=300)
        plt.close()

    def plot_cdf_alpha(self):
        """
        Plots the cumulative distribution function (CDF) of the alpha values.
        """
        plt.figure(figsize=(6, 6))
        sns.ecdfplot(self.df_alpha.iloc[:, 0], stat='proportion')
        plt.xlabel('Alpha Value')
        plt.ylabel('Cumulative Probability')
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/CDF_Alpha.png', dpi=300)
        plt.close()

    def plot_cdf_beta(self):
        """
        Plots the cumulative distribution function (CDF) of the beta values.
        """
        plt.figure(figsize=(6, 6))
        sns.ecdfplot(self.df_beta['Value'], stat='proportion')
        plt.xlabel('Beta Value')
        plt.ylabel('Cumulative Probability')
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/CDF_Beta.png', dpi=300)
        plt.close()

    def plot_time_wise_residuals(self):
        """
        Plots the residuals over time for each mRNA.
        """
        plt.figure(figsize=(6, 6))
        # Generate a colormap for unique mRNAs with residuals > 0.5
        unique_colors = plt.cm.tab10(np.linspace(0, 1, len(self.df.index)))
        default_color = 'lightgray'
        # Calculate a single mean absolute value across all time points for all mRNAs
        mean_residuals = self.df.abs().mean(axis=1)
        for i, mRNA in enumerate(self.df.index):
            if any(self.df.iloc[i] > mean_residuals.iloc[i]):
                # Assign a unique color for mRNAs with residuals > 0.5
                color = unique_colors[i % len(unique_colors)]
                linestyle = '-'
            else:
                # Use the default color for other mRNAs
                color = default_color
                linestyle = ':'
                mRNA = ''  # Hide label for mRNAs with no residuals > 0.5

            plt.plot(self.df.columns, self.df.iloc[i], label=mRNA, color=color, linestyle=linestyle)

        plt.xlabel('Time Points', fontsize=7)
        plt.ylabel('Residuals', fontsize=7)
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        # plt.legend(loc='best', fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(f'{self.savepath}/Time_Wise_Residuals.png', dpi=300)
        plt.close()