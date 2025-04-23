
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import entropy
from adjustText import adjust_text
import adjustText
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kinopt.evol.config.constants import OUT_DIR


def goodnessoffit(estimated, observed):
    """
    Function to plot the goodness of fit for estimated and observed values.
    It creates scatter plots with confidence intervals and labels for genes outside the 95% CI.
    The function also calculates KL divergence and generates a heatmap for optimization progression.

    :param estimated:
    :param observed:
    """
    merged_data = estimated.rename(columns={"Gene": "GeneID"}).merge(observed, on=['GeneID', 'Psite'],
                                                                     suffixes=('_est', '_obs'))
    merged_data['Observed_Mean'] = merged_data.loc[:, 'x1_obs':'x14_obs'].mean(axis=1)
    merged_data['Estimated_Mean'] = merged_data.loc[:, 'x1_est':'x14_est'].mean(axis=1)
    min_val = min(merged_data.loc[:, 'x1_obs':'x14_obs'].values.min(),
                  merged_data.loc[:, 'x1_est':'x14_est'].values.min())
    max_val = max(merged_data.loc[:, 'x1_obs':'x14_obs'].values.max(),
                  merged_data.loc[:, 'x1_est':'x14_est'].values.max())
    colors = cm.tab20(range(len(merged_data)))

    ci_color_95 = 'red'
    ci_color_99 = 'gray'
    diagonal_color = 'gray'

    # Generate a palette with as many distinct colors as genes
    unique_genes_list = merged_data['GeneID'].unique()

    # Generate a categorical color palette with fully distinct colors (tab20 has 20 distinct colors)
    palette = sns.color_palette("tab20", len(unique_genes_list))

    # Map each gene to a unique color from the categorical palette
    gene_color_map = {gene: palette[i % len(palette)] for i, gene in enumerate(unique_genes_list)}

    # Calculate overall mean and standard deviation
    overall_std = merged_data.loc[:, 'x1_obs':'x14_obs'].values.std()

    # Calculate CI offsets for 95% and 99%
    ci_offset_95 = 1.96 * overall_std
    ci_offset_99 = 2.576 * overall_std

    plt.figure(figsize=(10, 10))

    # Plot Observed vs. Estimated for x1 to x14 values
    plotted_genes = set()
    text_annotations = []  # Collect text objects for adjustment
    for i, (gene, psite, obs_vals, est_vals) in enumerate(zip(merged_data['GeneID'],
                                                              merged_data['Psite'],
                                                              merged_data.loc[:, 'x1_obs':'x14_obs'].values,
                                                              merged_data.loc[:, 'x1_est':'x14_est'].values)):
        # Sort values for plotting
        sorted_indices = np.argsort(obs_vals)
        obs_vals_sorted = obs_vals[sorted_indices]
        est_vals_sorted = est_vals[sorted_indices]

        # Plot observed and estimated values
        plt.scatter(obs_vals_sorted, est_vals_sorted, color=gene_color_map[gene], alpha=0.5, s=100, edgecolor='black')

        # Add label only for genes outside the 95% CI
        for obs, est in zip(obs_vals_sorted, est_vals_sorted):
            if gene not in plotted_genes and (est > obs + ci_offset_95 or est < obs - ci_offset_95):
                plt.text(obs, est, gene, fontsize=8, color=gene_color_map[gene],
                         fontweight='bold', ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                plotted_genes.add(gene)

    # Adjust text positions to avoid overlap
    adjust_text(text_annotations, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Add diagonal line through the origin
    plt.plot([min_val, max_val], [min_val, max_val], color=diagonal_color, linestyle='-', linewidth=1.5)

    # Add lines parallel to the diagonal for 95% and 99% CI
    plt.plot([min_val, max_val], [min_val + ci_offset_95, max_val + ci_offset_95], color=ci_color_95, linestyle='--',
             linewidth=1, label='95% CI')
    plt.plot([min_val, max_val], [min_val - ci_offset_95, max_val - ci_offset_95], color=ci_color_95, linestyle='--',
             linewidth=1)
    plt.plot([min_val, max_val], [min_val + ci_offset_99, max_val + ci_offset_99], color=ci_color_99, linestyle='--',
             linewidth=1, label='99% CI')
    plt.plot([min_val, max_val], [min_val - ci_offset_99, max_val - ci_offset_99], color=ci_color_99, linestyle='--',
             linewidth=1)

    # Expand the axes limits slightly to include all points without clipping
    x_min = merged_data.loc[:, 'x1_obs':'x14_obs'].values.min() - 0.1 * (
                merged_data.loc[:, 'x1_obs':'x14_obs'].values.max() - merged_data.loc[:,
                                                                      'x1_obs':'x14_obs'].values.min())
    x_max = merged_data.loc[:, 'x1_obs':'x14_obs'].values.max() + 0.1 * (
                merged_data.loc[:, 'x1_obs':'x14_obs'].values.max() - merged_data.loc[:,
                                                                      'x1_obs':'x14_obs'].values.min())
    y_min = merged_data.loc[:, 'x1_est':'x14_est'].values.min() - 0.1 * (
                merged_data.loc[:, 'x1_est':'x14_est'].values.max() - merged_data.loc[:,
                                                                      'x1_est':'x14_est'].values.min())
    y_max = merged_data.loc[:, 'x1_est':'x14_est'].values.max() + 0.1 * (
                merged_data.loc[:, 'x1_est':'x14_est'].values.max() - merged_data.loc[:,
                                                                      'x1_est':'x14_est'].values.min())
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add labels and grid
    plt.xlabel("Observed")
    plt.ylabel("Fitted")
    plt.title("")
    # plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Goodness_of_Fit.png", dpi=300)
    plt.close()

    # KL Divergence
    normalized_obs = merged_data.loc[:, 'x1_obs':'x14_obs'].div(merged_data.loc[:, 'x1_obs':'x14_obs'].sum(axis=1),
                                                                axis=0)
    normalized_est = merged_data.loc[:, 'x1_est':'x14_est'].div(merged_data.loc[:, 'x1_est':'x14_est'].sum(axis=1),
                                                                axis=0)
    kl_divergence = normalized_obs.apply(lambda row: entropy(row, normalized_est.loc[row.name]), axis=1)

    # Sort before plotting
    kl_divergence_df = merged_data[['GeneID', 'Psite']].copy()
    kl_divergence_df['KL'] = kl_divergence.values
    kl_divergence_by_gene = kl_divergence_df.groupby('GeneID')['KL'].mean().sort_values()

    plt.figure(figsize=(8, 8))

    # Ensure distinct values for index (GeneID) on x-axis
    indices = kl_divergence_by_gene.index.tolist()
    values = kl_divergence_by_gene.values

    plt.scatter(indices, values, marker='s', linestyle='-', color='blue', label=r"$\bar{x}$")
    plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels
    plt.ylabel("Entropy")
    plt.title('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/kld.png', dpi=300)
    plt.close()


# Function to reshape alpha and beta values
def reshape_alpha_beta(alpha_values, beta_values):
    """
    Function to reshape alpha and beta values for plotting.
    It renames columns and creates a new 'Parameter' column for each type of value.

    :param alpha_values:
    :param beta_values:

    :returns:
    - pd.DataFrame: Reshaped DataFrame containing 'GeneID', 'Value', and 'Parameter' columns.
    """
    alpha_values['Gene'] = alpha_values['Gene'].astype(str)
    alpha_values['Psite'] = alpha_values['Psite'].astype(str)
    alpha_values['Kinase'] = alpha_values['Kinase'].astype(str)

    beta_values['Kinase'] = beta_values['Kinase'].astype(str)
    beta_values['Psite'] = beta_values['Psite'].astype(str)

    alpha_values_reshaped = alpha_values[['Gene', 'Psite', 'Alpha']].rename(
        columns={'Gene': 'GeneID', 'Alpha': 'Value'})
    alpha_values_reshaped['Parameter'] = 'α_' + alpha_values['Gene'] + '_' + alpha_values['Psite'] + '_' + \
                                         alpha_values['Kinase']

    beta_values_reshaped = beta_values[['Kinase', 'Psite', 'Beta']].rename(
        columns={'Kinase': 'GeneID', 'Beta': 'Value'})
    beta_values_reshaped['Parameter'] = 'β_' + beta_values['Kinase'] + '_' + beta_values['Psite']

    return pd.concat([alpha_values_reshaped, beta_values_reshaped], ignore_index=True)


# Function to perform PCA analysis
def perform_pca(df):
    """
    Perform PCA analysis on the given DataFrame.
    The DataFrame should contain a 'Value' column for PCA analysis.
    The function returns a DataFrame with PCA results and additional columns for type and gene/psite information.

    :param df: DataFrame containing the data for PCA analysis.
    :return: DataFrame with PCA results and additional columns.
    """
    numeric_df = df[['Value']].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(pca_result, columns=['PCA'])
    result_df = pd.concat([df[['Parameter']], pca_df], axis=1)
    result_df['Type'] = result_df['Parameter'].apply(lambda x: 'Alpha' if x.startswith('α') else 'Beta')
    result_df['Gene_Psite'] = result_df['Parameter'].str.extract(r'α_(.*?)_|β_(.*)')[0].combine_first(
        result_df['Parameter'].str.extract(r'β_(.*?)_')[0]
    )
    return result_df.sort_values(by=['Gene_Psite', 'Type'])


# Function to plot PCA or t-SNE results
def plot_pca(result_df_sorted, y_axis_column):
    """
    Plot PCA or t-SNE results for each gene/psite.
    The function creates scatter plots with different markers for alpha and beta parameters,
    and adds labels for each point.
    The function also adjusts text labels to avoid overlap using the adjustText library.

    :param result_df_sorted: DataFrame containing PCA or t-SNE results.
    :param y_axis_column: Column name for the y-axis values in the plot.
    """

    for gene_psite, group in result_df_sorted.groupby('Gene_Psite'):
        plt.figure(figsize=(8, 8))
        alpha_marker = '^'
        beta_marker = 'v'

        texts = []
        for param_type, marker in [('Alpha', alpha_marker), ('Beta', beta_marker)]:
            subset = group[group['Type'] == param_type]
            if not subset.empty:
                plt.scatter(subset['Parameter'], subset[y_axis_column], alpha=0.7, marker=marker, label=param_type)
                for _, row in subset.iterrows():
                    if param_type == 'Alpha':
                        label = "_".join(row['Parameter'].split('_')[2:5])
                    else:
                        label = "_".join(row['Parameter'].split('_')[1:4])
                    texts.append(plt.text(row['Parameter'], row[y_axis_column], label, fontsize=8, alpha=0.7))

        adjustText.adjust_text(
            texts, arrowprops=dict(arrowstyle="-", color='gray', alpha=0.5, lw=0.5, shrinkA=5, shrinkB=5)
        )
        plt.title(f'{gene_psite}', fontsize=12)
        plt.ylabel(y_axis_column, fontsize=8)
        plt.xticks([])
        plt.legend(title='Type', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{y_axis_column}_{gene_psite}.png', format='png', dpi=300)
        plt.close()


# Function to perform t-SNE analysis
def perform_tsne(scaled_data, df):
    """
    Perform t-SNE analysis on the given scaled data.
    The function returns a DataFrame with t-SNE results and additional columns for type and gene/psite information.

    :param scaled_data:
    :param df:

    :return:
    - pd.DataFrame: DataFrame with t-SNE results and additional columns.
    """
    tsne = TSNE(n_components=1, perplexity=30, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)

    tsne_df = pd.DataFrame(tsne_result, columns=['tSNE'])
    tsne_result_df = pd.concat([df[['Parameter']], tsne_df], axis=1)
    tsne_result_df['Type'] = tsne_result_df['Parameter'].apply(lambda x: 'Alpha' if x.startswith('α') else 'Beta')
    tsne_result_df['Gene_Psite'] = tsne_result_df['Parameter'].str.extract(r'α_(.*?)_|β_(.*)')[0].combine_first(
        tsne_result_df['Parameter'].str.extract(r'β_(.*?)_')[0]
    )
    return tsne_result_df.sort_values(by=['Gene_Psite', 'Type'])


# Function to plot CDF, KDE, Boxplot, and Hierarchical Clustering
def additional_plots(df, scaled_data, alpha_values, beta_values, residuals_df):
    """
    Function to create additional plots including CDF, KDE, Boxplot, and Hierarchical Clustering.

    :param df:
    :param scaled_data:
    :param alpha_values:
    :param beta_values:
    :param residuals_df:
    """
    alpha_df = df[df['Parameter'].str.startswith('α')]
    beta_df = df[df['Parameter'].str.startswith('β')]

    # KDE Plot
    plt.figure(figsize=(8, 8))
    sns.kdeplot(alpha_df['Value'], color='blue', label='α', fill=True)
    sns.kdeplot(beta_df['Value'], color='green', label='β', fill=True)
    plt.title('')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/distribution_parameters.png', format='png', dpi=300)
    plt.close()

    # Box Plot
    plt.figure(figsize=(8, 8))
    sns.boxplot(
        x='Type',
        y='Value',
        data=pd.concat([alpha_df, beta_df]).assign(Type=lambda df: df['Parameter'].str[0])
    )
    plt.title('')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/boxplot_parameters.png', format='png', dpi=300)
    plt.close()

    # CDF Plots
    plt.figure(figsize=(8, 8))
    sns.ecdfplot(alpha_df['Value'], color='blue', label='α')
    sns.ecdfplot(beta_df['Value'], color='green', label='β')
    plt.title('')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/cdf_parameters.png', format='png', dpi=300)
    plt.close()

    # Prepare data
    alpha_data = alpha_values[['Gene', 'Alpha']].rename(columns={'Gene': 'Group', 'Alpha': 'Value'})
    alpha_data['Parameter'] = r'$\alpha$'
    beta_data = beta_values[['Kinase', 'Beta']].rename(columns={'Kinase': 'Group', 'Beta': 'Value'})
    beta_data['Parameter'] = r'$\beta$'
    combined_data = pd.concat([alpha_data, beta_data])
    sorted_data = combined_data.sort_values(by='Value')

    # Plot the violin plot
    plt.figure(figsize=(8, 8))
    sns.violinplot(data=sorted_data, y='Group', x='Value', hue='Parameter', split=True, palette='tab20', orient='h')
    plt.yticks(rotation=0)
    plt.title("")
    plt.xlabel("Estimated Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/violin_parameters.png', format='png', dpi=300)
    plt.close()

    # Rename the columns of time points in the residuals DataFrame to 1, 2, ..., 14
    # Create a mapping of known column names to their replacements
    column_mapping = {f"x{i}": str(i) for i in range(1, 15)}
    # Rename the columns directly using the mapping
    residuals_df.rename(columns=column_mapping, inplace=True)

    # Define consistent time columns
    time_columns = [str(i) for i in range(1, 15)]

    # Heatmap data for optimization progression
    heatmap_data = residuals_df.set_index('Gene')[time_columns]

    # Optimization progression heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        annot=False,
        cbar_kws={'label': 'Magnitude'}
    )
    plt.title("")
    plt.xlabel("Time")
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/time_residuals.png', format='png', dpi=300)
    plt.close()

    # Variance across time points
    time_point_variance = residuals_df.set_index('Gene')[time_columns].var(axis=0)

    # Identify the top 5 highest variance time points
    top_5_variances = time_point_variance.nlargest(5)

    # Plot the variance across time points with top 5 highlighted
    plt.figure(figsize=(8, 8))
    bars = plt.bar(time_point_variance.index, time_point_variance, color='lightblue', edgecolor='black')
    for time_point in top_5_variances.index:
        bars[time_point_variance.index.get_loc(time_point)].set_color('coral')

    plt.title("")
    plt.xlabel("Time")
    plt.ylabel(r"$\mathrm{Var}(\text{residuals})$")
    plt.xticks(rotation=45)

    # Annotate the top 5 points
    for time_point, variance in top_5_variances.items():
        plt.text(time_point_variance.index.get_loc(time_point), variance + 0.01, f"{variance:.2f}",
                 ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/variance_residuals.png', format='png', dpi=300)
    plt.close()

    # Error trends over time
    # Mean absolute residuals across all Genes for each time point
    mean_absolute_error = residuals_df[time_columns].abs().mean(axis=0)

    # Identify top 3 lowest and top 3 highest MAE time points
    top_3_highest_mae = mean_absolute_error.nlargest(3)
    top_3_lowest_mae = mean_absolute_error.nsmallest(3)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_absolute_error.index, mean_absolute_error, marker='o', color='lightblue', linestyle='-')

    # Highlight top 3 highest with upward triangles and lowest with downward triangles
    for time_point, value in top_3_highest_mae.items():
        plt.scatter(mean_absolute_error.index.get_loc(time_point), value, color='red', s=100, marker='^', zorder=5)
    for time_point, value in top_3_lowest_mae.items():
        plt.scatter(mean_absolute_error.index.get_loc(time_point), value, color='green', s=100, marker='v', zorder=5)

    plt.title("")
    plt.xlabel("Time")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/error_trends.png', format='png', dpi=300)
    plt.close()

    # Residual profiles across Genes
    # Sum residuals across time points for each gene
    residual_profiles = residuals_df.set_index('Gene')[time_columns].sum(axis=1)

    # Sort and identify top 5 and bottom 5 Genes with highest and lowest cumulative residuals
    sorted_residual_profiles = residual_profiles.sort_values(ascending=False)
    top_5_residuals = sorted_residual_profiles.head(5)
    bottom_5_residuals = sorted_residual_profiles.tail(5)

    plt.figure(figsize=(8, 8))

    # Assign unique colors to the top 5 and bottom 5 bars
    top_colors = ['red', 'blue', 'green', 'orange', 'purple']
    bottom_colors = ['cyan', 'magenta', 'yellow', 'brown', 'pink']
    default_color = 'teal'

    # Create bars with distinct colors for the top 5 and bottom 5
    for index, (gene, value) in enumerate(sorted_residual_profiles.items()):
        if gene in top_5_residuals.index:
            color_index = top_5_residuals.index.tolist().index(gene)
            plt.bar(index, value, color=top_colors[color_index], edgecolor='black',
                    label=gene if gene not in plt.gca().get_legend_handles_labels()[1] else None)
        elif gene in bottom_5_residuals.index:
            color_index = bottom_5_residuals.index.tolist().index(gene)
            plt.bar(index, value, color=bottom_colors[color_index], edgecolor='black',
                    label=gene if gene not in plt.gca().get_legend_handles_labels()[1] else None)
        else:
            plt.bar(index, value, color=default_color, edgecolor='black')

    # Add a single legend for the top 5 and bottom 5 Genes
    plt.legend(title="Protein")

    # Remove x-axis labels and ticks for a cleaner look
    plt.xticks([], [])
    plt.title("")
    plt.ylabel("Cumulative Residuals")
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/residual_profiles.png', format='png', dpi=300)
    plt.close()


# Create a Sankey diagram from the data and functions
def create_sankey_from_network(output_dir, data, title):
    """
    Creates a Sankey diagram from the given data and saves it as an HTML file.

    This function processes the input data to generate nodes and links for a Sankey diagram.
    It assigns colors to nodes and links based on their attributes and values, and uses Plotly
    to render the diagram. The resulting diagram is saved as an HTML file in the specified output directory.

    :param output_dir: str
        The directory where the Sankey diagram HTML file will be saved.
    :param data: pd.DataFrame
        A DataFrame containing the data for the Sankey diagram. It must include the following columns:
        - 'Source': The source node of the link.
        - 'Target': The target node of the link.
        - 'Value': The value of the link, which determines the flow size.
    :param title: str
        The title of the Sankey diagram.

    The function performs the following steps:
    1. Initializes nodes and links for the Sankey diagram.
    2. Maps node labels to indices and assigns colors to nodes.
    3. Processes the data to create links between nodes, assigning colors based on link values.
    4. Builds the Sankey diagram using Plotly.
    5. Adds a color bar to explain the flow gradient.
    6. Saves the Sankey diagram as an HTML file in the specified output directory.
    """

    # Initialize nodes and links for Sankey diagram
    nodes = []
    links = []

    # Create a mapping for node indices
    node_indices = {}
    index = 0

    # Define color scale
    cmap = plt.cm.tab20  # Choose a colormap
    norm = mcolors.Normalize(vmin=data['Value'].abs().min(), vmax=data['Value'].abs().max())

    # Process the data to extract nodes and edges
    for _, row in data.iterrows():
        source = row['Source']
        target = row['Target']
        value = abs(row['Value']) * 100  # Use absolute value for Sankey flow size

        # Add source and target nodes if not already added
        if source not in node_indices:
            nodes.append({"label": source, "color": "green" if "Kinase" in source else "red"})
            node_indices[source] = index
            index += 1
        if target not in node_indices:
            nodes.append({"label": target, "color": "gray"})
            node_indices[target] = index
            index += 1

        # Generate color for the link based on value
        flow_color = mcolors.rgb2hex(cmap(norm(value)))

        # Add a link between the source and target
        links.append({
            "source": node_indices[source],
            "target": node_indices[target],
            "value": value,
            "color": flow_color
        })

    # Build the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node["label"] for node in nodes],
            color=[node["color"] for node in nodes]
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            color=[link["color"] for link in links]  # Set link colors
        )
    ))
    # Add a color bar to explain the flow gradient
    colorbar = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            cmin=data['Value'].abs().min(),
            cmax=data['Value'].abs().max(),
            colorbar=dict(
                title="Edge Value"
            )
        ),
        hoverinfo='none'
    )
    # Add the color bar scatter plot
    fig.add_trace(colorbar)
    # Update layout
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title_text=title, font_size=10)
    fig.write_html(f"{output_dir}/sankey.html")


# Function to extract important connections and save them to a CSV file
def important_connections(output_dir, data, top_n=20):
    """
    Extracts the top N most important connections based on their absolute values
    and saves them to a CSV file.

    :param output_dir: str
        The directory where the CSV file will be saved.
    :param data: pd.DataFrame
        A DataFrame containing the connections with columns 'Source', 'Target', and 'Value'.
    :param top_n: int, optional
        The number of top connections to extract (default is 20).

    The function sorts the connections by their absolute values in descending order,
    selects the top N connections, and saves them to a CSV file named 'top_connections.csv'
    in the specified output directory.
    """

    sorted_edges = data.sort_values(by="Value", key=abs, ascending=False).head(top_n)
    important_connections = sorted_edges[["Source", "Target", "Value"]]
    important_connections.to_csv(f"{output_dir}/top_connections.csv", index=False)