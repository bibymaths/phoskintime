
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kinopt.fitanalysis.helpers.postfit import goodnessoffit, reshape_alpha_beta, perform_pca, plot_pca, perform_tsne, \
    additional_plots, create_sankey_from_network, important_connections
from kinopt.local.config.constants import OUT_FILE, OUT_DIR


def optimization_performance():
    """
    Executes the optimization performance analysis pipeline.

    This function reads input data from an Excel file, performs various analyses such as PCA, t-SNE,
    and residual analysis, and generates visualizations and output files. It also creates a Sankey
    diagram and extracts important connections for further analysis.

    Steps performed:
    1. Reads input data from the specified Excel file.
    2. Computes goodness of fit between estimated and observed data.
    3. Reshapes alpha and beta values for further analysis.
    4. Performs PCA and t-SNE on the data and generates corresponding plots.
    5. Generates additional plots for residuals and other metrics.
    6. Processes alpha values to create a kinase-substrate relationship file.
    7. Creates a Sankey diagram to visualize phosphorylation connections.
    8. Extracts the top N important connections and saves them to a CSV file.

    Outputs:
    - PCA and t-SNE plots.
    - Residual analysis plots.
    - `kinase_substrate.csv` file containing kinase-substrate relationships.
    - Sankey diagram saved as an HTML file.
    - CSV file of top important connections.

    :raises FileNotFoundError: If the input Excel file is not found.
    :raises ValueError: If required columns are missing in the input data.
    """

    file_path = OUT_FILE
    output_dir = OUT_DIR
    alpha_values = pd.read_excel(file_path, sheet_name='Alpha Values')
    beta_values = pd.read_excel(file_path, sheet_name='Beta Values')
    estimated_df = pd.read_excel(file_path, sheet_name='Estimated')
    observed_df = pd.read_excel(file_path, sheet_name='Observed')
    residuals_df = pd.read_excel(file_path, sheet_name='Residuals')
    goodnessoffit(estimated_df, observed_df)
    df = reshape_alpha_beta(alpha_values, beta_values)
    result_df_sorted = perform_pca(df)
    plot_pca(result_df_sorted, 'PCA')
    numeric_df = df[['Value']].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    tsne_result_df_sorted = perform_tsne(scaled_data, df)
    plot_pca(tsne_result_df_sorted, 'tSNE')
    additional_plots(df, scaled_data, alpha_values, beta_values, residuals_df)
    combined_alpha = alpha_values.rename(columns={'Gene': 'Target', 'Kinase': 'Source', 'Alpha': 'Value'})
    combined_alpha['Type'] = 'Alpha'
    data = combined_alpha.dropna(subset=['Source', 'Target'])
    # Renaming and transforming columns
    ks = data.rename(columns={
        'Source': 'KinaseID',
        'Target': 'TargetID',
        'Value': 'KinaseActivity',
        'Psite': 'PhosphoSiteID'
    })
    # Removing underscores in the 'PhosphoSiteID' column
    ks['PhosphoSiteID'] = ks['PhosphoSiteID'].str.replace('_', '', regex=False)
    # Keeping only the specified columns
    ks = ks[['KinaseID', 'TargetID', 'KinaseActivity', 'PhosphoSiteID']]
    ks.to_csv(f'{OUT_DIR}/kinase_substrate.csv')
    create_sankey_from_network(output_dir, data, "Phosphorylation Connections")
    important_connections(output_dir, data)