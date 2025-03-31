
import pandas as pd
from sklearn.preprocessing import StandardScaler

from abopt.fitanalysis.helpers.postfit import goodnessoffit, reshape_alpha_beta, perform_pca, plot_pca, perform_tsne, \
    additional_plots, create_sankey_from_network, important_connections
from abopt.local.config.constants import OUT_FILE, OUT_DIR


def main():
    file_path = OUT_FILE
    output_dir = OUT_DIR
    alpha_values = pd.read_excel(file_path, sheet_name='Alpha Values')
    beta_values = pd.read_excel(file_path, sheet_name='Beta Values')
    estimated_df = pd.read_excel(file_path, sheet_name='Estimated Values')
    observed_df = pd.read_excel(file_path, sheet_name='Observed Values')
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
    combined_alpha = alpha_values.rename(columns={'Protein': 'Target', 'Kinase': 'Source', 'Alpha': 'Value'})
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

if __name__ == "__main__":
    main()