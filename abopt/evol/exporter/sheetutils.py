import pandas as pd
from abopt.evol.exporter.helpers import build_genes_data
from abopt.evol.exporter.plotout import plot_residuals_for_gene
from abopt.evol.config.logconf import setup_logger
logger = setup_logger()

def output_results(P_initial, P_init_dense, P_estimated, residuals, alpha_values, beta_values,
                   result, timepoints, OUT_FILE):

    # Build a genes_data dictionary from computed metrics.
    genes_data = build_genes_data(P_initial, P_init_dense, P_estimated, residuals)

    # For each gene, call the plotting functions.
    for gene, gene_data in genes_data.items():
        plot_residuals_for_gene(gene, gene_data)

        # Write results to Excel.
    output_file = OUT_FILE
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Alpha Values
        # Create a DataFrame for alphas
        alpha_data = []
        for (gene, psite), kinases in alpha_values.items():
            for kinase, value in kinases.items():
                alpha_data.append({'Protein': gene, 'Psite': psite, 'Kinase': kinase, 'Alpha': value})
        alpha_df = pd.DataFrame(alpha_data)

        # Write alpha values to a separate sheet
        alpha_df.to_excel(writer, sheet_name="Alpha Values", index=False)

        # Sheet 2: Beta Values
        # Create a DataFrame for betas
        # Collect beta data for each kinase and psite combination
        beta_data = []
        for (kinase, psite), value in beta_values.items():
            beta_data.append({'Kinase': kinase, 'Psite': psite,
                              'Beta': float(value.item()) if hasattr(value, "item") else float(value)})

        # Create DataFrame from the collected data
        beta_df = pd.DataFrame(beta_data)

        # Write beta values to a separate sheet
        beta_df.to_excel(writer, sheet_name="Beta Values", index=False)

        # Sheet 4: Residuals with Gene and Psite Labels
        residuals_data = []
        for i, ((gene, psite), data) in enumerate(P_initial.items()):
            residual_row = {'Gene': gene, 'Psite': psite}
            residual_row.update({time: residuals[i, t] for t, time in enumerate(timepoints)})
            residuals_data.append(residual_row)
        residuals_df = pd.DataFrame(residuals_data)
        residuals_df.to_excel(writer, sheet_name="Residuals", index=False)

        # Sheet 5: Estimated Values with Gene and Psite Labels
        estimated_data = []
        for i, ((gene, psite), data) in enumerate(P_initial.items()):
            estimated_row = {'Gene': gene, 'Psite': psite}
            estimated_row.update({time: P_estimated[i, t] for t, time in enumerate(timepoints)})
            estimated_data.append(estimated_row)
        estimated_df = pd.DataFrame(estimated_data)
        estimated_df.to_excel(writer, sheet_name="Estimated Values", index=False)

    logger.info(f"Optimization results saved for ODE modelling.")