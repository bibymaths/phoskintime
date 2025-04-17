import pandas as pd
from kinopt.local.config.constants import TIME_POINTS, OUT_FILE
from kinopt.local.exporter.helpers import build_genes_data
from kinopt.local.exporter.plotout import *
from kinopt.local.config.logconf import setup_logger
logger = setup_logger()

def output_results(P_initial, P_init_dense, P_estimated, residuals, alpha_values, beta_values,
                   result, mse, rmse, mae, mape, r_squared):
    """
    Function to output the results of the optimization process.
    It logs the optimized alpha and beta values, optimization summary,
    error metrics, and generates plots for each gene.
    It also writes the results to an Excel file with multiple sheets.

    The sheets include:
    - Alpha Values: Optimized alpha values for each gene and psite.
    - Beta Values: Optimized beta values for each kinase and psite.
    - Summary: Summary of the optimization process.
    - Observed: Observed time-series data for each gene and psite.
    - Estimated: Estimated time-series data for each gene and psite.
    - Residuals: Residuals for each gene and psite.

    :param P_initial:
    :param P_init_dense:
    :param P_estimated:
    :param residuals:
    :param alpha_values:
    :param beta_values:
    :param result:
    :param mse:
    :param rmse:
    :param mae:
    :param mape:
    :param r_squared:
    """
    logger.info("Optimized Alpha values:")
    for (gene, psite), kinases in alpha_values.items():
        logger.info(f"Protein {gene}, Psite {psite}:")
        for kinase, value in kinases.items():
            logger.info(f"  Kinase {kinase}: {value:.2f}")
    logger.info("Optimized Beta values:")
    for (kinase, psite), value in beta_values.items():
        logger.info(f"Kinase {kinase}, Psite {psite}: {value:.2f}")
    logger.info("--- Optimization Summary ---")
    logger.info(f"Success: {'Success' if result.success else 'Failure'}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Iterations: {result.nit}")
    logger.info(f"Function Evaluations: {result.nfev}")
    logger.info(f"SSE: {result.fun:.2f}")
    logger.info("--- Error Metrics ---")
    logger.info(f"MSE: {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"R-squared: {r_squared:.2f}")

    # Build a genes_data dictionary from computed metrics.
    genes_data = build_genes_data(P_initial, P_init_dense, P_estimated, residuals)

    # For each gene, call the plotting functions.
    for gene, data in genes_data.items():
        plot_fits_for_gene(gene, data, TIME_POINTS)
        plot_cumulative_residuals(gene, data, TIME_POINTS)
        plot_autocorrelation_residuals(gene, data, TIME_POINTS)
        plot_histogram_residuals(gene, data, TIME_POINTS)
        plot_qqplot_residuals(gene, data, TIME_POINTS)

    # Write results to Excel.
    output_file = OUT_FILE
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        alpha_list = []
        for (gene, psite), kinases in alpha_values.items():
            for kinase, value in kinases.items():
                alpha_list.append({'Gene': gene, 'Psite': psite, 'Kinase': kinase, 'Alpha': value})
        pd.DataFrame(alpha_list).to_excel(writer, sheet_name="Alpha Values", index=False)

        beta_list = []
        for (kinase, psite), value in beta_values.items():
            beta_list.append({'Kinase': kinase, 'Psite': psite, 'Beta': value})
        pd.DataFrame(beta_list).to_excel(writer, sheet_name="Beta Values", index=False)

        summary = {
            'Metric': ["Success", "Message", "Iterations", "Function Evaluations", "SSE", "MSE", "RMSE", "MAE", "MAPE",
                       "R-squared"],
            'Value': [("Success" if result.success else "Failure"), result.message, result.nit, result.nfev, result.fun,
                      mse, rmse, mae, mape, r_squared]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        timepoints = [f'x{i + 1}' for i in range(P_init_dense.shape[1])]

        # For Observed sheet: use "GeneID" as the column name.
        observed_keys_df = pd.DataFrame(list(P_initial.keys()), columns=["GeneID", "Psite"])
        observed_df = pd.DataFrame(P_init_dense, columns=timepoints)
        observed_df = pd.concat([observed_keys_df, observed_df], axis=1)
        observed_df.to_excel(writer, sheet_name="Observed", index=False)

        # For Estimated sheet: use "gene" as the column name.
        estimated_keys_df = pd.DataFrame(list(P_initial.keys()), columns=["Gene", "Psite"])
        estimated_df = pd.DataFrame(P_estimated, columns=timepoints)
        estimated_df = pd.concat([estimated_keys_df, estimated_df], axis=1)
        estimated_df.to_excel(writer, sheet_name="Estimated", index=False)

        # For Residuals sheet: use "gene" as the column name.
        residuals_keys_df = pd.DataFrame(list(P_initial.keys()), columns=["Gene", "Psite"])
        residuals_df = pd.DataFrame(residuals, columns=timepoints)
        residuals_df = pd.concat([residuals_keys_df, residuals_df], axis=1)
        residuals_df.to_excel(writer, sheet_name="Residuals", index=False)

    logger.info(f"Optimization results saved for ODE modelling.")