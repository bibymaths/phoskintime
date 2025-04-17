import numpy as np
from kinopt.evol.objfn import estimated_series
from kinopt.evol.config.logconf import setup_logger
logger = setup_logger()

def extract_parameters(P_initial, gene_psite_counts, K_index, optimized_params):
    """
    Extracts the optimized alpha and beta values from the optimized parameters.
    The function organizes the values into dictionaries for easy access and
    interpretation. The alpha values are associated with gene-psite pairs and
    their corresponding kinases, while the beta values are associated with
    kinase-psite pairs. The function also logs the optimized values for
    transparency and debugging purposes.

    The function takes the following parameters:
    :param P_initial:
    :param gene_psite_counts:
    :param K_index:
    :param optimized_params:

    :return:
    - alpha_values: Dictionary mapping (gene, psite) to a dictionary of kinases and their alpha values.
    - beta_values: Dictionary mapping (kinase, psite) to their corresponding beta values.
    """
    alpha_values = {}
    alpha_start = 0
    for idx, count in enumerate(gene_psite_counts):
        gene, psite = list(P_initial.keys())[idx]
        kinases = P_initial[(gene, psite)]['Kinases']
        alpha_values[(gene, psite)] = dict(zip(kinases, optimized_params[alpha_start:alpha_start + count]))
        alpha_start += count

    # Extract betas for kinase-psite
    beta_values = {}
    beta_start = sum(gene_psite_counts)
    for kinase, kinase_psites in K_index.items():
        for psite, _ in kinase_psites:
            count = 1  # Each psite in beta_counts has a count of 1
            beta_values[(kinase, psite)] = optimized_params[beta_start:beta_start + count]
            beta_start += count
            # Display optimized values
    logger.info("Optimized Alpha Values:")
    for (gene, psite), kinases in alpha_values.items():
        logger.info(f"Protein {gene}, Psite {psite}:")
        for kinase, value in kinases.items():
            logger.info(f"  Kinase {kinase}: {value}")

    logger.info("Optimized Beta Values:")
    for (kinase, psite), value in beta_values.items():
        logger.info(f"Kinase {kinase}, Psite {psite}: {value}")
    return alpha_values, beta_values

def compute_metrics(optimized_params, P_initial, P_initial_array, K_index, K_array,
                    gene_psite_counts, beta_counts, n):
    """
    Computes various error metrics to evaluate the performance of the optimization process.
    The function calculates the Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared value.
    These metrics provide insights into the accuracy of the estimated time series compared to the observed data.

    The function takes the following parameters:
    :param optimized_params:
    :param P_initial:
    :param P_initial_array:
    :param K_index:
    :param K_array:
    :param gene_psite_counts:
    :param beta_counts:
    :param n:

    :return:
    - P_estimated: Estimated time series matrix for all gene-psite combinations.
    - residuals: Residuals between observed and estimated values.
    - mse: Mean Squared Error.
    - rmse: Root Mean Squared Error.
    - mae: Mean Absolute Error.
    - mape: Mean Absolute Percentage Error.
    - r_squared: R-squared value.
    """
    P_estimated = estimated_series(
        optimized_params, P_initial, K_index, K_array, gene_psite_counts, beta_counts
    )
    residuals = P_initial_array - P_estimated
    mse = np.sum((residuals) ** 2) / n
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / P_initial_array)) * 100
    r_squared = 1 - (np.sum((residuals) ** 2) / np.sum((P_initial_array - np.mean(P_initial_array)) ** 2))
    logger.info("--- Error Metrics ---")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    logger.info(f"R-squared (R^2): {r_squared:.4f}")
    return P_estimated, residuals, mse, rmse, mae, mape, r_squared