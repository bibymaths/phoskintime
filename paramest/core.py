import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from knockout import apply_knockout, generate_knockout_combinations
from config.constants import get_param_names, generate_labels, OUT_DIR, SENSITIVITY_ANALYSIS, TIME_POINTS
from models.diagram import illustrate
from paramest.toggle import estimate_parameters
from sensitivity import sensitivity_analysis
from models import solve_ode
from steady import initial_condition
from plotting import Plotter
from config.logconf import setup_logger

logger = setup_logger()

def process_gene(
        gene,
        protein_data,
        kinase_data,
        mrna_data,
        time_points,
        bounds,
        bootstraps=0,
        out_dir=OUT_DIR
):
    """
    Process a single gene by estimating its parameters and generating plots.

    Args:
        gene (str): Gene name.
        protein_data (pd.DataFrame): DataFrame containing protein-only data.
        kinase_data (pd.DataFrame): DataFrame containing kinase data.
        mrna_data (pd.DataFrame): DataFrame containing mRNA data.
        time_points (list): List of time points for the experiment.
        bounds (tuple): Bounds for parameter estimation.
        bootstraps (int, optional): Number of bootstrap iterations. Defaults to 0.
        out_dir (str, optional): Output directory for saving results. Defaults to OUT_DIR.

    Returns:
        - gene: The gene being processed.
        - estimated_params: Estimated parameters for the gene.
        - model_fits: Model fits for the gene.
        - seq_model_fit: Sequential model fit for the gene.
        - errors: Error metrics (MSE, MAE).
        - final_params: Final estimated parameters.
        - param_df: DataFrame of estimated parameters.
        - gene_psite_data: Dictionary of gene-specific data.
        - psite_labels: Labels for phosphorylation sites.
        - pca_result: PCA result for the gene.
        - ev: Explained variance for PCA.
        - tsne_result: t-SNE result for the gene.
        - perturbation_analysis: Sensitivity analysis results.
        - perturbation_curves_params: Trajectories with parameters for sensitivity analysis.
        - knockout_results: Dictionary of knockout results.
        - regularization: Regularization value used in parameter estimation.

    """
    # Extract protein data
    protein_data = protein_data[protein_data['Psite'].isna() & (protein_data['GeneID'] == gene)]

    # Extract protein-group phospho data
    gene_data = kinase_data[kinase_data['Gene'] == gene]

    # Extract mRNA data
    rna_data = mrna_data[mrna_data['mRNA'] == gene]

    # Get the number of phosphorylation sites
    num_psites = gene_data.shape[0]

    # Get the residue and position values
    psite_values = gene_data['Psite'].values

    Pr_data = protein_data.iloc[:, 2:].values

    # Get the FC value for TIME_POINTS
    P_data = gene_data.iloc[:, 2:].values

    # Get the FC value for TIME_POINTS_RNA
    R_data = rna_data.iloc[:, 1:].values

    # Get initial conditions
    init_cond = initial_condition(num_psites)

    logger.info(f"[{gene}]      Fitting to data...")

    # Estimate parameters
    model_fits, estimated_params, seq_model_fit, errors, regularization_val = estimate_parameters(
        gene, Pr_data, P_data, R_data, init_cond, num_psites, time_points, bounds, bootstraps
    )

    # Error Metrics
    mse = mean_squared_error(np.concatenate((R_data.flatten(), P_data.flatten())), seq_model_fit.flatten())
    mae = mean_absolute_error(np.concatenate((R_data.flatten(), P_data.flatten())), seq_model_fit.flatten())

    logger.info("           --------------------------------")
    logger.info(f"[{gene}]      MSE: {mse:.4f} | MAE: {mae:.4f}")
    logger.info("           --------------------------------")

    # Solve Full ODE with Final Params
    final_params = estimated_params[-1]

    # Insert labels of the protein and phosphorylation sites
    gene_psite_dict_local = {'Protein': gene}
    for i, name in enumerate(get_param_names(num_psites)):
        gene_psite_dict_local[name] = [final_params[i]]

    # Solve ODE with final parameters
    sol_full, _ = solve_ode(final_params, init_cond, num_psites, time_points)

    # Generate Labels
    labels = generate_labels(num_psites)

    logger.info(f"[{gene}]      Generating plots...")

    # Generate phosphorylation ODE diagram
    illustrate(gene, num_psites)

    # Create plotting instance
    plotter = Plotter(gene, out_dir)

    # Plot PCA
    pca_result, ev = plotter.plot_pca(sol_full, components=3)

    # Plot t-SNE
    tsne_result = plotter.plot_tsne(sol_full, perplexity=5)

    # Plot parallel coordinates
    plotter.plot_parallel(sol_full, labels)

    # Plot PCA components
    plotter.pca_components(sol_full, target_variance=0.99)

    # Plot ODE model fits
    plotter.plot_model_fit(seq_model_fit, P_data, R_data.flatten(), sol_full, num_psites, psite_values, time_points)

    # Simulate wild-type
    sol_wt, p_fit_wt = solve_ode(final_params, init_cond, num_psites, time_points)

    # Generate combinations for knockouts
    knockout_combinations = generate_knockout_combinations(num_psites)
    knockout_results = {}

    # Loop through those combinations
    for knockout_setting in knockout_combinations:

        # Apply knockout settings
        final_params_ko = apply_knockout(final_params, knockout_setting, num_psites)

        # Solve ODE with knockout settings
        sol_ko, p_fit_ko = solve_ode(final_params_ko, init_cond, num_psites, time_points)

        # Create a descriptive title for the plot and report
        knockout_name = []
        if knockout_setting['transcription']:
            knockout_name.append("Transcription KO")
        if knockout_setting['translation']:
            knockout_name.append("Translation KO")
        phospho = knockout_setting['phosphorylation']
        if phospho is True:
            knockout_name.append("Phospho KO")
        elif isinstance(phospho, list) and phospho:
            knockout_name.append(f"PhosphoSite KO {','.join(psite_values[p] for p in phospho)}")
        if not knockout_name:
            knockout_name = ["WT"]

        # Save the knockout result
        knockout_results["_".join(knockout_name)] = {
            "knockout_setting": knockout_setting,
            "sol_ko": sol_ko,
            "p_fit_ko": p_fit_ko,
        }

        # Update the file names based on KO
        plotter.gene = f"{gene}_knockouts_" + "_".join(knockout_name)

        # Create the dictionary to pass
        knockout_dict = {
            'WT': (time_points, sol_wt, p_fit_wt),
            'KO': (time_points, sol_ko, p_fit_ko),
        }

        # Plot the knockout results
        plotter.plot_knockouts(knockout_dict, num_psites, psite_values)

    # Save Parameters
    df_params = pd.DataFrame(estimated_params, columns=get_param_names(num_psites))
    df_params.insert(0, "Time", time_points[:len(estimated_params)])
    df_params['Regularization'] = regularization_val
    param_path = os.path.join(out_dir, f"{gene}_parameters.xlsx")
    df_params.to_excel(param_path, index=False)

    perturbation_analysis = None
    trajectories_w_params = None

    if SENSITIVITY_ANALYSIS:
        # Perform Sensitivity Analysis
        # Perturbation of parameters around the estimated values
        perturbation_analysis, trajectories_w_params = sensitivity_analysis(P_data, R_data, final_params, time_points,
                                                                            num_psites, psite_values, labels, init_cond,
                                                                            gene)

    # Return Results
    return {
        "gene": gene,
        "labels": labels,
        "psite_labels": psite_values,
        "estimated_params": estimated_params,
        "model_fits": sol_full,
        "seq_model_fit": seq_model_fit[9:].reshape(num_psites, len(TIME_POINTS)),
        "observed_data": P_data,
        "errors": errors,
        "final_params": final_params,
        "param_df": df_params,
        "gene_psite_data": gene_psite_dict_local,
        "mse": mse,
        "mae": mae,
        "pca_result": pca_result,
        "ev": ev,
        "tsne_result": tsne_result,
        "perturbation_analysis": perturbation_analysis if SENSITIVITY_ANALYSIS else None,
        "perturbation_curves_params": trajectories_w_params if SENSITIVITY_ANALYSIS else None,
        "knockout_results": knockout_results,
        "regularization": regularization_val
    }


def process_gene_wrapper(gene, protein_data, kinase_data, mrna_data, time_points, bounds, bootstraps, out_dir=OUT_DIR):
    """
    Wrapper function to process a gene.

    Args:
        gene (str): Gene name.
        protein_data (pd.DataFrame): DataFrame containing protein-only data.
        kinase_data (pd.DataFrame): DataFrame containing kinase data.
        mrna_data (pd.DataFrame): DataFrame containing mRNA data.
        time_points (list): List of time points for the experiment.
        bounds (tuple): Bounds for parameter estimation.
        bootstraps (int, optional): Number of bootstrap iterations. Defaults to 0.
        out_dir (str, optional): Output directory for saving results. Defaults to OUT_DIR.

    Returns:
        dict: A dictionary containing the results of the gene processing.
    """
    return process_gene(
        gene=gene,
        protein_data=protein_data,
        kinase_data=kinase_data,
        mrna_data=mrna_data,
        time_points=time_points,
        bounds=bounds,
        bootstraps=bootstraps,
        out_dir=out_dir
    )
