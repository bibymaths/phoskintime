import os
import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from knockout import apply_knockout, generate_knockout_combinations
from config.constants import get_param_names, generate_labels, OUT_DIR, SENSITIVITY_ANALYSIS
from models.diagram import illustrate
from paramest.toggle import estimate_parameters
from sensitivity import sensitivity_analysis
from models import solve_ode
from steady import initial_condition
from plotting import Plotter
from config.logconf import setup_logger

logger = setup_logger()


# ----------------------------------
# Early-Weighted Scheme
# ----------------------------------
@njit
def early_emphasis(P_data, time_points, num_psites):
    if P_data.ndim == 1:
        P_data = P_data.reshape(1, P_data.size)

    n_times = len(time_points)
    custom_weights = np.ones((num_psites, n_times))
    time_diffs = np.empty(n_times)
    time_diffs[0] = 0.0
    for j in range(1, n_times):
        time_diffs[j] = time_points[j] - time_points[j - 1]

    for i in range(num_psites):
        limit = min(5, n_times)
        for j in range(1, limit):
            data_based_weight = 1.0 / (abs(P_data[i, j]) + 1e-5)
            time_based_weight = 1.0 / (time_diffs[j] + 1e-5)
            custom_weights[i, j] = data_based_weight * time_based_weight
        for j in range(5, n_times):
            custom_weights[i, j] = 1.0

    return custom_weights.ravel()


def process_gene(
        gene,
        kinase_data,
        mrna_data,
        time_points,
        bounds,
        fixed_params,
        desired_times=None,
        time_fixed=None,
        bootstraps=0,
        out_dir=OUT_DIR
):
    """
    Process a single gene by estimating its parameters and generating plots.
    This function extracts gene-specific data, estimates parameters using the
    specified estimation mode, and generates plots for the estimated parameters
    and the model fits. It also calculates error metrics and saves the results
    to Excel files.

    :param gene:
    :param kinase_data:
    :param mrna_data:
    :param time_points:
    :param bounds:
    :param fixed_params:
    :param desired_times:
    :param time_fixed:
    :param bootstraps:
    :param out_dir:
    :return:
        - gene: The gene being processed.
        - estimated_params: Estimated parameters for the gene.
        - model_fits: Model fits for the gene.
        - seq_model_fit: Sequential model fit for the gene.
        - errors: Error metrics (MSE, MAE).
        - final_params: Final estimated parameters.
        - profiles: Adaptive profile estimates (if applicable).
        - profiles_df: DataFrame of adaptive profile estimates (if applicable).
        - param_df: DataFrame of estimated parameters.
        - gene_psite_data: Dictionary of gene-specific data.
    """
    # Extract protein-group data
    gene_data = kinase_data[kinase_data['Gene'] == gene]

    # Extract mRNA data
    rna_data = mrna_data[mrna_data['mRNA'] == gene]

    # Get the number of phosphorylation sites
    num_psites = gene_data.shape[0]

    # Get the residue and position values
    psite_values = gene_data['Psite'].values

    # Get initial conditions
    init_cond = initial_condition(num_psites)

    # Get the FC value for TIME_POINTS
    P_data = gene_data.iloc[:, 2:].values

    # Get the FC value for TIME_POINTS_RNA
    R_data = rna_data.iloc[:, 1:].values

    logger.info(f"[{gene}]      Fitting to data...")

    # Estimate parameters
    model_fits, estimated_params, seq_model_fit, errors = estimate_parameters(
        gene, P_data, R_data, init_cond, num_psites, time_points, bounds, bootstraps
    )

    # Error Metrics
    mse = mean_squared_error(np.concatenate((R_data.flatten(), P_data.flatten())), seq_model_fit.flatten())
    mae = mean_absolute_error(np.concatenate((R_data.flatten(), P_data.flatten())), seq_model_fit.flatten())

    logger.info(f"[{gene}]      MSE: {mse:.4f} | MAE: {mae:.4f}")

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

        # Update the file names dynamically
        plotter.gene = f"{gene}_" + "_".join(knockout_name)

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
    param_path = os.path.join(out_dir, f"{gene}_parameters.xlsx")
    df_params.to_excel(param_path, index=False)

    perturbation_analysis = None
    trajectories_w_params = None
    if SENSITIVITY_ANALYSIS:
        # Perform Sensitivity Analysis
        # Perturbation of parameters around the estimated values
        perturbation_analysis, trajectories_w_params = sensitivity_analysis(P_data, R_data, final_params, time_points,
                                                                            num_psites, psite_values, init_cond, gene)

    # Return Results
    return {
        "gene": gene,
        "labels": labels,
        "psite_labels": psite_values,
        "estimated_params": estimated_params,
        "model_fits": sol_full,
        "seq_model_fit": seq_model_fit[9:].reshape(num_psites, 14),
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
        "knockout_results": knockout_results
    }


def process_gene_wrapper(gene, kinase_data, mrna_data, time_points, bounds, fixed_params,
                         desired_times, time_fixed, bootstraps, out_dir=OUT_DIR):
    """
    Wrapper function to process a gene. This function is a placeholder for
    any additional processing or modifications needed before calling the
    main processing function.

    :param gene:
    :param kinase_data:
    :param mrna_data:
    :param time_points:
    :param bounds:
    :param fixed_params:
    :param desired_times:
    :param time_fixed:
    :param bootstraps:
    :param out_dir:
    :return:
        - result: Dictionary containing the results of the gene processing.
    """
    return process_gene(
        gene=gene,
        kinase_data=kinase_data,
        mrna_data=mrna_data,
        time_points=time_points,
        bounds=bounds,
        fixed_params=fixed_params,
        desired_times=desired_times,
        time_fixed=time_fixed,
        bootstraps=bootstraps,
        out_dir=out_dir
    )
