
import os
import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config.constants import get_param_names, generate_labels, OUT_DIR, ESTIMATION_MODE
from models.diagram import illustrate
from paramest.toggle import estimate_parameters
from paramest.adapest import estimate_profiles
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
    measurement_data,
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
    :param measurement_data:
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
    # 1. Extract Gene-specific Data
    gene_data = measurement_data[measurement_data['Gene'] == gene]
    num_psites = gene_data.shape[0]
    psite_values = gene_data['Psite'].values
    init_cond = initial_condition(num_psites)
    P_data = gene_data.iloc[:, 2:].values

    # 2. Choose estimation mode
    estimation_mode = ESTIMATION_MODE

    model_fits, estimated_params, seq_model_fit, errors = estimate_parameters(
        estimation_mode, gene, P_data, init_cond, num_psites, time_points, bounds, fixed_params, bootstraps
    )

    # 7. Error Metrics
    mse = mean_squared_error(P_data.flatten(), seq_model_fit.flatten())
    mae = mean_absolute_error(P_data.flatten(), seq_model_fit.flatten())
    logger.info(f"{gene} → MSE: {mse:.4f}, MAE: {mae:.4f}")

    # 3. Adaptive Profile Estimation (Optional)
    profiles_df, profiles_dict = None, None
    if desired_times is not None and time_fixed is not None:
        profiles_df, profiles_dict = estimate_profiles(
            gene, measurement_data, init_cond, num_psites,
            time_points, desired_times, bounds, fixed_params,
            bootstraps, time_fixed
        )
        # Save profile Excel
        profile_path = os.path.join(out_dir, f"{gene}_profiles.xlsx")
        profiles_df.to_excel(profile_path, index=False)
        # logger.info(f"Profiled Estimates: {profile_path}")

    # 4. Solve Full ODE with Final Params
    final_params = estimated_params[-1]
    gene_psite_dict_local = {'Protein': gene}
    for i, name in enumerate(get_param_names(num_psites)):
        gene_psite_dict_local[name] = [final_params[i]]

    sol_full, _ = solve_ode(final_params, init_cond, num_psites, time_points)

    # 5. Plotting Outputs
    labels = generate_labels(num_psites)
    illustrate(gene, num_psites)
    # Create a single Plotter instance.
    plotter = Plotter(gene, out_dir)
    # Call plot_all with all necessary data.
    plotter.plot_all(solution=sol_full, labels=labels,
                     estimated_params=estimated_params, time_points=time_points,
                     P_data=P_data, seq_model_fit=seq_model_fit,
                     psite_labels=psite_values, perplexity=5, components=3, target_variance=0.99)

    # 6. Save Sequential Parameters to Excel
    df_params = pd.DataFrame(estimated_params, columns=get_param_names(num_psites))
    df_params.insert(0, "Time", time_points[:len(estimated_params)])
    param_path = os.path.join(out_dir, f"{gene}_parameters.xlsx")
    df_params.to_excel(param_path, index=False)
    # logger.info(f"Estimated Parameters: {param_path}")

    # 8. Return Results
    return {
        "gene": gene,
        "estimated_params": estimated_params,
        "model_fits": model_fits,
        "seq_model_fit": seq_model_fit,
        "errors": errors,
        "final_params": final_params,
        "profiles": profiles_dict,
        "profiles_df": profiles_df,
        "param_df": df_params,
        "gene_psite_data": gene_psite_dict_local,
        "mse": mse,
        "mae": mae
    }

def process_gene_wrapper(gene, measurement_data, time_points, bounds, fixed_params,
                         desired_times, time_fixed, bootstraps, out_dir=OUT_DIR):
    """
    Wrapper function to process a gene. This function is a placeholder for
    any additional processing or modifications needed before calling the
    main processing function.

    :param gene:
    :param measurement_data:
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
        measurement_data=measurement_data,
        time_points=time_points,
        bounds=bounds,
        fixed_params=fixed_params,
        desired_times=desired_times,
        time_fixed=time_fixed,
        bootstraps=bootstraps,
        out_dir=out_dir
    )