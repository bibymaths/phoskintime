from paramest.normest import normest


def estimate_parameters(gene, p_data, r_data, init_cond, num_psites, time_points, bounds, bootstraps):
    """

    This function allows for the selection of the estimation mode
    and handles the parameter estimation process accordingly.

    It uses the sequential estimation method for "sequential" mode (deprecated)
    and the normal estimation method for "normal" mode.

    Args:
        - gene: The gene name.
        - p_data: phosphorylation data (DataFrame or numpy array).
        - r_data: mRNA data (DataFrame or numpy array).
        - init_cond: Initial condition for the ODE solver.
        - num_psites: Number of phosphorylation sites.
        - time_points: Array of time points to use.
        - bounds: Dictionary of parameter bounds.
        - fixed_params: Dictionary of fixed parameters.
        - bootstraps: Number of bootstrapping iterations (only used in normal mode).
    :returns:
        - model_fits: List with the ODE solution and model predictions.
        - estimated_params: List with the full estimated parameter vector.
        - seq_model_fit: Sequential model fit for the gene.
        - errors: Error metrics (MSE, MAE).
    """

    # For normal estimation, we use the provided bounds and fixed parameters
    estimated_params, model_fits, errors = normest(
        gene, p_data, r_data, init_cond, num_psites, time_points, bounds, bootstraps
    )

    # For normal estimation, model_fits[0][1] is already an array of shape (num_psites, len(time_points))
    seq_model_fit = model_fits[0][1]

    return model_fits, estimated_params, seq_model_fit, errors
