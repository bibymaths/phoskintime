from paramest.normest import normest


def estimate_parameters(gene, pr_data, p_data, r_data, init_cond, num_psites, time_points, bounds, bootstraps):
    """

    This function allows for the selection of the estimation mode
    and handles the parameter estimation process accordingly.

    Args:
        gene (str): Gene name.
        pr_data (array): Array of protein data.
        p_data (array): Array of protein-phospho data.
        r_data (array): Array of RNA data.
        init_cond (array): Initial conditions for the model.
        num_psites (int): Number of phosphorylation sites.
        time_points (array): Time points for the data.
        bounds (tuple): Bounds for the parameter estimation.
        bootstraps (int): Number of bootstrap samples.

    Returns:
        model_fits (list): List of model fits.
        estimated_params (array): Estimated parameters.
        seq_model_fit (array): Sequence model fit.
        errors (array): Errors in the estimation.
        reg_term (float): Regularization term.
    """

    # For normal estimation, we use the provided bounds and fixed parameters
    estimated_params, model_fits, errors, reg_term = normest(
        gene, pr_data, p_data, r_data, init_cond, num_psites, time_points, bounds, bootstraps
    )

    # For normal estimation, model_fits[0][1] is already an array of shape (num_psites, len(time_points))
    seq_model_fit = model_fits[0][1]

    return model_fits, estimated_params, seq_model_fit, errors, reg_term
