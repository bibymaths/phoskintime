import numpy as np

from paramest.normest import normest
from paramest.seqest import sequential_estimation


def estimate_parameters(mode, gene, p_data, init_cond, num_psites, time_points, bounds, fixed_params, bootstraps):
    """
    Toggle between sequential and normal (all timepoints) estimation.

    This function allows for the selection of the estimation mode
    and handles the parameter estimation process accordingly.

    It uses the sequential estimation method for "sequential" mode
    and the normal estimation method for "normal" mode.

    Args:
        - mode: The estimation mode, either "sequential" or "normal".
        - gene: The gene name.
        - p_data: Measurement data (DataFrame or numpy array).
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

    if mode == "sequential":

        # For sequential estimation, we need to set up the model function
        estimated_params, model_fits, errors = sequential_estimation(
            p_data, time_points, init_cond, bounds, fixed_params, num_psites, gene
        )

        # For sequential estimation, assemble the fitted predictions at each time point:
        seq_model_fit = np.zeros((num_psites, len(time_points)))

        # Iterate over the model fits and extract the last column (predictions)

        for i, (_, P_fitted) in enumerate(model_fits):
            # P_fitted is expected to be of shape (num_psites, len(time_points))
            seq_model_fit[:, i] = P_fitted[:, -1]

    elif mode == "normal":

        # For normal estimation, we use the provided bounds and fixed parameters
        estimated_params, model_fits, errors = normest(
            gene, p_data, init_cond, num_psites, time_points, bounds, bootstraps
        )

        # For normal estimation, model_fits[0][1] is already an array of shape (num_psites, len(time_points))
        seq_model_fit = model_fits[0][1]

    else:
        raise ValueError("Invalid estimation mode. Choose 'sequential' or 'normal'.")

    return model_fits, estimated_params, seq_model_fit, errors
