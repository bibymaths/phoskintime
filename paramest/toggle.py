import numpy as np

from paramest.normest import normest
from paramest.seqest import sequential_estimation


def estimate_parameters(mode, gene, p_data, init_cond, num_psites, time_points, bounds, fixed_params, bootstraps):
    """
    Toggle between sequential and normal (all timepoints) estimation.

    Parameters:
      - mode: a string, either "sequential" or "normal"
      - gene, P_data, init_cond, num_psites, time_points, bounds, fixed_params, bootstraps:
                as in your process_gene function.

    Returns:
      - estimated_params: List of estimated parameter vectors.
      - seq_model_fit: Array of model predictions with shape (num_psites, len(time_points)).
      - errors: Error metrics from the estimation.
    """
    if mode == "sequential":
        estimated_params, model_fits, errors = sequential_estimation(
            p_data, time_points, init_cond, bounds, fixed_params, num_psites, gene
        )
        # For sequential estimation, assemble the fitted predictions at each time point:
        seq_model_fit = np.zeros((num_psites, len(time_points)))
        for i, (_, P_fitted) in enumerate(model_fits):
            seq_model_fit[:, i] = P_fitted[:, -1]
    elif mode == "normal":
        estimated_params, model_fits, errors = normest(
            gene, p_data, init_cond, num_psites, time_points, bounds, bootstraps
        )
        # For normal estimation, model_fits[0][1] is already an array of shape (num_psites, len(time_points))
        seq_model_fit = model_fits[0][1]
    else:
        raise ValueError("Invalid estimation mode. Choose 'sequential' or 'normal'.")

    return model_fits, estimated_params, seq_model_fit, errors
