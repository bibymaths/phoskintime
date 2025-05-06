import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from config.constants import USE_CUSTOM_WEIGHTS

current_dir = Path(__file__).resolve().parent


@njit
def early_emphasis(p_data, time_points, num_psites):
    """
    Function that calculates custom weights for early time points in a dataset.

    Args:
        p_data: 2D numpy array of shape (num_psites, n_times)
        time_points: 1D numpy array of time points
        num_psites: Number of phosphorylation sites

    Returns:
        custom_weights: 1D numpy array of weights for early time points
    """
    if p_data.ndim == 1:
        p_data = p_data.reshape(1, p_data.size)

    n_times = len(time_points)
    custom_weights = np.ones((num_psites, n_times))

    time_diffs = np.empty(n_times)
    time_diffs[0] = 0.0

    # Calculate time differences
    for j in range(1, n_times):
        # Subtract the previous time point from the current one
        time_diffs[j] = time_points[j] - time_points[j - 1]

    for i in range(num_psites):
        # Emphasize early time points - first five
        limit = min(8, n_times)
        # Compute weights for early time points
        for j in range(1, limit):
            # Calculate the data-based and time-based weights
            data_based_weight = 1.0 / (abs(p_data[i, j]) + 1e-5)
            time_based_weight = 1.0 / (time_diffs[j] + 1e-5)
            # Combine the weights
            custom_weights[i, j] = data_based_weight * time_based_weight
        for j in range(5, n_times):
            # For later time points, use a fixed weight
            custom_weights[i, j] = 1.0

    return custom_weights.ravel()


def get_protein_weights(
        gene,
        input1_path=Path(__file__).resolve().parent.parent / 'processing' / 'input1_wstd.csv',
        input2_path=Path(__file__).resolve().parent.parent / 'kinopt' / 'data' / 'input2.csv'
):
    """
    Function to extract weights for a specific gene from the input files.

    Args:
        gene (str): Gene ID to filter the weights.
        input1_path (Path): Path to the input1_wstd.csv file.
        input2_path (Path): Path to the input2.csv file.

    Returns:
        weights (numpy.ndarray): Extracted weights for the specified gene.
    """

    # Load input1_wstd and input2
    input1 = pd.read_csv(input1_path)
    input2 = pd.read_csv(input2_path)

    # Clean column names
    input1.columns = input1.columns.str.strip()
    input2.columns = input2.columns.str.strip()

    # Filter input2 for the given GeneID
    input2_gene = input2[input2['GeneID'] == gene]

    if input2_gene.empty:
        raise ValueError(f"No entries for GeneID {gene} found in input2.csv")

    # Merge to get corresponding x1_std to x14_std values
    merged = pd.merge(
        input2_gene, input1,
        on=['GeneID', 'Psite'],
        how='left'
    )

    if merged.isnull().any().any():
        missing = merged[merged.isnull().any(axis=1)][['GeneID', 'Psite']]
        raise ValueError(f"Missing (GeneID, Psite) pairs for {gene} in input1_wstd.csv:\n{missing}")

    # Extract weights
    std_columns = [f'x{i}_std' for i in range(1, 15)]
    weights = merged[std_columns].to_numpy().flatten()

    return weights


def full_weight(p_data_weight, use_regularization, reg_len):
    """
    Function to create a full weight array for parameter estimation.

    Args:
        p_data_weight (numpy.ndarray): The weight data to be processed.
        use_regularization (bool): Flag to indicate if regularization is used.
        reg_len (int): Length of the regularization term.

    Returns:
        numpy.ndarray: The full weight array.
    """
    base = np.concatenate([np.ones(9), p_data_weight])
    if use_regularization:
        base = np.concatenate([base, np.ones(reg_len)])
    return base


def get_weight_options(target, t_target, num_psites, use_regularization, reg_len, early_weights, ms_gauss_weights):
    """
    Function to calculate weights for parameter estimation based on the target data and time points.

    Args:
        target (numpy.ndarray): The target data for which weights are calculated.
        t_target (numpy.ndarray): The time points corresponding to the target data.
        num_psites (int): Number of phosphorylation sites.
        use_regularization (bool): Flag to indicate if regularization is used.
        reg_len (int): Length of the regularization term.
        early_weights (numpy.ndarray): Weights for early time points.
        ms_gauss_weights (numpy.ndarray): Weights based on Gaussian distribution.

    Returns:
        dict: A dictionary containing different weight options.
    """
    time_indices = np.tile(np.arange(1, len(t_target) + 1), num_psites)

    log_scale = np.log1p(np.abs(target))
    sqrt_signal = np.sqrt(np.maximum(np.abs(target), 1e-5))

    # fallback for flat region penalty
    if len(target) >= 2:
        grad = np.gradient(target)
        flat_region_penalty = 1 / np.maximum(np.abs(grad), 1e-5)
    else:
        # Early time points—biologically inspired
        flat_region_penalty = 1 / np.maximum(np.abs(target), 1e-5)  # inverse signal intensity

    base_weights = {
        "inverse": full_weight(1 / np.maximum(np.abs(target[9:]), 1e-5), use_regularization, reg_len),
        "exponential_decay": full_weight(np.exp(-0.5 * target[9:]), use_regularization, reg_len),
        "inverse_log_scale": full_weight(1 / np.maximum(log_scale[9:], 1e-5), use_regularization, reg_len),
        "inverse_time_diff": full_weight(1 / np.maximum(np.abs(np.diff(target[9:], prepend=target[9])), 1e-5),
                                         use_regularization, reg_len),
        "inverse_moving_avg": full_weight(1 / np.maximum(np.abs(target[9:] - uniform_filter1d(target[9:], 3)), 1e-5),
                                          use_regularization, reg_len),

        "sigmoid_decay": full_weight(1 / (1 + np.exp((time_indices - 5))), use_regularization, reg_len),
        "exponential_early_decay": full_weight(np.exp(-0.5 * time_indices), use_regularization, reg_len),
        "polynomial_time_decay": full_weight(1 / (1 + 0.5 * time_indices), use_regularization, reg_len),

        "signal_noise": full_weight(1 / sqrt_signal[9:], use_regularization, reg_len),
        "inverse_variance": full_weight(1 / (np.maximum(np.abs(target[9:]), 1e-5) ** 0.7), use_regularization, reg_len),
        "flat_penalty": full_weight(flat_region_penalty[9:], use_regularization, reg_len) if flat_region_penalty.shape[
                                                                                                 0] == target.shape[
                                                                                                 0] else flat_region_penalty,

        "steady_decay": full_weight(np.exp(-0.1 * time_indices), use_regularization, reg_len),
        "inverse_square_root_data": full_weight(1 / sqrt_signal[9:], use_regularization, reg_len),

        "early_moderate_decay": full_weight(
            np.linspace(1.0, 0.3, len(time_indices)),
            use_regularization,
            reg_len
        ),

        "early_steep_decay": full_weight(
            np.concatenate([
                np.full(min(8, len(time_indices)), 0.05),
                np.full(min(2, max(len(time_indices) - 8, 0)), 0.2),
                np.ones(max(len(time_indices) - 10, 0))
            ]),
            use_regularization,
            reg_len
        ),

        "early_emphasis": full_weight(early_weights, use_regularization, reg_len),
        "uncertainties_from_data": full_weight(ms_gauss_weights, use_regularization, reg_len),
    }

    if not USE_CUSTOM_WEIGHTS:
        base_weights = {"uncertainties_from_data": base_weights["uncertainties_from_data"]}

    return base_weights
