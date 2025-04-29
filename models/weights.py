
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
    The weights are based on the data values and the time differences between points.

    The weights are calculated in a way that emphasizes early time points,
    while also considering the data values and time intervals.

    :param p_data:
    :param time_points:
    :param num_psites:
    :return: flattened array of custom weights
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

    return np.concatenate(np.ones(9), custom_weights.ravel())


def get_protein_weights(
    gene,
    input1_path=Path(__file__).resolve().parent.parent / 'processing' / 'input1_wstd.csv',
    input2_path=Path(__file__).resolve().parent.parent / 'kinopt' / 'data' / 'input2.csv'
):
    """
    Extracts x1_std to x14_std weights for a single GeneID.

    Args:
        gene: GeneID (str) to process
        input1_path: Path to input1_wstd.csv
        input2_path: Path to input2.csv

    Returns:
        Flattened numpy array of weights for the specific GeneID
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

def get_weight_options(target, t_target, num_psites, use_regularization, reg_len, early_weights, ms_gauss_weights):
    """
    Function to calculate weights for parameter estimation based on the target data and time points.
    The weights are designed to emphasize early time points and account for noise in the data.
    The function also includes options for regularization and custom early point emphasis.

    The following are the weighting schemes:
    - Inverse data: 1 / abs(target)
    - Exponential decay: exp(-0.5 * target)
    - Log scale: 1 / log(1 + abs(target))
    - Time difference: 1 / abs(time_diff)
    - Moving average: 1 / abs(target - moving_avg)
    - Sigmoid time decay: 1 / (1 + exp(time_indices - 5))
    - Exponential early emphasis: exp(-0.5 * time_indices)
    - Polynomial decay: 1 / (1 + 0.5 * time_indices)
    - MS SNR model: 1 / sqrt(signal)
    - MS inverse variance: 1 / (abs(target) ** 0.7)
    - Flat region penalty: 1 / abs(grad)
    - Steady state decay: exp(-0.1 * time_indices)
    - Combined data time: 1 / (abs(target) * (1 + 0.5 * time_indices))
    - Inverse sqrt data: 1 / sqrt(abs(target))
    - Early emphasis moderate: ones
    - Early emphasis steep decay: ones
    - Custom early points emphasis: early_weights

    :param target:
    :param t_target:
    :param num_psites:
    :param use_regularization:
    :param reg_len:
    :param early_weights:
    :return: dictionary of weights for parameter estimation
    """
    time_indices = np.tile(np.arange(1, len(t_target) + 1), num_psites)
    log_scale = np.log1p(np.abs(target))
    sqrt_signal = np.sqrt(np.maximum(np.abs(target), 1e-5))

    # Noise-aware weights
    signal_noise_model = 1 / sqrt_signal  # MS-like error model
    inverse_variance_model = 1 / (np.maximum(np.abs(target), 1e-5) ** 0.7)  # empirical fit

    # Biological modeling
    early_sigmoid = 1 / (1 + np.exp((time_indices - 5)))
    steady_state_decay = np.exp(-0.1 * time_indices)  # Less weight to late points
    # fallback for flat region penalty
    if len(target) >= 2:
        grad = np.gradient(target)
        flat_region_penalty = 1 / np.maximum(np.abs(grad), 1e-5)
    else:
        # Early time points—biologically inspired
        flat_region_penalty = 1 / np.maximum(np.abs(target), 1e-5)  # inverse signal intensity

    base_weights = {
        "inverse": 1 / np.maximum(np.abs(target), 1e-5),
        "exponential_decay": np.exp(-0.5 * target),
        "inverse_log_scale": 1 / np.maximum(log_scale, 1e-5),
        "inverse_time_diff": 1 / np.maximum(np.abs(np.diff(target, prepend=target[0])), 1e-5),
        "inverse_moving_avg": 1 / np.maximum(np.abs(target - uniform_filter1d(target, 3)), 1e-5),

        "sigmoid_decay": early_sigmoid,
        "exponential_early_decay": np.exp(-0.5 * time_indices),
        "polynomial_time_decay": 1 / (1 + 0.5 * time_indices),

        "signal_noise": signal_noise_model,
        "inverse_variance": inverse_variance_model,
        "flat_penalty": flat_region_penalty,
        "steady_decay": steady_state_decay,

        "combined_data_time": 1 / (np.maximum(np.abs(target), 1e-5) * (1 + 0.5 * time_indices)),
        "inverse_square_root_data": 1 / sqrt_signal,

        "early_moderate_decay": np.ones_like(target),
        "early_steep_decay": (
            np.tile(np.concatenate([
                np.full(8, 0.05), np.full(2, 0.2), np.ones(max(0, len(t_target) * num_psites - 10))]), 1)
            if len(t_target) * num_psites >= 10 else np.ones(len(target))
        ),
        "early_emphasis": early_weights,
        "uncertainties_from_data": ms_gauss_weights,
    }

    for key in base_weights:
        base_weights[key] = np.concatenate([np.ones(9), base_weights[key]])

    if not USE_CUSTOM_WEIGHTS:
        base_weights = {"uncertainties_from_data": base_weights["uncertainties_from_data"]}

    if use_regularization:
        base_weights = {
            k: np.concatenate([v, np.ones(reg_len)])
            for k, v in base_weights.items()
        }

    return base_weights
