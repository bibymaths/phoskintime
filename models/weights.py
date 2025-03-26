
import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d

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

def get_weight_options(target, t_target, num_psites, use_regularization, reg_len, early_emphasis):
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
        "inverse_data": 1 / np.maximum(np.abs(target), 1e-5),
        "exp_decay": np.exp(-0.5 * target),
        "log_scale": 1 / np.maximum(log_scale, 1e-5),
        "time_diff": 1 / np.maximum(np.abs(np.diff(target, prepend=target[0])), 1e-5),
        "moving_avg": 1 / np.maximum(np.abs(target - uniform_filter1d(target, 3)), 1e-5),

        "sigmoid_time_decay": early_sigmoid,
        "exponential_early_emphasis": np.exp(-0.5 * time_indices),
        "polynomial_decay": 1 / (1 + 0.5 * time_indices),

        "ms_snr_model": signal_noise_model,
        "ms_inverse_variance": inverse_variance_model,
        "flat_region_penalty": flat_region_penalty,
        "steady_state_decay": steady_state_decay,

        "combined_data_time": 1 / (np.maximum(np.abs(target), 1e-5) * (1 + 0.5 * time_indices)),
        "inverse_sqrt_data": 1 / sqrt_signal,

        "early_emphasis_moderate": np.ones_like(target),
        "early_emphasis_steep_decay": (
            np.tile(np.concatenate([
                np.full(8, 0.05), np.full(2, 0.2), np.ones(max(0, len(t_target) * num_psites - 10))]), 1)
            if len(t_target) * num_psites >= 10 else np.ones(len(target))
        ),
        "custom_early_points_emphasis": early_emphasis
    }

    if use_regularization:
        base_weights = {
            k: np.concatenate([v, np.ones(reg_len)])
            for k, v in base_weights.items()
        }

    return base_weights
