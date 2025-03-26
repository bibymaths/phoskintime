# weights.py
import numpy as np
from numba import njit

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

def get_weight_options(target, residuals, t_target, num_psites, use_regularization, reg_len, early_emphasis):
    base_weights = {
        "inverse_data": 1 / np.maximum(np.abs(target), 1e-5),
        "exp_decay": np.exp(-0.5 * target),
        "log_scale": 1 / np.maximum(np.log1p(np.abs(target)), 1e-5),
        "time_diff": 1 / np.maximum(np.abs(np.diff(target, prepend=target[0])), 1e-5),
        "early_emphasis_moderate": np.ones_like(target),
        "early_emphasis_steep_decay": (
            np.tile(np.concatenate([
                np.full(8, 0.05), np.full(2, 0.2), np.ones(max(0, len(t_target) * num_psites - 10))]), 1)
            if len(t_target) * num_psites >= 10 else np.ones(len(target))
        ),
        "exponential_early_emphasis": np.exp(-0.5 * np.tile(np.arange(1, len(t_target) + 1), num_psites)),
        "polynomial_decay": 1 / (1 + 0.5 * np.tile(np.arange(1, len(t_target) + 1), num_psites)),
        "custom_early_points_emphasis": early_emphasis
    }

    if use_regularization:
        base_weights = {k: np.concatenate([v, np.ones(reg_len)]) for k, v in base_weights.items()}

    return base_weights
