import numpy as np
from itertools import combinations

from config.logconf import setup_logger

logger = setup_logger()


def initial_condition(num_psites: int) -> list:
    """Positive normalized initial condition for random phosphorylation states."""
    if num_psites < 0:
        raise ValueError("num_psites must be non-negative")
    subsets = [comb for k in range(1, num_psites + 1) for comb in combinations(range(1, num_psites + 1), k)]
    R = 1.0
    P = 1.0 / max(1, len(subsets) + 1)
    phos = np.full(len(subsets), P, dtype=float)
    return [R, P] + phos.tolist()
