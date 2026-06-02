import numpy as np
from config.helpers import generate_randmod_subsets

from config.logconf import setup_logger

logger = setup_logger()


def initial_condition(num_psites: int) -> list:
    """Positive normalized initial condition for random phosphorylation states."""
    if num_psites < 0:
        raise ValueError("num_psites must be non-negative")
    subsets = generate_randmod_subsets(num_psites)
    R = 1.0
    P = 1.0 / max(1, len(subsets) + 1)
    phos = np.full(len(subsets), P, dtype=float)
    return [R, P] + phos.tolist()
