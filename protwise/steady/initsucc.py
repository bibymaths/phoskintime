import numpy as np

from config.logconf import setup_logger

logger = setup_logger()


def initial_condition(num_psites: int) -> list:
    """Analytical positive steady-state initial condition for successive model."""
    if num_psites < 0:
        raise ValueError("num_psites must be non-negative")
    R = 1.0
    P = 1.0 / (1.0 + 0.5 * float(num_psites))
    P_sites = np.full(num_psites, 0.5 * P, dtype=float)
    return [R, P] + P_sites.tolist()
