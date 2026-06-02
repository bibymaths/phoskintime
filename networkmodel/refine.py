"""Scalar-objective refinement compatibility helpers."""
from __future__ import annotations

import numpy as np
from config.config import setup_logger
from networkmodel.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def get_refined_bounds(X, current_xl, current_xu, idx=None, padding=0.2):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    p_min, p_max = np.min(X, axis=0), np.max(X, axis=0)
    span = np.maximum(p_max - p_min, 1e-2)
    return np.maximum(current_xl, p_min - padding * span), np.minimum(current_xu, p_max + padding * span)


def run_iterative_refinement(problem, res, args, idx=None, max_passes=1, padding=0.25):
    logger.warning(
        "[Deprecated Config] refine is accepted but scalar JAXopt mode already performs deterministic local optimization; returning current result.")
    return res
