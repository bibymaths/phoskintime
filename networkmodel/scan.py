"""Compatibility hyperparameter scan entry point for the scalar JAXopt path."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_hyperparameter_scan(args, sys, loss_data, defaults, solver_times, runner, slices, xl, xu):
    """Return existing lambda settings; legacy evolutionary scans are mapped safely."""
    logger.warning(
        "[Deprecated Config] hyperparameter_scan is accepted but legacy evolutionary scans are disabled in the JAXopt/Diffrax path.")
    return {
        "lambda_protein": getattr(args, "lambda_protein", 1.0),
        "lambda_phospho": getattr(args, "lambda_phospho", 1.0),
        "lambda_rna": getattr(args, "lambda_rna", 1.0),
        "lambda_prior": getattr(args, "lambda_prior", 0.0),
    }
