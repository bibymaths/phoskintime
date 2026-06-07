"""Expose a compatibility hyperparameter-scan entry point for the scalar JAXopt path; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules."""
from __future__ import annotations

from config.config import setup_logger
from networkmodel.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def run_hyperparameter_scan(args, sys, loss_data, defaults, solver_times, runner, slices, xl, xu):
    """Run the scalar compatibility hyperparameter scan
    
    Args:
        args: Positional arguments forwarded to the runner.
        sys: Input value used by this routine.
        loss_data: Input value used by this routine.
        defaults: Input value used by this routine.
        solver_times: Input value used by this routine.
        runner: Input value used by this routine.
        slices: Input value used by this routine.
        xl: Input value used by this routine.
        xu: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    logger.warning(
        "[Deprecated Config] hyperparameter_scan is accepted but legacy evolutionary scans are disabled in the JAXopt/Diffrax path.")
    return {
        "lambda_protein": getattr(args, "lambda_protein", 1.0),
        "lambda_phospho": getattr(args, "lambda_phospho", 1.0),
        "lambda_rna": getattr(args, "lambda_rna", 1.0),
        "lambda_prior": getattr(args, "lambda_prior", 0.0),
    }
