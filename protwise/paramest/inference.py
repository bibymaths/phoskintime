"""Protwise wrappers around shared JAXopt/Diffrax inference utilities."""
from __future__ import annotations

from networkmodel.BayesianInference import (
    InferenceContext,
    configure_jax_parallelism,
    generate_multistart_initials,
    run_multistart,
    run_numpyro_posterior,
    run_profile_likelihood,
)

__all__ = [
    "InferenceContext",
    "configure_jax_parallelism",
    "generate_multistart_initials",
    "run_multistart",
    "run_profile_likelihood",
    "run_numpyro_posterior",
]
