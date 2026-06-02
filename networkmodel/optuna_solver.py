"""Compatibility entry point for legacy optimizer configuration.

The active PhosKinTime networkmodel execution path uses the scalar JAXopt/Diffrax
implementation in :mod:`networkmodel.optproblem`. This module is retained so old
imports fail with an actionable migration message instead of an optional dependency
import error.
"""
from __future__ import annotations


def run_optuna_solver(*args, **kwargs):
    raise RuntimeError(
        "The legacy alternative optimizer path has been retired for PhosKinTime. "
        "Use networkmodel.optproblem.GlobalODEScalarObjective through networkmodel.runner; "
        "old optimizer config values are accepted there and mapped to JAXopt."
    )
