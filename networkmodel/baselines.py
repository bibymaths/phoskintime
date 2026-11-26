"""Baseline (fold-change reference) times and the one fold-change function shared by the fit, the export and the
posterior/profile rebuild (KNOWN_ISSUES N7).

Baseline times are defined once: t = 0 (the first protein / phospho time) for protein and phospho, and the first RNA
time for RNA. Data are scaled to their own value at these times when scaling_method = "fc_start"
(networkmodel.utils.process_and_scale_raw_data); model trajectories are divided by their own value at the same times.
"""
from __future__ import annotations

import numpy as np

from networkmodel.config import TIME_POINTS_PHOSPHO, TIME_POINTS_PROTEIN, TIME_POINTS_RNA
from networkmodel.utils import _base_idx

BASELINE_TIMES = {
    "protein": float(np.asarray(TIME_POINTS_PROTEIN, dtype=float)[0]),
    "phospho": float(np.asarray(TIME_POINTS_PHOSPHO, dtype=float)[0]),
    "rna": float(np.asarray(TIME_POINTS_RNA, dtype=float)[0]),
}

# Model fold changes use this floor on numerator and denominator (unchanged from the former inline code).
FC_FLOOR = 1e-12
# Data baselines with |value| below this (or not finite) are not divided by: the series is skipped and counted.
DATA_BASELINE_NEAR_ZERO = 1e-12


def baseline_index(times, layer: str):
    """Index of the baseline time of ``layer`` ("protein", "phospho", "rna") in ``times``."""
    return _base_idx(np.asarray(times, dtype=float), BASELINE_TIMES[layer])


def model_fold_change(values, base, xp=np):
    """Model fold change used by the loss (xp = jax.numpy) and the export (xp = numpy)."""
    return xp.maximum(values, FC_FLOOR) / xp.maximum(base, FC_FLOOR)
