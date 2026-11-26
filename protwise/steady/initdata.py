"""Data-anchored initial conditions for the protwise models (KNOWN_ISSUES P4, owner decision: option b).

The state at the initial time t0 is taken from the data at t0, under each model's own observation operator, so the
model reproduces the t0 observations exactly:
- R(0) = 1.0: there is no RNA datum at t0 (the first RNA time is later); fold-change convention. (Choice recorded as
  pending for the owner: R(0) = 1 vs the first RNA value.)
- P(0) = protein fold change at t0.
- distmod / succmod: X_i(0) = fold change of site i at t0 (these models read site i from state i).
- randmod: the single-site subset {i} holds site i's t0 value and every multi-site subset starts at 0, so the
  site-level aggregate at t0 equals the data (allocation rule; pending owner confirmation).
A missing or non-finite t0 value keeps the former constant initial value for that state and is logged.
"""
from __future__ import annotations

import numpy as np

from config.logconf import setup_logger

logger = setup_logger()


def _t0(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    a = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
    return a[:, 0]


def data_initial_condition(gene, model, num_psites, protein_t0_series, site_series, fallback):
    """Initial state from the t0 data; ``fallback`` is the former constant initial condition (same layout)."""
    y0 = np.asarray(fallback, dtype=np.float64).copy()
    used, kept = [], []
    p0 = _t0(protein_t0_series)
    if p0 is not None and np.isfinite(p0[0]) and p0[0] > 0:
        y0[1] = float(p0[0])
        used.append("protein")
    else:
        kept.append("protein")
    s0 = _t0(site_series) if num_psites else None
    if num_psites:
        if model == "randmod":
            from config.helpers import randmod_subset_masks

            masks = randmod_subset_masks(num_psites)
            for k, mask in enumerate(masks):
                y0[2 + k] = 0.0 if bin(mask).count("1") > 1 else y0[2 + k]
            for i in range(num_psites):
                k = masks.index(1 << i)
                if s0 is not None and i < s0.size and np.isfinite(s0[i]) and s0[i] >= 0:
                    y0[2 + k] = float(s0[i])
                    used.append(f"site{i}")
                else:
                    kept.append(f"site{i}")
        else:
            for i in range(num_psites):
                if s0 is not None and i < s0.size and np.isfinite(s0[i]) and s0[i] >= 0:
                    y0[2 + i] = float(s0[i])
                    used.append(f"site{i}")
                else:
                    kept.append(f"site{i}")
    if kept:
        logger.warning("[%s] Initial condition: no usable t0 datum for %s; kept the former constant (P4).", gene, kept)
    return y0.tolist()
