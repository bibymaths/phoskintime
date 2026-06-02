"""Diffrax-backed protwise model solver helpers."""
from __future__ import annotations

import numpy as np
from networkmodel.jax_backend import solve_diffrax, DiffraxSolverConfig
from config.constants import NORMALIZE_MODEL_OUTPUT


def solve_protwise_ode(params, init_cond, num_psites, t):
    sol = np.asarray(solve_diffrax(np.asarray(init_cond, dtype=np.float64), np.asarray(t, dtype=np.float64), params=np.asarray(params, dtype=np.float64), config=DiffraxSolverConfig()), dtype=np.float64)
    sol = np.clip(sol, 0, None)
    if NORMALIZE_MODEL_OUTPUT:
        norm_init = np.asarray(init_cond, dtype=sol.dtype)
        norm_init = np.where(norm_init == 0, 1.0, norm_init)
        sol = sol / norm_init[np.newaxis, :]
    r_fitted = sol[5:, 0].T if sol.shape[0] > 5 else sol[:, 0].T
    pr_fitted = sol[:, 1].T if sol.shape[1] > 1 else np.asarray([], dtype=sol.dtype)
    p_fitted = sol[:, 2:2 + num_psites].T if num_psites else np.asarray([], dtype=sol.dtype)
    return sol, np.concatenate((np.ravel(r_fitted), np.ravel(pr_fitted), np.ravel(p_fitted)))
