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


# --- Round 2, item 13: initial-condition policy (measured t0 → steady state or estimate) --------------------------

def measured_t0_vector(model, num_psites, protein_t0_series, site_series):
    """t0 values per state (NaN where not measured). R has no t0 datum (the first RNA time is later)."""
    n = 2 + ((1 << int(num_psites)) - 1 if model == "randmod" else int(num_psites))
    m = np.full(n, np.nan)
    p0 = _t0(protein_t0_series)
    if p0 is not None and np.isfinite(p0[0]) and p0[0] > 0:
        m[1] = float(p0[0])
    s0 = _t0(site_series) if num_psites else None
    if num_psites and s0 is not None:
        if model == "randmod":
            # Independence rule (Round 2 item 14): the protein t0 value is taken as the total, occupancies are the
            # site t0 values divided by it (projected into [0, 1], logged); see randmod_independent_states.
            if np.all(np.isfinite(s0[:num_psites])) and np.isfinite(m[1]):
                total = m[1]
                occ = s0[:num_psites] / total
                occ_p = np.clip(occ, 0.0, 1.0)
                if not np.allclose(occ_p, occ):
                    logger.warning("[IC] randmod occupancies %s outside [0, 1]; projected to %s (Round 2 item 14).",
                                   np.round(occ, 6).tolist(), np.round(occ_p, 6).tolist())
                m[1], m[2:] = randmod_independent_states(total, occ_p)
        else:
            for i in range(num_psites):
                if i < s0.size and np.isfinite(s0[i]) and s0[i] >= 0:
                    m[2 + i] = float(s0[i])
    return m


def randmod_independent_states(total, occ):
    """randmod t0 states by the INDEPENDENCE rule (Round 2 item 14): sites are phosphorylated independently, so the
    mass of subset m is total · Π_{j∈m} occ_j · Π_{j∉m} (1 − occ_j). Returns (unphosphorylated mass, subset masses in
    randmod_subset_masks order). States sum to ``total``; the marginal of site j is total · occ_j."""
    from config.helpers import randmod_subset_masks

    occ = np.asarray(occ, dtype=np.float64)
    n = occ.size
    masks = randmod_subset_masks(n)

    def mass(m):
        return float(total) * float(np.prod([occ[j] if m >> j & 1 else 1.0 - occ[j] for j in range(n)]))

    return mass(0), np.asarray([mass(m) for m in masks], dtype=np.float64)


class InitialCondition:
    """Callable initial state y0(params[, ic_params]) for a protwise gene under ``policy`` (Round 2, item 13)."""

    def __init__(self, gene, model, num_psites, protein_t0_series, site_series, policy="steady_state"):
        from common.initial_conditions import rules_for

        self.gene, self.model, self.num_psites, self.policy = gene, model, int(num_psites), policy
        self.measured = measured_t0_vector(model, num_psites, protein_t0_series, site_series)
        self.state_dim = self.measured.size
        self.rules = rules_for(self.measured, policy)
        self.n_estimate = sum(r == "estimate" for r in self.rules)
        self.estimated = None  # filled after a fit with policy "estimate"

    def __call__(self, params, ic_params=None, xp=None):
        import jax.numpy as jnp

        from common.initial_conditions import initial_state, linear_steady_state
        from protwise.models.diffrax_solver import make_local_model_rhs

        xp = jnp if xp is None else xp
        if self.policy == "steady_state":
            rhs = make_local_model_rhs(self.model, self.num_psites)
            steady = linear_steady_state(lambda y: rhs(0.0, y, jnp.asarray(params, dtype=jnp.float64)),
                                         self.state_dim, jnp)
            return initial_state(self.measured, policy="steady_state", steady=steady, xp=jnp)
        est = self.estimated if ic_params is None else ic_params
        full = jnp.zeros(self.state_dim, dtype=jnp.float64)
        if self.n_estimate:
            idx = np.flatnonzero(~np.isfinite(self.measured))
            full = full.at[idx].set(jnp.asarray(est, dtype=jnp.float64))
        return initial_state(self.measured, policy="estimate", estimated=full, xp=jnp)

    def value(self, params):
        """Numeric y0 at the given (final) parameters."""
        return np.asarray(self(params), dtype=np.float64)

    def record(self):
        names = ["R", "P"] + [f"X{i}" for i in range(self.state_dim - 2)]
        rec = {"policy": self.policy, "rules": dict(zip(names, self.rules))}
        if self.estimated is not None:
            rec["estimated"] = [float(v) for v in np.asarray(self.estimated)]
        return rec
