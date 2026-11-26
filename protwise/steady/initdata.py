"""Data-anchored initial conditions for the protwise models (KNOWN_ISSUES P4, owner decision: option b).

The state at the initial time t0 is taken from the data at t0, under each model's own observation operator, so the
model reproduces the t0 observations exactly:
- R(0) = 1.0: there is no RNA datum at t0 (the first RNA time is later); fold-change convention. (Choice recorded as
  pending for the owner: R(0) = 1 vs the first RNA value.)
- P(0) = protein fold change at t0.
- distmod / succmod: X_i(0) = fold change of site i at t0 (these models read site i from state i).
- randmod: independence rule (Round 2 item 14, see randmod_independent_states / measured_t0_vector).
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
            # Independence rule (Round 2 item 14): the total T is chosen so that the unphosphorylated state equals the
            # protein t0 datum (the randmod protein observable) and occ_j = s_j/T ∈ [0, 1); then the protein datum and
            # every site marginal are reproduced exactly (randmod_total_for, randmod_independent_states).
            if s0.size >= num_psites and np.all(np.isfinite(s0[:num_psites])) and np.all(s0[:num_psites] >= 0) \
                    and np.isfinite(m[1]):
                total = randmod_total_for(m[1], s0[:num_psites])
                p_unphos, m[2:] = randmod_independent_states(total, s0[:num_psites] / total)
                m[1] = p_unphos
            else:
                logger.warning("[IC] randmod: incomplete or negative site t0 data; subset states use the ic_policy "
                               "(Round 2 item 14).")
        else:
            for i in range(num_psites):
                if i < s0.size and np.isfinite(s0[i]) and s0[i] >= 0:
                    m[2 + i] = float(s0[i])
    return m


def randmod_total_for(p0, sites):
    """Total T with T·Π(1 − s_j/T) = p0 (the unphosphorylated mass equals the protein datum p0; Round 2 item 14).

    g(T) = Π(T − s_j)/T^{n−1} is strictly increasing on T > max s_j (d log g/dT > 1/T), from 0 to ∞, so the root is
    unique; found by bisection to machine precision. For one site T = p0 + s_1 (as in distmod)."""
    s = np.asarray(sites, dtype=np.float64)
    if not (p0 > 0) or np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError(f"randmod total needs p0 > 0 and finite non-negative site values, got {p0}, {s.tolist()}.")
    n = s.size

    def g(T):
        return float(np.prod(T - s)) / T ** (n - 1)

    lo = float(s.max()) if s.size else 0.0
    hi = max(lo, 1e-300) * 2.0 + float(p0) + float(s.sum())
    while g(hi) < p0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        lo, hi = (mid, hi) if g(mid) < p0 else (lo, mid)
    return 0.5 * (lo + hi)


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
