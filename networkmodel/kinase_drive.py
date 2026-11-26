"""Kinase drive K(t): monotone cubic Hermite interpolation (Fritsch–Carlson, as scipy.interpolate.PchipInterpolator)
of the observed kinase fold-change series (KNOWN_ISSUES N17, owner decision: PCHIP).

One definition is used everywhere: the JAX right-hand side of the fit and the exports evaluate the same coefficients
with the same formula (xp = jax.numpy or numpy).

Extrapolation after the last knot ([networkmodel] drive_extrapolation, owner decision):
- "hold" (default): K(t) = K(t_last). Justified for this dataset because the stimulation is sustained (the
  stimulus is not withdrawn during or after the measured time course).
- "baseline": K(t) = 1 + (K(t_last) − 1)·exp(−(t − t_last)/τ), a decay to fold change 1 with time constant
  τ = [networkmodel] drive_baseline_tau (minutes).
Before the first knot the first value is held. Simulations past the last knot log a WARNING (simulate_diffrax).
"""
from __future__ import annotations

import numpy as np

EXTRAPOLATION_POLICIES = ("hold", "baseline")


def pchip_slopes(x, Y):
    """Fritsch–Carlson derivatives at the knots for every row of Y (identical to scipy's PchipInterpolator)."""
    x = np.asarray(x, dtype=np.float64)
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    h = np.diff(x)
    m = np.diff(Y, axis=1) / h
    d = np.zeros_like(Y)
    if x.size == 2:
        d[:] = m[:, :1]
        return d
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    mk, mk1 = m[:, :-1], m[:, 1:]
    same = np.sign(mk) * np.sign(mk1) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        whmean = (w1 / mk + w2 / mk1) / (w1 + w2)
        d[:, 1:-1] = np.where(same, 1.0 / whmean, 0.0)

    def _edge(h0, h1, m0, m1):
        dd = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        dd = np.where(np.sign(dd) != np.sign(m0), 0.0, dd)
        return np.where((np.sign(m0) != np.sign(m1)) & (np.abs(dd) > 3.0 * np.abs(m0)), 3.0 * m0, dd)

    d[:, 0] = _edge(h[0], h[1], m[:, 0], m[:, 1])
    d[:, -1] = _edge(h[-1], h[-2], m[:, -1], m[:, -2])
    return d


class KinaseDrive:
    """K(t) for all kinases from knots ``grid`` (n,) and values ``Kmat`` (n_kinases, n)."""

    def __init__(self, grid, Kmat, extrapolation=None, tau=None):
        if extrapolation is None or tau is None:
            from networkmodel.config import DRIVE_BASELINE_TAU, DRIVE_EXTRAPOLATION

            extrapolation = DRIVE_EXTRAPOLATION if extrapolation is None else extrapolation
            tau = DRIVE_BASELINE_TAU if tau is None else tau
        if extrapolation not in EXTRAPOLATION_POLICIES:
            raise ValueError(f"drive_extrapolation must be one of {EXTRAPOLATION_POLICIES}, got {extrapolation!r}.")
        if extrapolation == "baseline" and not float(tau) > 0.0:
            raise ValueError(f"drive_baseline_tau must be > 0 for drive_extrapolation = 'baseline', got {tau}.")
        self.extrapolation = str(extrapolation)
        self.tau = float(tau)
        self.grid = np.asarray(grid, dtype=np.float64)
        self.Kmat = np.atleast_2d(np.asarray(Kmat, dtype=np.float64))
        if self.grid.ndim != 1 or self.grid.size < 2 or np.any(np.diff(self.grid) <= 0):
            raise ValueError("Kinase drive knots must be a strictly increasing grid with at least two points.")
        self.slopes = pchip_slopes(self.grid, self.Kmat)

    def evaluate(self, t, xp=np, grid=None, Kmat=None, slopes=None):
        """K(t) for a scalar t (vector over kinases). Pass xp=jax.numpy and jnp copies of grid/Kmat/slopes in JAX."""
        g = self.grid if grid is None else grid
        K = self.Kmat if Kmat is None else Kmat
        d = self.slopes if slopes is None else slopes
        n = g.shape[0]
        i = xp.clip(xp.searchsorted(g, t, side="right") - 1, 0, n - 2)
        x0, x1 = g[i], g[i + 1]
        h = x1 - x0
        s = xp.clip((t - x0) / h, 0.0, 1.0)  # hold the end values outside the knot range
        h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
        h10 = s * (1.0 - s) ** 2
        h01 = s * s * (3.0 - 2.0 * s)
        h11 = s * s * (s - 1.0)
        inside = h00 * K[:, i] + h10 * h * d[:, i] + h01 * K[:, i + 1] + h11 * h * d[:, i + 1]
        if self.extrapolation == "hold":
            return inside
        last = K[:, n - 1]
        decay = 1.0 + (last - 1.0) * xp.exp(-xp.maximum(t - g[n - 1], 0.0) / self.tau)
        return xp.where(t > g[n - 1], decay, inside)

    def evaluate_many(self, times):
        """K at many times, shape (n_kinases, n_times) (numpy)."""
        return np.stack([self.evaluate(float(t)) for t in np.asarray(times, dtype=np.float64)], axis=1)
