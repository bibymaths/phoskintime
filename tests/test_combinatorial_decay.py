"""N9: in the combinatorial model (MODEL 2) protein degradation D_i applies ONCE per molecule:
decay rate of state m = D_i + sum_{j in m} Dp_i[j]. Checks (a)-(d) of the owner's decision."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp


def _sys(model, n_sites):
    nst = 2 ** n_sites
    idx = SimpleNamespace(N=1, proteins=["P"], kinases=["K"], p2i={"P": 0}, k2i={"K": 0},
                          offset_y=np.array([0]), offset_s=np.array([0]), n_sites=np.array([n_sites]),
                          n_states=np.array([nst]))
    W = np.ones((max(n_sites, 1), 1))[:n_sites]
    return SimpleNamespace(idx=idx, W_global=sp.csr_matrix(W if n_sites else np.zeros((0, 1))),
                           tf_mat=sp.csr_matrix(np.zeros((1, 1))), tf_deg=np.ones(1),
                           kin=SimpleNamespace(grid=np.array([0.0, 1.0]), Kmat=np.ones((1, 2))))


def _rhs(monkeypatch, model, n_sites):
    import networkmodel.config as nc
    from networkmodel.backend import ensure_jax_float64, make_networkmodel_rhs

    ensure_jax_float64()
    monkeypatch.setattr(nc, "MODEL", model)
    return make_networkmodel_rhs(_sys(model, n_sites))


def _par(n_sites, **kw):
    import jax.numpy as jnp

    par = {k: jnp.zeros(1) for k in ("A_i", "B_i", "C_i", "D_i", "E_i")}
    par.update(c_k=jnp.zeros(1), Dp_i=jnp.zeros(n_sites), tf_scale=jnp.asarray(1.0))
    par.update({k: jnp.asarray(v, dtype=jnp.float64) for k, v in kw.items()})
    return par


def _solve(rhs, y0, par, ts, rtol=1e-12):
    import jax.numpy as jnp

    from networkmodel.backend import DiffraxSolverConfig, solve_diffrax

    cfg = DiffraxSolverConfig(solver_name="Kvaerno5", rtol=rtol, atol=1e-16, max_steps=1_000_000)
    return np.asarray(solve_diffrax(jnp.asarray(y0), jnp.asarray(ts), params=par, rhs=rhs, config=cfg))


def test_a_total_protein_decays_at_D(monkeypatch):
    """(a) all transition/synthesis rates zero, Dp = 0: total protein decays at exactly D_i for any initial
    distribution over the 2^3 states."""
    rhs = _rhs(monkeypatch, 2, 3)
    D = 0.3
    y0 = np.concatenate([[0.0], np.random.default_rng(1).uniform(0.1, 1.0, 8)])  # R = 0, all masks populated
    ts = np.linspace(0.0, 10.0, 21)
    Y = _solve(rhs, y0, _par(3, D_i=[D]), ts, rtol=1e-14)
    total = Y[:, 1:].sum(axis=1)
    np.testing.assert_allclose(total, total[0] * np.exp(-D * ts), rtol=1e-10)


def test_b_single_site_combinatorial_equals_distributive(monkeypatch):
    """(b) with one site, combinatorial and distributive describe the same system (same state order R, P0, P1)."""
    import jax.numpy as jnp

    rhs_c, rhs_d = _rhs(monkeypatch, 2, 1), None
    rhs_d = _rhs(monkeypatch, 0, 1)
    rng = np.random.default_rng(2)
    for _ in range(20):
        par = _par(1, A_i=[rng.uniform(.1, 1)], B_i=[rng.uniform(.1, 1)], C_i=[rng.uniform(.1, 1)],
                   D_i=[rng.uniform(.01, .5)], E_i=[rng.uniform(0, 2)], c_k=[rng.uniform(.1, 2)],
                   Dp_i=[rng.uniform(.01, 2)])
        y = jnp.asarray(rng.uniform(0.1, 2, 3))
        np.testing.assert_allclose(np.asarray(rhs_c(0.5, y, par)), np.asarray(rhs_d(0.5, y, par)), rtol=1e-12,
                                   atol=1e-15)
    ts = np.linspace(0, 20, 41)
    y0 = np.array([1.0, 0.8, 0.3])
    np.testing.assert_allclose(_solve(rhs_c, y0, par, ts), _solve(rhs_d, y0, par, ts), rtol=1e-12, atol=1e-14)


def test_c_mass_conserved_without_degradation(monkeypatch):
    """(c) degradation and Dp zero: phosphorylation/dephosphorylation alone conserve total protein."""
    import jax.numpy as jnp

    rhs = _rhs(monkeypatch, 2, 3)
    rng = np.random.default_rng(3)
    for _ in range(20):
        par = _par(3, E_i=[rng.uniform(0, 2)], c_k=[rng.uniform(.1, 3)])
        y = jnp.asarray(np.concatenate([[0.0], rng.uniform(0.1, 1, 8)]))
        dy = np.asarray(rhs(0.5, y, par))
        assert abs(dy[1:].sum()) <= 1e-13 * np.abs(dy[1:]).sum() + 1e-15


@pytest.mark.parametrize("kernel", ["jax", "numba"])
def test_per_state_decay_rate(monkeypatch, kernel):
    """With S = E = C = 0: dy_m = -(D + sum_{j in m} Dp_j) * y_m for every mask m (both kernels)."""
    import jax.numpy as jnp

    rng = np.random.default_rng(4)
    D, Dp = 0.17, rng.uniform(0.05, 1.0, 3)
    y = np.concatenate([[0.0], rng.uniform(0.1, 1.0, 8)])
    if kernel == "jax":
        dy = np.asarray(_rhs(monkeypatch, 2, 3)(0.5, jnp.asarray(y), _par(3, D_i=[D], Dp_i=Dp)))
    else:
        from networkmodel.models import combinatorial_rhs

        dy = np.zeros_like(y)
        z = np.zeros(1)
        combinatorial_rhs(y, dy, z, z, z, np.array([D]), Dp, z, 1.0, z, np.zeros(3), np.array([0], np.int32),
                          np.array([0], np.int32), np.array([3], np.int32), np.array([8], np.int32))
    for m in range(8):
        rate = D + sum(Dp[j] for j in range(3) if m >> j & 1)
        np.testing.assert_allclose(dy[1 + m], -rate * y[1 + m], rtol=1e-14, atol=1e-16)
