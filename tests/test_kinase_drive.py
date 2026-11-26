"""N17 (owner decision: PCHIP): the kinase drive is monotone cubic Hermite (Fritsch–Carlson) in JAX, identical to
scipy's PchipInterpolator, homogeneous in K, positive for positive data, and the export uses the same function."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.interpolate import PchipInterpolator

GRID = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])


def _K(seed=0):
    rng = np.random.default_rng(seed)
    K = rng.uniform(0.2, 3.0, (4, GRID.size))
    K[1] = np.linspace(1.0, 2.0, GRID.size)  # monotone
    K[2, 4:8] = 1.5  # flat stretch
    return K


def test_matches_scipy_pchip_and_is_homogeneous():
    from networkmodel.kinase_drive import KinaseDrive

    K = _K()
    t = np.unique(np.concatenate([np.linspace(0.0, 960.0, 5001), GRID]))
    got = KinaseDrive(GRID, K).evaluate_many(t)
    ref = PchipInterpolator(GRID, K, axis=1)(t)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))
    for c in (0.01, 2.7, 150.0):
        np.testing.assert_allclose(KinaseDrive(GRID, c * K).evaluate_many(t), c * got, rtol=1e-12, atol=0)


def test_positive_data_give_positive_drive_and_hold_after_last_knot():
    from networkmodel.kinase_drive import KinaseDrive

    K = _K(3)
    drv = KinaseDrive(GRID, K)
    t = np.linspace(0.0, 960.0, 20001)
    assert drv.evaluate_many(t).min() > 0.0
    np.testing.assert_array_equal(drv.evaluate(5000.0), K[:, -1])


def test_jax_rhs_uses_the_pchip_drive(monkeypatch):
    import jax.numpy as jnp

    import networkmodel.config as nc
    from networkmodel.backend import ensure_jax_float64, make_networkmodel_rhs

    ensure_jax_float64()
    monkeypatch.setattr(nc, "MODEL", 0)
    K = _K(1)
    ref = PchipInterpolator(GRID, K, axis=1)
    for j in range(K.shape[0]):
        W = np.zeros((1, K.shape[0]))
        W[0, j] = 1.0
        idx = SimpleNamespace(N=1, proteins=["P"], kinases=[f"K{i}" for i in range(K.shape[0])], p2i={"P": 0},
                              k2i={f"K{i}": i for i in range(K.shape[0])}, offset_y=np.array([0]),
                              offset_s=np.array([0]), n_sites=np.array([1]), n_states=np.array([2]))
        sysn = SimpleNamespace(idx=idx, W_global=sp.csr_matrix(W), tf_mat=sp.csr_matrix(np.zeros((1, 1))),
                               tf_deg=np.ones(1), kin=SimpleNamespace(grid=GRID, Kmat=K))
        rhs = make_networkmodel_rhs(sysn)
        par = {k: jnp.zeros(1) for k in ("A_i", "B_i", "C_i", "D_i", "Dp_i", "E_i")}
        par.update(c_k=jnp.ones(K.shape[0]), tf_scale=jnp.asarray(1.0))
        y = jnp.asarray([0.0, 1.0, 0.0])  # dy[site] = S * P = K_j(t)
        for t in (0.6, 3.0, 45.0, 700.0):
            np.testing.assert_allclose(float(rhs(jnp.float64(t), y, par)[2]), ref(t)[j], rtol=1e-12)


def test_export_uses_the_same_drive():
    import networkmodel.export as ex

    src = inspect.getsource(ex.export_kinase_activities)
    assert "interp1d" not in src and "KinaseDrive" in src
