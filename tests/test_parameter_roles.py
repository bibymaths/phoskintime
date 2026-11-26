"""N8: pin the meaning of E_i, Dp_i and D_i in the networkmodel JAX RHS (distributive, one protein, one site).
E_i returns the phospho form to P; Dp_i removes the phospho form only; D_i removes every state."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rhs_parity", ROOT / "tests" / "test_rhs_parity.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


@pytest.fixture()
def one_site_rhs(monkeypatch):
    import jax.numpy as jnp
    import scipy.sparse as sp
    from types import SimpleNamespace

    import networkmodel.config as nc
    from networkmodel.backend import ensure_jax_float64, make_networkmodel_rhs

    ensure_jax_float64()
    monkeypatch.setattr(nc, "MODEL", 0)
    idx = SimpleNamespace(N=1, proteins=["P"], kinases=["K"], p2i={"P": 0}, k2i={"K": 0},
                          offset_y=np.array([0]), offset_s=np.array([0]), n_sites=np.array([1]),
                          n_states=np.array([2]))
    sys = SimpleNamespace(idx=idx, W_global=sp.csr_matrix(np.zeros((1, 1))), tf_mat=sp.csr_matrix(np.zeros((1, 1))),
                          tf_deg=np.ones(1), kin=SimpleNamespace(grid=np.array([0.0, 1.0]), Kmat=np.ones((1, 2))))
    rhs = make_networkmodel_rhs(sys)

    def f(**rates):
        par = {k: jnp.zeros(1) for k in ("c_k", "A_i", "B_i", "C_i", "D_i", "Dp_i", "E_i")}
        par["tf_scale"] = jnp.asarray(1.0)
        par.update({k: jnp.asarray([v]) for k, v in rates.items()})
        y = jnp.asarray([0.0, 2.0, 3.0])  # R, P (unphosphorylated), X (phospho site)
        return np.asarray(rhs(jnp.float64(0.5), y, par))

    return f


def test_E_returns_phospho_form_to_P(one_site_rhs):
    np.testing.assert_allclose(one_site_rhs(E_i=0.7), [0.0, 0.7 * 3.0, -0.7 * 3.0], atol=1e-15)


def test_Dp_removes_only_the_phospho_form(one_site_rhs):
    np.testing.assert_allclose(one_site_rhs(Dp_i=0.4), [0.0, 0.0, -0.4 * 3.0], atol=1e-15)


def test_D_degrades_every_protein_state(one_site_rhs):
    np.testing.assert_allclose(one_site_rhs(D_i=0.2), [0.0, -0.2 * 2.0, -0.2 * 3.0], atol=1e-15)
