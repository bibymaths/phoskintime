"""DECISIONS_C1 item 5: the saturating model's tf_scale override (max(tf_hi, 6) / max(tf_hi, 10); added in 1f48b59
with the comment "bounded forward flux allows more ... TF scale") is removed: saturation acts on translation, not on
the TF modulation of mRNA synthesis, so tf_scale (and the A_i box it feeds) follow one rule for every model."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_design_bounds import GRIDS, _inputs


def _bounds(monkeypatch, model):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", model)
    import scipy.sparse as sp

    idx, df_prot, df_rna, _, _ = _inputs()
    tf_mat = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))  # one TF edge: A_i depends on tf_max (review)
    kin = SimpleNamespace(Kmat=np.array([[1.1, 0.9], [0.9, 1.1]]))
    return u.calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin, time_points=GRIDS)


@pytest.mark.parametrize("model", [1, 2, 4])
def test_tf_scale_and_a_i_bounds_do_not_depend_on_the_model(monkeypatch, model):
    ref = _bounds(monkeypatch, 0)
    b = _bounds(monkeypatch, model)
    assert b["tf_scale"] == ref["tf_scale"]
    assert np.allclose(b["A_i"], ref["A_i"], rtol=1e-12)
