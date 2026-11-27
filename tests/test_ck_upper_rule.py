"""N25/N23 (P3 decision 3): the c_k upper bound is the continuous rule clip(0.1 / var(Kmat), 5, 15)."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_design_bounds import GRIDS, _inputs



def _ck_hi(monkeypatch, var, model=0):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", model)
    idx, df_prot, df_rna, tf_mat, _ = _inputs()
    # a 2 x 2 drive matrix with population variance exactly var: values 1 +- sqrt(var)
    a = np.sqrt(var)
    kin = SimpleNamespace(Kmat=np.array([[1 + a, 1 - a], [1 - a, 1 + a]]))
    assert np.isclose(np.var(kin.Kmat), var)
    return u.calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin, time_points=GRIDS)["c_k"][1]


@pytest.mark.parametrize("var,expected", [(0.0222, 5.0), (0.001, 15.0)])
def test_ck_upper_end_points(monkeypatch, var, expected):
    assert np.isclose(_ck_hi(monkeypatch, var), expected, rtol=1e-12)


@pytest.mark.parametrize("var,expected", [(0.00906, 0.1 / 0.00906), (0.01, 10.0)])
def test_ck_upper_is_continuous_rule(monkeypatch, var, expected):
    assert np.isclose(_ck_hi(monkeypatch, var), expected, rtol=1e-12)


def test_no_jump_at_former_threshold(monkeypatch):
    lo, hi = _ck_hi(monkeypatch, 0.02 - 1e-9), _ck_hi(monkeypatch, 0.02 + 1e-9)
    assert abs(lo - hi) < 1e-5


def test_model_adjustments_apply_to_the_rule(monkeypatch):
    v = 0.01  # rule: 10
    assert np.isclose(_ck_hi(monkeypatch, v, model=1), max(3.0, 0.75 * 10.0))
    assert np.isclose(_ck_hi(monkeypatch, v, model=4), min(10.0, 1.5 * 10.0))


def test_combinatorial_clamp_unchanged(monkeypatch):
    assert np.isclose(_ck_hi(monkeypatch, 0.01, model=2), 2.5)
