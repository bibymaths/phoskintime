"""N24 (P3 decision 4): parameters without data influence are detected at network construction, fixed via the
mask (lower = upper = start value) and reported as "not identifiable: no data influence".

Threshold: an edge weight counts as absent if |W| <= 1e-12 * max(1, max |W|). c_k[k] has no data influence if every
edge of kinase k is absent and k drives no TF input (driver protein with a TF edge). A site whose edges are all absent
has no kinase inflow: reported (its Dp_i still sets the decay of the site's initial value, so it is not masked).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp



def _net(tf_col_for_k1=False):
    # proteins P0 (sites a, b), P1 (site c); kinases K0, K1, K2; K1 is also protein P1 (a driver)
    idx = SimpleNamespace(
        N=2, proteins=["P0", "P1"], sites=[["a", "b"], ["c"]], kinases=["K0", "P1", "K2"],
        k2i={"K0": 0, "P1": 1, "K2": 2}, p2i={"P0": 0, "P1": 1}, proxy_map={}, total_sites=3)
    W = np.array([[0.7, 0.0, 2.7e-17],   # P0_a: K0, K2 (~0)
                  [0.3, 0.0, 1.0e-16],   # P0_b: K0, K2 (~0)
                  [0.0, 0.0, 0.0]])      # P1_c: no inflow
    tf = np.zeros((2, 2))
    if tf_col_for_k1:
        tf[0, 1] = 1.0  # P1 regulates P0's transcription
    return idx, sp.csr_matrix(W), sp.csr_matrix(tf)


def test_detects_zero_weight_kinases_and_inflowless_sites():
    from networkmodel.utils import zero_influence_parameters

    idx, W, tf = _net()
    z = zero_influence_parameters(idx, W, tf)
    assert z["threshold"] == 1e-12
    assert z["c_k"] == ["P1", "K2"]          # P1 has no edges and no TF output; K2 only ~0 weights
    assert z["sites_without_kinase_inflow"] == ["P1_c"]


def test_driver_with_tf_output_is_not_zero_influence():
    from networkmodel.utils import zero_influence_parameters

    idx, W, tf = _net(tf_col_for_k1=True)
    assert zero_influence_parameters(idx, W, tf)["c_k"] == ["K2"]


def test_mask_fixes_the_coordinates():
    from networkmodel.utils import apply_zero_influence_mask, zero_influence_parameters

    idx, W, tf = _net()
    slices = {"c_k": slice(2, 5)}
    theta0 = np.arange(6, dtype=float)
    xl, xu = np.full(6, -9.0), np.full(6, 9.0)
    xl2, xu2, mask, names = apply_zero_influence_mask(zero_influence_parameters(idx, W, tf), idx, slices,
                                                      theta0, xl, xu)
    assert names == ["c_k[P1]", "c_k[K2]"]
    assert mask.tolist() == [False, False, False, True, True, False]
    np.testing.assert_array_equal(xl2[mask], theta0[mask])
    np.testing.assert_array_equal(xu2[mask], theta0[mask])
    np.testing.assert_array_equal(xl2[~mask], xl[~mask])
    assert (xl == -9.0).all()  # inputs not modified
