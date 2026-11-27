"""N22 (P3 decision 1): first-order rate constants are bounded by the sampling design.

k_min = -ln(1 - delta) / T and k_max = -ln(delta) / dt_min, with T the span and dt_min the shortest interval of the
configured observation grid of the modality that sees the rate; delta = 0.1 (no replicate data reach networkmodel).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


GRIDS = {
    "protein": [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    "phospho": [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
    "rna": [4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0],
}


def _inputs(var_scale=0.1):
    idx = SimpleNamespace(N=2, n_sites=[1, 1])
    df_prot = pd.DataFrame({"fc": [1.0, 1.2, 0.8]})
    df_rna = pd.DataFrame({"fc": [1.0, 1.5, 0.6]})
    tf_mat = sp.csr_matrix(np.zeros((2, 2)))
    kmat = 1.0 + var_scale * np.array([[0.0, 1.0, -1.0], [0.0, 0.5, 0.5]])
    return idx, df_prot, df_rna, tf_mat, SimpleNamespace(Kmat=kmat)


def _design(grid, delta=0.1):
    t = np.unique(np.asarray(grid, float))
    return -np.log(1.0 - delta) / (t.max() - t.min()), -np.log(delta) / np.min(np.diff(t))


@pytest.mark.parametrize("model", [0, 1, 2, 4])
def test_rate_groups_use_sampling_design(monkeypatch, model):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", model)
    b = u.calculate_bio_bounds(*_inputs(), time_points=GRIDS)
    lo_r, hi_r = _design(GRIDS["rna"])
    lo_p, hi_p = _design(GRIDS["protein"])
    lo_ph, hi_ph = _design(GRIDS["phospho"])
    np.testing.assert_allclose(b["B_i"], (lo_r, hi_r), rtol=1e-12)
    np.testing.assert_allclose(b["D_i"], (lo_p, hi_p), rtol=1e-12)
    np.testing.assert_allclose(b["Dp_i"], (lo_ph, hi_ph), rtol=1e-12)
    np.testing.assert_allclose(b["E_i"], (lo_ph, hi_ph), rtol=1e-12)
    np.testing.assert_allclose(b["c_k"][0], lo_ph, rtol=1e-12)  # c_k upper: decision 3 rule


def test_design_values_for_the_configured_grids():
    import networkmodel.utils as u

    info = u.design_rate_bounds(GRIDS)
    # delta = 0.1: T = 960 / 956 min, dt_min = 0.25 / 4 min
    assert info["delta"] == 0.1 and info["delta_source"].startswith("assumption")
    np.testing.assert_allclose(info["groups"]["D_i"]["k_min"], -np.log(0.9) / 960.0, rtol=1e-12)
    np.testing.assert_allclose(info["groups"]["D_i"]["k_max"], -np.log(0.1) / 0.25, rtol=1e-12)
    np.testing.assert_allclose(info["groups"]["B_i"]["k_min"], -np.log(0.9) / 956.0, rtol=1e-12)
    np.testing.assert_allclose(info["groups"]["B_i"]["k_max"], -np.log(0.1) / 4.0, rtol=1e-12)
    g = info["groups"]["Dp_i"]
    np.testing.assert_allclose(g["half_life_max_min"], np.log(2) / g["k_min"], rtol=1e-12)
    np.testing.assert_allclose(g["half_life_min_min"], np.log(2) / g["k_max"], rtol=1e-12)
    assert set(info["groups"]) == {"B_i", "D_i", "E_i", "Dp_i", "c_k"}


def test_default_grids_come_from_config():
    import networkmodel.config as nmc
    import networkmodel.utils as u

    b = u.calculate_bio_bounds(*_inputs())
    np.testing.assert_allclose(b["B_i"], _design(nmc.TIME_POINTS_RNA), rtol=1e-12)
    np.testing.assert_allclose(b["D_i"], _design(nmc.TIME_POINTS_PROTEIN), rtol=1e-12)
