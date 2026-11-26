"""N20: for the sequential model (MODEL 1) the observable of measured site j is the site occupancy, i.e. the sum
of all states in which site j is phosphorylated: state k (= k+1 sites phosphorylated, in site order) contains
site j iff k >= j. Same rule in the loss, the export and the initial conditions. Combinatorial (MODEL 2) already
sums over the masks containing the site."""
from types import SimpleNamespace

import numpy as np
import pytest

# one protein, 3 sites: y = [R, P0, P1, P2, P3]
Y = np.array([[1.0, 2.0, 0.6, 0.3, 0.1],
              [1.5, 1.0, 0.9, 0.5, 0.4]])


def _occ(row, j):
    return row[2 + j:].sum()


@pytest.mark.parametrize("site", [0, 1, 2])
def test_loss_observable_is_site_occupancy(site):
    import jax.numpy as jnp

    from networkmodel.backend import _global_networkmodel_observable

    got = float(_global_networkmodel_observable(
        jnp.asarray(Y), jnp.array([0]), jnp.array([3]), jnp.array([3]), 0, 1, layer="phospho", site_idx=site,
        base_idx=0, layout="sequential", max_count=3))
    np.testing.assert_allclose(got, _occ(Y[1], site) / _occ(Y[0], site), rtol=1e-12)


def test_cache_marks_sequential_layout(monkeypatch):
    import inspect

    import networkmodel.cache as cache

    assert '"sequential" if MODEL == 1' in inspect.getsource(cache)


def test_export_measurement_is_site_occupancy(monkeypatch):
    import networkmodel.simulate as sim

    monkeypatch.setattr(sim, "MODEL", 1)
    idx = SimpleNamespace(proteins=["G"], offset_y=np.array([0]), n_sites=np.array([3]), n_states=np.array([8]),
                          sites=[["S1", "S2", "S3"]])
    times = np.array([0.0, 4.0])
    out = sim.measure_trajectory(Y, idx, times, times, times, times)
    pho = out[2] if isinstance(out, tuple) else out["phospho"]
    for j, s in enumerate(["S1", "S2", "S3"]):
        fc = pho.loc[(pho["psite"] == s) & (pho["time"] == 4.0), "pred_fc"].item()
        np.testing.assert_allclose(fc, _occ(Y[1], j) / _occ(Y[0], j), rtol=1e-12)


def test_initial_conditions_invert_occupancy(monkeypatch):
    import networkmodel.InitialConditions as ic

    monkeypatch.setattr(ic, "MODEL", 1)
    occ = np.array([0.20, 0.12, 0.05])
    states = ic.sequential_states_from_occupancy(occ)
    np.testing.assert_allclose([states[j:].sum() for j in range(3)], occ, rtol=1e-12)
    # non-monotone occupancy is projected (running minimum), never negative
    states2 = ic.sequential_states_from_occupancy(np.array([0.1, 0.3, 0.05]))
    assert (states2 >= 0).all()
