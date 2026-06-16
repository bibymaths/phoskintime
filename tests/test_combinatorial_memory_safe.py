import numpy as np
import pandas as pd
import pytest

from networkmodel.models import iter_random_transitions_for_sites, build_random_transitions
from networkmodel.simulate import combinatorial_site_signals_streaming


def _dense_transitions(n_sites):
    out = []
    for m in range(1 << n_sites):
        for j in range(n_sites):
            if (m & (1 << j)) == 0:
                out.append((m, m | (1 << j), j))
    return out


@pytest.mark.parametrize("n_sites", [2, 3, 4])
def test_streaming_bit_extraction_matches_dense(n_sites):
    rng = np.random.default_rng(123 + n_sites)
    states = rng.normal(size=(5, 1 << n_sites))
    m = np.arange(1 << n_sites, dtype=np.uint32)[:, None]
    j = np.arange(n_sites, dtype=np.uint32)[None, :]
    bits = ((m >> j) & 1).astype(np.float64)
    expected = states @ bits
    actual = combinatorial_site_signals_streaming(states, n_sites)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("n_sites", [0, 1, 2, 3, 4])
def test_transition_iterator_matches_dense_order(n_sites):
    assert list(iter_random_transitions_for_sites(n_sites)) == _dense_transitions(n_sites)


def test_build_random_transitions_keeps_small_dense_compatibility():
    class Idx:
        N = 1
        n_sites = np.array([2], dtype=np.int32)

    frm, to, site, off, ntr, dense = build_random_transitions(Idx())
    expected = _dense_transitions(2)
    assert off.tolist() == [0]
    assert ntr.tolist() == [len(expected)]
    assert dense.tolist() == [True]
    assert list(zip(frm.tolist(), to.tolist(), site.tolist())) == expected


def test_build_random_transitions_avoids_large_dense_storage():
    class Idx:
        N = 1
        n_sites = np.array([5], dtype=np.int32)

    frm, to, site, off, ntr, dense = build_random_transitions(Idx(), dense_threshold_sites=4)
    assert frm.size == to.size == site.size == 0
    assert off.tolist() == [0]
    assert ntr.tolist() == [5 * (1 << 4)]
    assert dense.tolist() == [False]


def test_tiny_combinatorial_smoke_signals():
    states = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 7.0]])
    signals = combinatorial_site_signals_streaming(states, 2)
    expected = np.array([[6.0, 7.0], [10.0, 12.0]])
    np.testing.assert_allclose(signals, expected, rtol=0.0, atol=1e-12)


def test_memory_guard_message(monkeypatch):
    import networkmodel.network as network

    monkeypatch.setattr(network, "MODEL", 2)
    monkeypatch.setattr(network, "COMBINATORIAL_MAX_STATES_PER_PROTEIN", 4)
    interactions = pd.DataFrame(
        {"protein": ["P", "P", "P"], "psite": ["S1", "S2", "S3"], "kinase": ["K", "K", "K"]}
    )
    with pytest.raises(MemoryError, match="MODEL=2 scales"):
        network.Index(interactions)
