"""P2: the flat protwise fit vector [RNA | protein | phospho] is split from the configured grids,
not from the hard-coded 9/14/23 offsets."""
import numpy as np
import pytest

@pytest.mark.parametrize("n_rna,n_t,num_psites", [(9, 14, 3), (5, 10, 2), (0, 7, 1), (4, 6, 0)])
def test_split_flat_fit_roundtrip(n_rna, n_t, num_psites):
    from protwise.models.diffrax_solver import split_flat_fit

    r = np.arange(n_rna, dtype=float)
    pr = 100 + np.arange(n_t, dtype=float)
    ph = 1000 + np.arange(num_psites * n_t, dtype=float).reshape(num_psites, n_t)
    flat = np.concatenate([r, pr, ph.ravel()])
    r2, pr2, ph2 = split_flat_fit(flat, num_psites, n_t, n_rna)
    np.testing.assert_array_equal(r2, r)
    np.testing.assert_array_equal(pr2, pr)
    np.testing.assert_array_equal(ph2, ph)


def test_split_flat_fit_rejects_wrong_length():
    from protwise.models.diffrax_solver import split_flat_fit

    with pytest.raises(ValueError):
        split_flat_fit(np.zeros(10), 1, 4, 3)


def test_split_flat_fit_defaults_to_configured_rna_grid():
    from config.constants import TIME_POINTS, TIME_POINTS_RNA
    from protwise.models.diffrax_solver import split_flat_fit

    n_r, n_t = len(TIME_POINTS_RNA), len(TIME_POINTS)
    r, pr, ph = split_flat_fit(np.arange(n_r + n_t + 2 * n_t, dtype=float), 2, n_t)
    assert r.size == n_r and pr.size == n_t and ph.shape == (2, n_t)


@pytest.mark.parametrize("layers", [("r", "p", "ph"), ("p", "ph"), ("r", "ph"), ("ph",)])
def test_phospho_tail_independent_of_present_layers(layers):
    from protwise.paramest.core import _phospho_tail

    n_t, ns = 14, 3
    ph = 1000 + np.arange(ns * n_t, dtype=float)
    parts = {"r": np.arange(9.0), "p": 100 + np.arange(n_t, dtype=float), "ph": ph}
    flat = np.concatenate([parts[k] for k in layers])
    np.testing.assert_array_equal(_phospho_tail(flat, ns, n_t), ph.reshape(ns, n_t))
