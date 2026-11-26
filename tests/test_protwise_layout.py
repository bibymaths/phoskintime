"""P2: the flat protwise fit vector [RNA | protein | phospho] is split from the configured grids,
not from the hard-coded 9/14/23 offsets."""
import numpy as np
import pytest

P2 = pytest.mark.xfail(reason="P2: hard-coded 9/14/23 offsets", strict=True)


@P2
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


@P2
def test_split_flat_fit_rejects_wrong_length():
    from protwise.models.diffrax_solver import split_flat_fit

    with pytest.raises(ValueError):
        split_flat_fit(np.zeros(10), 1, 4, 3)


@P2
def test_split_flat_fit_defaults_to_configured_rna_grid():
    from config.constants import TIME_POINTS, TIME_POINTS_RNA
    from protwise.models.diffrax_solver import split_flat_fit

    n_r, n_t = len(TIME_POINTS_RNA), len(TIME_POINTS)
    r, pr, ph = split_flat_fit(np.arange(n_r + n_t + 2 * n_t, dtype=float), 2, n_t)
    assert r.size == n_r and pr.size == n_t and ph.shape == (2, n_t)
