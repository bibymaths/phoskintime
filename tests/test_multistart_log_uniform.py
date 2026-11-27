"""DECISIONS_C1 item 3: networkmodel multistart starts are log-uniform in physical space (softplus of the raw
coordinate) within the bounds, replacing uniform-in-raw (uniform-in-softplus-coordinate) draws."""
import numpy as np
import pytest


def _sp(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inv_sp(y):
    y = np.asarray(y, dtype=float)
    return y + np.log(-np.expm1(-y))


def test_starts_are_log_uniform_in_physical_space():
    from scipy.stats import kstest

    from networkmodel.BayesianInference import generate_multistart_initials

    lo_p, hi_p = np.array([1e-4, 0.1]), np.array([50.0, 3.0])  # physical bounds
    lo, hi = _inv_sp(lo_p), _inv_sp(hi_p)
    starts = generate_multistart_initials(_inv_sp([1.0, 1.0]), lo, hi, n_starts=4001, seed=3)
    s = np.array(starts[1:])
    assert np.all(s >= lo) and np.all(s <= hi)
    for j in range(2):
        u = (np.log(_sp(s[:, j])) - np.log(lo_p[j])) / (np.log(hi_p[j]) - np.log(lo_p[j]))
        assert kstest(u, "uniform").pvalue > 1e-3
    # a 6-decade box: about half of the starts lie below the geometric midpoint (uniform-in-raw puts ~0 % there)
    assert 0.45 < np.mean(_sp(s[:, 0]) < np.sqrt(lo_p[0] * hi_p[0])) < 0.55


def test_first_start_and_fixed_coordinates_and_seed():
    from networkmodel.BayesianInference import generate_multistart_initials

    lo, hi = _inv_sp([1e-3, 1e-3, 1e-3]), _inv_sp([10.0, 10.0, 10.0])
    th0 = _inv_sp([0.5, 2.0, 0.3])
    mask = np.array([False, True, False])
    fv = th0.copy()
    a = generate_multistart_initials(th0, lo, hi, 6, seed=11, fixed_mask=mask, fixed_values=fv)
    b = generate_multistart_initials(th0, lo, hi, 6, seed=11, fixed_mask=mask, fixed_values=fv)
    np.testing.assert_array_equal(a[0], th0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert all(x[1] == th0[1] for x in a)
    # the draws are not the old uniform-in-raw draws
    old = np.random.default_rng(11).uniform(lo, hi)
    assert not np.allclose(a[1][[0, 2]], old[[0, 2]])
