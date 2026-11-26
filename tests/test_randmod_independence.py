"""Round 2 item 14: randmod initial subset states by the independence rule
mass(m) = total · Π_{j∈m} occ_j · Π_{j∉m} (1 − occ_j); states sum to the total, site marginals recovered exactly."""
import numpy as np
import pytest



@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_independence_rule_sums_and_marginals(n):
    from config.helpers import randmod_subset_masks
    from protwise.steady.initdata import randmod_independent_states

    rng = np.random.default_rng(n)
    occ = rng.uniform(0.05, 0.95, n)
    total = 1.7
    p0, subsets = randmod_independent_states(total, occ)
    masks = randmod_subset_masks(n)
    assert p0 + subsets.sum() == pytest.approx(total, rel=1e-14)
    for j in range(n):
        marg = sum(subsets[k] for k, m in enumerate(masks) if m >> j & 1)
        assert marg == pytest.approx(total * occ[j], rel=1e-13)
    assert p0 == pytest.approx(total * np.prod(1 - occ), rel=1e-14)


def test_measured_vector_reproduces_protein_and_site_data():
    """The unphosphorylated state equals the protein datum and every site marginal equals its datum, also when a
    site value exceeds the protein value (fold changes; no projection)."""
    from config.helpers import randmod_subset_masks
    from protwise.steady.initdata import measured_t0_vector

    pr, sites = 1.216, np.array([1.559, 0.494, 1.49])  # ABL2-like t0 values
    m = measured_t0_vector("randmod", 3, np.array([[pr, 1.0]]), np.column_stack([sites, np.ones(3)]))
    assert m[1] == pytest.approx(pr, rel=1e-12)
    masks = randmod_subset_masks(3)
    for j in range(3):
        assert sum(m[2 + k] for k, mk in enumerate(masks) if mk >> j & 1) == pytest.approx(sites[j], rel=1e-12)
    assert np.all(m[1:] >= 0)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_total_root_property(n):
    from protwise.steady.initdata import randmod_total_for

    rng = np.random.default_rng(10 + n)
    s, p0 = rng.uniform(0, 3, n), rng.uniform(0.1, 3)
    T = randmod_total_for(p0, s)
    assert T * np.prod(1 - s / T) == pytest.approx(p0, rel=1e-12)
    if n == 1:
        assert T == pytest.approx(p0 + s[0], rel=1e-12)
