"""Round 2 item 14: randmod initial subset states by the independence rule
mass(m) = total · Π_{j∈m} occ_j · Π_{j∉m} (1 − occ_j); states sum to the total, site marginals recovered exactly."""
import numpy as np
import pytest

R14 = pytest.mark.xfail(reason="R2-14: randmod t0 states use single-site allocation", strict=True)


@R14
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


@R14
def test_measured_vector_uses_independence_and_projects_out_of_range():
    from protwise.steady.initdata import measured_t0_vector

    m = measured_t0_vector("randmod", 2, np.array([[2.0, 1.0]]), np.array([[0.5, 1.0], [3.0, 1.0]]))
    # occ = [0.25, 1.5 -> 1.0 (projected)]; total 2.0
    assert m[1] == pytest.approx(0.0, abs=1e-15)  # unphosphorylated: total·(1−0.25)·(1−1)
    assert m[2:].sum() + m[1] == pytest.approx(2.0)
