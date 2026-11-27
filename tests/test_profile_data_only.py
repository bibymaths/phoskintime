"""DECISIONS_C1 items 1 and 2: data-only profile likelihoods.

Item 1 (C1-1): the prior is off for the reference optimum and every profile refit; the reference is the data-only
minimum (a refit from theta-hat with the prior off); threshold = chi2(1, 0.95) * f_ref / n_eff (Gaussian
approximation; the variance is estimated from the data residuals); labels say "approximate; no proper likelihood
yet (N6)".
Item 2 (C1-2): log-spaced grid in physical space centred on the reference value, out to both bounds, >= 3 points
within one decade; only converged refits count; a side with fewer than 2 converged points is "not assessed";
points below the reference objective are flagged, not refitted.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

CHI2 = 3.841458820694124


# ---------------------------------------------------------------- item 1
def test_threshold_is_chi2_times_reference_over_n_eff():
    from networkmodel.utils import profile_threshold

    assert np.isclose(profile_threshold(f_ref=0.04, n_eff=100), CHI2 * 0.04 / 100)
    assert profile_threshold(f_ref=float("nan"), n_eff=10) != profile_threshold(f_ref=float("nan"), n_eff=10)


def test_n_eff_counts_residuals_of_active_terms_without_baseline_rows():
    from networkmodel.utils import n_eff_from_loss_data

    ld = {"t_prot": np.array([0, 1, 2, 0, 1]), "obs_prot": np.array([1.0, 1.2, 0.9, 1.0, 1.1]),
          "w_prot": np.ones(5), "prot_base_idx": 0,
          "t_rna": np.array([0, 3]), "obs_rna": np.array([1.0, 2.0]), "w_rna": np.ones(2), "rna_base_idx": 0,
          "t_pho": np.array([0, 1, 2]), "obs_pho": np.array([0.8, 1.0, 1.0]), "w_pho": np.ones(3), "pho_base_idx": 0}
    mode = SimpleNamespace(fit_protein=True, fit_mrna=False, fit_phospho=True)
    # protein: 5 rows - 2 baseline rows (obs == 1 at base: residual 0 by construction) = 3
    # phospho: the baseline row has obs 0.8 (a real residual) -> counts; 3
    assert n_eff_from_loss_data(ld, mode) == 6


def test_reference_is_data_only_refit_from_theta_hat():
    from networkmodel.ProfileWorker import reference_refit

    import jax.numpy as jnp

    a = np.array([0.5, -1.0, 2.0])
    obj = lambda th: jnp.sum((jnp.asarray(th) - a) ** 2)  # data-only objective, minimum at a
    theta_hat = a + np.array([0.3, 0.2, -0.1])  # MAP: pulled away from a by the prior
    ref = reference_refit(obj, theta_hat, -5 * np.ones(3), 5 * np.ones(3), fixed_mask=None, maxiter=500,
                          tol=1e-10, optimizer="lbfgsb")
    np.testing.assert_allclose(ref["theta_ref"], a, atol=1e-6)
    assert ref["f_ref"] < float(obj(theta_hat)) and ref["converged"]
    assert np.isclose(ref["f_at_theta_hat"], float(obj(theta_hat)))


def test_profile_note_says_approximate_no_proper_likelihood():
    from networkmodel.BayesianInference import PROFILE_SCALE_NOTE
    from networkmodel.utils import PROFILE_RULE

    for s in (PROFILE_SCALE_NOTE, PROFILE_RULE):
        assert "approximate; no proper likelihood yet (N6)" in s
        assert "prior off" in s


# ---------------------------------------------------------------- item 2
@pytest.mark.parametrize("v,lo,hi", [(0.07, 1e-4, 50.0), (2.0, 1.5, 3.0), (1e-3, 1e-4, 1e4)])
def test_grid_is_log_spaced_to_both_bounds_with_three_points_per_decade(v, lo, hi):
    from networkmodel.utils import profile_grid_physical

    below, above = profile_grid_physical(v, lo, hi, n_per_side=5)
    for side, bound in ((below, lo), (above, hi)):
        assert np.isclose(side[-1], bound, rtol=1e-12)  # out to the bound
        d = np.abs(np.log10(side / v))
        assert np.all(np.diff(d) > 0)  # ordered outward from the centre
        assert np.sum(d <= 1.0 + 1e-12) >= min(3, len(side))  # >= 3 within one decade
    assert np.all(below < v) and np.all(above > v) and np.all(below >= lo) and np.all(above <= hi)


def test_grid_side_is_empty_at_a_bound():
    from networkmodel.utils import profile_grid_physical

    below, above = profile_grid_physical(1e-4, 1e-4, 10.0, n_per_side=5)
    assert len(below) == 0 and len(above) == 5


def _prof(below_f, above_f, conv_b=None, conv_a=None):
    nb, na = len(below_f), len(above_f)
    return pd.DataFrame({
        "side": ["below"] * nb + ["above"] * na,
        "grid_value_physical": list(np.geomspace(0.5, 0.01, nb)) + list(np.geomspace(2, 100, na)),
        "objective_value": list(below_f) + list(above_f),
        "converged": list(conv_b if conv_b is not None else [True] * nb)
        + list(conv_a if conv_a is not None else [True] * na),
        "parameter_name": "p"})


@pytest.mark.parametrize("b,a,label", [
    ([1.2, 2.0], [1.3, 3.0], "estimate (two-sided)"),
    ([1.0, 1.0], [1.3, 3.0], "upper limit only"),
    ([1.3, 3.0], [1.0, 1.0], "lower limit only"),
    ([1.0, 1.0], [1.0, 1.0], "not identifiable (flat)"),
    ([1.0], [1.3, 3.0], "upper limit only (lower side not assessed)"),
    ([1.3, 3.0], [], "lower limit only (upper side not assessed)"),
    ([1.0], [1.0, 1.0], "not assessed"),
])
def test_side_rules(b, a, label):
    from networkmodel.utils import identifiability_from_profile

    out = identifiability_from_profile(_prof(b, a), f_ref=1.0, threshold=0.1)
    assert out["label"] == label


def test_non_converged_points_do_not_count():
    from networkmodel.utils import identifiability_from_profile

    out = identifiability_from_profile(_prof([1.5, 5.0], [1.5, 5.0], conv_b=[False, False], conv_a=[True, False]),
                                       f_ref=1.0, threshold=0.1)
    assert out["label"] == "not assessed"
    assert out["sides"] == {"below": "not assessed", "above": "not assessed"}


def test_nonfinite_reference_or_threshold_is_not_assessed():
    from networkmodel.utils import identifiability_from_profile

    for f_ref, thr in ((float("nan"), 0.1), (1.0, float("nan")), (float("inf"), 0.1)):
        assert identifiability_from_profile(_prof([2, 3], [2, 3]), f_ref=f_ref, threshold=thr)["label"] == \
            "not assessed"


def test_points_better_than_reference_are_flagged():
    from networkmodel.utils import identifiability_from_profile

    out = identifiability_from_profile(_prof([0.9, 2.0], [1.3, 3.0]), f_ref=1.0, threshold=0.1)
    assert out["label"] == "estimate (two-sided)"  # labelled against the reference, but flagged
    (bp,) = out["better_points"]
    assert bp["parameter"] == "p" and np.isclose(bp["delta"], -0.1) and np.isclose(bp["grid_value_physical"], 0.5)


# ---------------------------------------------------------------- review C1-review MJ-1, MJ-2
def test_n_eff_skips_modalities_with_zero_weight():
    from networkmodel.utils import n_eff_from_loss_data

    ld = {"t_prot": np.array([1, 2]), "obs_prot": np.array([1.2, 0.9]), "w_prot": np.ones(2), "prot_base_idx": 0,
          "t_rna": np.array([1, 3]), "obs_rna": np.array([1.5, 2.0]), "w_rna": np.ones(2), "rna_base_idx": 0}
    mode = SimpleNamespace(fit_protein=True, fit_mrna=True, fit_phospho=False)
    assert n_eff_from_loss_data(ld, mode, weights={"protein": 1.0, "rna": 0.0, "phospho": 1.0}) == 2
    assert n_eff_from_loss_data(ld, mode, weights={"protein": 1.0, "rna": 1.0}) == 4


def test_non_converged_points_below_the_reference_are_flagged_too():
    from networkmodel.utils import identifiability_from_profile

    out = identifiability_from_profile(_prof([0.8, 2.0, 3.0], [1.3, 3.0], conv_b=[False, True, True]),
                                       f_ref=1.0, threshold=0.1)
    assert out["label"] == "estimate (two-sided)"
    (bp,) = out["better_points"]
    assert np.isclose(bp["delta"], -0.2) and bp["converged"] is False
