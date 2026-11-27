"""DECISIONS_C1_FINAL items 2-3.

Item 2 (C1F-2): the data-only reference is refitted from theta-hat and the best kept multistart solutions; the best
usable start is kept (usable: converged, KKT <= 1e-6, or near_converged, <= 1e-5). Profile points: converged or
near_converged are usable; a side built on near-converged points (or a near-converged reference) gives a label
"... (approximate: near-converged)"; "not assessed" only without a usable reference or with < 2 usable points.
Item 3 (C1F-3): profile interval (threshold crossings), value_fit, value_data_only, fit_inside_interval; outside ->
status "... ; fitted value set by the prior" and not a point estimate.
"""
import numpy as np
import pandas as pd

from tests.test_parameter_status import _case


def _prof(below, above, st_b=None, st_a=None, center=1.0):
    nb, na = len(below), len(above)
    gb = center * 10.0 ** (-np.arange(1, nb + 1) / 3.0)
    ga = center * 10.0 ** (np.arange(1, na + 1) / 3.0)
    return pd.DataFrame({
        "side": ["below"] * nb + ["above"] * na,
        "grid_value_physical": list(gb) + list(ga),
        "objective_value": list(below) + list(above),
        "convergence_status": list(st_b or ["converged"] * nb) + list(st_a or ["converged"] * na),
        "parameter_name": "p"})


# ---------------------------------------------------------------- item 2
def test_reference_selection_keeps_best_usable_start():
    from networkmodel.ProfileWorker import select_reference

    recs = [{"start": "theta_hat", "f": 0.50, "status": "not_converged"},
            {"start": "ms1", "f": 0.40, "status": "near_converged"},
            {"start": "ms2", "f": 0.45, "status": "converged"},
            {"start": "ms3", "f": 0.30, "status": "not_converged"}]
    assert select_reference(recs)["start"] == "ms1"
    assert select_reference([{"start": "a", "f": 0.1, "status": "not_converged"}]) is None


def test_near_converged_points_count_and_mark_the_label_approximate():
    from networkmodel.utils import identifiability_from_profile

    p = _prof([1.2, 2.0], [1.3, 3.0], st_b=["near_converged", "converged"])
    out = identifiability_from_profile(p, f_ref=1.0, threshold=0.1)
    assert out["label"] == "estimate (two-sided) (approximate: near-converged)"
    assert out["n_usable"] == {"below": 2, "above": 2}
    # all converged and a converged reference: no suffix
    assert identifiability_from_profile(_prof([1.2, 2.0], [1.3, 3.0]), f_ref=1.0, threshold=0.1)["label"] == \
        "estimate (two-sided)"
    # a near-converged reference makes every label approximate
    assert identifiability_from_profile(_prof([1.2, 2.0], [1.3, 3.0]), f_ref=1.0, threshold=0.1,
                                        reference_status="near_converged")["label"].endswith(
        "(approximate: near-converged)")


def test_not_converged_points_are_not_usable():
    from networkmodel.utils import identifiability_from_profile

    p = _prof([1.2, 2.0], [1.3, 3.0], st_b=["not_converged", "converged"])
    assert identifiability_from_profile(p, f_ref=1.0, threshold=0.1)["label"] == \
        "upper limit only (lower side not assessed)"


# ---------------------------------------------------------------- item 3
def test_profile_interval_from_threshold_crossings():
    from networkmodel.utils import identifiability_from_profile

    # below: d = 0.05 at 10^-1/3, 0.15 at 10^-2/3 -> crossing of 0.1 halfway in log10: 10^-0.5
    # above: d = 0.3 at 10^1/3 -> crossing between the centre (d = 0) and the first point: 10^(1/9)
    p = _prof([1.05, 1.15], [1.3, 3.0])
    out = identifiability_from_profile(p, f_ref=1.0, threshold=0.1, center_physical=1.0, bounds_physical=(1e-3, 1e3))
    lo, hi = out["interval"]
    assert np.isclose(lo, 10 ** -0.5) and np.isclose(hi, 10 ** (1 / 9))
    # a flat side extends to its bound; an unassessed side gives NaN
    flat = identifiability_from_profile(_prof([1.0, 1.0], [1.3, 3.0]), f_ref=1.0, threshold=0.1, center_physical=1.0,
                                        bounds_physical=(1e-3, 1e3))
    assert flat["interval"][0] == 1e-3
    na = identifiability_from_profile(_prof([1.0], [1.3, 3.0]), f_ref=1.0, threshold=0.1, center_physical=1.0,
                                      bounds_physical=(1e-3, 1e3))
    assert np.isnan(na["interval"][0])


def test_parameter_status_reports_fit_inside_interval():
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    v_k1 = float(np.logaddexp(0.0, th[names.index("c_k[K1]")]))
    ident = {"c_k[K1]": "estimate (two-sided)", "tf_scale": "estimate (two-sided)"}
    prof = {"c_k[K1]": {"value_data_only": v_k1 * 1.1, "lower": v_k1 / 2, "upper": v_k1 * 2},
            "tf_scale": {"value_data_only": 3.0, "lower": 2.5, "upper": 4.0}}
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"], identifiability=ident,
                          profile_info=prof).set_index("parameter")
    for c in ("value_fit", "value_data_only", "profile_lower", "profile_upper", "fit_inside_interval"):
        assert c in st.columns
    assert st.loc["c_k[K1]", "fit_inside_interval"] == True  # noqa: E712
    assert st.loc["c_k[K1]", "is_estimate"]
    tf = st.loc["tf_scale"]
    assert tf["fit_inside_interval"] == False  # noqa: E712
    assert "fitted value set by the prior" in tf["status"] and "fitted value set by the prior" in tf["label"]
    assert not tf["is_estimate"]
    assert pd.isna(st.loc["Dp_i[P_s1]", "fit_inside_interval"])


def test_fit_outside_on_its_own_side_counts_when_the_other_side_is_open():
    """B4 E_i[ABL2]: data-only optimum at the lower bound (lower limit NaN), fit 1.08 above the upper limit 0.90."""
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    prof = {"tf_scale": {"value_data_only": 0.5, "lower": float("nan"), "upper": 0.9},   # fit 1.0 > 0.9
            "c_k[K1]": {"value_data_only": 1e-3, "lower": float("nan"), "upper": 5.0}}  # fit inside on its side
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"], profile_info=prof).set_index(
        "parameter")
    assert st.loc["tf_scale", "fit_inside_interval"] == False  # noqa: E712
    assert st.loc["c_k[K1]", "fit_inside_interval"] == True  # noqa: E712


def test_profile_of_a_parameter_with_an_empty_grid_is_not_assessed(tmp_path):
    from networkmodel.ProfileWorker import empty_profile_frame
    from networkmodel.utils import identifiability_from_profile

    df = empty_profile_frame("c_k[K3]", 11)
    assert len(df) == 0 and {"side", "objective_value", "convergence_status", "grid_value_physical"} <= set(df)
    assert identifiability_from_profile(df, f_ref=1.0, threshold=0.1)["label"] == "not assessed"
