"""Pre-merge item 5: identifiability labels. With profile likelihoods: "estimate (two-sided)", "upper limit only",
"lower limit only" or "not identifiable (flat)"; without: "not assessed", never "estimate"."""

from tests.test_parameter_status import _case



# Label rules from data-only profiles: tests/test_profile_data_only.py (DECISIONS_C1 items 1-2).


def test_without_profiles_nothing_is_called_estimate():
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"]).set_index("parameter")
    assert not (st["label"] == "estimate").any()
    for n in ("c_k[K1]", "Dp_i[P_s1]", "tf_scale"):
        assert st.loc[n, "label"] == "not assessed" and not st.loc[n, "is_estimate"]
    assert (st["identifiability"] == "not assessed").all()


def test_with_profiles_interior_labels_follow_the_profile():
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    ident = {"c_k[K1]": "estimate (two-sided)", "tf_scale": "upper limit only"}
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"],
                          identifiability=ident).set_index("parameter")
    assert st.loc["c_k[K1]", "label"] == "estimate (two-sided)" and st.loc["c_k[K1]", "is_estimate"]
    assert st.loc["tf_scale", "label"] == "upper limit only" and not st.loc["tf_scale", "is_estimate"]
    assert st.loc["Dp_i[P_s1]", "label"] == "not assessed"
    # a parameter at a design bound keeps its bound label; its profile result is in the identifiability column
    assert st.loc["D_i[P]", "status"] == "below_resolvable_rate"
