"""Pre-merge item 5: identifiability labels. With profile likelihoods: "estimate (two-sided)", "upper limit only",
"lower limit only" or "not identifiable (flat)"; without: "not assessed", never "estimate"."""
import numpy as np
import pandas as pd
import pytest

from tests.test_parameter_status import _case

pytestmark = pytest.mark.xfail(strict=True, reason="pre-merge 5: identifiability labels")


def _profile(shape, th=0.0, f0=1.0):
    g = np.linspace(-3, 3, 13) + th
    x = g - th
    f = {"two": f0 + x ** 2, "upper": f0 + np.where(x > 0, x ** 2, 0.0), "lower": f0 + np.where(x < 0, x ** 2, 0.0),
         "flat": f0 + 1e-6 * x}[shape]
    return pd.DataFrame({"grid_value": g, "objective_value": f})


@pytest.mark.parametrize("shape,label", [("two", "estimate (two-sided)"), ("upper", "upper limit only"),
                                         ("lower", "lower limit only"), ("flat", "not identifiable (flat)")])
def test_label_from_profile(shape, label):
    from networkmodel.utils import identifiability_from_profile

    assert identifiability_from_profile(_profile(shape), theta_hat_raw=0.0, f_hat=1.0, threshold=0.5) == label


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


def test_profile_threshold_formula():
    from networkmodel.utils import profile_threshold

    # chi2_1(0.95) x data misfit per observation (pooled Gaussian approximation)
    assert np.isclose(profile_threshold(total=0.05, prior=0.01, n_obs=100), 3.841458820694124 * 0.04 / 100)
