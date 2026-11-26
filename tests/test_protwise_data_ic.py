"""P4 (option b): protwise initial conditions come from the t0 data, so the model reproduces the t0 observations."""
import inspect

import numpy as np
import pytest

P4 = pytest.mark.xfail(reason="P4: fixed constant initial conditions", strict=True)


@P4
@pytest.mark.parametrize("model", ["distmod", "succmod"])
def test_ic_equals_t0_data(model):
    from protwise.steady.initdata import data_initial_condition

    pr = np.array([[1.3, 1.1, 0.9]])
    ph = np.array([[0.7, 1.0, 1.2], [1.6, 1.5, 1.4]])
    y0 = data_initial_condition("G", model, 2, pr, ph, [1.0, 0.5, 0.25, 0.25])
    np.testing.assert_allclose(y0, [1.0, 1.3, 0.7, 1.6])


@P4
def test_randmod_single_site_subsets_and_fallback():
    from protwise.steady.initdata import data_initial_condition

    pr = np.array([[np.nan, 1.0]])
    ph = np.array([[0.4, 1.0], [0.9, 1.0]])
    fallback = [1.0, 0.25, 0.25, 0.25, 0.25]  # R, P, subsets {1},{2},{1,2}
    y0 = data_initial_condition("G", "randmod", 2, pr, ph, fallback)
    np.testing.assert_allclose(y0, [1.0, 0.25, 0.4, 0.9, 0.0])  # protein t0 missing -> fallback kept


@P4
def test_core_uses_data_initial_condition():
    import protwise.paramest.core as core

    assert "data_initial_condition(" in inspect.getsource(core.process_gene)
