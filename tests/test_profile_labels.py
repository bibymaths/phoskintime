"""N13 option A: profiles are labelled as a training-objective difference (not -2 log L), report the fitted
optimum and the objective there, and flag per-point convergence."""
import numpy as np
import pandas as pd


def test_annotate_profile_columns():
    from networkmodel.BayesianInference import annotate_profile

    df = pd.DataFrame({"grid_value": [0.0, 1.0, 2.0], "objective_value": [0.3, 0.2, np.inf],
                       "delta_objective": [0.1, 0.0, np.inf], "iterations": [10, 100, -1],
                       "success": [True, True, False]})
    out = annotate_profile(df, theta_ref_raw=0.8, f_ref=0.15, theta_hat_raw=0.9, maxiter=100)
    assert out["objective_scale"].str.contains("NOT -2 log L").all()
    np.testing.assert_allclose(out["delta_vs_reference"][:2], [0.15, 0.05])
    assert out["converged"].tolist() == [True, False, False]
    assert (out["theta_hat_raw"] == 0.9).all() and (out["f_ref"] == 0.15).all() and (out["theta_ref_raw"] == 0.8).all()


def test_objective_at_reports_nan_on_failure():
    from networkmodel.BayesianInference import objective_at

    def bad(_):
        raise RuntimeError("solver failed")

    assert np.isnan(objective_at(bad, np.zeros(2)))
    assert objective_at(lambda th: (th ** 2).sum(), np.array([1.0, 2.0])) == 5.0
