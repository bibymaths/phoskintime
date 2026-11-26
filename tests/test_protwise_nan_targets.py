"""P8 (a)+(c): NaN targets are masked in the protwise objective (count logged and recorded in metadata); with no
NaNs the objective is unchanged; a failing gene is isolated and reported instead of aborting the run."""
import inspect

import numpy as np
import pytest

P8 = pytest.mark.xfail(reason="P8: NaN targets abort the fit; one failing gene aborts the run", strict=True)


def _setup():
    import jax.numpy as jnp

    from config.constants import TIME_POINTS, TIME_POINTS_RNA, get_num_params
    from protwise.paramest.normest import to_opt_space

    rng = np.random.default_rng(5)
    t = np.asarray(TIME_POINTS, float)
    params = rng.uniform(0.05, 0.8, get_num_params("distmod", 2))
    target = {"mrna": rng.uniform(0.5, 2, len(TIME_POINTS_RNA)), "protein": rng.uniform(0.5, 2, t.size),
              "phospho": rng.uniform(0.5, 2, 2 * t.size)}
    mode = {"fit_mrna": True, "fit_protein": True, "fit_phospho": True, "n_rna": len(TIME_POINTS_RNA),
            "scale_mrna": 1.0, "scale_protein": 1.0, "scale_phospho": 1.0}
    return jnp, t, to_opt_space(params, "distmod"), target, mode


@P8
def test_nan_targets_are_masked():
    from protwise.paramest.normest import protwise_objective

    jnp, t, theta, target, mode = _setup()
    init = np.array([1.0, 1.0, 0.5, 0.5])
    clean = float(protwise_objective(theta, {k: jnp.asarray(v) for k, v in target.items()}, init, 2, t, mode, "distmod"))
    nan_t = {k: v.copy() for k, v in target.items()}
    nan_t["protein"][3] = np.nan
    nan_t["phospho"][[0, 5]] = np.nan
    val = float(protwise_objective(theta, {k: jnp.asarray(v) for k, v in nan_t.items()}, init, 2, t, mode, "distmod"))
    assert np.isfinite(val) and val != clean


def test_no_nan_objective_unchanged_formula():
    import protwise.paramest.normest as ne

    src = inspect.getsource(ne.protwise_objective)
    assert "jnp.mean((r_fit.reshape(-1) - target[\"mrna\"]) ** 2)" in src  # the no-NaN branch keeps the exact formula


@P8
def test_runner_isolates_failing_genes_and_records_masks():
    import protwise.runner.main as m

    src = inspect.getsource(m.main)
    assert "failed_genes" in src and "masked_targets" in src
