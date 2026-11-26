"""S7: one seed is threaded through the protwise Morris analysis and the networkmodel sensitivity
analysis (sampling AND the bootstrap confidence intervals of SALib's analyze), honouring --seed."""
import inspect

import numpy as np
import pytest

S7 = pytest.mark.xfail(reason="S7: Morris seeds not threaded through", strict=True)


def test_salib_morris_is_deterministic_for_a_fixed_seed():
    from SALib.analyze import morris as ma
    from SALib.sample import morris as ms

    problem = {"num_vars": 3, "names": ["a", "b", "c"], "bounds": [[0, 1]] * 3}

    def run(seed):
        X = ms.sample(problem, N=6, num_levels=4, local_optimization=True, seed=seed)
        Y = X @ np.array([1.0, 2.0, 0.5]) + np.sin(X[:, 0])
        return X, ma.analyze(problem, X, Y, num_levels=4, seed=seed)

    (X1, S1), (X2, S2) = run(5), run(5)
    np.testing.assert_array_equal(X1, X2)
    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(np.asarray(S1[key]), np.asarray(S2[key]))


@S7
def test_protwise_seed_reaches_morris_sample_and_analyze(monkeypatch):
    import protwise.sensitivity.analysis as an

    assert "seed" in inspect.signature(an._sensitivity_analysis).parameters
    src = inspect.getsource(an._sensitivity_analysis)
    assert "seed=seed" in src.split("morris.sample(")[1].split(")")[0]
    assert "seed=seed" in src.split("Si = analyze(")[1].split(")")[0]


@S7
def test_protwise_cli_seed_reaches_process_gene():
    from config.config import extract_config, parse_args
    from protwise.paramest import core

    cfg = extract_config(parse_args(["--seed", "123"]))
    assert cfg["seed"] == 123
    assert "seed" in inspect.signature(core.process_gene).parameters
    assert "seed" in inspect.signature(core.process_gene_wrapper).parameters


@S7
def test_networkmodel_sensitivity_takes_the_run_seed():
    import networkmodel.sensitivity as ns

    assert "seed" in inspect.signature(ns.run_sensitivity_analysis).parameters
    src = inspect.getsource(ns.run_sensitivity_analysis)
    assert "seed=seed" in src.split("morris.sample(")[1].split(")")[0]
    assert "seed=seed" in src.split("analyze(")[1].split(")")[0]
    runner_src = inspect.getsource(__import__("networkmodel.runner", fromlist=["main"]).main)
    call = runner_src.split("run_sensitivity_analysis(")[1].split(")")[0]
    assert "seed=args.seed" in call
