from __future__ import annotations

import inspect

from config_loader import load_config_toml
import networkmodel.config as config
import networkmodel.runner as runner


def test_inference_config_defaults_are_loaded():
    cfg = load_config_toml("config.toml")

    assert cfg.n_starts == 1
    assert cfg.profile_likelihood is False
    assert cfg.profile_grid_size == 5
    assert cfg.posterior_sampling is False
    assert cfg.posterior_num_warmup == 20
    assert cfg.posterior_num_samples == 30

    assert config.N_STARTS == 1
    assert config.PROFILE_LIKELIHOOD is False
    assert config.PROFILE_GRID_SIZE == 5
    assert config.POSTERIOR_SAMPLING is False
    assert config.SENSITIVITY_ANALYSIS is False
    assert config.HYPERPARAM_SCAN is False
    assert config.POSTERIOR_NUM_WARMUP == 20
    assert config.POSTERIOR_NUM_SAMPLES == 30


def test_legacy_networkmodel_config_fields_are_not_exported():
    cfg = load_config_toml("config.toml")

    for field in (
        "population_size",
        "use_custom_solver",
        "optimizer",
        "study_name",
        "sampler",
        "pruner",
        "n_trials",
        "refine",
        "num_refine",
    ):
        assert not hasattr(cfg, field)

    for constant in (
        "POPULATION_SIZE",
        "USE_CUSTOM_SOLVER",
        "OPTIMIZER",
        "STUDY_NAME",
        "SAMPLER",
        "PRUNER",
        "N_TRIALS",
        "REFINE",
        "NUM_REFINE",
    ):
        assert not hasattr(config, constant)


def test_runner_wires_inference_without_new_legacy_config_imports():
    source = inspect.getsource(runner)

    assert "from networkmodel.BayesianInference import" in source
    assert "from networkmodel.inference import" not in source
    assert "InferenceContext" in source
    assert "run_multistart" in source
    assert "run_profile_likelihood" in source
    assert "run_numpyro_posterior" in source
    assert "configure_jax_parallelism" in source
    assert "parser.add_argument(\"--pop\"" not in source
    assert "parser.add_argument(\"--refine\"" not in source
