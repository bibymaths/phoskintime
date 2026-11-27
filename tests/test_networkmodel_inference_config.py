from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

from config_loader import load_config_toml
import networkmodel.config as config
import networkmodel.runner as runner

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_KEYS_TOML = REPO_ROOT / "tests" / "fixtures" / "networkmodel_inference_keys.toml"


def test_loader_reads_inference_keys_from_given_toml():
    """load_config_toml reads the inference keys from the [networkmodel] table of the given TOML file."""
    cfg = load_config_toml(str(INFERENCE_KEYS_TOML))

    assert cfg.n_starts == 1
    assert cfg.profile_likelihood is False
    assert cfg.profile_grid_size == 5
    assert cfg.posterior_sampling is False
    assert cfg.posterior_num_warmup == 20
    assert cfg.posterior_num_samples == 30


def test_networkmodel_config_module_reads_inference_keys_from_given_toml(tmp_path):
    """networkmodel.config exposes the inference keys of the TOML named by PHOSKINTIME_NETWORKMODEL_CONFIG.

    The module reads its TOML at import time, so it is imported in a fresh subprocess
    (cwd = tmp_path) to keep the module already imported by this test session untouched.
    """
    probe = (
        "import json, networkmodel.config as c; print(json.dumps({k: getattr(c, k) for k in ("
        "'N_STARTS', 'PROFILE_LIKELIHOOD', 'PROFILE_GRID_SIZE', 'POSTERIOR_SAMPLING', "
        "'SENSITIVITY_ANALYSIS', 'HYPERPARAM_SCAN', 'POSTERIOR_NUM_WARMUP', 'POSTERIOR_NUM_SAMPLES')}))"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PHOSKINTIME_NETWORKMODEL_CONFIG=str(INFERENCE_KEYS_TOML))
    out = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True).stdout
    values = json.loads(out.strip().splitlines()[-1])

    assert values["N_STARTS"] == 1
    assert values["PROFILE_LIKELIHOOD"] is False
    assert values["PROFILE_GRID_SIZE"] == 5
    assert values["POSTERIOR_SAMPLING"] is False
    assert values["SENSITIVITY_ANALYSIS"] is False
    assert values["HYPERPARAM_SCAN"] is False
    assert values["POSTERIOR_NUM_WARMUP"] == 20
    assert values["POSTERIOR_NUM_SAMPLES"] == 30


def test_legacy_networkmodel_config_fields_are_not_exported():
    cfg = load_config_toml("config.toml")

    for field in (
            "population_size",
            "use_custom_solver",
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


def test_optimizer_key_is_the_bounded_optimizer_choice(tmp_path):
    """[networkmodel] optimizer (O01) selects the bounded optimizer; the legacy pymoo/optuna values are rejected."""
    import pytest

    cfg = load_config_toml("config.toml")
    assert cfg.optimizer == "lbfgsb"  # default L-BFGS-B since N22 P3 decision 6 (owner decision)
    text = open("config.toml").read()
    for value in ("projected_gradient", "pymoo"):
        p = tmp_path / f"c_{value}.toml"
        p.write_text(text.replace('optimizer = "lbfgsb"', f'optimizer = "{value}"', 1))
        if value == "projected_gradient":
            assert load_config_toml(p).optimizer == "projected_gradient"
        else:
            with pytest.raises(ValueError, match="optimizer must be"):
                load_config_toml(p)
