from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _write_network_config(path: Path, outdir: Path) -> None:
    path.write_text(
        f'''
[networkmodel]
kinase_net = "custom/kinase_network.csv"
tf_net = "custom/tf_network.csv"
ms = "custom/protein.csv"
rna = "custom/rna.csv"
phospho = "custom/phospho.csv"
kinopt = "custom/kinopt.xlsx"
tfopt = "custom/tfopt.xlsx"
output_dir = "{outdir.as_posix()}"
cores = 7
seed = 123
n_gen = 44
lambda_prior = 0.31
lambda_protein = 0.41
lambda_rna = 0.51
lambda_phospho = 0.61
normalize_fc_steady = true
use_initial_condition_from_data = false
model = "sequential"
n_starts = 3
profile_likelihood = true
profile_indices = "1,2"
profile_grid_size = 5
posterior_sampling = true
posterior_num_warmup = 6
posterior_num_samples = 7
weighting_method_protein = "variance"
weighting_method_rna = "uniform"
sensitivity_metric = "total_signal"

[networkmodel.timepoints]
protein = [0, 10, 20]
rna = [4, 14, 24]
phospho_protein = [0, 5, 15]

[networkmodel.solver]
absolute_tolerance = 1e-6
relative_tolerance = 1e-5
max_timesteps = 12345
''',
        encoding="utf-8",
    )


def test_importing_runner_does_not_import_networkmodel_config_eagerly():
    sys.modules.pop("networkmodel.config", None)

    import networkmodel.runner as runner

    assert runner.parse_config_path(["--conf", "custom.toml"]) == Path("custom.toml")
    assert "networkmodel.config" not in sys.modules


def test_custom_conf_is_used_for_networkmodel_defaults(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from networkmodel import runner

    conf = tmp_path / "custom_config.toml"
    outdir = tmp_path / "configured-output"
    _write_network_config(conf, outdir)

    args, _ = runner.parse_runtime_args(["--conf", str(conf)])

    assert args.config_source == "custom"
    assert Path(args.resolved_config_path) == conf.resolve()
    assert args.kinase_net == "custom/kinase_network.csv"
    assert args.tf_net == "custom/tf_network.csv"
    assert args.output_dir == outdir.as_posix()
    assert args.cores == 7
    assert args.n_gen == 44
    assert args.lambda_prior == pytest.approx(0.31)
    assert args.lambda_protein == pytest.approx(0.41)
    assert args.lambda_rna == pytest.approx(0.51)
    assert args.lambda_phospho == pytest.approx(0.61)
    assert args.model_code == 1
    assert args.time_points_protein.tolist() == [0.0, 10.0, 20.0]
    assert args.time_points_rna.tolist() == [4.0, 14.0, 24.0]
    assert args.time_points_phospho.tolist() == [0.0, 5.0, 15.0]
    assert args.raw_config.ode_abs_tol == pytest.approx(1e-6)
    assert args.raw_config.ode_rel_tol == pytest.approx(1e-5)
    assert args.raw_config.ode_max_steps == 12345


def test_cli_flags_override_custom_conf(tmp_path):
    pytest.importorskip("numpy")
    from networkmodel import runner

    conf = tmp_path / "custom_config.toml"
    configured_outdir = tmp_path / "configured-output"
    cli_outdir = tmp_path / "cli-output"
    _write_network_config(conf, configured_outdir)

    args, _ = runner.parse_runtime_args([
        "--conf", str(conf),
        "--solver", "optuna",
        "--output-dir", str(cli_outdir),
        "--lambda-rna", "9.5",
        "--cores", "2",
    ])

    assert args.output_dir == str(cli_outdir)
    assert args.solver == "optuna"
    assert args.lambda_rna == pytest.approx(9.5)
    assert args.cores == 2
    assert args.kinase_net == "custom/kinase_network.csv"


def test_default_config_is_used_when_conf_is_omitted():
    pytest.importorskip("numpy")
    from networkmodel import runner

    args, _ = runner.parse_runtime_args([])

    assert args.config_source == "default"
    assert Path(args.resolved_config_path) == runner.DEFAULT_CONFIG_PATH.resolve()
    assert args.conf is None


def test_networkmodel_initialization_writes_resolved_config_and_metadata(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("numba")
    from networkmodel import runner

    conf = tmp_path / "custom_config.toml"
    outdir = tmp_path / "configured-output"
    _write_network_config(conf, outdir)

    args, _ = runner.parse_runtime_args(["--conf", str(conf)])
    runner.initialize_run_contract(args)

    metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    assert (outdir / "config_resolved.yaml").is_file()
    assert metadata["supplied_config_path"] == str(conf)
    assert metadata["resolved_config_path"] == str(conf.resolve())
    assert metadata["config_source"] == "custom"
    assert metadata["effective_inputs"]["kinase_net"] == "custom/kinase_network.csv"
    assert metadata["effective_settings"]["lambda_rna"] == pytest.approx(0.51)
    assert metadata["effective_settings"]["time_points_protein"] == [0.0, 10.0, 20.0]
