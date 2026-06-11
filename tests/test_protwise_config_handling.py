from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


def _write_protwise_config(path: Path, outdir: Path) -> None:
    path.write_text(
        f'''
[paths]
data_dir = "data"
results_dir = "{outdir.parent.as_posix()}"
logs_dir = "{(outdir.parent / 'logs').as_posix()}"
ode_data_dir = "data"

[ode]
model = "succmod"
dev_test = true
y_metric = "variance"
alpha_ci = 0.9

[ode.bounds]
mRNA_prod = 11
mRNA_deg = 12
protein_prod = 13
protein_deg = 14
phospho_prod = 15
phospho_deg = 16

[ode.bootstrap]
n = 17

[ode.time]
protein = [0, 2, 4, 8]
rna = [4, 8, 16]

[ode.fit]
use_regularization = false

[ode.fit.composite_weights]
rmse = 1.1
mae = 1.2
var = 1.3
mse = 1.4
l2 = 1.5

[ode.sensitivity]
enabled = false
perturbation = 0.25

[ode.sensitivity.morris]
num_trajectories = 12
num_levels = 6

[ode.inputs]
protein_excel = "custom/protein.csv"
psite_excel = "custom/psite.xlsx"
rna_excel = "custom/rna.xlsx"

[ode.output]
out_dir_name = "{outdir.name}"
out_xlsx_name = "custom_results.xlsx"
''',
        encoding="utf-8",
    )


def test_custom_protwise_conf_is_honored(tmp_path):
    pytest.importorskip("numpy")
    from config.config import parse_args, extract_config

    conf = tmp_path / "protwise.toml"
    outdir = tmp_path / "configured-out"
    _write_protwise_config(conf, outdir)

    args = parse_args(["--conf", str(conf)])
    config = extract_config(args)

    assert config["config_source"] == "custom"
    assert Path(config["resolved_config_path"]) == conf.resolve()
    assert config["input_excel_protein"].endswith("custom/protein.csv")
    assert config["input_excel_psite"].endswith("custom/psite.xlsx")
    assert config["input_excel_rna"].endswith("custom/rna.xlsx")
    assert config["bounds"]["A"] == (0.0, 11.0)
    assert config["bounds"]["B"] == (0.0, 12.0)
    assert config["bounds"]["C"] == (0.0, 13.0)
    assert config["bounds"]["D"] == (0.0, 14.0)
    assert config["bounds"]["S(i)"] == (0.0, 15.0)
    assert config["bounds"]["D(i)"] == (0.0, 16.0)
    assert config["bootstraps"] == 17
    assert config["time_points"].tolist() == [0.0, 2.0, 4.0, 8.0]
    assert Path(config["outdir"]) == outdir


def test_cli_flags_override_custom_protwise_conf(tmp_path):
    pytest.importorskip("numpy")
    from config.config import parse_args, extract_config

    conf = tmp_path / "protwise.toml"
    configured_out = tmp_path / "configured-out"
    cli_out = tmp_path / "cli-out"
    _write_protwise_config(conf, configured_out)

    args = parse_args([
        "--conf", str(conf),
        "--A-bound", "0,99",
        "--bootstraps", "3",
        "--outdir", str(cli_out),
    ])
    config = extract_config(args)

    assert config["bounds"]["A"] == (0.0, 99.0)
    assert config["bootstraps"] == 3
    assert Path(config["outdir"]) == cli_out
    assert config["input_excel_protein"].endswith("custom/protein.csv")


def test_default_protwise_config_is_used_without_conf():
    pytest.importorskip("numpy")
    from config.config import parse_args, extract_config, default_config_path

    args = parse_args([])
    config = extract_config(args)

    assert config["config_source"] == "default"
    assert Path(config["resolved_config_path"]) == default_config_path().resolve()
    assert config["supplied_config_path"] is None


def test_config_cli_model_forwards_custom_conf(monkeypatch, tmp_path):
    pytest.importorskip("typer")
    from config import cli

    calls = []
    monkeypatch.setattr(cli, "_run", lambda cmd: calls.append(cmd))
    conf = tmp_path / "protwise.toml"
    conf.write_text("[ode]\n", encoding="utf-8")

    cli.model(conf=conf, outdir=tmp_path / "out")

    assert calls == [["protwise.runner.main", "--conf", str(conf), "--outdir", str(tmp_path / "out")]]


def test_importing_protwise_runner_does_not_parse_defaults(monkeypatch):
    import importlib

    runner = importlib.import_module("protwise.runner.main")

    assert runner._parse_config_path(["--conf", "custom.toml"]) == Path("custom.toml").resolve()


def test_protwise_run_contract_records_custom_config(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("numba")
    from config.config import parse_args, extract_config
    from protwise.runner.main import initialize_run_contract

    conf = tmp_path / "protwise.toml"
    outdir = tmp_path / "configured-out"
    _write_protwise_config(conf, outdir)
    args = parse_args(["--conf", str(conf)])
    config = extract_config(args)

    initialize_run_contract(config, args, logging.getLogger("protwise-test"))

    metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    assert (outdir / "config_resolved.yaml").is_file()
    assert metadata["supplied_config_path"] == str(conf)
    assert metadata["resolved_config_path"] == str(conf.resolve())
    assert metadata["config_source"] == "custom"
    assert metadata["effective_inputs"]["protein"].endswith("custom/protein.csv")
    assert metadata["effective_bounds"]["A"] == [0.0, 11.0]
    assert metadata["effective_bootstraps"] == 17
    assert metadata["effective_time_grid"] == [0.0, 2.0, 4.0, 8.0]
