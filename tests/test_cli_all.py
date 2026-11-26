"""`phoskintime all`: prep -> validate external KinOpt/TFOpt result files -> model (decision X-D4).

`_run` is replaced by a recorder, so no stage is executed.
"""
from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

import config.cli as cli


def _workbook(path, id_cols, n_time):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({**{c: ["A"] for c in id_cols}, **{f"x{i}": [1.0] for i in range(1, n_time + 1)}}) \
            .to_excel(writer, sheet_name="Estimated", index=False)
    return path


def _setup(tmp_path, monkeypatch, *, kinopt=True, tfopt=True, malformed_tfopt=False):
    calls = []
    monkeypatch.setattr(cli, "_run", lambda cmd: calls.append(list(cmd)))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    if kinopt:
        _workbook(tmp_path / "data" / "kinopt_results.xlsx", ["Gene", "Psite"], 14)
    if tfopt:
        tf = tmp_path / "data" / "tfopt_results.xlsx"
        if malformed_tfopt:
            with pd.ExcelWriter(tf) as writer:
                pd.DataFrame({"mRNA": ["A"]}).to_excel(writer, sheet_name="Observed", index=False)
        else:
            _workbook(tf, ["mRNA"], 9)
    conf = tmp_path / "model.toml"
    conf.write_text('[ode.inputs]\npsite_excel = "data/kinopt_results.xlsx"\nrna_excel = "data/tfopt_results.xlsx"\n')
    return calls, conf


def test_all_runs_prep_then_model_when_result_files_are_valid(tmp_path, monkeypatch):
    calls, conf = _setup(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli.app, ["all", "--model-conf", str(conf)])

    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == ["processing.cleanup", "protwise.runner.main"]
    assert "KinOpt results OK" in result.output and "TFOpt results OK" in result.output


def test_all_stops_before_model_when_a_result_file_is_missing(tmp_path, monkeypatch):
    calls, conf = _setup(tmp_path, monkeypatch, kinopt=False)

    result = CliRunner().invoke(cli.app, ["all", "--model-conf", str(conf)])

    assert result.exit_code == 2
    assert [c[0] for c in calls] == ["processing.cleanup"]
    assert "external_tools.md" in result.output and "KinOpt" in result.output


def test_all_stops_before_model_when_a_result_file_is_malformed(tmp_path, monkeypatch):
    calls, conf = _setup(tmp_path, monkeypatch, malformed_tfopt=True)

    result = CliRunner().invoke(cli.app, ["all", "--model-conf", str(conf)])

    assert result.exit_code == 2
    assert [c[0] for c in calls] == ["processing.cleanup"]
    assert "Estimated" in result.output and "TFOpt" in result.output


def test_cli_has_no_kinopt_or_tfopt_commands():
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "kinopt" not in result.output.lower().replace("kinopt/tfopt", "")
    for name in ("prep", "model", "networkmodel", "clean", "all"):
        assert name in result.output
