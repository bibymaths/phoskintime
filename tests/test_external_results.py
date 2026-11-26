"""File contract for external KinOpt/TFOpt result workbooks (common/external_results.py).

The tiny fixture workbooks are written to tmp_path at test time (no binary fixtures in git).
"""
from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from common.external_results import (
    KINOPT,
    TFOPT,
    ExternalResultsError,
    ExternalResultsMissing,
    read_external_results,
)


def kinopt_sheets():
    return {
        "Alpha Values": pd.DataFrame({"Gene": ["egfr", "MAPK1"], "Psite": ["Y_1068", "T_185"],
                                      "Kinase": ["src", "MAP2K1"], "Alpha": [0.25, 0.75]}),
        "Beta Values": pd.DataFrame({"Kinase": ["SRC", "SRC", "MAP2K1"], "Psite": [None, "Y_419", None],
                                     "Beta": [1.5, 0.3, 2.0]}),
        "Estimated": pd.DataFrame({"Gene": ["EGFR"], "Psite": ["Y_1068"], **{f"x{i}": [1.0] for i in range(1, 15)}}),
    }


def tfopt_sheets():
    return {
        "Alpha Values": pd.DataFrame({"mRNA": ["fos", "JUN"], "TF": ["elk1", "ELK1"], "Value": [0.6, 0.4]}),
        "Beta Values": pd.DataFrame({"TF": ["ELK1", "ELK1"], "PSite": [None, "S_383"], "Value": [0.9, 0.1]}),
        "Estimated": pd.DataFrame({"mRNA": ["FOS"], **{f"x{i}": [1.0] for i in range(1, 10)}}),
    }


def write_workbook(path, sheets):
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return path


# --------------------------------------------------------------------------- contract module

def test_valid_workbook_returns_sheets_sha256_and_no_metadata(tmp_path):
    path = write_workbook(tmp_path / "kinopt_results.xlsx", kinopt_sheets())

    res = read_external_results(path, KINOPT, sheets=("Alpha Values", "Beta Values"))

    assert set(res.sheets) == {"Alpha Values", "Beta Values"}
    assert list(res.sheets["Alpha Values"].columns) == ["Gene", "Psite", "Kinase", "Alpha"]
    assert res.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert res.metadata is None
    prov = res.provenance(role="kinase priors")
    assert prov["status"] == "loaded" and prov["sha256"] == res.sha256 and prov["tool"] == "KinOpt"


def test_optional_metadata_sheet_is_returned(tmp_path):
    sheets = tfopt_sheets()
    sheets["Metadata"] = pd.DataFrame({"key": ["tool", "version", "git_commit"], "value": ["TFOpt", "0.1.0", "6cfc8bd"]})
    path = write_workbook(tmp_path / "tfopt_results.xlsx", sheets)

    res = read_external_results(path, TFOPT, sheets=("Alpha Values",))

    assert res.metadata == {"tool": "TFOpt", "version": "0.1.0", "git_commit": "6cfc8bd"}


@pytest.mark.parametrize("path", [None, "", "does/not/exist.xlsx"])
def test_absent_file_raises_missing(path):
    with pytest.raises(ExternalResultsMissing) as exc:
        read_external_results(path, KINOPT, sheets=("Alpha Values",))
    assert isinstance(exc.value, FileNotFoundError)
    assert "External tools" in str(exc.value)


def test_missing_sheet_names_file_and_sheet(tmp_path):
    sheets = kinopt_sheets()
    del sheets["Beta Values"]
    path = write_workbook(tmp_path / "kinopt_results.xlsx", sheets)

    with pytest.raises(ExternalResultsError) as exc:
        read_external_results(path, KINOPT, sheets=("Alpha Values", "Beta Values"))
    msg = str(exc.value)
    assert str(path) in msg and "Beta Values" in msg and "KinOpt" in msg


def test_missing_column_names_file_sheet_and_columns(tmp_path):
    sheets = tfopt_sheets()
    sheets["Alpha Values"] = sheets["Alpha Values"].drop(columns=["TF"])
    path = write_workbook(tmp_path / "tfopt_results.xlsx", sheets)

    with pytest.raises(ExternalResultsError) as exc:
        read_external_results(path, TFOPT, sheets=("Alpha Values",))
    msg = str(exc.value)
    assert str(path) in msg and "'Alpha Values'" in msg and "['TF']" in msg


def test_unreadable_file_raises_contract_error(tmp_path):
    path = tmp_path / "kinopt_results.xlsx"
    path.write_text("not an excel workbook")

    with pytest.raises(ExternalResultsError) as exc:
        read_external_results(path, KINOPT, sheets=("Alpha Values",))
    assert str(path) in str(exc.value)


def test_required_columns_override_checks_sheet_presence_only(tmp_path):
    sheets = kinopt_sheets()
    sheets["Estimated"] = sheets["Estimated"].drop(columns=["Gene", "Psite"])
    path = write_workbook(tmp_path / "kinopt_results.xlsx", sheets)

    res = read_external_results(path, KINOPT, sheets=("Estimated",), required_columns={"Estimated": ()})

    assert "Gene" not in res.sheets["Estimated"].columns


# --------------------------------------------------------------------------- networkmodel/io.py

@pytest.fixture()
def nm_io(monkeypatch):
    import networkmodel.io as io_mod
    monkeypatch.setattr(io_mod, "logger", logging.getLogger("test_external_results.networkmodel_io"))
    return io_mod


def network_args(tmp_path, kinopt, tfopt):
    kin = tmp_path / "input2.csv"
    pd.DataFrame({"GeneID": ["EGFR", "MAPK1"], "Psite": ["Y_1068", "T_185"],
                  "Kinase": ["{SRC}", "{MAP2K1, SRC}"]}).to_csv(kin, index=False)
    tf = tmp_path / "input4.csv"
    pd.DataFrame({"Source": ["ELK1", "ELK1"], "Target": ["FOS", "JUN"]}).to_csv(tf, index=False)
    ms = tmp_path / "input1.csv"
    pd.DataFrame({"GeneID": ["EGFR", "EGFR"], "Psite": ["", "Y_1068"],
                  **{f"x{i}": [1.0, 1.0] for i in range(1, 15)}}).to_csv(ms, index=False)
    return SimpleNamespace(kinase_net=str(kin), tf_net=str(tf), ms=str(ms), rna="", phospho="",
                           kinopt=None if kinopt is None else str(kinopt),
                           tfopt=None if tfopt is None else str(tfopt))


def test_networkmodel_loads_priors_from_valid_workbooks(tmp_path, nm_io):
    kin_xlsx = write_workbook(tmp_path / "kinopt_results.xlsx", kinopt_sheets())
    tf_xlsx = write_workbook(tmp_path / "tfopt_results.xlsx", tfopt_sheets())

    df_kin, df_tf, *_rest, kin_beta, tf_beta = nm_io.load_data(network_args(tmp_path, kin_xlsx, tf_xlsx))

    alpha = {(r.protein, r.psite, r.kinase): r.alpha for r in df_kin.itertuples()}
    assert alpha == {("EGFR", "Y_1068", "SRC"): 0.25, ("MAPK1", "T_185", "MAP2K1"): 0.75,
                     ("MAPK1", "T_185", "SRC"): 1.0}  # edge absent from the workbook -> 1.0
    assert kin_beta == {"SRC": 1.5, "MAP2K1": 2.0}  # only rows with an empty Psite
    assert {(r.tf, r.target): r.alpha for r in df_tf.itertuples()} == {("ELK1", "FOS"): 0.6, ("ELK1", "JUN"): 0.4}
    assert tf_beta == {"ELK1": 0.9}
    statuses = [(p["tool"], p["status"]) for p in nm_io.EXTERNAL_RESULTS_PROVENANCE]
    assert statuses == [("KinOpt", "loaded"), ("TFOpt", "loaded")]
    assert nm_io.EXTERNAL_RESULTS_PROVENANCE[0]["sha256"] == hashlib.sha256(kin_xlsx.read_bytes()).hexdigest()


def test_networkmodel_absent_workbooks_keep_fallback_and_warn(tmp_path, nm_io, caplog):
    with caplog.at_level(logging.WARNING, logger="test_external_results.networkmodel_io"):
        df_kin, df_tf, *_rest, kin_beta, tf_beta = nm_io.load_data(
            network_args(tmp_path, None, tmp_path / "missing_tfopt.xlsx"))

    assert set(df_kin["alpha"]) == {1.0} and set(df_tf["alpha"]) == {1.0}
    assert kin_beta == {} and tf_beta == {}
    assert sum("priors: none" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING) == 2
    assert [(p["tool"], p["status"], p["priors"]) for p in nm_io.EXTERNAL_RESULTS_PROVENANCE] == [
        ("KinOpt", "absent", "none"), ("TFOpt", "absent", "none")]


def test_networkmodel_malformed_workbook_raises(tmp_path, nm_io):
    sheets = kinopt_sheets()
    del sheets["Beta Values"]
    kin_xlsx = write_workbook(tmp_path / "kinopt_results.xlsx", sheets)

    with pytest.raises(ExternalResultsError, match="Beta Values"):
        nm_io.load_data(network_args(tmp_path, kin_xlsx, None))


# --------------------------------------------------------------------------- viewer inspection

def test_inspect_returns_present_sheets_and_lists_problems_without_raising(tmp_path):
    from common.external_results import inspect_external_results
    sheets = kinopt_sheets()
    sheets["Observed"] = pd.DataFrame({"GeneID": ["EGFR"], "Psite": ["Y_1068"]})
    del sheets["Beta Values"]
    sheets["Alpha Values"] = sheets["Alpha Values"].drop(columns=["Kinase"])
    path = write_workbook(tmp_path / "kinopt_results.xlsx", sheets)

    res = inspect_external_results(path, KINOPT, ("Alpha Values", "Beta Values", "Observed", "Residuals"))

    assert set(res.sheets) == {"Alpha Values", "Observed"}
    assert res.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    joined = " | ".join(res.problems)
    assert "missing contract sheet 'Beta Values'" in joined
    assert "sheet 'Alpha Values': missing contract column(s) ['Kinase']" in joined


def test_inspect_valid_workbook_has_no_problems(tmp_path):
    from common.external_results import inspect_external_results
    path = write_workbook(tmp_path / "tfopt_results.xlsx", tfopt_sheets())

    res = inspect_external_results(path, TFOPT, ("Alpha Values", "Beta Values", "Estimated"))

    assert res.problems == [] and set(res.sheets) == {"Alpha Values", "Beta Values", "Estimated"}


def test_inspect_unreadable_workbook_reports_problem(tmp_path):
    from common.external_results import inspect_external_results
    path = tmp_path / "kinopt_results.xlsx"
    path.write_bytes(b"xlsx")  # the dashboard tests use such placeholder files

    res = inspect_external_results(path, KINOPT, ("Alpha Values",))

    assert res.sheets == {} and len(res.problems) == 1 and "could not be read" in res.problems[0]
