from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_model_to_sbml import discover_model_outputs  # noqa: E402


def _params(results, family="kinopt"):
    found = discover_model_outputs(results, model_family=family)
    assert found
    return found[0]


def test_kinopt_results_numeric_pollution_is_skipped(tmp_path):
    p = tmp_path / "kinopt_results.csv"
    p.write_text("time,observed,estimated,residual,rmse\n0,1.0,0.9,0.1,0.2\n1,1.2,1.1,0.1,0.2\n")
    result = _params(tmp_path)
    assert result.parameters == {}
    assert result.skipped_tables
    polluted = {"time", "observed", "estimated", "residual", "rmse"}
    assert polluted.isdisjoint(result.parameters)


def test_project_style_alpha_table_uses_cli_family_fallback(tmp_path):
    tables = tmp_path / "results" / "run_001" / "tables"
    tables.mkdir(parents=True)
    (tables / "alpha_values.csv").write_text("Protein,Psite,Kinase,Alpha\nP1,S1,K1,0.7\nP1,S1,K2,0.3\n")
    result = _params(tmp_path / "results" / "run_001")
    assert result.model_family == "kinopt"
    assert result.metadata["family_source"] == "--model-family"
    assert result.parameters["alpha__protein_P1__kinase_K1__psite_S1"] == pytest.approx(0.7)
    assert result.parameters["alpha__protein_P1__kinase_K2__psite_S1"] == pytest.approx(0.3)


def test_project_style_beta_table_uses_cli_family_fallback(tmp_path):
    tables = tmp_path / "results" / "run_001" / "tables"
    tables.mkdir(parents=True)
    (tables / "beta_values.csv").write_text("Kinase,Psite,Beta\nK1,S3,1.25\n")
    result = _params(tmp_path / "results" / "run_001")
    assert result.model_family == "kinopt"
    assert result.parameters["beta__kinase_K1__psite_S3"] == pytest.approx(1.25)


def test_workbook_sheet_filtering(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    xlsx = tmp_path / "kinopt_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame({"Protein": ["P1"], "Psite": ["S1"], "Kinase": ["K1"], "Alpha": [0.8]}).to_excel(writer, sheet_name="alpha_values", index=False)
        pd.DataFrame({"time": [0], "pred_fc": [1.2]}).to_excel(writer, sheet_name="fitted_trajectories", index=False)
        pd.DataFrame({"rmse": [0.1], "score": [3.0]}).to_excel(writer, sheet_name="model_metrics", index=False)
        pd.DataFrame({"residual": [0.2]}).to_excel(writer, sheet_name="residuals", index=False)
    result = _params(tmp_path)
    assert result.parameters == {"alpha__protein_P1__kinase_K1__psite_S1": pytest.approx(0.8)}
    skipped = {(s.sheet, s.reason) for s in result.skipped_tables}
    assert any(sheet == "fitted_trajectories" for sheet, _ in skipped)
    assert any(sheet == "model_metrics" for sheet, _ in skipped)
    assert any(sheet == "residuals" for sheet, _ in skipped)


def test_valid_parameter_table_without_cli_family_is_unknown_not_dropped(tmp_path):
    tables = tmp_path / "results" / "run_001" / "tables"
    tables.mkdir(parents=True)
    (tables / "alpha_values.csv").write_text("Protein,Psite,Kinase,Alpha\nP1,S1,K1,0.7\n")
    found = discover_model_outputs(tmp_path / "results" / "run_001")
    assert found
    result = found[0]
    assert result.model_family == "unknown"
    assert result.parameters["alpha__protein_P1__kinase_K1__psite_S1"] == pytest.approx(0.7)
    assert any("--model-family" in w for w in result.warnings)
