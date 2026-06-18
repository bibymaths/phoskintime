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


def test_networkmodel_combinatorial_layout_parses_parameters_and_skips_outputs(tmp_path):
    (tmp_path / "fitted_params_picked.json").write_text('{"A_i": [1.0], "B_i": [0.2], "tf_scale": 3.0}')
    (tmp_path / "model_parameters_genes.csv").write_text("protein,param,value\nP1,A_i,1.0\nP1,B_i,0.2\n")
    (tmp_path / "model_parameters_genes_psites.csv").write_text("protein,psite,param,value\nP1,S1,Dp_i,0.4\n")
    (tmp_path / "model_parameters_kinases.csv").write_text("kinase,c_k\nK1,2.5\n")
    (tmp_path / "S_rates_picked.csv").write_text("protein,psite,time,S\nP1,S1,0,0.9\n")
    (tmp_path / "model_trajectories.csv").write_text("protein,time,pred_fc\nP1,0,1.2\n")
    (tmp_path / "residuals_table.csv").write_text("protein,time,residual\nP1,0,0.1\n")
    result = discover_model_outputs(tmp_path, model_family="networkmodel")[0]
    assert result.model_family == "networkmodel"
    assert result.parameters["A_i_0"] == pytest.approx(1.0)
    assert any("model_parameters_genes" in key for key in result.parameters)
    assert any("model_parameters_genes_psites" in key for key in result.parameters)
    assert any("model_parameters_kinases" in key for key in result.parameters)
    assert any("S_rate" in key for key in result.parameters)
    skipped = {Path(s.file).name for s in result.skipped_tables}
    assert {"model_trajectories.csv", "residuals_table.csv"}.issubset(skipped)


def test_tfopt_results_workbook_sheet_filtering(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    xlsx = tmp_path / "tfopt_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame({"mRNA": ["G1"], "TF": ["T1"], "Value": [0.6]}).to_excel(writer, sheet_name="Alpha Values", index=False)
        pd.DataFrame({"TF": ["T1"], "PSite": [""], "Value": [1.0]}).to_excel(writer, sheet_name="Beta Values", index=False)
        pd.DataFrame({"mRNA": ["G1"], "x1": [1.2]}).to_excel(writer, sheet_name="Estimated", index=False)
        pd.DataFrame({"Metric": ["MSE"], "Value": [0.1]}).to_excel(writer, sheet_name="Optimization Results", index=False)
    result = discover_model_outputs(tmp_path, model_family="tfopt")[0]
    assert any(key.startswith("alpha__") for key in result.parameters)
    assert any(key.startswith("beta__") for key in result.parameters)
    assert not any("MSE" in key or "Estimated" in key for key in result.parameters)


def test_protwise_kinase_parameter_workbook_and_nonparameter_files(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    kinase_dir = tmp_path / "ABL2"
    kinase_dir.mkdir()
    with pd.ExcelWriter(kinase_dir / "ABL2_parameters.xlsx", engine="openpyxl") as writer:
        pd.DataFrame({"Time": [0, 1], "A": [0.1, 0.2], "B": [0.3, 0.4], "Regularization": [1.0, 1.0]}).to_excel(writer, index=False)
    (kinase_dir / "ABL2_confidence_intervals.csv").write_text("parameter,ci_low,ci_high\nA,0.1,0.3\n")
    (kinase_dir / "ABL2_model_fit_.png").write_bytes(b"png")
    result = discover_model_outputs(tmp_path, model_family="protwise")[0]
    assert result.parameters["A__gene_ABL2"] == pytest.approx(0.2)
    assert result.parameters["B__gene_ABL2"] == pytest.approx(0.4)
    assert not any("ci_low" in key or "ci_high" in key for key in result.parameters)
