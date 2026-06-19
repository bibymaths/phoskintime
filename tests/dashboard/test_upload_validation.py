from __future__ import annotations

from dashboard.components.validation_panel import validate_dashboard_setup
from dashboard.config_utils import validate_input_assignments
from dashboard.registry import get_workflow


def test_validate_input_assignments_allows_supported_config_toml(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[x]\ny = 1\n", encoding="utf-8")
    workflow = get_workflow("networkmodel")

    assert validate_input_assignments(workflow, {"config": str(config)}) == []


def test_validate_input_assignments_reports_bad_extension(tmp_path):
    bad = tmp_path / "kinase.xlsx"
    bad.write_bytes(b"x")
    workflow = get_workflow("networkmodel")

    problems = validate_input_assignments(workflow, {"kinase_network": str(bad)})

    assert any("must use one of" in problem for problem in problems)


def test_validation_panel_helper_reports_duplicate_and_unknown_role(tmp_path):
    one = tmp_path / "a file.csv"
    two = tmp_path / "a-file.csv"
    one.write_text("a\n1\n", encoding="utf-8")
    two.write_text("a\n2\n", encoding="utf-8")
    workflow = get_workflow("tfopt-local")

    problems = validate_dashboard_setup(workflow, [one, two], {"unknown": str(one)})

    assert any("Duplicate filenames" in problem for problem in problems)
    assert any("Unsupported input role" in problem for problem in problems)


def _write_file(path):
    path.write_text("col\nvalue\n", encoding="utf-8")
    return path


def _validation_problems(workflow_key: str, role: str, path):
    return validate_input_assignments(get_workflow(workflow_key), {role: str(path)})


def test_protwise_protein_input_accepts_csv_and_rejects_xlsx(tmp_path):
    csv = _write_file(tmp_path / "protein.csv")
    xlsx = _write_file(tmp_path / "protein.xlsx")

    assert _validation_problems("protwise-model", "protein_file", csv) == []
    assert any(".csv" in problem for problem in _validation_problems("protwise-model", "protein_file", xlsx))


def test_networkmodel_kinase_network_accepts_only_csv(tmp_path):
    csv = _write_file(tmp_path / "kinase.csv")
    tsv = _write_file(tmp_path / "kinase.tsv")
    xlsx = _write_file(tmp_path / "kinase.xlsx")

    assert _validation_problems("networkmodel", "kinase_network", csv) == []
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "kinase_network", tsv))
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "kinase_network", xlsx))


def test_networkmodel_tf_network_accepts_only_csv(tmp_path):
    csv = _write_file(tmp_path / "tf.csv")
    tsv = _write_file(tmp_path / "tf.tsv")
    xlsx = _write_file(tmp_path / "tf.xlsx")

    assert _validation_problems("networkmodel", "tf_network", csv) == []
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "tf_network", tsv))
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "tf_network", xlsx))


def test_networkmodel_ms_protein_data_accepts_only_csv(tmp_path):
    csv = _write_file(tmp_path / "protein.csv")
    tsv = _write_file(tmp_path / "protein.tsv")
    xlsx = _write_file(tmp_path / "protein.xlsx")

    assert _validation_problems("networkmodel", "protein_file", csv) == []
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "protein_file", tsv))
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "protein_file", xlsx))


def test_networkmodel_rna_data_accepts_only_csv(tmp_path):
    csv = _write_file(tmp_path / "rna.csv")
    tsv = _write_file(tmp_path / "rna.tsv")
    xlsx = _write_file(tmp_path / "rna.xlsx")

    assert _validation_problems("networkmodel", "rna_file", csv) == []
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "rna_file", tsv))
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "rna_file", xlsx))


def test_networkmodel_phospho_data_accepts_only_csv(tmp_path):
    csv = _write_file(tmp_path / "phospho.csv")
    tsv = _write_file(tmp_path / "phospho.tsv")
    xlsx = _write_file(tmp_path / "phospho.xlsx")

    assert _validation_problems("networkmodel", "phosphosite_file", csv) == []
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "phosphosite_file", tsv))
    assert any(".csv" in problem for problem in _validation_problems("networkmodel", "phosphosite_file", xlsx))


def test_workflow_config_inputs_accept_only_toml(tmp_path):
    toml = _write_file(tmp_path / "config.toml")
    yaml = _write_file(tmp_path / "config.yaml")
    yml = _write_file(tmp_path / "config.yml")
    json_config = _write_file(tmp_path / "config.json")

    for workflow_key, role in (
            ("kinopt-local", "config"),
            ("tfopt-local", "config"),
            ("protwise-model", "config"),
            ("networkmodel", "config"),
            ("phoskintime-all", "tf_config"),
            ("phoskintime-all", "kin_config"),
            ("phoskintime-all", "model_config"),
    ):
        assert _validation_problems(workflow_key, role, toml) == []
        for invalid in (yaml, yml, json_config):
            problems = _validation_problems(workflow_key, role, invalid)
            assert any(".toml" in problem for problem in problems)
