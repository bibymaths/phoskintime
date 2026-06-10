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
