from __future__ import annotations

import json

from dashboard.registry import get_workflow, infer_workflow, launchable_workflows, registered_workflows
from dashboard.result_parser import discover_result_directory


def test_infer_workflow_from_metadata(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({"workflow": "tfopt.local"}), encoding="utf-8")

    descriptor = infer_workflow(discover_result_directory(root))

    assert descriptor.key in {"tfopt.local", "tfopt-local"}
    assert descriptor.label == "TFOpt local"


def test_infer_workflow_from_legacy_networkmodel_tables(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "scalar_objective.csv").write_text("x\n", encoding="utf-8")

    descriptor = infer_workflow(discover_result_directory(root))

    assert descriptor.key == "networkmodel.runner"


def test_registered_workflows_contains_result_and_launcher_entries():
    keys = {descriptor.key for descriptor in registered_workflows()}

    assert {"kinopt.local", "tfopt.local", "protwise.runner", "networkmodel.runner", "unknown"} <= keys
    assert {"prep", "kinopt-local", "tfopt-local", "protwise-model", "networkmodel", "phoskintime-all"} <= keys


def test_launchable_registry_entries_include_command_metadata():
    workflows = {workflow.key: workflow for workflow in launchable_workflows()}

    kinopt = workflows["kinopt-local"]
    assert kinopt.pixi_task == "kinopt-local"
    assert kinopt.python_module == "kinopt.local"
    assert kinopt.output_dir_arg == "--outdir"
    assert kinopt.accepted_arguments
    assert get_workflow("networkmodel").output_dir_arg == "--output-dir"


def _spec(workflow_key: str, role: str):
    workflow = get_workflow(workflow_key)
    return next(spec for spec in workflow.input_assignments if spec.role == role)


def test_protwise_protein_input_matches_csv_backend_reader():
    spec = _spec("protwise-model", "protein_file")

    assert spec.extensions == (".csv",)
    assert "CSV" in spec.label
    assert "Excel" not in spec.description


def test_networkmodel_input_specs_match_csv_backend_readers():
    roles = ("kinase_network", "tf_network", "protein_file", "rna_file", "phosphosite_file")

    for role in roles:
        spec = _spec("networkmodel", role)
        assert spec.extensions == (".csv",)
        assert "CSV" in spec.label
