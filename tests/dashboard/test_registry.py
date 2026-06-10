from __future__ import annotations

import json

from dashboard.registry import infer_workflow, registered_workflows
from dashboard.result_parser import discover_result_directory


def test_infer_workflow_from_metadata(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({"workflow": "tfopt.local"}), encoding="utf-8")

    descriptor = infer_workflow(discover_result_directory(root))

    assert descriptor.key == "tfopt.local"
    assert descriptor.label == "TFOpt local"


def test_infer_workflow_from_legacy_networkmodel_tables(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "scalar_objective.csv").write_text("x\n", encoding="utf-8")

    descriptor = infer_workflow(discover_result_directory(root))

    assert descriptor.key == "networkmodel.runner"


def test_registered_workflows_contains_unknown_fallback():
    keys = {descriptor.key for descriptor in registered_workflows()}

    assert {"kinopt.local", "tfopt.local", "protwise.runner", "networkmodel.runner", "unknown"} <= keys
