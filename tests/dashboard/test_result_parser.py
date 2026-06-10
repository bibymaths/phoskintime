from __future__ import annotations

import json

from dashboard.result_parser import discover_result_directory


def test_discovers_standard_result_contract(tmp_path):
    root = tmp_path / "run1"
    for dirname in ("tables", "plots", "logs", "reports", "artifacts"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps({"workflow": "networkmodel.runner"}), encoding="utf-8")
    (root / "command.txt").write_text("python -m networkmodel.runner", encoding="utf-8")
    (root / "console.log").write_text("done", encoding="utf-8")
    (root / "config_resolved.yaml").write_text("x: 1\n", encoding="utf-8")
    (root / "tables" / "summary.csv").write_text("a\n1\n", encoding="utf-8")
    (root / "tables" / "summary.tsv").write_text("a\t b\n", encoding="utf-8")
    (root / "tables" / "workbook.xlsx").write_bytes(b"placeholder")
    (root / "plots" / "fit.png").write_bytes(b"png")
    (root / "plots" / "interactive.html").write_text("<html></html>", encoding="utf-8")
    (root / "logs" / "worker.log").write_text("worker", encoding="utf-8")
    (root / "reports" / "report.md").write_text("# Report", encoding="utf-8")
    (root / "reports" / "report.pdf").write_bytes(b"pdf")
    (root / "artifacts" / "state.pkl").write_bytes(b"pickle")

    inventory = discover_result_directory(root)

    assert inventory.metadata == root.resolve() / "metadata.json"
    assert inventory.command == root.resolve() / "command.txt"
    assert inventory.console_log == root.resolve() / "console.log"
    assert inventory.config == root.resolve() / "config_resolved.yaml"
    assert {item.relative_path for item in inventory.tables} == {"tables/summary.csv", "tables/summary.tsv", "tables/workbook.xlsx"}
    assert {item.relative_path for item in inventory.plots} == {"plots/fit.png", "plots/interactive.html"}
    assert {item.relative_path for item in inventory.logs} == {"console.log", "logs/worker.log"}
    assert {item.relative_path for item in inventory.reports} == {"reports/report.md", "reports/report.pdf"}
    assert {item.relative_path for item in inventory.artifacts} == {"artifacts/state.pkl"}
    assert inventory.missing_expected == []


def test_discovers_legacy_networkmodel_outputs(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    for name in ("scalar_objective.csv", "convergence_history.csv", "pred_prot_picked.csv", "pred_rna_picked.csv", "pred_phospho_picked.csv"):
        (root / name).write_text("value\n1\n", encoding="utf-8")
    (root / "optimization").mkdir()
    (root / "optimization" / "multistart_summary.csv").write_text("x\n", encoding="utf-8")
    (root / "profiles").mkdir()
    (root / "profiles" / "profile_likelihood_summary.csv").write_text("x\n", encoding="utf-8")
    (root / "posterior").mkdir()
    (root / "posterior" / "posterior_summary.csv").write_text("x\n", encoding="utf-8")
    (root / "plots").mkdir()
    (root / "plots" / "ranked_objective.png").write_bytes(b"png")

    inventory = discover_result_directory(root)

    table_names = {item.name for item in inventory.tables}
    assert "scalar_objective.csv" in table_names
    assert "convergence_history.csv" in table_names
    assert "pred_prot_picked.csv" in table_names
    assert "multistart_summary.csv" in table_names
    assert "profile_likelihood_summary.csv" in table_names
    assert "posterior_summary.csv" in table_names
    assert {item.relative_path for item in inventory.plots} == {"plots/ranked_objective.png"}
    assert "metadata.json" in inventory.missing_expected


def test_discovers_legacy_local_excel_outputs(tmp_path):
    root = tmp_path / "legacy-local"
    root.mkdir()
    (root / "kinopt_results.xlsx").write_bytes(b"xlsx")
    (root / "tfopt_results.xlsx").write_bytes(b"xlsx")

    inventory = discover_result_directory(root)

    assert {item.name for item in inventory.tables} == {"kinopt_results.xlsx", "tfopt_results.xlsx"}
