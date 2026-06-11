from __future__ import annotations

import ast
from pathlib import Path

import pytest
from dashboard.workflow_panels.analysis import build_analysis_command, discover_analysis_outputs
from dashboard.workflow_panels.kinopt import discover_kinopt_panel
from dashboard.workflow_panels.networkmodel import discover_networkmodel_panel
from dashboard.workflow_panels.protwise import discover_protwise_panel
from dashboard.workflow_panels.tfopt import discover_tfopt_panel


def _standard_dirs(root):
    for name in ("tables", "plots", "logs", "reports", "artifacts"):
        (root / name).mkdir(parents=True, exist_ok=True)


def test_kinopt_and_tfopt_panel_discovery(tmp_path):
    kin = tmp_path / "kin"
    tf = tmp_path / "tf"
    _standard_dirs(kin)
    _standard_dirs(tf)
    (kin / "tables" / "kinopt_results.xlsx").write_bytes(b"xlsx")
    (kin / "plots" / "fit.png").write_bytes(b"png")
    (tf / "tfopt_results.xlsx").write_bytes(b"xlsx")

    kin_data = discover_kinopt_panel(kin)
    tf_data = discover_tfopt_panel(tf)

    assert kin_data.primary_result == kin.resolve() / "tables" / "kinopt_results.xlsx"
    assert len(kin_data.plots) == 1
    assert tf_data.primary_result == tf.resolve() / "tfopt_results.xlsx"
    assert not tf_data.messages


def test_missing_files_do_not_crash_panel_discovery(tmp_path):
    root = tmp_path / "empty"
    _standard_dirs(root)

    kin_data = discover_kinopt_panel(root)
    tf_data = discover_tfopt_panel(root)
    prot_data = discover_protwise_panel(root)
    net_data = discover_networkmodel_panel(root)

    assert kin_data.messages
    assert tf_data.messages
    assert prot_data.messages
    assert net_data.messages


def test_protwise_and_networkmodel_discovery(tmp_path):
    prot = tmp_path / "protwise"
    net = tmp_path / "network"
    _standard_dirs(prot)
    _standard_dirs(net)
    (prot / "tables" / "fit_residuals.csv").write_text("a\n1\n", encoding="utf-8")
    (prot / "plots" / "sensitivity.png").write_bytes(b"png")
    (net / "scalar_objective.csv").write_text("scalar_objective\n1\n", encoding="utf-8")
    (net / "pred_prot_picked.csv").write_text("protein,pred_fc\nA,1\n", encoding="utf-8")
    (net / "artifacts" / "dashboard_bundle.pkl").write_bytes(b"bundle")
    (net / "optimization").mkdir()
    (net / "optimization" / "best_fit.csv").write_text("x\n1\n", encoding="utf-8")

    prot_data = discover_protwise_panel(prot)
    net_data = discover_networkmodel_panel(net)

    assert "tables/fit_residuals.csv" in prot_data.tables
    assert len(prot_data.plots) == 1
    assert net_data.primary_result == net.resolve() / "artifacts" / "dashboard_bundle.pkl"
    assert "scalar_objective.csv" in net_data.tables
    assert "optimization/best_fit.csv" in net_data.tables


def test_analysis_command_wrapper_and_output_discovery(tmp_path):
    root = tmp_path / "run"
    _standard_dirs(root)
    command = build_analysis_command(
        "export-subnetworks",
        {"input2": "input2.csv", "input4": "input4.csv", "hops": "auto"},
        root,
    )

    assert command[:2] == ["python", "scripts/export_subnetworks.py"]
    assert "--input2" in command
    assert "--outdir" in command
    outdir = root / "artifacts" / "analysis" / "export-subnetworks"
    assert str(outdir) in command
    (outdir / "index.csv").write_text("x\n", encoding="utf-8")

    outputs = discover_analysis_outputs(root)

    assert outputs["export-subnetworks"] == [outdir / "index.csv"]


def _command_options(command):
    return {part for part in command[2:] if part.startswith("--")}


def test_advanced_analysis_commands_match_supported_cli_flags(tmp_path):
    root = tmp_path / "run"
    _standard_dirs(root)

    tf_counts = build_analysis_command(
        "tf-kin-counts",
        {"tfopt_xlsx": "tf.xlsx", "kinopt_xlsx": "kin.xlsx"},
        root,
    )
    assert _command_options(tf_counts) == {"--tfopt-xlsx", "--kinopt-xlsx", "--out-dir"}

    curve = build_analysis_command(
        "curve-similarity",
        {"tfopt_xlsx": "tf.xlsx", "kinopt_xlsx": "kin.xlsx"},
        root,
    )
    assert _command_options(curve) == {"--tfopt-xlsx", "--kinopt-xlsx", "--out-dir"}

    mechanistic = build_analysis_command(
        "mechanistic-insights",
        {
            "kinase_net": "kinase.csv",
            "tf_net": "tf.csv",
            "ms": "protein.csv",
            "rna": "rna.csv",
            "phospho": "phospho.csv",
            "kinopt": "kinopt.json",
            "tfopt": "tfopt.json",
            "cores": 2,
        },
        root,
    )
    assert "--output-dir" in mechanistic
    assert "--results-dir" not in mechanistic
    assert "--out-dir" not in mechanistic
    assert str(root) in mechanistic




def _script_argparse_flags(script: str) -> set[str]:
    tree = ast.parse(Path(script).read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return flags


def test_generated_analysis_commands_use_target_script_argparse_flags(tmp_path):
    root = tmp_path / "run"
    _standard_dirs(root)
    command_values = {
        "tf-kin-counts": {"tfopt_xlsx": "tf.xlsx", "kinopt_xlsx": "kin.xlsx"},
        "curve-similarity": {"tfopt_xlsx": "tf.xlsx", "kinopt_xlsx": "kin.xlsx"},
        "export-subnetworks": {"input2": "input2.csv", "input4": "input4.csv", "hops": "auto"},
        "protein-accumulators": {"prot": "prot.csv", "rna": "rna.csv", "threshold": 2.0},
        "mechanistic-insights": {
            "kinase_net": "kinase.csv",
            "tf_net": "tf.csv",
            "ms": "protein.csv",
            "rna": "rna.csv",
            "phospho": "phospho.csv",
            "kinopt": "kinopt.json",
            "tfopt": "tfopt.json",
            "cores": 2,
        },
        "temporal-sensitivity": {"results_dir": str(root), "samples": 8},
    }

    for task_key, values in command_values.items():
        command = build_analysis_command(task_key, values, root)
        accepted = _script_argparse_flags(command[1])
        assert _command_options(command) <= accepted, task_key


def test_analyze_tf_kin_counts_cli_uses_selected_paths(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    import importlib

    module = importlib.import_module("scripts.analyze_tf_kin_counts")
    calls = {}

    def fake_main(*, tfopt_xlsx, kinopt_xlsx, out_dir):
        calls["tfopt_xlsx"] = tfopt_xlsx
        calls["kinopt_xlsx"] = kinopt_xlsx
        calls["out_dir"] = out_dir

    monkeypatch.setattr(module, "main", fake_main)
    tf = tmp_path / "tf.xlsx"
    kin = tmp_path / "kin.xlsx"
    out = tmp_path / "out"

    module.cli(["--tfopt-xlsx", str(tf), "--kinopt-xlsx", str(kin), "--out-dir", str(out)])

    assert calls == {"tfopt_xlsx": str(tf), "kinopt_xlsx": str(kin), "out_dir": str(out)}


def test_curve_similarity_cli_uses_selected_paths(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import importlib

    module = importlib.import_module("scripts.curve_similarity")
    calls = {}

    def fake_main(*, tfopt_xlsx, kinopt_xlsx, out_dir):
        calls["tfopt_xlsx"] = tfopt_xlsx
        calls["kinopt_xlsx"] = kinopt_xlsx
        calls["out_dir"] = out_dir

    monkeypatch.setattr(module, "main", fake_main)
    tf = tmp_path / "tf.xlsx"
    kin = tmp_path / "kin.xlsx"
    out = tmp_path / "out"

    module.cli(["--tfopt-xlsx", str(tf), "--kinopt-xlsx", str(kin), "--out-dir", str(out)])

    assert calls == {"tfopt_xlsx": str(tf), "kinopt_xlsx": str(kin), "out_dir": str(out)}
