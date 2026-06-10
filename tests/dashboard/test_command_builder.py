from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.command_builder import build_output_dir, build_workflow_command, sanitize_run_name, workflow_arguments
from dashboard.registry import get_workflow, launchable_workflows, pixi_environments


def test_command_construction_for_each_launchable_workflow(tmp_path):
    keys = {workflow.key for workflow in launchable_workflows()}
    assert {"prep", "kinopt-local", "tfopt-local", "protwise-model", "networkmodel", "phoskintime-all"} <= keys

    for key in keys:
        built = build_workflow_command(key, repo_root=tmp_path, pixi_environment="dev", run_name="my run", use_pixi=True)
        assert built.command[:4] == ["pixi", "run", "-e", "dev"]
        assert built.command[4:7] == ["python", "-m", built.workflow.python_module]
        if built.workflow.output_dir_arg:
            assert built.workflow.output_dir_arg in built.command
            assert str(built.outdir) in built.command
        assert built.outdir == tmp_path / "results" / built.workflow.key / "my-run"


def test_builds_direct_python_command_without_pixi(tmp_path):
    built = build_workflow_command("networkmodel", repo_root=tmp_path, run_name="n", use_pixi=False, argument_values={"cores": 2, "scan": True})

    assert built.command[:3] == ["python", "-m", "networkmodel.runner"]
    assert "pixi" not in built.command
    assert ["--cores", "2"] == built.command[built.command.index("--cores"):built.command.index("--cores") + 2]
    assert "--scan" in built.command
    assert "--output-dir" in built.command


def test_phoskintime_all_places_subcommand_before_options(tmp_path):
    built = build_workflow_command("phoskintime-all", repo_root=tmp_path, run_name="all", use_pixi=False)

    assert built.command[:4] == ["python", "-m", "config.cli", "all"]
    assert "--outdir" in built.command


def test_structured_arguments_reject_unknown_values():
    workflow = get_workflow("tfopt-local")

    with pytest.raises(ValueError, match="Unsupported arguments"):
        workflow_arguments(workflow, {"raw_shell": "rm -rf ."})


def test_run_name_is_sanitized_and_output_dir_stays_under_base(tmp_path):
    assert sanitize_run_name("../../bad run!!") == "bad-run"

    outdir = build_output_dir(tmp_path, "kinopt-local", "../../bad run!!")

    assert outdir == tmp_path / "results" / "kinopt-local" / "bad-run"


def test_pixi_environment_selection_reads_defined_environments(tmp_path):
    pixi = tmp_path / "pixi.toml"
    pixi.write_text('[environments]\ndefault = {}\ndev = {}\nviz = {}\n', encoding="utf-8")

    assert pixi_environments(pixi) == ["default", "dev", "viz"]


def test_command_generation_from_assigned_inputs(tmp_path):
    kinase = tmp_path / "kinase.csv"
    config = tmp_path / "config.toml"
    kinase.write_text("a\n", encoding="utf-8")
    config.write_text("[x]\n", encoding="utf-8")

    built = build_workflow_command(
        "networkmodel",
        repo_root=tmp_path,
        run_name="assigned",
        use_pixi=False,
        input_assignments={"kinase_network": kinase, "config": config, "networkmodel_result_dir": tmp_path},
    )

    assert ["--kinase-net", str(kinase)] == built.command[built.command.index("--kinase-net"):built.command.index("--kinase-net") + 2]
    assert ["--conf", str(config)] == built.command[built.command.index("--conf"):built.command.index("--conf") + 2]
    assert "networkmodel_result_dir" not in built.command


def test_unsupported_input_roles_are_not_generated(tmp_path):
    with pytest.raises(ValueError, match="Unsupported input roles"):
        build_workflow_command("networkmodel", repo_root=tmp_path, input_assignments={"shell": "bad"})


def test_networkmodel_command_with_csv_assignments_is_unchanged(tmp_path):
    kinase = tmp_path / "kinase.csv"
    tf = tmp_path / "tf.csv"
    protein = tmp_path / "protein.csv"
    rna = tmp_path / "rna.csv"
    phospho = tmp_path / "phospho.csv"
    for path in (kinase, tf, protein, rna, phospho):
        path.write_text("col\nvalue\n", encoding="utf-8")

    built = build_workflow_command(
        "networkmodel",
        repo_root=tmp_path,
        run_name="csv-inputs",
        use_pixi=False,
        input_assignments={
            "kinase_network": kinase,
            "tf_network": tf,
            "protein_file": protein,
            "rna_file": rna,
            "phosphosite_file": phospho,
        },
    )

    assert ["--kinase-net", str(kinase)] == built.command[built.command.index("--kinase-net"):built.command.index("--kinase-net") + 2]
    assert ["--tf-net", str(tf)] == built.command[built.command.index("--tf-net"):built.command.index("--tf-net") + 2]
    assert ["--ms", str(protein)] == built.command[built.command.index("--ms"):built.command.index("--ms") + 2]
    assert ["--rna", str(rna)] == built.command[built.command.index("--rna"):built.command.index("--rna") + 2]
    assert ["--phospho", str(phospho)] == built.command[built.command.index("--phospho"):built.command.index("--phospho") + 2]
