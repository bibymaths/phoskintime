from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dashboard.result_parser import ResultInventory

ArgKind = Literal["str", "int", "float", "bool", "path"]


@dataclass(frozen=True)
class ArgumentSpec:
    """Structured argument accepted by a registered workflow launcher."""

    name: str
    flag: str
    kind: ArgKind = "str"
    description: str = ""
    default: str | int | float | bool | None = None
    required: bool = False


@dataclass(frozen=True)
class InputSpec:
    """Workflow input role that can be assigned from uploads or selected files."""

    role: str
    label: str
    argument_name: str | None = None
    description: str = ""
    required: bool = False
    extensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowDescriptor:
    key: str
    label: str
    description: str
    pixi_task: str | None = None
    python_module: str | None = None
    module_args: tuple[str, ...] = ()
    accepted_arguments: tuple[ArgumentSpec, ...] = ()
    required_inputs: tuple[str, ...] = ()
    input_assignments: tuple[InputSpec, ...] = ()
    output_dir_arg: str | None = "--outdir"
    expected_output_folder: str | None = None
    result_workflow_keys: tuple[str, ...] = field(default_factory=tuple)
    safe_for_dashboard: bool = True


WORKFLOWS: dict[str, WorkflowDescriptor] = {
    # Result-browser descriptors retained for backwards-compatible metadata inference.
    "kinopt.local": WorkflowDescriptor(
        "kinopt.local", "KinOpt local", "Local kinase-phosphorylation optimisation results.",
        result_workflow_keys=("kinopt.local",), expected_output_folder="results",
    ),
    "tfopt.local": WorkflowDescriptor(
        "tfopt.local", "TFOpt local", "Local transcription-factor optimisation results.",
        result_workflow_keys=("tfopt.local",), expected_output_folder="results",
    ),
    "protwise.runner": WorkflowDescriptor(
        "protwise.runner", "ProtWise", "ProtWise ODE modelling results.",
        result_workflow_keys=("protwise.runner",), expected_output_folder="results",
    ),
    "networkmodel.runner": WorkflowDescriptor(
        "networkmodel.runner", "Network model", "Integrated global network model results.",
        result_workflow_keys=("networkmodel.runner",), expected_output_folder="results",
    ),
    "unknown": WorkflowDescriptor(
        "unknown", "Unknown workflow", "Result directory without recognised workflow metadata.",
        output_dir_arg=None, safe_for_dashboard=False,
    ),
    # Launchable workflows for Phase 2.
    "prep": WorkflowDescriptor(
        key="prep",
        label="Preprocessing",
        description="Run preprocessing cleanup using the existing processing.cleanup module.",
        pixi_task="prep",
        python_module="processing.cleanup",
        output_dir_arg=None,
        expected_output_folder="results/prep",
        result_workflow_keys=("prep",),
    ),
    "kinopt-local": WorkflowDescriptor(
        key="kinopt-local",
        label="KinOpt local",
        description="Run local kinase-phosphorylation optimisation via kinopt.local.",
        pixi_task="kinopt-local",
        python_module="kinopt.local",
        accepted_arguments=(
            ArgumentSpec("conf", "--conf", "path", "Optional TOML/YAML configuration file."),
            ArgumentSpec("lower_bound", "--lower_bound", "float", "Lower optimisation bound."),
            ArgumentSpec("upper_bound", "--upper_bound", "float", "Upper optimisation bound."),
            ArgumentSpec("loss_type", "--loss_type", "str", "Loss function name."),
            ArgumentSpec("method", "--method", "str", "Optimisation method."),
        ),
        required_inputs=("input1.csv", "input2.csv"),
        input_assignments=(
            InputSpec("config", "Config file", "conf", "Optional kinopt config file.", extensions=(".toml", ".yaml", ".yml", ".json")),
        ),
        output_dir_arg="--outdir",
        expected_output_folder="results/kinopt-local",
        result_workflow_keys=("kinopt.local",),
    ),
    "tfopt-local": WorkflowDescriptor(
        key="tfopt-local",
        label="TFOpt local",
        description="Run local transcription-factor optimisation via tfopt.local.",
        pixi_task="tfopt-local",
        python_module="tfopt.local",
        accepted_arguments=(
            ArgumentSpec("conf", "--conf", "path", "Optional TOML/YAML configuration file."),
            ArgumentSpec("lower_bound", "--lower_bound", "float", "Lower optimisation bound."),
            ArgumentSpec("upper_bound", "--upper_bound", "float", "Upper optimisation bound."),
            ArgumentSpec("loss_type", "--loss_type", "int", "Loss function identifier."),
        ),
        required_inputs=("input1.csv", "input3.csv", "input4.csv"),
        input_assignments=(
            InputSpec("config", "Config file", "conf", "Optional tfopt config file.", extensions=(".toml", ".yaml", ".yml", ".json")),
        ),
        output_dir_arg="--outdir",
        expected_output_folder="results/tfopt-local",
        result_workflow_keys=("tfopt.local",),
    ),
    "protwise-model": WorkflowDescriptor(
        key="protwise-model",
        label="ProtWise model",
        description="Run the ProtWise ODE model via protwise.runner.main.",
        pixi_task="model",
        python_module="protwise.runner.main",
        accepted_arguments=(
            ArgumentSpec("conf", "--conf", "path", "Optional model configuration file."),
            ArgumentSpec("bootstraps", "--bootstraps", "int", "Bootstrap iterations."),
            ArgumentSpec("input_excel_protein", "--input-excel-protein", "path", "Protein input CSV file."),
            ArgumentSpec("input_excel_psite", "--input-excel-psite", "path", "Phosphosite input Excel file."),
            ArgumentSpec("input_excel_rna", "--input-excel-rna", "path", "RNA input Excel file."),
        ),
        required_inputs=("protein CSV", "phosphosite Excel", "RNA Excel"),
        input_assignments=(
            InputSpec("config", "Config file", "conf", "Optional ProtWise config file.", extensions=(".toml", ".yaml", ".yml", ".json")),
            InputSpec("protein_file", "Protein CSV file", "input_excel_protein", "Protein input CSV file read by ProtWise.", extensions=(".csv",)),
            InputSpec("phosphosite_file", "Phosphosite file", "input_excel_psite", "Phosphosite input Excel file.", extensions=(".xlsx",)),
            InputSpec("rna_file", "RNA/mRNA file", "input_excel_rna", "RNA input Excel file.", extensions=(".xlsx",)),
        ),
        output_dir_arg="--outdir",
        expected_output_folder="results/protwise-model",
        result_workflow_keys=("protwise.runner",),
    ),
    "networkmodel": WorkflowDescriptor(
        key="networkmodel",
        label="Network model",
        description="Run the integrated global model via networkmodel.runner.",
        pixi_task="networkmodel",
        python_module="networkmodel.runner",
        accepted_arguments=(
            ArgumentSpec("conf", "--conf", "path", "Optional global model config file."),
            ArgumentSpec("kinase_net", "--kinase-net", "path", "Kinase network CSV."),
            ArgumentSpec("tf_net", "--tf-net", "path", "TF network CSV."),
            ArgumentSpec("ms", "--ms", "path", "Protein/MS data file."),
            ArgumentSpec("rna", "--rna", "path", "RNA data file."),
            ArgumentSpec("phospho", "--phospho", "path", "Phosphoproteomics data file."),
            ArgumentSpec("cores", "--cores", "int", "Worker/core count."),
            ArgumentSpec("n_gen", "--n-gen", "int", "Maximum optimiser iterations."),
            ArgumentSpec("seed", "--seed", "int", "Random seed."),
            ArgumentSpec("scan", "--scan", "bool", "Run hyperparameter scan."),
            ArgumentSpec("sensitivity", "--sensitivity", "bool", "Run sensitivity analysis."),
        ),
        required_inputs=("kinase network", "TF network", "MS/protein data", "RNA data", "phospho data"),
        input_assignments=(
            InputSpec("config", "Config file", "conf", "Optional networkmodel config file.", extensions=(".toml", ".yaml", ".yml", ".json")),
            InputSpec("kinase_network", "Kinase network CSV file", "kinase_net", "Kinase network CSV file read by networkmodel.", extensions=(".csv",)),
            InputSpec("tf_network", "TF network CSV file", "tf_net", "TF network CSV file read by networkmodel.", extensions=(".csv",)),
            InputSpec("protein_file", "Protein/MS CSV file", "ms", "Protein/MS CSV data file read by networkmodel.", extensions=(".csv",)),
            InputSpec("rna_file", "RNA/mRNA CSV file", "rna", "RNA CSV data file read by networkmodel.", extensions=(".csv",)),
            InputSpec("phosphosite_file", "Phosphoproteomics CSV file", "phospho", "Phosphoproteomics CSV data file read by networkmodel.", extensions=(".csv",)),
            InputSpec("previous_kinopt", "Previous KinOpt result", "kinopt", "Previous kinopt Excel result.", extensions=(".xlsx",)),
            InputSpec("previous_tfopt", "Previous TFOpt result", "tfopt", "Previous tfopt Excel result.", extensions=(".xlsx",)),
            InputSpec("networkmodel_result_dir", "Networkmodel result directory", None, "Reference result directory for browsing; not passed to CLI."),
        ),
        output_dir_arg="--output-dir",
        expected_output_folder="results/networkmodel",
        result_workflow_keys=("networkmodel.runner",),
    ),
    "phoskintime-all": WorkflowDescriptor(
        key="phoskintime-all",
        label="PhosKinTime all",
        description="Run the existing config.cli all wrapper for preprocessing, local TF/KinOpt, and ProtWise model.",
        pixi_task="phoskintime-all",
        python_module="config.cli",
        module_args=("all",),
        accepted_arguments=(
            ArgumentSpec("tf_mode", "--tf-mode", "str", "TFOpt mode.", default="local"),
            ArgumentSpec("kin_mode", "--kin-mode", "str", "KinOpt mode.", default="local"),
            ArgumentSpec("tf_conf", "--tf-conf", "path", "TFOpt config file."),
            ArgumentSpec("kin_conf", "--kin-conf", "path", "KinOpt config file."),
            ArgumentSpec("model_conf", "--model-conf", "path", "ProtWise config file."),
        ),
        input_assignments=(
            InputSpec("tf_config", "TFOpt config file", "tf_conf", "Config for TFOpt stage.", extensions=(".toml", ".yaml", ".yml", ".json")),
            InputSpec("kin_config", "KinOpt config file", "kin_conf", "Config for KinOpt stage.", extensions=(".toml", ".yaml", ".yml", ".json")),
            InputSpec("model_config", "ProtWise config file", "model_conf", "Config for ProtWise stage.", extensions=(".toml", ".yaml", ".yml", ".json")),
        ),
        output_dir_arg="--outdir",
        expected_output_folder="results/phoskintime-all",
        result_workflow_keys=("phoskintime.all",),
        safe_for_dashboard=True,
    ),
}


def get_workflow(key: str) -> WorkflowDescriptor:
    """Return a registered workflow or raise a KeyError with a helpful message."""
    try:
        return WORKFLOWS[key]
    except KeyError as exc:
        known = ", ".join(sorted(WORKFLOWS))
        raise KeyError(f"Unknown workflow {key!r}. Known workflows: {known}") from exc


def launchable_workflows() -> list[WorkflowDescriptor]:
    """Return dashboard-safe workflows that have an executable Python module."""
    return [wf for wf in sorted(WORKFLOWS.values(), key=lambda item: item.key) if wf.safe_for_dashboard and wf.python_module]


def infer_workflow(inventory: ResultInventory) -> WorkflowDescriptor:
    """Infer workflow identity from metadata first, then legacy filenames."""
    if inventory.metadata is not None:
        try:
            metadata = json.loads(inventory.metadata.read_text(encoding="utf-8"))
            workflow = str(metadata.get("workflow", "")).strip()
            for descriptor in WORKFLOWS.values():
                if workflow == descriptor.key or workflow in descriptor.result_workflow_keys:
                    return descriptor
        except (OSError, json.JSONDecodeError):
            pass

    names = {item.name for item in inventory.tables}
    if "kinopt_results.xlsx" in names:
        return WORKFLOWS["kinopt.local"]
    if "tfopt_results.xlsx" in names:
        return WORKFLOWS["tfopt.local"]
    if {"scalar_objective.csv", "pred_prot_picked.csv", "pred_rna_picked.csv", "pred_phospho_picked.csv"} & names:
        return WORKFLOWS["networkmodel.runner"]
    return WORKFLOWS["unknown"]


def registered_workflows() -> list[WorkflowDescriptor]:
    """Return all known workflow descriptors for display or tests."""
    return [WORKFLOWS[key] for key in sorted(WORKFLOWS)]


def pixi_environments(pixi_toml: str | Path = "pixi.toml") -> list[str]:
    """Return Pixi environments defined by pixi.toml, always including default first."""
    path = Path(pixi_toml)
    envs = ["default"]
    if not path.is_file():
        return envs
    try:
        import tomllib
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        configured = data.get("environments", {}) or {}
        for name in configured:
            if name not in envs:
                envs.append(str(name))
    except Exception:
        return envs
    return envs
