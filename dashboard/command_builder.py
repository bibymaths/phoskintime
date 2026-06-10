from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashboard.registry import ArgumentSpec, WorkflowDescriptor, get_workflow

_SAFE_RUN_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class BuiltCommand:
    """A safely constructed workflow command."""

    workflow: WorkflowDescriptor
    command: list[str]
    outdir: Path
    pixi_environment: str

    @property
    def preview(self) -> str:
        return " ".join(shlex.quote(str(part)) for part in self.command)


def sanitize_run_name(name: str) -> str:
    """Return a filesystem-safe run name from dashboard input."""
    cleaned = _SAFE_RUN_CHARS.sub("-", name.strip()).strip(".-_")
    return cleaned or "run"


def build_output_dir(repo_root: str | Path, workflow_key: str, run_name: str, output_base: str | Path = "results") -> Path:
    """Build a project-local output directory for a workflow run."""
    root = Path(repo_root).resolve()
    base = Path(output_base).expanduser()
    if not base.is_absolute():
        base = root / base
    safe_name = sanitize_run_name(run_name)
    return (base / workflow_key / safe_name).resolve()


def _coerce_argument(spec: ArgumentSpec, value: Any) -> list[str]:
    if value is None or value == "":
        if spec.required:
            raise ValueError(f"Missing required argument: {spec.name}")
        return []
    if spec.kind == "bool":
        enabled = value
        if isinstance(value, str):
            enabled = value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return [spec.flag] if bool(enabled) else []
    if spec.kind == "int":
        return [spec.flag, str(int(value))]
    if spec.kind == "float":
        return [spec.flag, str(float(value))]
    return [spec.flag, str(value)]


def workflow_arguments(workflow: WorkflowDescriptor, values: dict[str, Any] | None = None) -> list[str]:
    """Convert structured argument values into a safe argv fragment."""
    values = values or {}
    accepted = {spec.name: spec for spec in workflow.accepted_arguments}
    unknown = sorted(set(values) - set(accepted))
    if unknown:
        raise ValueError(f"Unsupported arguments for {workflow.key}: {', '.join(unknown)}")

    args: list[str] = []
    for spec in workflow.accepted_arguments:
        value = values.get(spec.name, spec.default)
        args.extend(_coerce_argument(spec, value))
    return args



def arguments_from_input_assignments(workflow: WorkflowDescriptor, input_assignments: dict[str, str | Path] | None = None) -> dict[str, str]:
    """Map workflow input roles to supported CLI argument names."""
    input_assignments = input_assignments or {}
    specs = {spec.role: spec for spec in workflow.input_assignments}
    unknown = sorted(set(input_assignments) - set(specs))
    if unknown:
        raise ValueError(f"Unsupported input roles for {workflow.key}: {', '.join(unknown)}")

    mapped: dict[str, str] = {}
    for role, path in input_assignments.items():
        if path is None or str(path) == "":
            continue
        spec = specs[role]
        if spec.argument_name is None:
            continue
        mapped[spec.argument_name] = str(path)
    return mapped


def merge_argument_sources(
    workflow: WorkflowDescriptor,
    argument_values: dict[str, Any] | None = None,
    input_assignments: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Merge structured parameters with input assignments without inventing CLI options."""
    merged = dict(argument_values or {})
    for name, value in arguments_from_input_assignments(workflow, input_assignments).items():
        merged[name] = value
    return merged

def build_workflow_command(
    workflow_key: str,
    *,
    repo_root: str | Path = ".",
    pixi_environment: str = "default",
    run_name: str = "run",
    output_base: str | Path = "results",
    argument_values: dict[str, Any] | None = None,
    input_assignments: dict[str, str | Path] | None = None,
    use_pixi: bool = True,
) -> BuiltCommand:
    """Build a workflow command as argv list without shell interpolation."""
    workflow = get_workflow(workflow_key)
    if not workflow.python_module:
        raise ValueError(f"Workflow {workflow_key!r} does not define a Python module command.")

    outdir = build_output_dir(repo_root, workflow.key, run_name, output_base)
    command: list[str] = []
    if use_pixi:
        command.extend(["pixi", "run", "-e", pixi_environment])
    command.extend(["python", "-m", workflow.python_module])
    command.extend(workflow.module_args)
    merged_arguments = merge_argument_sources(workflow, argument_values, input_assignments)
    command.extend(workflow_arguments(workflow, merged_arguments))
    if workflow.output_dir_arg:
        command.extend([workflow.output_dir_arg, str(outdir)])
    return BuiltCommand(workflow=workflow, command=command, outdir=outdir, pixi_environment=pixi_environment)
