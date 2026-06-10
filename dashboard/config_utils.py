from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardSelection:
    """Serializable dashboard workflow setup selections."""

    workflow_key: str
    run_name: str
    pixi_environment: str = "default"
    arguments: dict[str, Any] = field(default_factory=dict)
    input_assignments: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_key": self.workflow_key,
            "run_name": self.run_name,
            "pixi_environment": self.pixi_environment,
            "arguments": self.arguments,
            "input_assignments": self.input_assignments,
        }


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a conservative flat YAML subset when PyYAML is unavailable."""
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("Only simple key: value YAML is supported without PyYAML")
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        elif value == "":
            parsed = None
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip('"\'')
        data[key.strip()] = parsed
    return data


def parse_config_file(path: str | Path) -> Any:
    """Parse supported dashboard config/preset files for preview."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except ImportError:
            return _parse_simple_yaml(text)
    if suffix == ".toml":
        import tomllib
        with file_path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix == ".txt":
        return text
    raise ValueError(f"Unsupported config extension: {suffix or '<none>'}")


def selection_to_preset(selection: DashboardSelection) -> dict[str, Any]:
    """Convert dashboard selections to a preset dictionary."""
    return selection.to_dict()


def save_preset(selection: DashboardSelection, path: str | Path) -> Path:
    """Save dashboard selections as JSON or simple YAML based on extension."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = selection_to_preset(selection)
    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            text = yaml.safe_dump(data, sort_keys=True)
        except ImportError:
            lines = [f"{key}: {json.dumps(value)}" for key, value in data.items()]
            text = "\n".join(lines) + "\n"
        target.write_text(text, encoding="utf-8")
    else:
        raise ValueError("Presets must be saved as .json, .yaml, or .yml")
    return target


def load_preset(path: str | Path) -> DashboardSelection:
    """Load a saved dashboard selection preset."""
    parsed = parse_config_file(path)
    if not isinstance(parsed, dict):
        raise ValueError("Preset must contain a mapping/object")
    return DashboardSelection(
        workflow_key=str(parsed.get("workflow_key", "")),
        run_name=str(parsed.get("run_name", "run")),
        pixi_environment=str(parsed.get("pixi_environment", "default")),
        arguments=dict(parsed.get("arguments", {}) or {}),
        input_assignments=dict(parsed.get("input_assignments", {}) or {}),
    )


def validate_input_assignments(workflow, assignments: dict[str, str]) -> list[str]:
    """Validate assigned input roles against workflow specs and basic file rules."""
    problems: list[str] = []
    specs = {spec.role: spec for spec in workflow.input_assignments}
    for role in sorted(set(assignments) - set(specs)):
        problems.append(f"Unsupported input role for {workflow.key}: {role}")
    for spec in workflow.input_assignments:
        raw_path = assignments.get(spec.role, "")
        if spec.required and not raw_path:
            problems.append(f"Missing required input: {spec.label}")
            continue
        if not raw_path:
            continue
        path = Path(raw_path)
        if spec.argument_name is None:
            if not path.exists():
                problems.append(f"{spec.label} does not exist: {path}")
            continue
        if not path.exists():
            problems.append(f"{spec.label} does not exist: {path}")
            continue
        if not path.is_file():
            problems.append(f"{spec.label} is not a file: {path}")
            continue
        if spec.extensions and path.suffix.lower() not in spec.extensions:
            problems.append(f"{spec.label} must use one of {', '.join(spec.extensions)}")
        try:
            if path.stat().st_size == 0:
                problems.append(f"{spec.label}: File is empty")
        except OSError as exc:
            problems.append(f"{spec.label}: Unreadable file: {exc}")
    return problems
