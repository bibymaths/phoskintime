from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dashboard.file_utils import DisplayFile
from dashboard.result_parser import discover_result_directory


@dataclass(frozen=True)
class WorkflowPanelData:
    root: Path
    primary_result: Path | None = None
    tables: dict[str, Path] = field(default_factory=dict)
    plots: list[DisplayFile] = field(default_factory=list)
    reports: list[DisplayFile] = field(default_factory=list)
    artifacts: list[DisplayFile] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.primary_result or self.tables or self.plots or self.reports or self.artifacts)


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    candidates = []
    for name in names:
        candidates.extend([root / name, root / "tables" / name, root / "artifacts" / name])
    return next((path for path in candidates if path.is_file()), None)


def _inventory(root: str | Path):
    return discover_result_directory(root)
