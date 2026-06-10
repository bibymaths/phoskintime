from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dashboard.file_utils import (
    DisplayFile,
    PLOT_EXTENSIONS,
    REPORT_EXTENSIONS,
    TABLE_EXTENSIONS,
    filter_by_suffix,
    iter_files,
    resolve_directory,
)

LEGACY_TABLE_NAMES = {
    "scalar_objective.csv",
    "convergence_history.csv",
    "pred_prot_picked.csv",
    "pred_rna_picked.csv",
    "pred_phospho_picked.csv",
    "kinopt_results.xlsx",
    "tfopt_results.xlsx",
}
LEGACY_ANALYSIS_DIRS = ("optimization", "profiles", "posterior", "plots")
PROVENANCE_FILES = ("metadata.json", "command.txt", "console.log", "config_resolved.yaml")


@dataclass(frozen=True)
class ResultInventory:
    """Lazy inventory of files in a PhosKinTime result directory."""

    root: Path
    metadata: Path | None = None
    command: Path | None = None
    console_log: Path | None = None
    config: Path | None = None
    tables: list[DisplayFile] = field(default_factory=list)
    plots: list[DisplayFile] = field(default_factory=list)
    logs: list[DisplayFile] = field(default_factory=list)
    reports: list[DisplayFile] = field(default_factory=list)
    artifacts: list[DisplayFile] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return any((self.metadata, self.command, self.console_log, self.config, self.tables, self.plots, self.logs, self.reports, self.artifacts))


def _display_files(root: Path, paths: list[Path], category: str) -> list[DisplayFile]:
    seen: set[Path] = set()
    files: list[DisplayFile] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        files.append(DisplayFile(path=path, root=root, category=category))
    return files


def _top_level_legacy_tables(root: Path) -> list[Path]:
    return sorted((root / name for name in LEGACY_TABLE_NAMES if (root / name).is_file()), key=lambda p: p.name.lower())


def _legacy_dir_files(root: Path, suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for dirname in LEGACY_ANALYSIS_DIRS:
        files.extend(filter_by_suffix(iter_files(root / dirname, recursive=True), suffixes))
    return files


def discover_result_directory(path: str | Path) -> ResultInventory:
    """Discover dashboard-readable files in a result directory without loading file contents."""
    root = resolve_directory(path)

    metadata = root / "metadata.json" if (root / "metadata.json").is_file() else None
    command = root / "command.txt" if (root / "command.txt").is_file() else None
    console_log = root / "console.log" if (root / "console.log").is_file() else None
    config = root / "config_resolved.yaml" if (root / "config_resolved.yaml").is_file() else None

    table_paths = filter_by_suffix(iter_files(root / "tables"), TABLE_EXTENSIONS)
    table_paths.extend(_top_level_legacy_tables(root))
    table_paths.extend(_legacy_dir_files(root, TABLE_EXTENSIONS))

    plot_paths = filter_by_suffix(iter_files(root / "plots"), PLOT_EXTENSIONS)
    plot_paths.extend(_legacy_dir_files(root, PLOT_EXTENSIONS))
    plot_paths.extend(filter_by_suffix([p for p in root.iterdir() if p.is_file()], PLOT_EXTENSIONS))

    log_paths = iter_files(root / "logs")
    if console_log is not None:
        log_paths.insert(0, console_log)

    report_paths = filter_by_suffix(iter_files(root / "reports"), REPORT_EXTENSIONS)
    report_paths.extend(
        filter_by_suffix(
            [p for p in root.iterdir() if p.is_file() and p.stem.lower() == "report"],
            REPORT_EXTENSIONS,
        )
    )

    artifact_paths = iter_files(root / "artifacts")

    missing = [name for name in PROVENANCE_FILES if not (root / name).is_file()]
    for dirname in ("tables", "plots", "logs", "reports", "artifacts"):
        if not (root / dirname).is_dir():
            missing.append(f"{dirname}/")

    return ResultInventory(
        root=root,
        metadata=metadata,
        command=command,
        console_log=console_log,
        config=config,
        tables=_display_files(root, table_paths, "table"),
        plots=_display_files(root, plot_paths, "plot"),
        logs=_display_files(root, log_paths, "log"),
        reports=_display_files(root, report_paths, "report"),
        artifacts=_display_files(root, artifact_paths, "artifact"),
        missing_expected=missing,
    )
