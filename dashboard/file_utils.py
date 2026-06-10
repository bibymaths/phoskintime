from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}
HTML_EXTENSIONS = {".html", ".htm"}
PLOT_EXTENSIONS = IMAGE_EXTENSIONS | HTML_EXTENSIONS
REPORT_EXTENSIONS = HTML_EXTENSIONS | {".md", ".pdf"}
TEXT_EXTENSIONS = {".txt", ".log", ".yaml", ".yml", ".json", ".md", ".csv", ".tsv"}

EXCLUDED_ZIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}


@dataclass(frozen=True)
class DisplayFile:
    """A discovered result file with paths suitable for UI labels and reading."""

    path: Path
    root: Path
    category: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def relative_path(self) -> str:
        try:
            return self.path.relative_to(self.root).as_posix()
        except ValueError:
            return self.path.as_posix()

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


def resolve_directory(path: str | Path) -> Path:
    """Resolve an existing result directory path."""
    directory = Path(path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Result directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Result path is not a directory: {directory}")
    return directory


def iter_files(directory: Path, patterns: Iterable[str] = ("*",), recursive: bool = False) -> list[Path]:
    """Return sorted files matching one or more glob patterns without reading file contents."""
    if not directory.is_dir():
        return []
    matches: set[Path] = set()
    for pattern in patterns:
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        matches.update(path for path in iterator if path.is_file())
    return sorted(matches, key=lambda path: path.as_posix().lower())


def filter_by_suffix(paths: Iterable[Path], suffixes: set[str]) -> list[Path]:
    """Filter paths by lower-case suffix and sort deterministically."""
    return sorted((p for p in paths if p.suffix.lower() in suffixes), key=lambda path: path.as_posix().lower())


def read_text_preview(path: Path, max_bytes: int = 512_000) -> tuple[str, bool]:
    """Read a bounded text preview and return whether truncation occurred."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    truncated = len(raw) > max_bytes or size > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace"), truncated


def create_result_zip(root: str | Path) -> bytes:
    """Create an in-memory ZIP archive for a result directory."""
    directory = resolve_directory(root)
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*"), key=lambda p: p.as_posix().lower()):
            if not path.is_file():
                continue
            rel = path.relative_to(directory)
            if any(part in EXCLUDED_ZIP_PARTS for part in rel.parts):
                continue
            archive.write(path, rel.as_posix())
    return buffer.getvalue()


def human_size(num_bytes: int) -> str:
    """Format a byte count for display."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"
