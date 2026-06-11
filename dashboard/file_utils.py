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
TEXT_EXTENSIONS = {".txt", ".log", ".yaml", ".yml", ".toml", ".json", ".md", ".csv", ".tsv"}

EXCLUDED_ZIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}

UPLOAD_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".yaml", ".yml", ".toml", ".json", ".txt"}
_FILENAME_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


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


def sanitize_filename(filename: str) -> str:
    """Return a safe filename while preserving a supported extension when present."""
    raw = Path(filename).name.strip().replace(" ", "-")
    cleaned = "".join(ch if ch in _FILENAME_SAFE_CHARS else "-" for ch in raw).strip(".-_")
    return cleaned or "uploaded-file"


def create_upload_dir(repo_root: str | Path, run_id: str, base_dir: str | Path = "dashboard_uploads") -> Path:
    """Create a per-run dashboard upload directory under the repository by default."""
    from dashboard.command_builder import sanitize_run_name

    root = Path(repo_root).resolve()
    base = Path(base_dir).expanduser()
    if not base.is_absolute():
        base = root / base
    directory = (base / sanitize_run_name(run_id)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def detect_duplicate_filenames(filenames: Iterable[str]) -> list[str]:
    """Return sanitized duplicate upload filenames."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in filenames:
        safe = sanitize_filename(name)
        if safe in seen:
            duplicates.add(safe)
        seen.add(safe)
    return sorted(duplicates)


def validate_upload_filename(filename: str) -> list[str]:
    """Validate a dashboard upload filename without reading content."""
    safe = sanitize_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        return [f"Unsupported extension for {filename!r}: {suffix or '<none>'}"]
    return []


def save_uploaded_file(uploaded_file, upload_dir: str | Path) -> Path:
    """Save a Streamlit-style uploaded file into the run upload directory."""
    directory = Path(upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = sanitize_filename(getattr(uploaded_file, "name", "uploaded-file"))
    problems = validate_upload_filename(name)
    if problems:
        raise ValueError("; ".join(problems))
    target = directory / name
    if hasattr(uploaded_file, "getbuffer"):
        data = bytes(uploaded_file.getbuffer())
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        data = bytes(uploaded_file)
    if not data:
        raise ValueError(f"Uploaded file is empty: {name}")
    target.write_bytes(data)
    return target


def validate_existing_file(path: str | Path) -> list[str]:
    """Detect basic file problems for saved uploads or selected paths."""
    file_path = Path(path)
    problems: list[str] = []
    if not file_path.exists():
        return [f"File does not exist: {file_path}"]
    if not file_path.is_file():
        return [f"Path is not a file: {file_path}"]
    if file_path.suffix.lower() not in UPLOAD_EXTENSIONS:
        problems.append(f"Unsupported extension: {file_path.suffix.lower() or '<none>'}")
    try:
        if file_path.stat().st_size == 0:
            problems.append("File is empty")
    except OSError as exc:
        problems.append(f"Unreadable file: {exc}")
    return problems


def preview_table(path: str | Path, max_rows: int = 50, sheet_name: str | int | None = 0):
    """Read a bounded preview of CSV/TSV/XLSX files using pandas."""
    import pandas as pd

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, nrows=max_rows)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t", nrows=max_rows)
    if suffix == ".xlsx":
        return pd.read_excel(file_path, sheet_name=0 if sheet_name is None else sheet_name, nrows=max_rows)
    raise ValueError(f"Preview is not supported for {suffix or '<none>'} files")
