from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STANDARD_SUBDIRS = ("tables", "plots", "logs", "reports", "artifacts")


def ensure_result_dir(outdir: str | Path) -> dict[str, Path]:
    root = Path(outdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {"root": root}
    for name in STANDARD_SUBDIRS:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p
    return paths


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_inputs(paths: Iterable[str | Path | None]) -> list[dict[str, Any]]:
    out = []
    for raw in paths:
        if raw is None:
            continue
        p = Path(raw).expanduser()
        item: dict[str, Any] = {"path": str(p)}
        if p.exists():
            try:
                item["resolved_path"] = str(p.resolve())
            except OSError:
                pass
            if p.is_file():
                item["sha256"] = file_sha256(p)
                item["size_bytes"] = p.stat().st_size
        else:
            item["missing"] = True
        out.append(item)
    return out


def package_version() -> str | None:
    try:
        import importlib.metadata as metadata
        return metadata.version("phoskintime")
    except Exception:
        pass
    try:
        import tomllib
        with Path("pixi.toml").open("rb") as fh:
            return tomllib.load(fh).get("workspace", {}).get("version")
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def pixi_environment() -> str | None:
    return os.environ.get("PIXI_ENVIRONMENT_NAME") or os.environ.get("PIXI_ENVIRONMENT")


def command_text(argv: Iterable[str] | None = None) -> str:
    args = list(sys.argv if argv is None else argv)
    return " ".join(shlex_quote(a) for a in args)


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(str(value))


def to_json_safe(value: Any) -> Any:
    """Convert values such as NumPy arrays/scalars and Paths to JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return to_json_safe(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return to_json_safe(vars(value))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _jsonable(value: Any) -> Any:
    return to_json_safe(value)


def write_command(outdir: str | Path, argv: Iterable[str] | None = None) -> Path:
    root = ensure_result_dir(outdir)["root"]
    path = root / "command.txt"
    path.write_text(command_text(argv) + "\n", encoding="utf-8")
    return path


def write_metadata(
    outdir: str | Path,
    workflow: str,
    args: Any | None = None,
    inputs: Iterable[str | Path | None] = (),
    extra: dict[str, Any] | None = None,
) -> Path:
    root = ensure_result_dir(outdir)["root"]
    metadata = {
        "workflow": workflow,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command_arguments": _jsonable(args) if args is not None else sys.argv[1:],
        "output_directory": str(root),
        "package_version": package_version(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pixi_environment": pixi_environment(),
        "inputs": describe_inputs(inputs),
    }
    if extra:
        metadata.update(_jsonable(extra))
    path = root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_resolved_config(outdir: str | Path, config: Any | None) -> Path | None:
    if config is None:
        return None
    root = ensure_result_dir(outdir)["root"]
    path = root / "config_resolved.yaml"
    try:
        import yaml  # type: ignore
        text = yaml.safe_dump(_jsonable(config), sort_keys=True)
    except Exception:
        text = json.dumps(_jsonable(config), indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return path


@contextmanager
def tee_console_log(outdir: str | Path):
    root = ensure_result_dir(outdir)["root"]
    log_path = root / "console.log"
    class Tee:
        def __init__(self, stream, fh):
            self.stream = stream
            self.fh = fh
        def write(self, data):
            self.stream.write(data)
            self.fh.write(data)
        def flush(self):
            self.stream.flush(); self.fh.flush()
    with log_path.open("a", encoding="utf-8") as fh:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(old_out, fh), Tee(old_err, fh)
        try:
            yield log_path
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def attach_file_console_logger(logger, outdir: str | Path, filename: str = "console.log"):
    import logging
    root = ensure_result_dir(outdir)["root"]
    log_path = root / filename
    abs_path = str(log_path.resolve())
    for handler in logger.handlers:
        if getattr(handler, "baseFilename", None) == abs_path:
            return log_path
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return log_path


def populate_standard_subdirs(outdir: str | Path, *, copy: bool = True) -> None:
    paths = ensure_result_dir(outdir)
    root = paths["root"]
    skip_names = set(STANDARD_SUBDIRS) | {"metadata.json", "command.txt", "console.log", "config_resolved.yaml"}
    table_ext = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".npy", ".npz"}
    plot_ext = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}
    report_names = {"report.html"}
    artifact_ext = {".pkl", ".pickle", ".joblib"}
    for p in list(root.iterdir()):
        if p.name in skip_names or p.is_dir():
            continue
        ext = p.suffix.lower()
        if p.name in report_names:
            dest_dir = paths["reports"]
        elif ext in table_ext:
            dest_dir = paths["tables"]
        elif ext in plot_ext:
            dest_dir = paths["plots"]
        elif ext in artifact_ext:
            dest_dir = paths["artifacts"]
        elif ext == ".log":
            dest_dir = paths["logs"]
        else:
            dest_dir = paths["artifacts"]
        dest = dest_dir / p.name
        if dest.exists():
            continue
        if copy:
            shutil.copy2(p, dest)
        else:
            shutil.move(str(p), str(dest))
