# config_loader.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # py<3.11


def _project_root() -> Path:
    """
    Find repo root robustly by walking upwards until config.toml is found.
    This avoids 'root = folder containing config_loader.py' which is wrong
    when config/ is a subdir.
    """
    start = Path(__file__).resolve().parent
    for p in [start, *start.parents]:
        if (p / "config.toml").is_file():
            return p
    return start


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=8)
def load(mode: str, section: str) -> dict[str, Any]:
    root = _project_root()
    with (root / "config.toml").open("rb") as f:
        raw = tomllib.load(f)

    base = raw.get(section, {}) or {}
    modes = (base.get("modes", {}) or {})
    merged = _deep_merge(base, modes.get(mode, {}))

    merged["_paths"] = raw.get("paths", {}) or {}
    merged["_root"] = str(root)

    return merged


def ensure_dirs() -> None:
    root = _project_root()
    with (root / "config.toml").open("rb") as f:
        raw = tomllib.load(f)

    paths = raw.get("paths", {}) or {}

    data_dir = paths.get("data_dir", "data")
    results_dir = paths.get("results_dir", "results")
    logs_dir = paths.get("logs_dir", "results/logs")
    ode_data_dir = paths.get("ode_data_dir", "data")

    (root / data_dir).mkdir(parents=True, exist_ok=True)
    (root / results_dir).mkdir(parents=True, exist_ok=True)
    (root / logs_dir).mkdir(parents=True, exist_ok=True)
    (root / ode_data_dir).mkdir(parents=True, exist_ok=True)