from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # py<3.11


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


@lru_cache(maxsize=4)
def load(mode: str, section: str) -> dict[str, Any]:
    root = _project_root()
    with (root / "config.toml").open("rb") as f:
        raw = tomllib.load(f)

    base = raw.get(section, {}) or {}
    modes = (base.get("modes", {}) or {})
    merged = _deep_merge(base, modes.get(mode, {}))
    merged["_paths"] = raw.get("paths", {}) or {}

    # ADD THIS LINE
    merged["_root"] = str(root)

    return merged



def ensure_dirs() -> None:
    root = _project_root()
    cfg = load("local", "tfopt")  # just to get `_paths`
    paths = cfg["_paths"]
    (root / paths.get("out_dir", "results")).mkdir(parents=True, exist_ok=True)
    (root / paths.get("data_dir", "data")).mkdir(parents=True, exist_ok=True)
    (root / paths.get("logs_dir", "results/logs")).mkdir(parents=True, exist_ok=True)
    (root / paths.get("ode_data_dir", "data/ode")).mkdir(parents=True, exist_ok=True)