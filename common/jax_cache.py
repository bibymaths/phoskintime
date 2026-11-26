"""Persistent JAX compilation cache shared by protwise, networkmodel and the notebooks.

Compiled XLA executables are written to disk and reused by later processes, so a repeated
run skips XLA compilation (tracing still happens). The cache stores compiled code only;
it does not change any computed value.

Configuration (environment variable ``PHOSKINTIME_JAX_CACHE_DIR``):

- unset: ``<repository root>/.jax_cache`` (git-ignored);
- a path: use that directory;
- ``off``, ``0``, ``false``, ``none`` or empty: disabled.

The cache must be configured before the first compilation in a process; ``networkmodel.backend``
calls :func:`configure_persistent_compilation_cache` when it is imported.
"""
from __future__ import annotations

import os
from pathlib import Path

JAX_CACHE_ENV = "PHOSKINTIME_JAX_CACHE_DIR"
DEFAULT_JAX_CACHE_DIR = Path(__file__).resolve().parents[1] / ".jax_cache"
_DISABLED = {"", "0", "off", "false", "none", "no"}


def resolve_cache_dir(value: str | os.PathLike | None = None) -> Path | None:
    """Return the cache directory for ``value`` (or the environment), or None when disabled."""
    raw = os.environ.get(JAX_CACHE_ENV) if value is None else value
    if raw is None:
        return DEFAULT_JAX_CACHE_DIR
    text = str(raw).strip()
    if text.lower() in _DISABLED:
        return None
    return Path(text).expanduser()


def configure_persistent_compilation_cache(value: str | os.PathLike | None = None) -> Path | None:
    """Point JAX's persistent compilation cache at the configured directory.

    Returns the directory in use, or None when the cache is disabled or cannot be created.
    """
    import jax

    path = resolve_cache_dir(value)
    if path is None:
        return None
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    jax.config.update("jax_compilation_cache_dir", str(path))
    # Cache every compilation: most compilations here take 0.1-3 s, below JAX's 1 s default.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    return path
