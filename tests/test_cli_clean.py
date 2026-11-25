"""`phoskintime clean` removes only caches and top-level build artifacts (AUDIT X2).

Runs only against a fake tree in tmp_path; the command is never invoked on the repository.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import config.cli as cli

# Paths that must survive `clean`: everything under .pixi/ and .git/, nested build/,
# dist/ and *.egg-info directories, and ordinary files.
KEEP = [
    ".pixi/envs/dev/lib/python3.12/site-packages/pip/_internal/operations/build/wheel.py",
    ".pixi/envs/dev/share/zoneinfo/build/tzdata.zi",
    ".pixi/envs/dev/lib/python3.12/site-packages/llvmlite-0.47.0-py3.12.egg-info/PKG-INFO",
    ".pixi/envs/dev/lib/python3.12/site-packages/numpy/__pycache__/version.cpython-312.pyc",
    ".pixi/envs/dev/lib/python3.12/site-packages/numba/cached.nbc",
    ".pixi/envs/dev/dist/info.txt",
    ".git/objects/build/abc",
    ".git/hooks/__pycache__/hook.cpython-312.pyc",
    "networkmodel/sub/nested.egg-info/PKG-INFO",
    "networkmodel/build/keep.txt",
    "scripts/dist/keep.txt",
    "networkmodel/backend.py",
    "data/input1.csv",
]

# Paths that `clean` must remove.
REMOVE = [
    "build/lib/x.py",
    "dist/phoskintime-0.0.0.tar.gz",
    "phoskintime.egg-info/PKG-INFO",
    "networkmodel/__pycache__/backend.cpython-312.pyc",
    "networkmodel/sub/stale.pyc",
    "protwise/models/kernel.nbc",
    ".pytest_cache/v/cache/lastfailed",
    "tests/.mypy_cache/3.12/x.json",
]

# The top-level entries that clean_targets reports for REMOVE.
EXPECTED_TARGETS = {
    "build",
    "dist",
    "phoskintime.egg-info",
    "networkmodel/__pycache__",
    "networkmodel/sub/stale.pyc",
    "protwise/models/kernel.nbc",
    ".pytest_cache",
    "tests/.mypy_cache",
}


def _make_tree(root: Path) -> None:
    for rel in KEEP + REMOVE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def test_clean_targets_selects_only_allowed_paths(tmp_path):
    _make_tree(tmp_path)

    targets = {p.relative_to(tmp_path).as_posix() for p in cli.clean_targets(tmp_path)}

    assert targets == EXPECTED_TARGETS


def test_clean_command_removes_only_allowed_paths(tmp_path, monkeypatch):
    _make_tree(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    result = CliRunner().invoke(cli.app, ["clean"])

    assert result.exit_code == 0, result.output
    for rel in KEEP:
        assert (tmp_path / rel).exists(), f"clean removed a protected path: {rel}"
    for rel in REMOVE:
        assert not (tmp_path / rel).exists(), f"clean left behind: {rel}"
    assert f"Cleaned up {len(EXPECTED_TARGETS)} cached files and directories." in result.output
