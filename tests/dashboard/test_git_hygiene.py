from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_generated_dashboard_artifacts_are_ignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "/dashboard_uploads/" in gitignore
    assert "/results/" in gitignore or "/*results" in gitignore
    assert "logs/" in gitignore
    assert "*.zip" in gitignore
    assert "/.streamlit/" in gitignore or ".streamlit/" in gitignore


def test_no_generated_dashboard_folders_are_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()

    forbidden_prefixes = ("dashboard_uploads/", "results/", "logs/", ".streamlit/")
    assert not [path for path in tracked if path.startswith(forbidden_prefixes)]
