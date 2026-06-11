from __future__ import annotations

import importlib
import pkgutil
import sys

import dashboard


def test_dashboard_modules_import_without_starting_streamlit(monkeypatch):
    """Import dashboard modules without requiring a Streamlit server or optional UI package at import time."""
    sys.modules.pop("streamlit", None)
    imported = []
    for module_info in pkgutil.walk_packages(dashboard.__path__, prefix="dashboard."):
        module = importlib.import_module(module_info.name)
        imported.append(module.__name__)

    assert "dashboard.app" in imported
    assert "dashboard.components.table_viewer" in imported
    assert "streamlit" not in sys.modules
