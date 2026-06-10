from __future__ import annotations

from pathlib import Path

from dashboard.workflow_panels.analysis import ANALYSIS_TASKS, build_analysis_command, discover_analysis_outputs


def render_analysis_panel(result_dir: str | Path) -> None:
    """Compatibility wrapper for rendering advanced analysis controls."""
    from dashboard.workflow_panels.analysis import render

    render(result_dir)


__all__ = ["ANALYSIS_TASKS", "build_analysis_command", "discover_analysis_outputs", "render_analysis_panel"]
