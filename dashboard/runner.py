from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from common.results import ensure_result_dir, write_command, write_metadata
from dashboard.command_builder import BuiltCommand

RunStatus = Literal["running", "success", "failure", "cancelled"]


@dataclass(frozen=True)
class RunEvent:
    """A single launcher event for the dashboard console/status panels."""

    status: RunStatus
    line: str = ""
    returncode: int | None = None
    outdir: Path | None = None


def _merge_metadata(outdir: Path, updates: dict) -> None:
    path = outdir / "metadata.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        existing = {}
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_run_provenance(built: BuiltCommand, extra: dict | None = None) -> Path:
    """Create result folders and initial launcher provenance before subprocess execution."""
    outdir = ensure_result_dir(built.outdir)["root"]
    write_command(outdir, built.command)
    write_metadata(
        outdir,
        workflow=built.workflow.result_workflow_keys[0] if built.workflow.result_workflow_keys else built.workflow.key,
        args={
            "launcher_workflow": built.workflow.key,
            "pixi_environment": built.pixi_environment,
            "command": built.command,
        },
        extra={"launcher": "dashboard", "launcher_status": "running", **(extra or {})},
    )
    return outdir


def stream_command(
    command: list[str],
    *,
    cwd: str | Path,
    outdir: str | Path,
    env: dict[str, str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[RunEvent]:
    """Run a command with shell=False, streaming merged stdout/stderr and writing console.log."""
    root = ensure_result_dir(outdir)["root"]
    console_path = root / "console.log"
    with console_path.open("a", encoding="utf-8") as console:
        console.write(f"\n[dashboard] start {datetime.now(timezone.utc).isoformat()}\n")
        console.write("[dashboard] command " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=Path(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
            env=env,
        )
        cancelled = False
        assert process.stdout is not None
        for line in process.stdout:
            console.write(line)
            console.flush()
            yield RunEvent(status="running", line=line, outdir=root)
            if cancel_check is not None and cancel_check():
                cancelled = True
                process.terminate()
                break
        returncode = process.wait()
        if cancelled:
            status: RunStatus = "cancelled"
        else:
            status = "success" if returncode == 0 else "failure"
        console.write(f"[dashboard] {status} returncode={returncode}\n")
        console.flush()
    yield RunEvent(status=status, returncode=returncode, outdir=root)


def run_built_command(
    built: BuiltCommand,
    *,
    repo_root: str | Path,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[RunEvent]:
    """Prepare provenance, execute a built workflow command, and update run metadata."""
    outdir = prepare_run_provenance(built)
    final_event: RunEvent | None = None
    for event in stream_command(built.command, cwd=repo_root, outdir=outdir, cancel_check=cancel_check):
        final_event = event
        yield event
    if final_event is not None and final_event.status in {"success", "failure", "cancelled"}:
        _merge_metadata(
            outdir,
            {
                "launcher_status": final_event.status,
                "launcher_returncode": final_event.returncode,
                "launcher_completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def log_tail(outdir: str | Path, lines: int = 40) -> str:
    """Return a bounded console.log tail for failure display."""
    path = Path(outdir) / "console.log"
    if not path.is_file():
        return ""
    return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-lines:])
