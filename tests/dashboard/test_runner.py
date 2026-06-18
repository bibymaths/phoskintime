from __future__ import annotations

import json
import sys

from dashboard.command_builder import build_workflow_command
from dashboard.runner import RunEvent, log_tail, run_built_command, stream_command


def test_stream_command_uses_shell_false(monkeypatch, tmp_path):
    calls = {}

    class FakeProcess:
        stdout = iter(["hello\n"])

        def wait(self):
            return 0

        def terminate(self):
            calls["terminated"] = True

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["shell"] = kwargs.get("shell")
        return FakeProcess()

    monkeypatch.setattr("dashboard.runner.subprocess.Popen", fake_popen)

    events = list(stream_command(["python", "-c", "print('hello')"], cwd=tmp_path, outdir=tmp_path / "run"))

    assert calls["shell"] is False
    assert events[-1] == RunEvent(status="success", returncode=0, outdir=(tmp_path / "run").resolve())
    assert "hello" in (tmp_path / "run" / "console.log").read_text(encoding="utf-8")


def test_run_built_command_executes_tiny_python_and_writes_provenance(tmp_path):
    built = build_workflow_command("prep", repo_root=tmp_path, run_name="tiny", use_pixi=False)
    command = [sys.executable, "-c", "print('tiny ok')"]
    built = type(built)(workflow=built.workflow, command=command, outdir=built.outdir, pixi_environment="default")

    events = list(run_built_command(built, repo_root=tmp_path))

    assert events[-1].status == "success"
    assert events[-1].returncode == 0
    assert "tiny ok" in (built.outdir / "console.log").read_text(encoding="utf-8")
    command_text = (built.outdir / "command.txt").read_text(encoding="utf-8").strip()
    assert str(command[0]) in command_text
    assert "print" in command_text
    metadata = json.loads((built.outdir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["launcher"] == "dashboard"
    assert metadata["launcher_status"] == "success"
    assert metadata["launcher_returncode"] == 0


def test_run_built_command_reports_failure_and_log_tail(tmp_path):
    built = build_workflow_command("prep", repo_root=tmp_path, run_name="fail", use_pixi=False)
    command = [sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"]
    built = type(built)(workflow=built.workflow, command=command, outdir=built.outdir, pixi_environment="default")

    events = list(run_built_command(built, repo_root=tmp_path))

    assert events[-1].status == "failure"
    assert events[-1].returncode == 3
    assert "bad" in log_tail(built.outdir)


def test_stream_command_can_report_cancelled(monkeypatch, tmp_path):
    calls = {"checks": 0}

    class FakeProcess:
        stdout = iter(["line\n"])

        def wait(self):
            return -15

        def terminate(self):
            calls["terminated"] = True

    monkeypatch.setattr("dashboard.runner.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    def cancel_after_first_line():
        calls["checks"] += 1
        return True

    events = list(stream_command(["python"], cwd=tmp_path, outdir=tmp_path / "run", cancel_check=cancel_after_first_line))

    assert calls["terminated"] is True
    assert events[-1].status == "cancelled"
