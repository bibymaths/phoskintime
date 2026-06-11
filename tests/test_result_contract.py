from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.results import (
    STANDARD_SUBDIRS,
    ensure_result_dir,
    populate_standard_subdirs,
    write_command,
    write_metadata,
    write_resolved_config,
)


def test_result_contract_helpers_write_provenance(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("a,b\n1,2\n", encoding="utf-8")
    outdir = tmp_path / "run"

    paths = ensure_result_dir(outdir)
    write_command(outdir, ["python", "-m", "kinopt.local", "--outdir", str(outdir)])
    write_resolved_config(outdir, {"alpha": 1, "nested": {"beta": True}})
    write_metadata(outdir, "unit.workflow", args={"outdir": outdir}, inputs=[input_file])

    for name in STANDARD_SUBDIRS:
        assert paths[name].is_dir()
    assert (outdir / "command.txt").read_text(encoding="utf-8").startswith("python -m kinopt.local")
    assert (outdir / "config_resolved.yaml").is_file()

    metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["workflow"] == "unit.workflow"
    assert metadata["output_directory"] == str(outdir.resolve())
    assert metadata["python_version"]
    assert metadata["inputs"][0]["sha256"]


def test_metadata_and_resolved_config_preserve_numpy_arrays_as_lists(tmp_path):
    np = pytest.importorskip("numpy")
    outdir = tmp_path / "run"
    time_grid = np.asarray([0.0, 2.0, 4.0, 8.0], dtype=float)

    write_resolved_config(outdir, {"time_grid": time_grid, "scalar": np.float64(1.5)})
    write_metadata(outdir, "unit.workflow", extra={"effective_time_grid": time_grid})

    metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_time_grid"] == [0.0, 2.0, 4.0, 8.0]

    resolved = (outdir / "config_resolved.yaml").read_text(encoding="utf-8")
    assert "[ 0." not in resolved
    assert "0.0" in resolved
    assert "8.0" in resolved


def test_populate_standard_subdirs_copies_legacy_outputs(tmp_path):
    outdir = tmp_path / "run"
    ensure_result_dir(outdir)
    (outdir / "summary.csv").write_text("x\n1\n", encoding="utf-8")
    (outdir / "plot.png").write_bytes(b"png")
    (outdir / "report.html").write_text("<html></html>", encoding="utf-8")
    (outdir / "state.pkl").write_bytes(b"pickle")

    populate_standard_subdirs(outdir)

    assert (outdir / "tables" / "summary.csv").is_file()
    assert (outdir / "plots" / "plot.png").is_file()
    assert (outdir / "reports" / "report.html").is_file()
    assert (outdir / "artifacts" / "state.pkl").is_file()
    # Backward-compatible top-level names are retained.
    assert (outdir / "summary.csv").is_file()


def test_local_workflows_do_not_copy_outputs_to_data_ode():
    kinopt_main = Path("kinopt/local/__main__.py").read_text(encoding="utf-8")
    tfopt_main = Path("tfopt/local/__main__.py").read_text(encoding="utf-8")

    assert "ODE_DATA_DIR" not in kinopt_main
    assert "ODE_DATA_DIR" not in tfopt_main
    assert "shutil.copy" not in kinopt_main
    assert "shutil.copy" not in tfopt_main


def test_major_cli_parsers_expose_outdir_flags():
    parser_files = [
        Path("kinopt/local/config/constants.py"),
        Path("tfopt/local/config/constants.py"),
        Path("config/config.py"),
        Path("networkmodel/runner.py"),
        Path("config/cli.py"),
    ]
    for path in parser_files:
        text = path.read_text(encoding="utf-8")
        assert "--outdir" in text, path
        assert "--output-dir" in text, path
