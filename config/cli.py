"""
Command‑line entry point for the phoskintime pipeline.

Usage
--------------
# Run from the repository root through pixi (the `phoskintime` task is
# `python -m config.cli`; see pixi.toml).

# preprocessing -> validate KinOpt/TFOpt result files -> model
pixi run phoskintime all

# run only preprocessing
pixi run phoskintime prep

# KinOpt and TFOpt are separate tools (github.com/bibymaths/KinOpt and TFOpt).
# Their result workbooks are inputs here; see the 'External tools' docs page.

# run the model
pixi run phoskintime model

# run the integrated Global Model
pixi run phoskintime networkmodel

# remove caches and build artifacts
pixi run phoskintime clean
"""
import os
import shutil
from pathlib import Path
import subprocess as sp
import sys
import typer

# …/phoskintime
ROOT = Path(__file__).resolve().parent.parent
# current venv’s python
PY = sys.executable


def _run(cmd: list[str]) -> None:
    """
    Echo + run from the package root.
    Args:
        cmd: List of command line arguments.
    Returns:
        None
    """
    typer.echo(f"$ {' '.join(cmd)}")
    sp.check_call([PY, "-m", *cmd], cwd=ROOT)


def _python_module(module: str, cfg: Path | None, outdir: Path | None = None) -> list[str]:
    """
    Return `python module [--conf path]`.

    Args:
        module: Python module to run.
        cfg: Path to the configuration file (optional).
    Returns:
        List of command line arguments.
    """
    cmd = [module]
    if cfg is not None:
        cmd += ["--conf", str(cfg)]
    if outdir is not None:
        cmd += ["--outdir", str(outdir)]
    return cmd


app = typer.Typer(help="CLI shortcuts for the phoskintime workflow")


@app.command()
def prep():
    """
    Preprocess data (processing.cleanup).
    """
    _run(["processing.cleanup"])


@app.command()
def model(
        outdir: Path | None = typer.Option(None, "--outdir", "--output-dir", file_okay=False, dir_okay=True,
                                           help="Directory for workflow outputs."),
        conf: Path | None = typer.Option(
            None, "--conf", file_okay=True, dir_okay=False, writable=False,
            help="Path to model config file. Uses defaults if omitted."
        ),
):
    """
    Run the model (runner.main).

    Args:
        conf: Path to model config file. Uses defaults if omitted.
    Returns:
        None
    """
    _run(_python_module("protwise.runner.main", conf, outdir))


@app.command()
def networkmodel(
        outdir: Path | None = typer.Option(None, "--outdir", "--output-dir", file_okay=False, dir_okay=True,
                                           help="Directory for workflow outputs."),
        conf: Path | None = typer.Option(
            "config.toml", "--conf", file_okay=True, dir_okay=False, writable=False,
            help="Path to global model config file. Uses config.toml by default."
        ),
):
    """
    Run the integrated Global Model (networkmodel.runner).

    This runs the unified optimization pipeline defined in the [networkmodel]
    section of your configuration.
    """
    # Assuming the package is named 'networkmodel' and has a 'runner.py'
    _run(_python_module("networkmodel.runner", conf, outdir))


# Caches removed anywhere in the repository (except inside SKIP_DIRS).
CLEAN_RECURSIVE_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache")
CLEAN_RECURSIVE_FILE_SUFFIXES = (".pyc", ".pyo", ".nbc")  # .nbc: Numba cache
# Build artifacts removed only at the repository top level.
CLEAN_TOPLEVEL_GLOBS = ("build", "dist", "*.egg-info")
# Never descend into these: the pixi environments and git metadata contain
# directories named build/, dist/ and *.egg-info that belong to them.
CLEAN_SKIP_DIRS = (".pixi", ".git")


def clean_targets(root: Path) -> list[Path]:
    """Return the paths `clean` removes under *root*.

    Caches are matched at any depth; build/, dist/ and *.egg-info only at the top
    level of *root*. Nothing inside .pixi/ or .git/ is ever matched.
    """
    root = Path(root)
    targets: list[Path] = []
    for pattern in CLEAN_TOPLEVEL_GLOBS:
        targets.extend(p for p in root.glob(pattern) if p.name not in CLEAN_SKIP_DIRS)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune skipped directories and directories that are removed as a whole.
        keep = []
        for d in dirnames:
            if d in CLEAN_SKIP_DIRS:
                continue
            if d in CLEAN_RECURSIVE_DIRS:
                targets.append(Path(dirpath) / d)
                continue
            keep.append(d)
        dirnames[:] = keep
        targets.extend(Path(dirpath) / f for f in filenames if f.endswith(CLEAN_RECURSIVE_FILE_SUFFIXES))
    return sorted(set(targets))


@app.command()
def clean():
    """
    Remove Python/Numba caches recursively, and build/, dist/ and *.egg-info at
    the repository top level. Never touches .pixi/ or .git/.
    """
    count = 0
    for path in clean_targets(ROOT):
        if not path.exists():  # already removed with a parent directory
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            count += 1
        except Exception as e:
            typer.echo(f"Error deleting {path}: {e}")

    typer.echo(f"Cleaned up {count} cached files and directories.")


def _external_result_paths(model_conf: Path | None) -> tuple[Path | None, Path | None]:
    """KinOpt/TFOpt workbook paths the model stage will read ([ode.inputs] psite_excel / rna_excel).

    Same resolution as protwise (config/config.py): [ode] overlaid with [ode.modes.local],
    relative paths resolved against the repository root.
    """
    import tomllib

    cfg = Path(model_conf).expanduser() if model_conf else ROOT / "config.toml"
    with cfg.open("rb") as fh:
        raw = tomllib.load(fh)
    ode = raw.get("ode", {}) or {}
    inputs = dict(ode.get("inputs", {}) or {})
    inputs.update((((ode.get("modes", {}) or {}).get("local", {}) or {}).get("inputs", {}) or {}))

    def resolve(value) -> Path | None:
        if not value:
            return None
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else ROOT / p

    return resolve(inputs.get("psite_excel")), resolve(inputs.get("rna_excel"))


def validate_external_results(model_conf: Path | None = None) -> None:
    """Check the KinOpt/TFOpt result workbooks the model stage reads; exit early if not usable.

    Applies the same check as protwise (common/external_results.py, "Estimated" sheet present).
    """
    from common.external_results import KINOPT, TFOPT, ExternalResultsError, ExternalResultsMissing, \
        read_external_results

    psite, rna = _external_result_paths(model_conf)
    for path, contract in ((psite, KINOPT), (rna, TFOPT)):
        try:
            res = read_external_results(path, contract, sheets=("Estimated",), required_columns={"Estimated": ()})
        except (ExternalResultsMissing, ExternalResultsError) as exc:
            typer.echo(f"[all] {contract.tool} result file not usable: {exc}", err=True)
            typer.echo("[all] Produce it with the standalone tool and point [ode.inputs] "
                       f"{'psite_excel' if contract is KINOPT else 'rna_excel'} at it. "
                       "See docs/Documentation/external_tools.md.", err=True)
            raise typer.Exit(code=2)
        typer.echo(f"[all] {contract.tool} results OK: {res.path} (sha256 {res.sha256})")


@app.command()
def all(
        model_conf: Path | None = typer.Option(None, help="model config file (also used to locate the "
                                                          "KinOpt/TFOpt result files, [ode.inputs])"),
        outdir: Path | None = typer.Option(None, "--outdir", "--output-dir", file_okay=False, dir_okay=True,
                                           help="Base directory for workflow outputs."),
):
    """
    Run preprocessing, check the external KinOpt/TFOpt result files, then the model.
    Preprocessing -> validate KinOpt/TFOpt results -> Model.

    KinOpt and TFOpt are separate tools; this command does not run them. If their result
    files are missing or malformed, it stops before the model stage.

    Note: This command does NOT run the global network simulation (networkmodel).
    Use ``python -m config.cli networkmodel`` (or ``pixi run networkmodel``) for that.

    Args:
        model_conf: Path to model config file. Uses defaults if omitted.
        outdir: Base directory for workflow outputs.
    Returns:
        None
    """
    prep()
    validate_external_results(model_conf)
    # typer's @app.command() returns the plain function, so call it directly
    # (the previous `<command>.callback(...)` calls raised AttributeError).
    model(conf=model_conf, outdir=(outdir / "protwise" if outdir else None))


if __name__ == "__main__":
    # Allow `python phoskintime ...`
    app()
